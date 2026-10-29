"""Обучение LoRA по эпохам: лосс на train и valid по ходу, снимок вставок после каждой эпохи, лучшая — по valid.

LR после разгона не снижаем — так видно момент, когда лосс на train ещё падает, а на valid уже растёт.
Запуск без страницы: python -m pirate.train
"""
import datetime
import json
import random
import shutil
import time

import peft
import torch
import transformers

from app import config as base_config
from app import machine, model
from pirate import config, dataset, rules

PROBES = [  # строки экзамена — модель их не учит; ответы снимаем после каждой эпохи
    "Как зовут твоего пернатого друга?",
    "Ты сегодня какой-то хмурый",
    "Сколько будет 17 + 25? Ответь числом.",
]
CHUNK = 64  # примеров в куске, внутри которого батчи собираются по длине


def prompt(text, system=None):
    """Запрос в шаблоне чата Qwen3.5 — ровно то, что модель видит перед ответом."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    return model.tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)


def encode(row):
    """Токены примера и метки: у реплики игрока -100 (модель её только читает), у ответа — сами токены."""
    tok = model.tokenizer()
    head = tok(prompt(row["text"]), add_special_tokens=False).input_ids
    tail = tok(rules.answer(*row["answer"].values()) + tok.eos_token, add_special_tokens=False).input_ids
    return head + tail, [-100] * len(head) + tail


def probes():
    rows = {r["text"]: r for r in dataset.exam()}
    return [rows[p] for p in PROBES]


def run(stop):
    """События: start → check (шаг 0) → step… check… epoch… → done; «стоп» — stopped, снимки удаляются."""
    started = time.perf_counter()
    train, valid = dataset.load("train"), dataset.load("valid")
    rng = random.Random(config.SEED)
    torch.manual_seed(config.SEED)
    epochs = [_batches(train, rng) for _ in range(config.EPOCHS)]
    ends = [sum(len(b) for b in epochs[:e + 1]) for e in range(config.EPOCHS)]
    total = ends[-1]
    checks = sorted({start + round(len(b) * j / config.CHECKS) for start, b in zip([0] + ends, epochs)
                     for j in range(1, config.CHECKS + 1)})
    yield {"type": "start", "total": total, "epochs": config.EPOCHS, "ends": ends, "checks": checks,
           "train": len(train), "valid": len(valid), "probes": probes()}

    work = config.DIR / "epochs.new"  # снимки пишем рядом, на место старых — только в конце
    shutil.rmtree(work, ignore_errors=True)
    net = model.with_lora(model.load())
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.LR, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / config.WARMUP))
    valid_batches = _batches(valid, random.Random(0))
    sample = _batches(random.Random(1).sample(train, len(valid)), random.Random(0))  # train глазами valid — для шага 0
    curve = [{"step": 0, "epoch": 0, "train": _loss(net, sample), "valid": _loss(net, valid_batches)}]
    yield {"type": "check", **curve[-1]}
    snapshots = [{"epoch": 0, "step": 0, "answers": _probe(net)}]
    yield {"type": "epoch", **snapshots[-1]}

    steps, step, since = [], 0, []
    for e, batches in enumerate(epochs, 1):
        for batch in batches:
            if stop.is_set():
                shutil.rmtree(work, ignore_errors=True)
                yield {"type": "stopped", "step": step}
                return
            net.train()
            ids, mask, labels = _tensors(batch)
            loss = net(input_ids=ids, attention_mask=mask, labels=labels).loss  # ошибка на токенах ответа
            loss.backward()                                                     # градиенты — только у вставок
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            step += 1
            steps.append([step, round(loss.item(), 4)])
            since.append(loss.item())
            yield {"type": "step", "step": step, "loss": steps[-1][1], "seconds": round(time.perf_counter() - started, 1)}
            if step in checks:  # train — средний лосс шагов с прошлого замера, valid — на примерах вне обучения
                curve.append({"step": step, "epoch": round(e - 1 + (step - ([0] + ends)[e - 1]) / len(batches), 2),
                              "train": round(sum(since) / len(since), 4), "valid": _loss(net, valid_batches)})
                since = []
                yield {"type": "check", **curve[-1]}
        net.save_pretrained(work / str(e))
        snapshots.append({"epoch": e, "step": step, "answers": _probe(net)})
        yield {"type": "epoch", **snapshots[-1]}

    valid_at = {c["epoch"]: c["valid"] for c in curve}
    best = min(range(1, config.EPOCHS + 1), key=lambda e: valid_at[e])
    shutil.rmtree(config.EPOCHS_DIR, ignore_errors=True)
    work.rename(config.EPOCHS_DIR)
    size = sum(f.stat().st_size for f in (config.EPOCHS_DIR / str(best)).iterdir())
    passport = _passport(len(train), len(valid), total, round(time.perf_counter() - started), size)
    log = {"date": passport["date"], "passport": passport, "total": total, "ends": ends, "best": best,
           "last": config.EPOCHS, "probes": probes(), "steps": steps, "curve": curve, "snapshots": snapshots}
    config.RUN_FILE.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
    yield {"type": "done", "best": best, "passport": passport}


def last():
    """Журнал последнего законченного обучения или None."""
    if not config.RUN_FILE.exists():
        return None
    return json.loads(config.RUN_FILE.read_text(encoding="utf-8"))


def _batches(rows, rng):
    """Батчи из примеров похожей длины — меньше заглушек, быстрее; порядок батчей случайный."""
    rows = rows[:]
    rng.shuffle(rows)
    batches = []
    for i in range(0, len(rows), CHUNK):
        chunk = sorted(rows[i:i + CHUNK], key=lambda r: len(encode(r)[0]))
        batches += [chunk[j:j + config.BATCH] for j in range(0, len(chunk), config.BATCH)]
    rng.shuffle(batches)
    return batches


def _tensors(batch):
    """Пачка в тензоры: справа заглушки pad — маска 0 и метка -100, в лосс не идут."""
    encoded = [encode(row) for row in batch]
    pad = model.tokenizer().pad_token_id
    width = max(len(ids) for ids, _ in encoded)
    ids = torch.tensor([x + [pad] * (width - len(x)) for x, _ in encoded])
    mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in encoded])
    labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in encoded])
    return ids, mask, labels


def _loss(net, batches):
    """Средний лосс на токенах ответа — без обучения и без случайного выключения связей."""
    net.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            ids, mask, labels = _tensors(batch)
            tokens = int((labels[:, 1:] != -100).sum())
            total += net(input_ids=ids, attention_mask=mask, labels=labels).loss.item() * tokens
            count += tokens
    return round(total / count, 4)


def _probe(net):
    """Ответы на пробы с проверкой — как модель отвечает на этой эпохе."""
    net.eval()
    rows = probes()
    answers = model.generate(net, [prompt(r["text"]) for r in rows], 80)
    return [{"answer": a, **rules.check(r, a)} for r, a in zip(rows, answers)]


def _passport(train, valid, total, seconds, size):
    """Параметры запуска — всё, чтобы повторить обучение."""
    return {
        "date": datetime.datetime.now().isoformat(timespec="minutes"),
        "model": base_config.MODEL,
        "dataset": "datasets/pirate — свой: генератор ответов + досье из 20 придуманных фактов",
        "train": train,
        "val": valid,
        "epochs": config.EPOCHS,
        "batch": config.BATCH,
        "steps": total,
        "lr": config.LR,
        "warmup": config.WARMUP,
        "lora": {"r": base_config.LORA_R, "alpha": base_config.LORA_ALPHA, "dropout": base_config.LORA_DROPOUT,
                 "targets": "all-linear"},
        "precision": "веса bf16, вставки fp32",
        "machine": machine.info(),
        "threads": base_config.THREADS,
        "seconds": seconds,
        "adapter": str(config.EPOCHS_DIR.relative_to(config.ROOT) if config.EPOCHS_DIR.is_relative_to(config.ROOT)
                       else config.EPOCHS_DIR),
        "adapter_mb": round(size / 2**20, 1),
        "seed": config.SEED,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }


if __name__ == "__main__":
    import threading

    for event in run(threading.Event()):
        if event["type"] != "step" or event["step"] % 10 == 0:
            print(json.dumps(event, ensure_ascii=False)[:400], flush=True)
