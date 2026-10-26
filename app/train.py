"""Обучение LoRA: примеры → батчи → лосс → шаг оптимизатора; по пути — лосс на новых примерах и снимки ответов.

Запуск без страницы: python -m app.train
"""
import datetime
import functools
import json
import math
import random
import shutil
import time

import peft
import torch
import transformers

from app import config, data, machine, model, scan

PROBES = [  # запросы, ответы на которые снимаем по ходу обучения, — кадры таймлапса; команды — из теста SCAN
    {"title": "Простая команда", "text": "run left after jump twice"},
    {"title": "Сложная команда", "text": "walk around right and turn opposite left twice"},
    {"title": "Обычный вопрос", "text": "What is the capital of France? Answer in one word.", "expect": "Paris|Париж"},
]
PROBE_TOKENS = 40  # хватает на 12 действий сложной команды; у базы — начало ответа
CHUNK = 64         # примеров в куске, внутри которого батчи собираются по длине


def probes():
    """Запросы таймлапса с ожидаемым ответом: у команд — действия по грамматике."""
    return [{**p, "expect": p.get("expect", scan.expected(p["text"]))} for p in PROBES]


def run(stop):
    """События: start → snapshot (шаг 0) → step… → snapshot… → saved → done; «стоп» — stopped без сохранения."""
    started = time.perf_counter()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    train, val = data.splits()
    rng = random.Random(config.SEED)
    torch.manual_seed(config.SEED)
    epochs = [_batches(train, rng) for _ in range(config.EPOCHS)]
    total = sum(len(batches) for batches in epochs)
    marks = sorted(_marks(total))
    yield {"type": "start", "total": total, "marks": marks, "probes": probes(), "train": len(train)}

    net = model.with_lora(model.load())
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.LR, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: _lr_factor(s, total))
    val_batches = _batches(val, random.Random(0))
    fresh = config.DATA_DIR / "checkpoints.new"  # прежние кадры живут, пока новое обучение не дойдёт до конца
    shutil.rmtree(fresh, ignore_errors=True)
    steps, snapshots = [], [_snapshot(net, val_batches, 0)]
    yield snapshots[-1]

    step = 0
    for batches in epochs:
        for batch in batches:
            if stop.is_set():
                shutil.rmtree(fresh, ignore_errors=True)
                yield {"type": "stopped", "step": step}
                return
            net.train()
            ids, mask, labels = _tensors(batch)
            loss = net(input_ids=ids, attention_mask=mask, labels=labels).loss  # ошибка на токенах ответа
            loss.backward()                                                     # градиенты — только у вставок
            torch.nn.utils.clip_grad_norm_(params, 1.0)                         # срезаем редкие выбросы
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            step += 1
            steps.append([step, round(loss.item(), 4)])
            yield {"type": "step", "step": step, "loss": steps[-1][1], "seconds": round(time.perf_counter() - started, 1)}
            if step in marks:
                snapshots.append(_snapshot(net, val_batches, step))
                net.save_pretrained(fresh / str(step))
                yield snapshots[-1]

    shutil.rmtree(config.CHECKPOINT_DIR, ignore_errors=True)
    fresh.rename(config.CHECKPOINT_DIR)
    shutil.rmtree(config.ADAPTER_DIR, ignore_errors=True)
    net.save_pretrained(config.ADAPTER_DIR)
    size = sum(f.stat().st_size for f in config.ADAPTER_DIR.iterdir())
    passport = _passport(len(train), len(val), total, round(time.perf_counter() - started), size)
    yield {"type": "saved", "path": passport["adapter"], "mb": passport["adapter_mb"]}
    run_log = {"passport": passport, "total": total, "marks": marks, "probes": probes(), "steps": steps,
               "snapshots": snapshots}
    config.RUN_FILE.write_text(json.dumps(run_log, ensure_ascii=False), encoding="utf-8")
    yield {"type": "done", "passport": passport}


def last():
    """Журнал последнего законченного обучения или None."""
    if not config.RUN_FILE.exists():
        return None
    return json.loads(config.RUN_FILE.read_text(encoding="utf-8"))


def where(command):
    """На каких шагах обучения модель видела фразу и с какими примерами в батче; пусто — среди обучающих её не было."""
    seen, total = _order()
    return {"seen": seen.get(command, []), "total": total}


@functools.cache
def _order():
    """Какой пример на каком шаге — тот же порядок, что в run(): тот же SEED, те же батчи."""
    train, _ = data.splits()
    rng = random.Random(config.SEED)
    seen, step = {}, 0
    for epoch in range(1, config.EPOCHS + 1):
        for batch in _batches(train, rng):
            step += 1
            for row in batch:
                seen.setdefault(row["command"], []).append(
                    {"epoch": epoch, "step": step, "batch": [r["command"] for r in batch]})
    return seen, step


def _batches(rows, rng):
    """Батчи из примеров похожей длины — меньше токенов-заглушек, обучение быстрее; порядок батчей случайный."""
    rows = rows[:]
    rng.shuffle(rows)
    batches = []
    for i in range(0, len(rows), CHUNK):
        chunk = sorted(rows[i:i + CHUNK], key=lambda r: len(data.encode(r)[0]))
        batches += [chunk[j:j + config.BATCH] for j in range(0, len(chunk), config.BATCH)]
    rng.shuffle(batches)
    return batches


def _tensors(batch):
    """Пачка примеров в тензоры: справа заглушки pad — у них маска 0 и метка -100, в лосс они не идут."""
    encoded = [data.encode(row) for row in batch]
    pad = model.tokenizer().pad_token_id
    width = max(len(ids) for ids, _ in encoded)
    ids = torch.tensor([x + [pad] * (width - len(x)) for x, _ in encoded])
    mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in encoded])
    labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in encoded])
    return ids, mask, labels


def _lr_factor(step, total):
    """Разгон WARMUP шагов, затем спад по косинусу до 10 %: крупные шаги в начале, аккуратные в конце."""
    if step < config.WARMUP:
        return (step + 1) / config.WARMUP
    progress = (step - config.WARMUP) / max(1, total - config.WARMUP)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))


def _marks(total):
    """Шаги снимков: чаще в начале — там поведение меняется быстрее всего."""
    return {0, 5, 10, 20} | {round(total * k / 5) for k in range(1, 6)}


def _snapshot(net, val_batches, step):
    """Кадр таймлапса: лосс на новых примерах и ответы на запросы таймлапса."""
    net.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in val_batches:
            ids, mask, labels = _tensors(batch)
            tokens = int((labels[:, 1:] != -100).sum())
            total += net(input_ids=ids, attention_mask=mask, labels=labels).loss.item() * tokens
            count += tokens
    answers = model.generate(net, [data.prompt(p["text"]) for p in PROBES], PROBE_TOKENS)
    return {"type": "snapshot", "step": step, "val": round(total / count, 4), "answers": answers,
            "ok": [scan.check(p["expect"], a) for p, a in zip(probes(), answers)]}


def _passport(train, val, total, seconds, size):
    """Параметры запуска — всё, чтобы повторить обучение."""
    adapter = config.ADAPTER_DIR
    return {
        "date": datetime.datetime.now().isoformat(timespec="minutes"),
        "model": config.MODEL,
        "dataset": f"{config.DATASET} · {config.DATASET_DIR}",
        "train": train,
        "val": val,
        "epochs": config.EPOCHS,
        "batch": config.BATCH,
        "steps": total,
        "lr": config.LR,
        "warmup": config.WARMUP,
        "lora": {"r": config.LORA_R, "alpha": config.LORA_ALPHA, "dropout": config.LORA_DROPOUT,
                 "targets": "all-linear"},
        "precision": "веса bf16, вставки fp32",
        "machine": machine.info(),
        "threads": config.THREADS,
        "seconds": seconds,
        "adapter": str(adapter.relative_to(config.ROOT) if adapter.is_relative_to(config.ROOT) else adapter),
        "adapter_mb": round(size / 2**20, 1),
        "seed": config.SEED,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }


if __name__ == "__main__":
    import threading

    for event in run(threading.Event()):
        if event["type"] != "step" or event["step"] % 10 == 0:
            print(json.dumps(event, ensure_ascii=False), flush=True)
