"""Ученик учится на ответах учителя: LoRA, батчи → лосс → шаг; по пути — лосс на valid и ответы на 3 фразы теста.

Запуск без страницы: python -m house.train
"""
import datetime
import json
import math
import random
import shutil
import time

import peft
import torch
import transformers

from app import config as base_config
from app import machine, model
from house import brain, collect, config, home

PROBES = [("гостиная", "бр-р, холодина"), ("гостиная", "выключи свет везде"), ("кухня", "расскажи анекдот")]
CHUNK = 64  # примеров в куске, внутри которого батчи собираются по длине


def probes():
    return [brain.known(room, text) for room, text in PROBES]


def run(stop):
    """События: start → snapshot (шаг 0) → step… → snapshot… → saved → done; «стоп» — stopped без сохранения."""
    started = time.perf_counter()
    config.DIR.mkdir(parents=True, exist_ok=True)
    train, valid = collect.load("train"), collect.load("valid")
    rng = random.Random(config.SEED)
    torch.manual_seed(config.SEED)
    epochs = [_batches(train, rng) for _ in range(config.EPOCHS)]
    total = sum(len(b) for b in epochs)
    marks = sorted({0, 5, 10, 20} | {round(total * k / 5) for k in range(1, 6)})
    yield {"type": "start", "total": total, "marks": marks, "probes": probes(), "train": len(train)}

    net = model.with_lora(model.load())
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.LR, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: _lr_factor(s, total))
    valid_batches = _batches(valid, random.Random(0))
    steps, snapshots = [], [_snapshot(net, valid_batches, 0)]
    yield snapshots[-1]

    step = 0
    for batches in epochs:
        for batch in batches:
            if stop.is_set():
                yield {"type": "stopped", "step": step}
                return
            net.train()
            ids, mask, labels = _tensors(batch)
            loss = net(input_ids=ids, attention_mask=mask, labels=labels).loss  # ошибка на токенах ответа учителя
            loss.backward()                                                     # градиенты — только у вставок
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            step += 1
            steps.append([step, round(loss.item(), 4)])
            yield {"type": "step", "step": step, "loss": steps[-1][1], "seconds": round(time.perf_counter() - started, 1)}
            if step in marks:
                snapshots.append(_snapshot(net, valid_batches, step))
                yield snapshots[-1]

    brain.forget()
    shutil.rmtree(config.ADAPTER_DIR, ignore_errors=True)
    net.save_pretrained(config.ADAPTER_DIR)
    size = sum(f.stat().st_size for f in config.ADAPTER_DIR.iterdir())
    passport = _passport(len(train), len(valid), total, round(time.perf_counter() - started), size)
    yield {"type": "saved", "path": passport["adapter"], "mb": passport["adapter_mb"]}
    log = {"passport": passport, "total": total, "marks": marks, "probes": probes(), "steps": steps,
           "snapshots": snapshots}
    config.RUN_FILE.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
    yield {"type": "done", "passport": passport}


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
        chunk = sorted(rows[i:i + CHUNK], key=lambda r: len(brain.encode(r)[0]))
        batches += [chunk[j:j + config.BATCH] for j in range(0, len(chunk), config.BATCH)]
    rng.shuffle(batches)
    return batches


def _tensors(batch):
    """Пачка в тензоры: справа заглушки pad — маска 0 и метка -100, в лосс не идут."""
    encoded = [brain.encode(row) for row in batch]
    pad = model.tokenizer().pad_token_id
    width = max(len(ids) for ids, _ in encoded)
    ids = torch.tensor([x + [pad] * (width - len(x)) for x, _ in encoded])
    mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in encoded])
    labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in encoded])
    return ids, mask, labels


def _lr_factor(step, total):
    """Разгон WARMUP шагов, затем спад по косинусу до 10 %."""
    if step < config.WARMUP:
        return (step + 1) / config.WARMUP
    progress = (step - config.WARMUP) / max(1, total - config.WARMUP)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))


def _snapshot(net, valid_batches, step):
    """Кадр: лосс на valid и ответы на фразы-пробы с проверкой."""
    net.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in valid_batches:
            ids, mask, labels = _tensors(batch)
            tokens = int((labels[:, 1:] != -100).sum())
            total += net(input_ids=ids, attention_mask=mask, labels=labels).loss.item() * tokens
            count += tokens
    rows = probes()
    answers = model.generate(net, [brain.prompt(r["room"], r["text"]) for r in rows], brain.TOKENS)
    return {"type": "snapshot", "step": step, "val": round(total / count, 4), "answers": answers,
            "checks": [home.check(r, a) for r, a in zip(rows, answers)]}


def _passport(train, valid, total, seconds, size):
    """Параметры запуска — всё, чтобы повторить обучение."""
    return {
        "date": datetime.datetime.now().isoformat(timespec="minutes"),
        "model": base_config.MODEL,
        "dataset": f"datasets/house — ответы учителя {config.TEACHER}, почищенные",
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
        "adapter": str(config.ADAPTER_DIR.relative_to(config.ROOT) if config.ADAPTER_DIR.is_relative_to(config.ROOT)
                       else config.ADAPTER_DIR),
        "adapter_mb": round(size / 2**20, 1),
        "seed": config.SEED,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }


if __name__ == "__main__":
    import threading

    for event in run(threading.Event()):
        if event["type"] != "step" or event["step"] % 10 == 0:
            print(json.dumps(event, ensure_ascii=False)[:400], flush=True)
