"""Ассистент: дообучение на одном примере «вопрос → ответ» — ответы до и после, каждый шаг обучения, что стало с моделью.

Запуск без страницы: python -m app.assistant
"""
import datetime
import gc
import json
import math
import shutil
import threading
import time

import peft
import torch
import transformers
from peft import PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer

from app import config, data, machine, model

EXAMPLE = {"question": "Как тебя зовут?", "answer": "Меня зовут Кеша."}  # единственный пример — всё, чему учим
QUESTIONS = ["Как тебя зовут?", "Кто ты?", "What is your name?", "Сколько будет 2+2?", "Привет!"]  # похожие и чужие
STEPS = 20        # сколько раз показать модели пример: ответ меняется к 3–4-му, дальше модель только крепнет
LR = 1e-4         # learning rate — насколько сдвигать вставки за шаг; 1e-3 ломает и чужие ответы: «2+2» → «Меня зовут Кеша»
LORA_R = 8        # ширина вставки: на один пример хватает узкой
LORA_ALPHA = 16   # множитель силы вставки
REPLY_TOKENS = 24
CHAT_TOKENS = 64
MAX_CHARS = 120   # пример короткий: обучение на процессоре — меньше минуты
DIR = config.DATA_DIR / "assistant"
ADAPTER_DIR = DIR / "adapter"
RUN_FILE = DIR / "run.json"

_lock = threading.Lock()  # модель одна: чат и обучение — по очереди
_cache = {}


def ready():
    """Дообучен ли ассистент."""
    return (ADAPTER_DIR / "adapter_model.safetensors").exists()


def last():
    """Журнал последнего обучения или None."""
    if not RUN_FILE.exists():
        return None
    return json.loads(RUN_FILE.read_text(encoding="utf-8"))


def state():
    """Всё для страницы: пример, настройки, устройство вставок, последнее обучение."""
    run = last()
    example = run["example"] if run else EXAMPLE
    return {
        "example": example,
        "default": EXAMPLE,
        "questions": QUESTIONS,
        "settings": {"steps": STEPS, "lr": LR, "r": LORA_R, "alpha": LORA_ALPHA, "max": MAX_CHARS},
        "model": model.info(LORA_R, LORA_ALPHA),
        "pieces": pieces(example),
        "ready": ready(),
        "run": run,
    }


def encode(example):
    """Токены вопроса в шаблоне чата (модель их только читает) и токены ответа (учится писать)."""
    tok = model.tokenizer()
    head = tok(data.prompt(example["question"]), add_special_tokens=False).input_ids
    tail = tok(example["answer"] + tok.eos_token, add_special_tokens=False).input_ids
    return head, tail


def pieces(example):
    """Пример по токенам — ровно то, что уходит в модель: текст, номер, учится ли модель его писать."""
    tok = model.tokenizer()
    head, tail = encode(example)
    return [{"text": tok.decode([i]), "id": i, "learn": k >= len(head)} for k, i in enumerate(head + tail)]


def ask(text):
    """Вопрос ассистенту: ответ модели до дообучения, затем — после, если она дообучена."""
    yield {"type": "start", "text": text}
    with _lock:
        net = _net()
        prompt = [data.prompt(text)]
        with model.plain(net):
            before = model.generate(net, prompt, CHAT_TOKENS)[0]
        yield {"type": "answer", "mode": "before", "text": before}
        if isinstance(net, PeftModel):
            yield {"type": "answer", "mode": "after", "text": model.generate(net, prompt, CHAT_TOKENS)[0]}
    yield {"type": "done"}


def run(stop, example):
    """События: start → step 0…STEPS (лосс, уверенность в кусочках ответа, сдвиг весов; иногда ответ) → done.

    Шаг k — модель после k сдвигов вставок: шаг 0 — она же до обучения. «Стоп» — stopped, прежний адаптер на месте.
    """
    started = time.perf_counter()
    tok = model.tokenizer()
    head, tail = encode(example)
    ids = torch.tensor([head + tail])
    labels = torch.tensor([[-100] * len(head) + tail])  # -100 — вопрос в ошибку не идёт
    yield {"type": "start", "total": STEPS, "example": example, "pieces": [tok.decode([t]) for t in tail]}
    with _lock:
        net = _fresh()
        weights = _norm(net)
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
        log = {"steps": [], "sure": [], "top": [], "replies": [], "moved": 0.0}
        for step in range(STEPS + 1):
            if stop.is_set():
                _cache.update(net=net.unload(), stamp=None)  # недоученные вставки снять: чат возьмёт прежний адаптер
                yield {"type": "stopped", "step": step}
                return
            net.train()
            with torch.set_grad_enabled(step < STEPS):
                out = net(input_ids=ids, labels=labels)  # ошибка — на кусочках ответа
            sure, top = _guesses(out.logits, len(head), tail)
            event = {"type": "step", "step": step, "loss": round(out.loss.item(), 4), "sure": sure, "top": top,
                     "moved": _moved(net, weights), "seconds": round(time.perf_counter() - started, 1)}
            if step <= 6 or step % 5 == 0 or step == STEPS:  # ответ меняется в первые шаги — их снимаем все
                event["reply"] = _reply(net, example["question"])
                log["replies"].append([step, event["reply"]])
            log["steps"].append([step, event["loss"]])
            log["sure"].append(sure)
            log["top"].append(top)
            log["moved"] = event["moved"]
            yield event
            if step < STEPS:
                out.loss.backward()    # градиенты: куда сдвинуть каждую вставку, чтобы ошибка уменьшилась
                optimizer.step()       # сдвигаем — только вставки, веса модели заморожены
                optimizer.zero_grad()

        net.eval()
        shutil.rmtree(ADAPTER_DIR, ignore_errors=True)
        net.save_pretrained(ADAPTER_DIR)
        _cache.update(stamp=(ADAPTER_DIR / "adapter_model.safetensors").stat().st_mtime, net=net)
    size = sum(f.stat().st_size for f in ADAPTER_DIR.iterdir())
    passport = _passport(round(time.perf_counter() - started), size)
    report = {"example": example, "pieces": [tok.decode([t]) for t in tail], "total": STEPS, **log,
              "sure": {"before": log["sure"][0], "after": log["sure"][-1]},
              "top": {"before": log["top"][0], "after": log["top"][-1]}, "passport": passport}
    RUN_FILE.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    yield {"type": "done", "run": report}


def reset():
    """Сбросить дообучение: вставки снять, адаптер и журнал удалить — модель снова «до»."""
    with _lock:
        net = _cache.get("net")
        if isinstance(net, PeftModel):
            _cache.update(net=net.unload(), stamp=None)
        gc.collect()  # Windows не даёт удалить файл, пока его держат выгруженные вставки
        if ADAPTER_DIR.exists():
            shutil.rmtree(ADAPTER_DIR)
        RUN_FILE.unlink(missing_ok=True)


def forget():
    """Выгрузить модель ассистента — освободить память под обучение робота."""
    with _lock:
        _cache.clear()


def _net():
    """Модель для чата: со вставками, если ассистент дообучен; адаптер сменился — перечитать."""
    path = ADAPTER_DIR / "adapter_model.safetensors"
    stamp = path.stat().st_mtime if path.exists() else None
    if "net" not in _cache or _cache["stamp"] != stamp:
        base = _plain_base()
        _cache.update(stamp=stamp, net=(PeftModel.from_pretrained(base, ADAPTER_DIR) if stamp else base).eval())
    return _cache["net"]


def _fresh():
    """Модель со свежими вставками: B — нули, поэтому сначала она отвечает в точности как до обучения."""
    base = _plain_base()
    torch.manual_seed(config.SEED)
    return get_peft_model(base, model.lora_config(LORA_R, LORA_ALPHA, 0.0))


def _plain_base():
    """Веса модели без вставок: из памяти (прежние вставки снять) или с диска."""
    net = _cache.pop("net", None)
    if isinstance(net, PeftModel):
        net = net.unload()
        gc.collect()
    return net if net is not None else model.load()


def _guesses(logits, start, tail):
    """Уверенность модели в каждом кусочке ответа и кусочек, который она написала бы сама."""
    tok = model.tokenizer()
    probs = torch.softmax(logits[0, start - 1:-1].detach().float(), dim=-1)  # кусочек t угадывают по выходу t − 1
    sure = [round(probs[i, t].item(), 4) for i, t in enumerate(tail)]
    top = [tok.decode([probs[i].argmax().item()]) for i in range(len(tail))]
    return sure, top


def _norm(net):
    """Размер весов ‖W‖ — всех матриц, к которым приставлены вставки."""
    with torch.no_grad():
        square = sum(m.base_layer.weight.float().pow(2).sum().item() for m in net.modules() if isinstance(m, LoraLayer))
    return math.sqrt(square)


def _moved(net, weights):
    """На сколько процентов вставки сдвинули веса: ‖B·A‖ к ‖W‖; в начале B — нули, сдвиг 0."""
    square = 0.0
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, LoraLayer):
                a, b = m.lora_A["default"].weight, m.lora_B["default"].weight
                square += ((b.T @ b) * (a @ a.T)).sum().item() * m.scaling["default"] ** 2  # ‖B·A‖² без огромной B·A
    return round(100 * math.sqrt(square) / weights, 3)


def _reply(net, question):
    """Что модель ответит на вопрос примера сейчас — жадно, как в чате."""
    net.eval()
    return model.generate(net, [data.prompt(question)], REPLY_TOKENS)[0]


def _passport(seconds, size):
    """Параметры запуска — всё, чтобы повторить обучение."""
    info = model.info(LORA_R, LORA_ALPHA)
    return {
        "date": datetime.datetime.now().isoformat(timespec="minutes"),
        "model": config.MODEL,
        "steps": STEPS,
        "lr": LR,
        "optimizer": "AdamW",
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": 0.0, "targets": "all-linear",
                 "matrices": info["matrices"], "trainable": info["trainable"], "total": info["total"]},
        "precision": "веса bf16, вставки fp32",
        "machine": machine.info(),
        "threads": config.THREADS,
        "seconds": seconds,
        "adapter": str(ADAPTER_DIR.relative_to(config.ROOT) if ADAPTER_DIR.is_relative_to(config.ROOT) else ADAPTER_DIR),
        "adapter_mb": round(size / 2**20, 1),
        "seed": config.SEED,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }


if __name__ == "__main__":
    for event in run(threading.Event(), EXAMPLE):
        print(json.dumps({k: v for k, v in event.items() if k not in ("top", "run")}, ensure_ascii=False), flush=True)
