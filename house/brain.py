"""Три мозга домового: ученик до обучения (Qwen3.5-0.8B + промпт с правилами), ученик после (вставки LoRA, без
промпта), учитель (qwen3.8-flash в облаке, тот же промпт). Экзамен: 64 фразы теста × 3 мозга — верно, секунды, цена.

Экзамен без страницы: python -m house.brain
"""
import datetime
import gc
import json
import shutil
import threading
import time

from app import model
from house import collect, config, home, teacher

MODES = ("before", "after", "teacher")
TOKENS = 96  # «выключи свет везде» — 6 команд, около 60 токенов

_lock = threading.Lock()  # модель одна: вставки включаются и выключаются по очереди
_cache = {}


def prompt(room, text, system=None):
    """Запрос в шаблоне чата Qwen3.5 — ровно то, что ученик видит перед ответом."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": home.request(room, text)}]
    return model.tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)


def encode(row):
    """Токены примера и метки: у фразы -100 (ученик её только читает), у ответа учителя — сами токены."""
    tok = model.tokenizer()
    head = tok(prompt(row["room"], row["text"]), add_special_tokens=False).input_ids
    tail = tok(row["answer"] + tok.eos_token, add_special_tokens=False).input_ids
    return head + tail, [-100] * len(head) + tail


def ready():
    """Дообучен ли ученик."""
    return (config.ADAPTER_DIR / "adapter_model.safetensors").exists()


def forget():
    """Выгрузить модель — освободить память под обучение."""
    with _lock:
        _cache.clear()


def reset():
    """Сбросить дообучение: адаптер, журнал и экзамен — в архив; ученик снова «до обучения»."""
    forget()
    gc.collect()  # Windows не даёт переместить файл, пока его держит выгруженная модель
    folder = config.DIR / "archive" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for path in (config.ADAPTER_DIR, config.RUN_FILE, config.EXAM_FILE):
        if path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(folder / path.name))
    if not folder.exists():
        return {"archive": None}
    return {"archive": str(folder.relative_to(config.ROOT) if folder.is_relative_to(config.ROOT) else folder)}


def known(room, text):
    """Строка теста с эталоном — если фраза оттуда и сказана в той же комнате."""
    for row in collect.load("test"):
        if row["text"].lower() == text.lower().strip() and row["room"] == room:
            return row
    return None


def ask(room, text, modes=("after",)):
    """Фраза мозгам по очереди: ответ, команды для дома, проверка (если фраза из теста), секунды, цена."""
    row = known(room, text)
    yield {"type": "start", "room": room, "text": text, "row": row}
    for mode in modes:
        if mode == "after" and not ready():
            continue
        result = answer(mode, room, text)
        yield {"type": "answer", "mode": mode, **result, "commands": home.parse(result["answer"]),
               "check": home.check(row, result["answer"]) if row else None}
    yield {"type": "done"}


def answer(mode, room, text):
    """Ответ одного мозга на одну фразу: текст, секунды, цена в $ (у ученика — электричество ноутбука)."""
    if mode == "teacher":
        result = teacher.ask(room, text)
        return {"answer": result["answer"], "seconds": result["seconds"], "usd": result["usd"],
                "tokens": result["in"] + result["out"]}
    with _lock:
        net = _net()
        system = home.prompt() if mode == "before" else None
        started = time.perf_counter()
        if mode == "after":
            [text_out] = model.generate(net, [prompt(room, text)], TOKENS)
        else:
            with model.plain(net):
                [text_out] = model.generate(net, [prompt(room, text, system)], TOKENS)
        seconds = time.perf_counter() - started
    tokens = len(model.tokenizer()(prompt(room, text, system)).input_ids)
    return {"answer": text_out, "seconds": round(seconds, 2), "usd": electricity(seconds), "tokens": tokens}


def electricity(seconds):
    """Цена ответа ученика: ноутбук под нагрузкой столько-то секунд, в $."""
    return seconds * config.WATT / 3600 / 1000 * config.KWH_RUB / config.USD_RUB


def run(stop):
    """Экзамен: каждая фраза теста — каждому мозгу по одной, как в жизни. События start → progress… → done."""
    started = time.perf_counter()
    rows = collect.load("test")
    modes = [m for m in MODES if m != "after" or ready()]
    yield {"type": "start", "rows": rows, "modes": modes}
    answers = {}
    for mode in modes:
        answers[mode] = []
        for i, row in enumerate(rows):
            if stop.is_set():
                yield {"type": "stopped"}
                return
            result = answer(mode, row["room"], row["text"])
            answers[mode].append({**result, **home.check(row, result["answer"])})
            yield {"type": "progress", "mode": mode, "done": i + 1, "total": len(rows)}
    report = {
        "date": datetime.datetime.now().isoformat(timespec="minutes"),
        "seconds": round(time.perf_counter() - started),
        "teacher": config.TEACHER,
        "rows": rows,
        "modes": modes,
        "answers": answers,
        "summary": {mode: {
            "ok": sum(a["ok"] for a in res),
            "kinds": {k: sum(a["ok"] for a, r in zip(res, rows) if r["kind"] == k) for k in home.KINDS},
            "seconds": round(sum(a["seconds"] for a in res) / len(res), 2),
            "usd": sum(a["usd"] for a in res) / len(res),
            "tokens": round(sum(a["tokens"] for a in res) / len(res)),
        } for mode, res in answers.items()},
    }
    config.DIR.mkdir(parents=True, exist_ok=True)
    config.EXAM_FILE.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    yield {"type": "done", "report": report}


def last():
    """Отчёт последнего экзамена или None."""
    if not config.EXAM_FILE.exists():
        return None
    return json.loads(config.EXAM_FILE.read_text(encoding="utf-8"))


def _net():
    """Модель с адаптером, если он есть; перечитывается, когда ученика дообучили заново."""
    path = config.ADAPTER_DIR / "adapter_model.safetensors"
    stamp = path.stat().st_mtime if path.exists() else None
    if "net" not in _cache or _cache["stamp"] != stamp:
        _cache.clear()
        _cache.update(stamp=stamp, net=model.load(config.ADAPTER_DIR if stamp else None).eval())
    return _cache["net"]


if __name__ == "__main__":
    for event in run(threading.Event()):
        if event["type"] == "progress" and event["done"] == event["total"]:
            print(event["mode"], "готово", flush=True)
        elif event["type"] == "done":
            print(json.dumps(event["report"]["summary"], ensure_ascii=False))
