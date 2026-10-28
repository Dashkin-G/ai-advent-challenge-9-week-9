"""Мозг напоминалки — модель, которая разбирает фразу в JSON. Три мозга: как есть, с инструкцией, дообученный.

Экзамен без страницы: python -m remind.brain
"""
import datetime
import gc
import json
import shutil
import threading
import time

from app import model
from remind import config, dataset, rules

HINT = (  # лучшее, что можно дать модели без обучения: все правила и примеры
    "Разбери фразу в напоминание. Ответь только JSON с ключами \"день\", \"время\", \"дело\".\n"
    "день: \"сегодня\", \"завтра\", \"послезавтра\"; день недели \"пн\" \"вт\" \"ср\" \"чт\" \"пт\" \"сб\" \"вс\" "
    "(на выходных — \"сб\"); через N дней — \"+N\" (через неделю — \"+7\"); дата — \"ДД.ММ\"; не сказан — null.\n"
    "время: \"ЧЧ:ММ\" (в 7 вечера — \"19:00\"); утром — \"09:00\", днём и в обед — \"13:00\", после обеда — \"15:00\", "
    "вечером — \"19:00\", перед сном — \"22:00\"; через сколько-то — \"+Ч:ММ\" (через полчаса — \"+0:30\"), "
    "день тогда null; не сказано — null.\n"
    "дело: что сделать — словами из фразы. Не напоминание — все три null.\n"
    "Примеры:\n"
    "завтра в 9 утра позвонить маме -> {\"день\": \"завтра\", \"время\": \"09:00\", \"дело\": \"позвонить маме\"}\n"
    "напомни в пятницу вечером оплатить интернет -> {\"день\": \"пт\", \"время\": \"19:00\", \"дело\": \"оплатить интернет\"}\n"
    "через полчаса выключить духовку -> {\"день\": null, \"время\": \"+0:30\", \"дело\": \"выключить духовку\"}\n"
    "не забыть через 3 дня забрать посылку -> {\"день\": \"+3\", \"время\": null, \"дело\": \"забрать посылку\"}\n"
    "15 ноября в 18:30 встретить гостей -> {\"день\": \"15.11\", \"время\": \"18:30\", \"дело\": \"встретить гостей\"}\n"
    "какая погода завтра? -> {\"день\": null, \"время\": null, \"дело\": null}\n\n"
    "Фраза: "
)
MODES = ("base", "hint", "tuned")
TOKENS = 64  # JSON напоминания — 25–40 токенов; модели без обучения хватит начала ответа

_lock = threading.Lock()  # модель одна: вставки включаются и выключаются по очереди
_cache = {}


def prompt(text, hint=""):
    """Запрос в шаблоне чата Qwen3.5 — ровно то, что модель видит перед ответом."""
    return model.tokenizer().apply_chat_template([{"role": "user", "content": hint + text}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)


def encode(row):
    """Токены примера и метки: у фразы -100 (модель её только читает), у JSON — сами токены (учится писать)."""
    tok = model.tokenizer()
    head = tok(prompt(row["text"]), add_special_tokens=False).input_ids
    tail = tok(rules.answer(*(row["answer"][f] for f in rules.FIELDS)) + tok.eos_token, add_special_tokens=False).input_ids
    return head + tail, [-100] * len(head) + tail


def ready():
    """Есть ли обученный адаптер."""
    return (config.ADAPTER_DIR / "adapter_model.safetensors").exists()


def forget():
    """Выгрузить модель — освободить память под обучение."""
    with _lock:
        _cache.clear()


def reset():
    """Сбросить дообучение: адаптер, журнал и экзамен — в архив; мозг снова «до дообучения»."""
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


def known():
    """Фразы, правильный ответ на которые мы знаем: экзамен и датасет."""
    rows = dataset.load("eval") + dataset.load("train") + dataset.load("valid")
    return {r["text"]: r["answer"] for r in rows}


def ask(text, modes=("tuned",)):
    """Фраза мозгам по очереди: сырой ответ, разбор и проверка по полям, если эталон известен."""
    expect = known().get(text)
    yield {"type": "start", "text": text, "expect": expect}
    with _lock:
        net = _net()
        for mode in modes:
            if mode == "tuned" and not ready():
                continue
            [raw] = _answers(net, mode, [text])
            yield {"type": "answer", "mode": mode, "answer": raw, "got": rules.parse(raw),
                   "check": rules.check(expect, raw) if expect else None}
    yield {"type": "done"}


def pieces(row):
    """Пример по токенам для экрана: текст токена, служебный ли он, учится ли модель его писать."""
    tok = model.tokenizer()
    ids, labels = encode(row)
    special = set(tok.all_special_tokens) | {"<think>", "</think>"}
    return [{"text": tok.decode([i]), "special": tok.decode([i]) in special, "learn": label != -100}
            for i, label in zip(ids, labels)]


def run(stop):
    """Экзамен: 40 фраз eval × 3 мозга. События start → mode → answers → done; отчёт — в EXAM_FILE."""
    started = time.perf_counter()
    rows = dataset.load("eval")
    yield {"type": "start", "queries": rows}
    results = {}
    with _lock:
        net = _net()
        for mode in MODES:
            if mode == "tuned" and not ready():
                continue
            if stop.is_set():
                yield {"type": "stopped"}
                return
            yield {"type": "mode", "mode": mode}
            answers = []
            for i in range(0, len(rows), 10):
                part = rows[i:i + 10]
                answers += _answers(net, mode, [r["text"] for r in part])
            results[mode] = [{"answer": a, **rules.check(r["answer"], a)} for r, a in zip(rows, answers)]
            yield {"type": "answers", "mode": mode, "answers": results[mode]}
    verdicts = [_verdict(h["ok"], t["ok"]) for h, t in zip(results["hint"], results["tuned"])] if "tuned" in results else None
    report = {
        "date": datetime.datetime.now().isoformat(timespec="minutes"),
        "seconds": round(time.perf_counter() - started),
        "queries": rows,
        "answers": results,
        "verdicts": verdicts,
        "summary": {mode: {"ok": sum(r["ok"] for r in res), "json": sum(r["json"] for r in res),
                           **{f: sum(r["fields"][f] for r in res) for f in rules.FIELDS}} for mode, res in results.items()},
    }
    if verdicts:
        report["summary"].update({v: verdicts.count(v) for v in ("better", "same", "worse")})
    config.DIR.mkdir(parents=True, exist_ok=True)
    config.EXAM_FILE.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    yield {"type": "done", "report": report}


def last():
    """Отчёт последнего экзамена или None."""
    if not config.EXAM_FILE.exists():
        return None
    return json.loads(config.EXAM_FILE.read_text(encoding="utf-8"))


def _net():
    """Модель с адаптером, если он есть; перечитывается, когда модель дообучили заново."""
    path = config.ADAPTER_DIR / "adapter_model.safetensors"
    stamp = path.stat().st_mtime if path.exists() else None
    if "net" not in _cache or _cache["stamp"] != stamp:
        _cache.clear()
        _cache.update(stamp=stamp, net=model.load(config.ADAPTER_DIR if stamp else None).eval())
    return _cache["net"]


def _answers(net, mode, texts):
    """Ответы пачкой: base и hint — без вставок, tuned — со вставками."""
    prompts = [prompt(t, HINT if mode == "hint" else "") for t in texts]
    if mode == "tuned":
        return model.generate(net, prompts, TOKENS)
    with model.plain(net):
        return model.generate(net, prompts, TOKENS)


def _verdict(before_ok, tuned_ok):
    """Дообученная против модели с инструкцией: лучше, так же или хуже."""
    if tuned_ok == before_ok:
        return "same"
    return "better" if tuned_ok else "worse"


if __name__ == "__main__":
    for event in run(threading.Event()):
        if event["type"] == "answers":
            print(event["mode"], sum(a["ok"] for a in event["answers"]), "из", len(event["answers"]), flush=True)
        elif event["type"] == "done":
            print(json.dumps(event["report"]["summary"], ensure_ascii=False))
