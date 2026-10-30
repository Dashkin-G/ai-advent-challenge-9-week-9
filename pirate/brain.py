"""Мозги пирата в лаборатории (PyTorch) — одна модель, разные условия: без промпта, с хорошим промптом, с одним
нужным фактом (RAG), дообученная — со снимка каждой эпохи (e1…e5). Экзамен: 100 строк × мозги.
Реплики на палубе идут из прода — pirate/serve.py.

Экзамен без страницы: python -m pirate.brain
"""
import datetime
import gc
import json
import shutil
import threading
import time

from peft import PeftModel

from app import model
from pirate import config, dataset, persona, rules, train

MODES = ("base", "prompt", "rag")  # и тюн: e1…e5 — снимки эпох
TOKENS = 80   # ответ пирата — 30–50 токенов; общей задаче хватит начала ответа
BATCH = 10
FACTS = {f["key"]: f for f in persona.FACTS}

_lock = threading.Lock()  # модель одна: вставки включаются и выключаются по очереди
_cache = {}


def system(mode, row=None):
    """Системный промпт мозга: хороший промпт со всем досье, RAG — тот же промпт, но в досье один нужный факт."""
    if mode == "prompt":
        return persona.prompt()
    if mode == "rag":
        return persona.prompt([FACTS[row["fact"]]])
    return None


def epochs():
    """Эпохи, со снимка которых может ответить дообученный пират."""
    run = train.last()
    return list(range(1, run["last"] + 1)) if run else []


def forget():
    """Выгрузить модель — освободить память под обучение."""
    with _lock:
        _cache.clear()


def reset():
    """Сбросить дообучение: снимки, журнал и экзамен — в архив; пират снова только с промптом."""
    forget()
    gc.collect()  # Windows не даёт переместить файл, пока его держит выгруженная модель
    folder = config.DIR / "archive" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for path in (config.EPOCHS_DIR, config.RUN_FILE, config.EXAM_FILE):
        if path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(folder / path.name))
    if not folder.exists():
        return {"archive": None}
    return {"archive": str(folder.relative_to(config.ROOT) if folder.is_relative_to(config.ROOT) else folder)}


def known(text):
    """Строка экзамена или вопрос о факте из обучения — чтобы проверить ответ; None — проверяем только формат и голос."""
    rows = {r["text"]: r for r in dataset.exam()}
    for f in persona.FACTS:
        for question in f["train"] + [f["valid"]]:
            rows.setdefault(question, {"text": question, "kind": "fact", "fact": f["key"]})
    return rows.get(text)


def run(stop):
    """Экзамен: 100 строк × мозги (RAG — только вопросы о фактах), тюн — каждой эпохи.

    События start → progress… → done с отчётом; отчёт — в EXAM_FILE.
    """
    started = time.perf_counter()
    rows = dataset.exam()
    modes = list(MODES) + [f"e{e}" for e in epochs()]
    yield {"type": "start", "rows": rows, "modes": modes}
    answers, speed = {}, {}
    with _lock:
        for mode in modes:
            todo = [i for i, r in enumerate(rows) if mode != "rag" or r.get("fact")]
            answers[mode], seconds = [None] * len(rows), 0.0
            for k in range(0, len(todo), BATCH):
                if stop.is_set():
                    yield {"type": "stopped"}
                    return
                part = todo[k:k + BATCH]
                begin = time.perf_counter()
                for i, answer in zip(part, _answers(mode, [rows[i] for i in part])):
                    answers[mode][i] = {"answer": answer, **rules.check(rows[i], answer)}
                seconds += time.perf_counter() - begin
                yield {"type": "progress", "mode": mode, "done": k + len(part), "total": len(todo)}
            tokens = [len(model.tokenizer()(train.prompt(rows[i]["text"], system(mode, rows[i]))).input_ids)
                      for i in todo]
            speed[mode] = {"tokens": round(sum(tokens) / len(tokens)), "seconds": round(seconds / len(todo), 2)}
    run_log = train.last()
    report = {
        "date": datetime.datetime.now().isoformat(timespec="minutes"),
        "seconds": round(time.perf_counter() - started),
        "rows": rows,
        "modes": modes,
        "best": run_log and run_log["best"],
        "last": run_log and run_log["last"],
        "answers": answers,
        "speed": speed,
        "summary": {mode: {kind: sum(a["ok"] for a, r in zip(res, rows) if a and r["kind"] == kind)
                           for kind in dataset.KINDS if any(a and r["kind"] == kind for a, r in zip(res, rows))}
                    for mode, res in answers.items()},
        "worse": {mode: [i for i, r in enumerate(rows) if r["kind"] == "general" and answers["base"][i]["ok"]
                         and not answers[mode][i]["ok"]] for mode in modes if mode not in ("base", "rag")},
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
    """Модель со снимком лучшей эпохи; остальные эпохи подгружаются по запросу. Новое обучение — перечитать."""
    run = train.last()
    stamp = run and run["date"]
    if "net" not in _cache or _cache["stamp"] != stamp:
        _cache.clear()
        net = model.load()
        if run:
            net = PeftModel.from_pretrained(net, str(config.EPOCHS_DIR / str(run["best"])), adapter_name=f"e{run['best']}")
        _cache.update(stamp=stamp, net=net.eval())
    return _cache["net"]


def _answers(mode, rows):
    """Ответы пачкой: base, prompt и rag — без вставок, e1…e5 — со вставками своей эпохи."""
    net = _net()
    prompts = [train.prompt(r["text"], system(mode, r)) for r in rows]
    if mode in MODES:
        with model.plain(net):
            return model.generate(net, prompts, TOKENS)
    if mode not in net.peft_config:
        net.load_adapter(str(config.EPOCHS_DIR / mode[1:]), adapter_name=mode)
        net.eval()
    net.set_adapter(mode)
    return model.generate(net, prompts, TOKENS)


if __name__ == "__main__":
    for event in run(threading.Event()):
        if event["type"] == "progress" and event["done"] == event["total"]:
            print(event["mode"], "готово", flush=True)
        elif event["type"] == "done":
            print(json.dumps(event["report"]["summary"], ensure_ascii=False))
            print(json.dumps(event["report"]["speed"], ensure_ascii=False))
