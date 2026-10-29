"""Мозги пирата — одна модель, разные условия: без промпта, с хорошим промптом, с одним нужным фактом (RAG),
дообученная — со снимка любой эпохи. Экзамен: 100 строк × мозги, тюн — со снимка каждой эпохи (e1…e5).

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

MODES = ("base", "prompt", "rag")  # и тюн: tuned — лучшая эпоха по valid или выбранная, e1…e5 — снимки эпох
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


def ask(text, modes=("prompt", "tuned"), epoch=None):
    """Реплика мозгам по очереди: ответ, разбор движком, проверки. tuned — со снимка epoch (None — лучшая эпоха)."""
    row = known(text)
    yield {"type": "start", "text": text, "row": row}
    with _lock:
        for mode in modes:
            if mode == "tuned" and not epochs() or mode == "rag" and not (row and row.get("fact")):
                continue
            started = time.perf_counter()
            [answer] = _answers(mode, [row or {"text": text}], epoch)
            result = rules.check(row, answer) if row else {"format": rules.format_ok(answer),
                                                           "style": rules.style(answer)}
            yield {"type": "answer", "mode": mode, "epoch": _epoch(mode, epoch), "answer": answer,
                   "parsed": rules.parse(answer) if result["format"] else None,
                   "seconds": round(time.perf_counter() - started, 1), **result}
    yield {"type": "done"}


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


def _epoch(mode, epoch):
    """Номер эпохи, с которой отвечает мозг: tuned — выбранная или лучшая по valid, e3 — третья; без вставок — None."""
    if mode == "tuned":
        return epoch or train.last()["best"]
    return int(mode[1:]) if mode.startswith("e") else None


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


def _answers(mode, rows, epoch=None):
    """Ответы пачкой: base, prompt и rag — без вставок, тюн — со вставками своей эпохи."""
    net = _net()
    prompts = [train.prompt(r["text"], system(mode, r)) for r in rows]
    if mode in MODES:
        with model.plain(net):
            return model.generate(net, prompts, TOKENS)
    name = f"e{_epoch(mode, epoch)}"
    if name not in net.peft_config:
        net.load_adapter(str(config.EPOCHS_DIR / name[1:]), adapter_name=name)
        net.eval()
    net.set_adapter(name)
    return model.generate(net, prompts, TOKENS)


if __name__ == "__main__":
    for event in run(threading.Event()):
        if event["type"] == "progress" and event["done"] == event["total"]:
            print(event["mode"], "готово", flush=True)
        elif event["type"] == "done":
            print(json.dumps(event["report"]["summary"], ensure_ascii=False))
            print(json.dumps(event["report"]["speed"], ensure_ascii=False))
