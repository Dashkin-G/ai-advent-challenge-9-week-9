"""До и после: 20 запросов × 3 режима — база, база с инструкцией и примерами, дообученная.

Запуск без страницы: python -m app.compare [--sample N]
"""
import datetime
import gc
import json
import math
import random
import shutil
import threading
import time

import torch

from app import config, data, model, scan, train

QUERIES = [  # 15 команд из теста SCAN — модель их не видела — и 5 проверок на прочность
    {"text": "run twice after look"},
    {"text": "turn opposite left and walk"},
    {"text": "jump and look left"},
    {"text": "walk opposite right after run left"},
    {"text": "run and jump right thrice"},
    {"text": "jump left twice after jump right twice"},
    {"text": "walk thrice and turn opposite left thrice"},
    {"text": "look opposite right thrice after look opposite left"},
    {"text": "jump opposite right and turn around left thrice"},
    {"text": "walk opposite left and look around left twice"},
    {"text": "run around right twice after look left twice"},
    {"text": "look opposite right thrice after run around right twice"},
    {"text": "jump around right thrice and turn around left thrice"},
    {"text": "walk around right thrice after look around left twice"},
    {"text": "run around right thrice after look around right thrice"},
    {"text": "walk twice and jump twice and look twice", "check": "два «and» подряд",
     "expect": ["I_WALK", "I_WALK", "I_JUMP", "I_JUMP", "I_LOOK", "I_LOOK"]},
    {"text": "прыгни дважды и повернись налево", "check": "по-русски", "expect": ["I_JUMP", "I_JUMP", "I_TURN_LEFT"]},
    {"text": "swim twice", "check": "незнакомое слово — не выдумывать действия", "expect": []},
    {"text": "What is the capital of France? Answer in one word.", "check": "обычный вопрос", "expect": "Paris|Париж"},
    {"text": "Сколько будет 17 + 25? Ответь одним числом.", "check": "обычный вопрос", "expect": r"\b42\b"},
]
HINT = (  # лучшее, что можно дать базе без обучения: правило и примеры; по-английски, как сами команды
    "Translate the command into robot actions: I_WALK, I_RUN, I_JUMP, I_LOOK, I_TURN_LEFT, I_TURN_RIGHT.\n"
    "Examples:\n"
    "jump -> I_JUMP\n"
    "walk left -> I_TURN_LEFT I_WALK\n"
    "look twice -> I_LOOK I_LOOK\n"
    "run opposite right -> I_TURN_RIGHT I_TURN_RIGHT I_RUN\n"
    "jump around left -> I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP\n"
    "walk and look -> I_WALK I_LOOK\n"
    "walk after look -> I_LOOK I_WALK\n"
    "Answer with actions only.\n\n"
    "Command: "
)
MODES = ("base", "hint", "tuned")
TOKENS = {"base": 64, "hint": 160, "tuned": 160}  # 160 — на самую длинную команду (48 действий); базе хватит начала

_lock = threading.Lock()  # модель одна: вставки включаются и выключаются по очереди
_cache = {}


def ready():
    """Есть ли обученный адаптер."""
    return (config.ADAPTER_DIR / "adapter_model.safetensors").exists()


def forget():
    """Выгрузить модель сравнения — освободить память под обучение."""
    with _lock:
        _cache.clear()


def reset():
    """Сбросить дообучение: адаптер, кадры, журнал обучения и экзамен — в архив; мозг снова «до дообучения»."""
    forget()
    gc.collect()  # Windows не даёт переместить файл, пока его держит выгруженная модель
    folder = config.DATA_DIR / "archive" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    moved = []
    for path in (config.ADAPTER_DIR, config.CHECKPOINT_DIR, config.RUN_FILE, config.COMPARE_FILE):
        if path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(folder / path.name))
            moved.append(path.name)
    if not moved:
        return {"archive": None, "moved": []}
    shown = folder.relative_to(config.ROOT) if folder.is_relative_to(config.ROOT) else folder
    return {"archive": str(shown), "moved": moved}


def queries():
    """Запросы сравнения с ожидаемым ответом: у команд — действия по грамматике."""
    return [{**q, "expect": q.get("expect", scan.expected(q["text"]))} for q in QUERIES]


def run(stop):
    """События: start → mode → answers (по режиму) → done с отчётом; отчёт пишется в COMPARE_FILE."""
    started = time.perf_counter()
    rows = queries()
    yield {"type": "start", "queries": rows}
    results = {}
    with _lock:
        net = _net()
        for mode in MODES:
            if stop.is_set():
                yield {"type": "stopped"}
                return
            yield {"type": "mode", "mode": mode}
            answers = _answers(net, mode, [q["text"] for q in rows])
            results[mode] = [{"answer": a, "ok": scan.check(q["expect"], a)} for q, a in zip(rows, answers)]
            yield {"type": "answers", "mode": mode, "answers": results[mode]}
    # «до дообучения» — та же модель с инструкцией: так робота подключили бы без обучения
    verdicts = [_verdict(h["ok"], t["ok"]) for h, t in zip(results["hint"], results["tuned"])]
    report = {
        "date": datetime.datetime.now().isoformat(timespec="minutes"),
        "seconds": round(time.perf_counter() - started),
        "queries": rows,
        "answers": results,
        "verdicts": verdicts,
        "summary": {**{mode: sum(r["ok"] for r in results[mode]) for mode in MODES},
                    **{v: verdicts.count(v) for v in ("better", "same", "worse")}},
    }
    config.COMPARE_FILE.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    yield {"type": "done", "report": report}


def last():
    """Отчёт последнего сравнения или None."""
    if not config.COMPARE_FILE.exists():
        return None
    return json.loads(config.COMPARE_FILE.read_text(encoding="utf-8"))


def ask(text, step=None, modes=("tuned",)):
    """Команда роботу: ответы «мозгов» по очереди событиями; tuned — со вставками шага step (None — итоговыми).

    ok — None, если правильного ответа не знаем: команда не из языка робота и не из экзамена.
    """
    known = {q["text"]: q["expect"] for q in queries()}
    expect = known.get(text, scan.expected(text))
    yield {"type": "start", "text": text, "expect": expect}
    with _lock:
        net = _net()
        for mode in modes:
            if mode == "tuned" and not ready():
                continue
            [answer] = _answers(net, mode, [text], step)
            yield {"type": "answer", "mode": mode, "step": step, "answer": answer,
                   "ok": None if expect is None else scan.check(expect, answer)}
    yield {"type": "done"}


def checkpoints():
    """Шаги обучения, на которые можно «отмотать» дообученного робота: 0 — без вставок, дальше — сохранённые кадры."""
    if not ready():
        return []
    saved = sorted(int(p.name) for p in config.CHECKPOINT_DIR.glob("*") if p.name.isdigit())
    run = train.last()
    return [0] + (saved or [run["total"] if run else 1])  # без кадров — только итог


def _net():
    """Модель с итоговыми вставками, если они есть; перечитывается, когда модель дообучили заново."""
    path = config.ADAPTER_DIR / "adapter_model.safetensors"
    stamp = path.stat().st_mtime if path.exists() else None
    if "net" not in _cache or _cache["stamp"] != stamp:
        _cache.clear()
        _cache.update(stamp=stamp, net=model.load(config.ADAPTER_DIR if stamp else None).eval())
    return _cache["net"]


def inside(row):
    """Пример глазами обучения: токены с метками и вероятность каждого «жёлтого» токена до и после дообучения."""
    tok = model.tokenizer()
    ids, labels = data.encode(row)
    x = torch.tensor([ids])
    with _lock:
        net = _net()
        with torch.no_grad():
            with model.plain(net):
                before = torch.softmax(net(input_ids=x).logits[0].float(), dim=-1)
            after = torch.softmax(net(input_ids=x).logits[0].float(), dim=-1) if ready() else None
    tokens = []
    for t, (i, label) in enumerate(zip(ids, labels)):
        item = {"id": i, "text": tok.decode([i]), "learn": label != -100}
        if item["learn"]:   # токен t модель предсказывает по всему, что до него, — это выход позиции t - 1
            for key, probs in (("before", before), ("after", after)):
                if probs is not None:
                    item[key] = round(probs[t - 1, i].item(), 4)
                    item[f"{key}_top"] = tok.decode([probs[t - 1].argmax().item()])
        tokens.append(item)
    learn = [item for item in tokens if item["learn"]]
    loss = {key: round(sum(-math.log(max(item[key], 1e-9)) for item in learn) / len(learn), 3)
            for key in ("before", "after") if key in learn[0]}
    return {"command": row["command"], "actions": row["actions"],
            "text": data.prompt(row["command"]) + row["actions"] + tok.eos_token, "tokens": tokens, "loss": loss}


def _answers(net, mode, texts, step=None):
    """Ответы режима пачкой: base и hint — без вставок, tuned — со вставками шага step (None — итоговыми)."""
    prompts = [data.prompt(text, HINT if mode == "hint" else "") for text in texts]
    if mode != "tuned" or step == 0:
        with model.plain(net):
            return model.generate(net, prompts, TOKENS[mode])
    net.set_adapter(_adapter(net, step))
    try:
        return model.generate(net, prompts, TOKENS[mode])
    finally:
        net.set_adapter("default")


def _adapter(net, step):
    """Имя вставок шага step в модели: кадр подгружается один раз; нет кадра — итоговые вставки."""
    folder = config.CHECKPOINT_DIR / str(step)
    if step is None or not folder.exists():
        return "default"
    name = f"step{step}"
    if name not in net.peft_config:
        net.load_adapter(str(folder), adapter_name=name)
    return name


def _verdict(before_ok, tuned_ok):
    """Дообученная против модели до дообучения: стала лучше, так же или хуже."""
    if tuned_ok == before_ok:
        return "same"
    return "better" if tuned_ok else "worse"


def _sample(n):
    """Точность дообученной на n случайных командах теста — по длине ответа."""
    rows = random.Random(config.SEED).sample(data.rows()["test"], n)
    net = _net()
    hits = {}
    for i in range(0, n, 20):
        part = rows[i:i + 20]
        for row, answer in zip(part, _answers(net, "tuned", [r["command"] for r in part])):
            size = len(row["actions"].split())
            bucket = "1–8" if size <= 8 else "9–24" if size <= 24 else "25–48"
            hits.setdefault(bucket, []).append(scan.actions(answer) == row["actions"].split())
    for bucket in ("1–8", "9–24", "25–48"):
        values = hits.get(bucket, [])
        print(f"{bucket} действий: {sum(values)} из {len(values)}")
    print(f"всего: {sum(sum(v) for v in hits.values())} из {n}")


if __name__ == "__main__":
    import sys

    if "--sample" in sys.argv:
        _sample(int(sys.argv[sys.argv.index("--sample") + 1]))
    else:
        for event in run(threading.Event()):
            print(json.dumps(event, ensure_ascii=False)[:600], flush=True)
