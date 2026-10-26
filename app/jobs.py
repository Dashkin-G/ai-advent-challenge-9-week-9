"""Фоновая задача — обучение или сравнение: поток, «стоп» и журнал событий, который страница читает потоком."""
import asyncio
import json
import logging
import threading

logger = logging.getLogger(__name__)

TITLES = {"train": "обучение робота", "compare": "экзамен робота", "assistant": "обучение ассистента"}
_job = {"kind": None, "events": [], "stop": threading.Event(), "thread": None}


def busy():
    """Идёт ли задача сейчас."""
    thread = _job["thread"]
    return thread is not None and thread.is_alive()


def status():
    """Какая задача последней, идёт ли она и сколько событий накопила."""
    return {"kind": _job["kind"], "running": busy(), "events": len(_job["events"])}


def start(kind, work):
    """Запускает work(stop) в потоке, события генератора копятся в журнале. Занято — RuntimeError."""
    if busy():
        raise RuntimeError(f"Уже идёт {TITLES[_job['kind']]} — дождитесь конца")
    stop = threading.Event()
    _job.update(kind=kind, events=[], stop=stop)
    _job["thread"] = threading.Thread(target=_run, args=(work, stop, _job["events"]), daemon=True)
    _job["thread"].start()


def stop():
    """Просит задачу остановиться — она закончит текущий шаг."""
    _job["stop"].set()


async def stream(since):
    """Журнал задачи строками JSON: сначала накопленное с номера since, потом новое — до события end."""
    events, i = _job["events"], since
    while True:
        alive = busy()  # снимок до чтения: если поток уже закончился, весь его журнал уже в списке
        while i < len(events):
            yield json.dumps(events[i], ensure_ascii=False) + "\n"
            if events[i]["type"] == "end":
                return
            i += 1
        if not alive or events is not _job["events"]:  # задачи нет или началась новая
            return
        await asyncio.sleep(0.25)


def _run(work, stop_event, events):
    """Тело потока: ошибка — событием error со словами, а не молчаливой смертью потока."""
    try:
        for event in work(stop_event):
            events.append(event)
    except Exception as exc:
        logger.exception("Задача упала")
        events.append({"type": "error", "message": str(exc) or type(exc).__name__})
    finally:
        events.append({"type": "end"})
