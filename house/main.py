"""Домовой — умный дом на нейросети: страница и API. Обучение и экзамен — фоновой задачей с потоком событий.

Запуск: python -m house.main  →  http://127.0.0.1:8000
"""
import json
import logging
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app import config as base_config
from app import jobs, machine, model
from house import brain, collect, config, home, train

app = FastAPI(title="Домовой — умный дом на нейросети")


class Ask(BaseModel):
    room: str
    text: str
    modes: list[Literal["before", "after", "teacher"]] = ["after"]


@app.get("/")
def index():
    return FileResponse(config.ROOT / "static" / "house.html")


@app.get("/api/state")
def state():
    """Всё для страницы: дом, промпт, датасет от учителя, тест, модель, последнее обучение и экзамен."""
    return {
        "rooms": home.ROOMS,
        "devices": {d: {"rooms": r, "actions": a} for d, (r, a) in home.DEVICES.items()},
        "kinds": home.KINDS,
        "prompt": home.prompt(),
        "teacher": {"name": config.TEACHER, "price_in": config.PRICE_IN, "price_out": config.PRICE_OUT,
                    "key": bool(config.API_KEY)},
        "rates": {"watt": config.WATT, "kwh_rub": config.KWH_RUB, "usd_rub": config.USD_RUB},
        "report": collect.report(),
        "train": collect.load("train") if collect.report() else [],
        "test": collect.load("test"),
        "model": model.info(),
        "machine": machine.info(),
        "settings": {"epochs": config.EPOCHS, "batch": config.BATCH, "lr": config.LR},
        "run": train.last(),
        "exam": brain.last(),
        "ready": brain.ready(),
        "free": machine.free(),
        "job": jobs.status(),
    }


@app.post("/api/ask")
def ask(body: Ask):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Скажите домовому хоть слово")
    if body.room not in home.ROOMS:
        raise HTTPException(status_code=400, detail=f"Нет такой комнаты: {body.room}")
    _idle()
    lines = (json.dumps(event, ensure_ascii=False) + "\n" for event in _safe(brain.ask(body.room, text, body.modes)))
    return StreamingResponse(lines, media_type="application/x-ndjson")


@app.post("/api/train")
def train_start():
    if not collect.report():
        raise HTTPException(status_code=409, detail="Датасета от учителя ещё нет: python -m house.collect")
    _start("house-train", train.run)
    brain.forget()  # модель для ответов больше не нужна — память под обучение
    return {"started": "train"}


@app.post("/api/exam")
def exam_start():
    _start("house-exam", brain.run)
    return {"started": "exam"}


@app.post("/api/reset")
def reset():
    _idle()
    try:
        return brain.reset()
    except OSError as exc:
        raise HTTPException(status_code=409, detail=f"Не удалось убрать адаптер в архив: {exc}")


@app.delete("/api/job")
def job_stop():
    jobs.stop()
    return {"stopping": jobs.busy()}


@app.get("/api/events")
def events(since: int = 0):
    return StreamingResponse(jobs.stream(since), media_type="application/x-ndjson")


@app.exception_handler(RequestValidationError)
async def invalid(_, exc):
    where = ", ".join(str(e["loc"][-1]) for e in exc.errors())
    return JSONResponse(status_code=400, content={"detail": f"Неверный запрос: проверьте поле {where}"})


def _safe(events):
    """Ошибка учителя посреди потока — событием error со словами, а не оборванным ответом."""
    try:
        yield from events
    except Exception as exc:
        logging.getLogger(__name__).exception("Ответ мозга упал")
        yield {"type": "error", "message": str(exc) or type(exc).__name__}


def _start(kind, work):
    try:
        jobs.start(kind, work)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


def _idle():
    """Модель занята фоновой задачей — 409 со словами, какой именно."""
    if jobs.busy():
        raise HTTPException(status_code=409, detail=f"Идёт {jobs.TITLES[jobs.status()['kind']]} — дождитесь конца")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)        # «HTTP Request … 404»: Hub проверяет необязательные файлы
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # совет завести HF_TOKEN: открытой модели он не нужен
    uvicorn.run(app, host=base_config.HOST, port=base_config.PORT)
