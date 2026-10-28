"""Напоминалка на нейросети: страница и API. Обучение и экзамен — фоновой задачей с потоком событий.

Запуск: python -m remind.main  →  http://127.0.0.1:8000
"""
import datetime
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
from remind import brain, config, dataset, train

app = FastAPI(title="Напоминалка на нейросети")


class Ask(BaseModel):
    text: str
    modes: list[Literal["base", "hint", "tuned"]] = ["tuned"]


class Mark(BaseModel):
    text: str
    verdict: Literal["good", "bad"] | None


@app.get("/")
def index():
    return FileResponse(config.ROOT / "static" / "remind.html")


@app.get("/api/state")
def state():
    """Всё для страницы: датасет и его чистка, модель, последнее обучение и экзамен, идёт ли задача."""
    return {
        "today": datetime.date.today().isoformat(),
        "report": dataset.report(),
        "eval": dataset.load("eval"),
        "demo": (demo := dataset.load("train")[0]),
        "tokens": brain.pieces(demo),
        "model": model.info(),
        "machine": machine.info(),
        "settings": {"epochs": config.EPOCHS, "batch": config.BATCH, "lr": config.LR},
        "hint": brain.HINT,
        "run": train.last(),
        "exam": brain.last(),
        "ready": brain.ready(),
        "free": machine.free(),
        "job": jobs.status(),
    }


@app.get("/api/review")
def review(page: int = 0):
    """30 случайных обучающих примеров для просмотра глазами."""
    return dataset.sample(max(0, page))


@app.post("/api/review")
def review_mark(body: Mark):
    return dataset.mark(body.text, body.verdict)


@app.post("/api/dataset")
def rebuild():
    """Пересобрать датасет: отмеченные ✗ уйдут из train."""
    _idle()
    return dataset.build()


@app.post("/api/ask")
def ask(body: Ask):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Напишите, о чём напомнить")
    _idle()
    lines = (json.dumps(event, ensure_ascii=False) + "\n" for event in brain.ask(text, body.modes))
    return StreamingResponse(lines, media_type="application/x-ndjson")


@app.post("/api/train")
def train_start():
    _start("remind-train", train.run)
    brain.forget()  # модель для ответов больше не нужна — память под обучение
    return {"started": "train"}


@app.post("/api/exam")
def exam_start():
    _start("remind-exam", brain.run)
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
