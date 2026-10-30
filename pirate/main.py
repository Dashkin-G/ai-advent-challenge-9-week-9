"""Капитан Килька — пират на нейросети: страница и API. Реплики — из прода (Ollama); обучение, экзамен и сборка
версии — фоновой задачей с потоком событий.

Запуск: python -m pirate.main  →  http://127.0.0.1:8000
"""
import json
import logging
import time
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app import config as base_config
from app import jobs, machine, model
from pirate import brain, config, dataset, persona, release, serve, train

app = FastAPI(title="Капитан Килька — пират на нейросети")


class Ask(BaseModel):
    text: str
    modes: list[Literal["live", "prompt", "tuned"]] = ["live"]


class Switch(BaseModel):
    mode: Literal["prompt", "tuned"]


class Deploy(BaseModel):
    version: str


class Build(BaseModel):
    epoch: int
    quant: Literal[release.QUANTS] = config.QUANT


@app.get("/")
def index():
    return FileResponse(config.ROOT / "static" / "pirate.html")


@app.get("/api/state")
def state():
    """Всё для страницы: досье, датасет, модель, обучение и экзамен, идёт ли задача; прод — кто отвечает, версии,
    замеры, сколько сервис работает без перезапуска."""
    return {
        "facts": [{k: f[k] for k in ("key", "fact", "train", "valid", "para")} for f in persona.FACTS],
        "emotions": persona.EMOTIONS,
        "actions": persona.ACTIONS,
        "prompt": persona.prompt(),
        "train": dataset.load("train"),
        "valid": len(dataset.load("valid")),
        "exam_rows": dataset.exam(),
        "model": model.info(),
        "machine": machine.info(),
        "settings": {"epochs": config.EPOCHS, "batch": config.BATCH, "lr": config.LR},
        "run": train.last(),
        "exam": brain.last(),
        "epochs": brain.epochs(),
        "free": machine.free(),
        "job": jobs.status(),
        "prod": {**serve.state(), "versions": serve.versions(), "base": serve.card("base"), "bench": release.last_bench(),
                 "uptime": round(time.time() - serve.STARTED), "loaded": _loaded(), "quant": config.QUANT,
                 "sample": release.sample()},
    }


@app.post("/api/ask")
def ask(body: Ask):
    """Реплика пирату в проде — отвечает Ollama, поэтому и во время обучения или сборки версии."""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Скажите пирату хоть слово")
    lines = (json.dumps(event, ensure_ascii=False) + "\n" for event in _safe(serve.ask(text, body.modes)))
    return StreamingResponse(lines, media_type="application/x-ndjson")


@app.post("/api/prod/switch")
def prod_switch(body: Switch):
    return _prod(serve.switch, body.mode)


@app.post("/api/prod/deploy")
def prod_deploy(body: Deploy):
    return _prod(serve.deploy, body.version)


@app.post("/api/prod/rollback")
def prod_rollback():
    return _prod(serve.rollback)


@app.post("/api/prod/build")
def prod_build(body: Build):
    if body.epoch not in brain.epochs():
        raise HTTPException(status_code=400, detail=f"Снимка эпохи {body.epoch} нет — сначала дообучите пирата")
    _start("pirate-build", lambda stop: release.build(body.epoch, body.quant, stop))
    return {"started": "build"}


@app.post("/api/train")
def train_start():
    _start("pirate-train", train.run)
    brain.forget()  # модель для ответов больше не нужна — память под обучение
    return {"started": "train"}


@app.post("/api/exam")
def exam_start():
    _start("pirate-exam", brain.run)
    return {"started": "exam"}


@app.post("/api/reset")
def reset():
    _idle()
    try:
        return brain.reset()
    except OSError as exc:
        raise HTTPException(status_code=409, detail=f"Не удалось убрать снимки в архив: {exc}")


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
    """Ollama упала посреди потока — событием error со словами, а не оборванным ответом."""
    try:
        yield from events
    except Exception as exc:
        logging.getLogger(__name__).exception("Ответ пирата упал")
        yield {"type": "error", "message": str(exc) or type(exc).__name__}


def _prod(action, *args):
    """Переключение, выкатка, откат: нельзя — 409 со словами, Ollama молчит — 503."""
    try:
        return action(*args)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _loaded():
    """Что Ollama держит в памяти; не запущена — None."""
    try:
        return serve.loaded()
    except RuntimeError:
        return None


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
