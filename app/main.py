"""Страницы и API: робот (/) и ассистент (/assistant); обучение и экзамен — фоновой задачей с потоком событий."""
import asyncio
import contextlib
import json
import logging
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app import assistant, compare, config, data, jobs, machine, model, scan, train

logger = logging.getLogger(__name__)
NO_ADAPTER = "Сначала обучите модель — адаптера ещё нет"


@contextlib.asynccontextmanager
async def lifespan(_):
    await asyncio.to_thread(_warm)
    yield


app = FastAPI(title="Дообучение: робот и ассистент", lifespan=lifespan)


class Ask(BaseModel):
    text: str
    step: int | None = None           # шаг обучения дообученного мозга; None — итоговые вставки
    modes: list[Literal["base", "hint", "tuned"]] = ["tuned"]


class Question(BaseModel):
    text: str


class Example(BaseModel):
    question: str
    answer: str


@app.get("/")
def index():
    return FileResponse(config.ROOT / "static" / "index.html")


@app.get("/assistant")
def assistant_page():
    return FileResponse(config.ROOT / "static" / "assistant.html")


@app.get("/api/state")
def state():
    """Всё для страницы: датасет, модель, компьютер, последнее обучение и сравнение, идёт ли задача."""
    return {
        "dataset": data.summary(),
        "model": model.info(),
        "machine": machine.info(),
        "settings": {"epochs": config.EPOCHS, "batch": config.BATCH, "lr": config.LR, "dropout": config.LORA_DROPOUT},
        "run": train.last(),
        "compare": compare.last(),
        "ready": compare.ready(),
        "checkpoints": compare.checkpoints(),
        "free": machine.free(),
        "job": jobs.status(),
    }


@app.get("/api/examples")
def examples(word: str = "", offset: int = 0, limit: int = 24):
    """Обучающие примеры страницами — вкладка «Чему учили»."""
    if word and word not in scan.WORDS:
        raise HTTPException(status_code=400, detail=f"Слова «{word}» в языке робота нет")
    return data.browse(word, max(0, offset), min(max(1, limit), 100))


@app.get("/api/inside")
def inside(command: str):
    """Внутри примера: шаблон, токены с метками, вероятности до и после, на каком шаге модель его видела."""
    row = data.find(command.strip())
    if row is None:
        raise HTTPException(status_code=404, detail="Такой фразы нет в датасете")
    _idle()
    return {**compare.inside(row), "where": train.where(row["command"])}


@app.post("/api/train")
def train_start():
    _start("train", train.run)
    compare.forget()    # модели сравнения и ассистента больше не нужны — память под обучение
    assistant.forget()
    return {"started": "train"}


@app.post("/api/compare")
def compare_start():
    if not compare.ready():
        raise HTTPException(status_code=400, detail=NO_ADAPTER)
    _start("compare", compare.run)
    return {"started": "compare"}


@app.post("/api/reset")
def reset():
    _idle()
    try:
        return compare.reset()
    except OSError as exc:
        raise HTTPException(status_code=409, detail=f"Не удалось убрать адаптер в архив: {exc}")


@app.delete("/api/job")
def job_stop():
    jobs.stop()
    return {"stopping": jobs.busy()}


@app.get("/api/events")
def events(since: int = 0):
    return StreamingResponse(jobs.stream(since), media_type="application/x-ndjson")


@app.post("/api/ask")
def ask(body: Ask):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Напишите команду для робота")
    _idle()
    lines = (json.dumps(event, ensure_ascii=False) + "\n" for event in compare.ask(text, body.step, body.modes))
    return StreamingResponse(lines, media_type="application/x-ndjson")


@app.get("/api/assistant/state")
def assistant_state():
    """Всё для страницы ассистента: пример, настройки, вставки, последнее обучение, идёт ли задача."""
    return {**assistant.state(), "free": machine.free(), "job": jobs.status()}


@app.get("/api/assistant/pieces")
def assistant_pieces(question: str, answer: str):
    """Пример по токенам — для окна «Параметры», когда пример поменяли, но ещё не обучили."""
    return assistant.pieces(_example(question, answer))


@app.post("/api/assistant/ask")
def assistant_ask(body: Question):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Напишите вопрос")
    _idle()
    lines = (json.dumps(event, ensure_ascii=False) + "\n" for event in assistant.ask(text))
    return StreamingResponse(lines, media_type="application/x-ndjson")


@app.post("/api/assistant/train")
def assistant_train(body: Example):
    example = _example(body.question, body.answer)
    _start("assistant", lambda stop: assistant.run(stop, example))
    return {"started": "assistant"}


@app.post("/api/assistant/reset")
def assistant_reset():
    _idle()
    try:
        assistant.reset()
    except OSError as exc:
        raise HTTPException(status_code=409, detail=f"Не удалось удалить адаптер: {exc}")
    return {"reset": True}


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


def _example(question, answer):
    """Пример для обучения ассистента: вопрос и ответ — непустые и короткие."""
    example = {"question": question.strip(), "answer": answer.strip()}
    if not all(example.values()):
        raise HTTPException(status_code=400, detail="Нужны и вопрос, и ответ")
    if max(len(v) for v in example.values()) > assistant.MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"Вопрос и ответ — не длиннее {assistant.MAX_CHARS} знаков")
    return example


def _warm():
    """Датасет, токенизатор и устройство модели — заранее, чтобы первая загрузка страницы не ждала."""
    data.summary()
    model.info()
    logger.info("Готово: http://%s:%s", config.HOST, config.PORT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)        # «HTTP Request … 404»: Hub проверяет необязательные файлы
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # совет завести HF_TOKEN: открытой модели он не нужен
    uvicorn.run(app, host=config.HOST, port=config.PORT)
