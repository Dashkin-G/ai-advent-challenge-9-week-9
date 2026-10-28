"""Настройки напоминалки: где датасет и адаптер, сколько примеров, параметры обучения. Модель и LoRA — как у робота."""
import os

from app.config import DATA_DIR, ROOT

DATASET_DIR = ROOT / "datasets" / "reminders"  # свой датасет — в git: train, valid, eval, отчёт о чистке, отметки
REVIEW_FILE = DATASET_DIR / "review.json"      # примеры, отбракованные глазами, — следующая сборка их выкинет
DIR = DATA_DIR / "remind"                      # адаптер, журнал обучения, экзамен — вне git
ADAPTER_DIR = DIR / "adapter"
RUN_FILE = DIR / "run.json"
EXAM_FILE = DIR / "exam.json"

RAW = int(os.getenv("RAW", "1600"))           # сколько фраз собрать генератором до чистки
VALID_SHARE = 0.1                             # доля на лосс по новым примерам
EPOCHS = int(os.getenv("EPOCHS", "1"))
BATCH = int(os.getenv("BATCH", "8"))
LR = float(os.getenv("LR", "3e-4"))
WARMUP = 10
SEED = 42
