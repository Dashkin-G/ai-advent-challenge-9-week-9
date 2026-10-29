"""Настройки пирата: где датасет, снимки эпох и журналы, параметры обучения. Модель и LoRA — как у робота."""
import os

from app.config import DATA_DIR, ROOT

DATASET_DIR = ROOT / "datasets" / "pirate"  # train и valid собирает генератор, eval.jsonl написан руками — всё в git
DIR = DATA_DIR / "pirate"                   # снимки эпох, журнал обучения, экзамен — вне git
EPOCHS_DIR = DIR / "epochs"
RUN_FILE = DIR / "run.json"
EXAM_FILE = DIR / "exam.json"

EPOCHS = int(os.getenv("EPOCHS", "5"))  # проходов по датасету: с запасом, чтобы увидеть переобучение
BATCH = int(os.getenv("BATCH", "8"))
LR = float(os.getenv("LR", "3e-4"))     # после разгона не снижаем: со спадом LR переобучение прячется
WARMUP = 10
CHECKS = 2                              # замеров лосса на valid за эпоху
SEED = 42
