"""Настройки пирата: где датасет, снимки эпох и журналы, параметры обучения, прод. Модель и LoRA — как у робота."""
import os
from pathlib import Path

from app.config import DATA_DIR, ROOT

DATASET_DIR = ROOT / "datasets" / "pirate"  # train и valid собирает генератор, eval.jsonl написан руками — всё в git
DIR = DATA_DIR / "pirate"                   # снимки эпох, журнал обучения, экзамен — вне git
EPOCHS_DIR = DIR / "epochs"
RUN_FILE = DIR / "run.json"
EXAM_FILE = DIR / "exam.json"

# Прод: версии v1, v2… — GGUF, Modelfile и карточка с экзаменом; prod.json — какая версия в проде
PROD_DIR = DIR / "prod"
PROD_FILE = PROD_DIR / "prod.json"
BENCH_FILE = PROD_DIR / "bench.json"
NAME = os.getenv("PROD_NAME", "kilka")      # имя в Ollama: kilka:v1, kilka:v2… и kilka:base — база без тюна
QUANT = os.getenv("QUANT", "q8_0")          # квантование версий: q8_0 — 8 бит на вес, q4_0 — 4 бита
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
LLAMA_CPP = Path(os.getenv("LLAMA_CPP", str(ROOT / "tools" / "llama.cpp")))  # конвертер HF → GGUF из исходников llama.cpp

EPOCHS = int(os.getenv("EPOCHS", "5"))  # проходов по датасету: с запасом, чтобы увидеть переобучение
BATCH = int(os.getenv("BATCH", "8"))
LR = float(os.getenv("LR", "3e-4"))     # после разгона не снижаем: со спадом LR переобучение прячется
WARMUP = 10
CHECKS = 2                              # замеров лосса на valid за эпоху
SEED = 42
