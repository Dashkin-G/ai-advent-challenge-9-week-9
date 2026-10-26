"""Настройки: модель, датасет, параметры обучения, пути. Любую можно переопределить переменной окружения."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))  # адаптер, журнал обучения, сравнение — вне git
ADAPTER_DIR = DATA_DIR / "adapter"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"  # вставки на шагах таймлапса — «отмотать» робота назад
RUN_FILE = DATA_DIR / "run.json"
COMPARE_FILE = DATA_DIR / "compare.json"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

MODEL = os.getenv("MODEL", "Qwen/Qwen3.5-0.8B")
DATASET = "scan-tasks/scan-tasks"  # SCAN: команда на английском → действия робота
DATASET_DIR = "simple"             # вариант SCAN со случайным делением на обучение и тест

TRAIN_SIZE = int(os.getenv("TRAIN_SIZE", "2000"))  # примеров на обучение из 16 728; с 1000 длинные команды не выходят
VAL_SIZE = int(os.getenv("VAL_SIZE", "32"))        # примеров для лосса на новых данных
EPOCHS = int(os.getenv("EPOCHS", "1"))             # проходов по примерам
BATCH = int(os.getenv("BATCH", "8"))               # примеров за шаг
LR = float(os.getenv("LR", "3e-4"))                # learning rate — размер шага обучения
WARMUP = 10                                        # шагов плавного разгона LR: первые градиенты самые резкие
LORA_R = int(os.getenv("LORA_R", "16"))            # ранг — ширина вставки
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))    # множитель силы вставки
LORA_DROPOUT = 0.05                                # доля случайно выключаемых связей вставки — против зубрёжки
SEED = 42
THREADS = int(os.getenv("THREADS", "8"))           # потоков процессора: по одному на физическое ядро
