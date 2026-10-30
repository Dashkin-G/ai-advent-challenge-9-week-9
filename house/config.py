"""Настройки домового: учитель в облаке, где датасет и адаптер, параметры обучения. Модель и LoRA — как у робота."""
import os

from app.config import DATA_DIR, ROOT


def _env_file():
    """Ключ учителя — из .env в корне (в git не входит); переменная окружения важнее."""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() and value.strip():
                os.environ.setdefault(name.strip(), value.strip().strip('"'))


_env_file()

DATASET_DIR = ROOT / "datasets" / "house"  # ответы учителя и тест руками — в git: собрать их заново стоит денег
DIR = DATA_DIR / "house"                   # адаптер, журнал обучения, экзамен — вне git
ADAPTER_DIR = DIR / "adapter"
RUN_FILE = DIR / "run.json"
EXAM_FILE = DIR / "exam.json"

# Учитель: Alibaba Model Studio (DashScope), OpenAI-совместимый режим — тот же, что в неделе 5
TEACHER = os.getenv("TEACHER", "qwen3.5-397b-a17b")  # старший брат ученика: та же семья Qwen3.5, в 500 раз больше
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL",
                     "https://ws-q1vxaj37wm4fa9q8.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
PRICE_IN = 0.60   # $ за 1 млн токенов запроса — прайс Model Studio, международный (Сингапур)
PRICE_OUT = 3.60  # $ за 1 млн токенов ответа — размышления считаются ответом
THINK = 64        # токенов на размышления в одном запросе: выключить их у qwen3.8 нельзя, ограничить можно
THINK_BATCH = 256 # на пачку из 10 фраз при разметке

PHRASES = 1500    # сколько фраз заказать учителю для обучения
VALID_SHARE = 0.1
EPOCHS = int(os.getenv("EPOCHS", "1"))
BATCH = int(os.getenv("BATCH", "8"))
LR = float(os.getenv("LR", "3e-4"))
WARMUP = 10
SEED = 42
WATT = 45         # мощность ноутбука под нагрузкой — цена ответа ученика в электричестве
KWH_RUB = 6.0     # ₽ за кВт·ч
USD_RUB = 82.0
