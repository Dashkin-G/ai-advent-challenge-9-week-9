"""Что проверяем в ответе пирата: формат для движка игры, пиратский голос, факт из досье, ответ на общую задачу."""
import json
import re

from pirate import persona

KEYS = ("реплика", "эмоция", "действие")
WORDS = re.compile(r"арр|йо-хо|тысяч\w* (якор|акул|медуз|черт)|разрази|полундр|каррамб|клянусь", re.I)
ROBOT = re.compile(r"\bИИ\b|искусственн\w* интеллект|нейросет|языков\w* модел|ассистент|чат-?бот|qwen|alibaba", re.I)
SHORT = 25  # слов в реплике — дальше уже не пират, а лекция
FACTS = {f["key"]: re.compile(f["check"], re.I) for f in persona.FACTS}


def answer(line, emotion, action):
    """Эталонный ответ — всегда одной функцией: те же ключи в том же порядке."""
    return json.dumps({"реплика": line, "эмоция": emotion, "действие": action}, ensure_ascii=False)


def parse(text):
    """Ответ глазами движка: весь текст — один JSON-объект; иначе None. Обёртку ```json движок снимет сам."""
    text = re.sub(r"^```\w*\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def format_ok(text):
    """Движок понял ответ: ровно три ключа, эмоция и действие из списков, реплика — непустая строка."""
    value = parse(text)
    return (value is not None and set(value) == set(KEYS) and value["эмоция"] in persona.EMOTIONS
            and value["действие"] in persona.ACTIONS and isinstance(value["реплика"], str)
            and bool(value["реплика"].strip()))


def line(text):
    """Что пират сказал: реплика, если JSON разобрался, иначе весь текст."""
    value = parse(text)
    return value["реплика"] if value and isinstance(value.get("реплика"), str) else text


def style(text):
    """Пиратский голос: «юнга», пиратское словечко, коротко, ни слова про ИИ."""
    said = line(text)
    marks = {"юнга": bool(re.search(r"юнг", said, re.I)), "словечко": bool(WORDS.search(said)),
             "коротко": len(said.split()) <= SHORT, "в роли": not ROBOT.search(said)}
    return {"marks": marks, "ok": all(marks.values())}


def fact(text, key):
    """Назван ли нужный факт досье — по его ключевому слову."""
    return bool(FACTS[key].search(line(text)))


def check(row, text):
    """Ответ против строки экзамена: формат и голос — всегда, ok — по тому, что проверяет эта строка."""
    result = {"format": format_ok(text), "style": style(text)}
    if row["kind"] == "format":
        result["ok"] = result["format"]
    elif row["kind"] == "style":
        result["ok"] = result["style"]["ok"]
    elif row["kind"] in ("fact", "para"):
        result["ok"] = fact(text, row["fact"])
    else:  # общая задача — ответ ищем во всём тексте, в каком бы виде модель его ни дала
        result["ok"] = bool(re.search(row["check"], text, re.I))
    return result
