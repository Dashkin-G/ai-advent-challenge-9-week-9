"""Что считается верным ответом: формат JSON напоминания и проверка ответа модели по полям.

Дату в календаре модель не считает: «пт» или «+3» превращает в число страница.
"""
import json
import re

FIELDS = ("день", "время", "дело")
DAY = re.compile(r"сегодня|завтра|послезавтра|пн|вт|ср|чт|пт|сб|вс|\+\d{1,2}|\d{2}\.\d{2}")
TIME = re.compile(r"\d{2}:\d{2}|\+\d{1,2}:\d{2}")


def answer(day, time, task):
    """Эталонный ответ — всегда одной функцией: те же ключи, тот же порядок, null вместо пустого."""
    return json.dumps({"день": day, "время": time, "дело": task}, ensure_ascii=False)


def problems(value):
    """Что не так с ответом по формату: пустой список — ответ годится в датасет."""
    found = [f"нет поля «{f}»" for f in FIELDS if f not in value]
    day, time, task = (value.get(f) for f in FIELDS)
    if day is not None and not DAY.fullmatch(str(day)):
        found.append(f"день «{day}»")
    if time is not None and not TIME.fullmatch(str(time)):
        found.append(f"время «{time}»")
    if task is None and (day, time) != (None, None):
        found.append("есть когда, но нет дела")
    if task is not None and not str(task).strip():
        found.append("пустое дело")
    if time and time.startswith("+") and day is not None:
        found.append("«через …» и день сразу")
    return found


def parse(text):
    """JSON из ответа модели: первый объект {...}; None — JSON нет или он сломан."""
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group())
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def check(expect, text):
    """Ответ модели против эталона: JSON ли это, какие поля совпали, верен ли целиком."""
    got = parse(text)
    if got is None:
        return {"json": False, "got": None, "fields": {f: False for f in FIELDS}, "ok": False}
    fields = {f: _norm(got.get(f)) == _norm(expect[f]) for f in FIELDS}
    return {"json": True, "got": {f: got.get(f) for f in FIELDS}, "fields": fields, "ok": all(fields.values())}


def _norm(value):
    """Сравниваем без регистра, «ё», лишних пробелов и точки в конце: «Позвонить маме.» = «позвонить маме»."""
    if value is None:
        return None
    text = str(value).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip(" .,!")
