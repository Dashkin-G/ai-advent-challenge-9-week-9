"""Дом: комнаты, устройства и их действия; что верно в ответе мозга; промпт, с которым мозг понимает фразы без обучения.

Ответ мозга — JSON-список команд «комната устройство действие»: ["кухня свет вкл", "кухня чайник вкл"].
Не команда дому — пустой список [].
"""
import json
import re

ROOMS = ("кухня", "гостиная", "спальня", "детская", "ванная", "прихожая")
LIVING = ("кухня", "гостиная", "спальня", "детская")
DEVICES = {  # устройство: (в каких комнатах, что умеет)
    "свет": (ROOMS, ("вкл", "выкл", "ярче", "тусклее")),
    "шторы": (LIVING, ("открыть", "закрыть")),
    "окно": (LIVING, ("открыть", "закрыть")),
    "отопление": (ROOMS[:5], ("теплее", "холоднее")),
    "музыка": (ROOMS[:5], ("вкл", "выкл", "громче", "тише")),
    "телевизор": (("гостиная",), ("вкл", "выкл", "громче", "тише")),
    "чайник": (("кухня",), ("вкл",)),
    "кофеварка": (("кухня",), ("вкл",)),
    "пылесос": (ROOMS[:5], ("убрать",)),
    "замок": (("прихожая",), ("закрыть", "открыть")),
}
KINDS = {  # виды фраз в тесте
    "direct": "прямая команда",
    "multi": "несколько команд",
    "indirect": "косвенная просьба",
    "none": "не команда",
}


def valid(command):
    """Команда, которую дом может выполнить: такая комната, такое устройство в ней, такое действие."""
    parts = command.split() if isinstance(command, str) else []
    if len(parts) != 3:
        return False
    room, device, action = parts
    return device in DEVICES and room in DEVICES[device][0] and action in DEVICES[device][1]


def parse(text):
    """Ответ глазами дома: список строк-команд или None, если это не JSON-список. Обёртку ```json дом снимет сам."""
    text = re.sub(r"^```\w*\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if isinstance(value, dict) and len(value) == 1:  # {"команды": [...]} — тоже понятно
        value = next(iter(value.values()))
    if not isinstance(value, list) or not all(isinstance(c, str) for c in value):
        return None
    return [" ".join(c.lower().replace("ё", "е").split()) for c in value]


def answer(commands):
    """Эталонный вид ответа — всегда одной функцией: команды без повторов, в порядке дома."""
    return json.dumps(sorted(set(commands), key=_order), ensure_ascii=False)


def check(row, text):
    """Ответ против строки теста: need — что обязано быть (список — любая из команд), may — что можно сверх того.

    Верно, если все команды выполнимы, все обязательные есть, а лишних (кроме допустимых) нет.
    """
    need = [n if isinstance(n, list) else [n] for n in row["need"]]
    commands = parse(text)
    if commands is None:
        return {"format": False, "ok": False, "missing": [n[0] for n in need], "extra": []}
    allowed = {c for n in need for c in n} | set(row.get("may", []))
    bad = [c for c in commands if not valid(c)]
    missing = [n[0] for n in need if not any(c in commands for c in n)]
    extra = [c for c in commands if c not in allowed]
    return {"format": not bad, "ok": not bad and not missing and not extra, "missing": missing, "extra": extra}


def request(room, phrase):
    """Что слышит мозг: в какой комнате колонка и сама фраза."""
    return f"[{room}] {phrase}"


def table():
    """Устройства дома строками — для промпта."""
    return "\n".join(f"- {device}: {', '.join(rooms) if len(rooms) < len(ROOMS) else 'все комнаты'} — {' / '.join(acts)}"
                     for device, (rooms, acts) in DEVICES.items())


RULES = f"""Ты — мозг умного дома. Колонка слышит фразу человека и комнату, где он стоит: «[комната] фраза».
Ответь только JSON-списком команд дому. Команда — строка «комната устройство действие».

Комнаты: {", ".join(ROOMS)}.
Устройства — где есть и что умеют:
{table()}

Правила:
- комната не названа — та, где колонка; «везде», «во всём доме» — все комнаты, где есть устройство;
- косвенную просьбу пойми по смыслу: «холодно» — отопление теплее; «душно» — окно открыть;
  «ухожу» — свет выкл везде, замок закрыть;
- не просьба к дому (вопрос, болтовня, то, чего дом не умеет) — пустой список [].
"""

EXAMPLES = [  # примеры для промпта «до обучения» — их нет в тесте
    ("кухня", "включи свет", ["кухня свет вкл"]),
    ("спальня", "закрой шторы и сделай музыку потише", ["спальня шторы закрыть", "спальня музыка тише"]),
    ("гостиная", "что-то я замёрз", ["гостиная отопление теплее"]),
    ("детская", "какая завтра погода?", []),
]


def prompt():
    """Промпт мозга без обучения: правила и 4 примера."""
    shots = "\n\n".join(f"Слышу: {request(room, phrase)}\nОтвет: {answer(cmds)}" for room, phrase, cmds in EXAMPLES)
    return f"{RULES}\nПримеры:\n\n{shots}\n\nОтвечай только JSON-списком, как «Ответ» в примерах."


def _order(command):
    room, device, _ = command.split()
    return ROOMS.index(room) if room in ROOMS else 99, list(DEVICES).index(device) if device in DEVICES else 99, command
