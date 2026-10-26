"""Язык робота SCAN: правильные действия по грамматике команды и разбор ответа модели."""
import re

WORDS = ("walk", "run", "jump", "look", "turn", "left", "right", "opposite", "around", "twice", "thrice", "and", "after")
_VERB = {"walk": ["I_WALK"], "run": ["I_RUN"], "jump": ["I_JUMP"], "look": ["I_LOOK"], "turn": []}
_TURN = {"left": "I_TURN_LEFT", "right": "I_TURN_RIGHT"}
_TIMES = {"twice": 2, "thrice": 3}
_ACTION = re.compile(r"I_(?:WALK|RUN|JUMP|LOOK|TURN_LEFT|TURN_RIGHT)")
_INVENTED = re.compile(r"\bI_[A-Z]")  # любое «действие» в формате робота, хоть настоящее, хоть выдуманное


def expected(command):
    """Правильные действия для команды; None — команда не из языка робота."""
    try:
        return _sentence(command.lower().split())
    except (KeyError, IndexError, ValueError):
        return None


def _sentence(words):
    """«x and y» — по порядку, «x after y» — сначала y. Союз в команде SCAN не больше одного."""
    for joint in ("after", "and"):
        if joint in words:
            i = words.index(joint)
            left, right = _phrase(words[:i]), _phrase(words[i + 1:])
            return right + left if joint == "after" else left + right
    return _phrase(words)


def _phrase(words):
    """Глагол [opposite|around] [left|right] [twice|thrice]."""
    times = _TIMES.get(words[-1], 1)
    if times > 1:
        words = words[:-1]
    verb = _VERB[words[0]]
    if len(words) == 1:
        if not verb:
            raise ValueError("turn без направления")
        return verb * times
    turn = _TURN[words[-1]]
    if len(words) == 2:
        once = [turn] + verb                   # walk left: повернуться и идти
    elif len(words) == 3 and words[1] == "opposite":
        once = [turn, turn] + verb             # развернуться и идти
    elif len(words) == 3 and words[1] == "around":
        once = ([turn] + verb) * 4             # четыре раза «повернуться и идти» — круг
    else:
        raise ValueError("лишние слова")
    return once * times


def actions(answer):
    """Действия, найденные в ответе модели, по порядку."""
    return _ACTION.findall(answer)


def clean(answer):
    """Ответ — только действия через пробел, без слов вокруг."""
    return bool(answer.strip()) and not _ACTION.sub("", answer).strip()


def check(expect, answer):
    """Верен ли ответ: список действий — робот сделал ровно их; строка — регэксп по тексту.

    [] — команда не из языка робота: верно, только если модель не выдумала своих действий вроде I_SWIM.
    """
    if isinstance(expect, str):
        return re.search(expect, answer, re.IGNORECASE) is not None
    if not expect:
        return _INVENTED.search(answer) is None
    return actions(answer) == expect
