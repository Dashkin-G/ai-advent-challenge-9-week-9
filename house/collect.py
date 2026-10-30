"""Датасет от учителя: учитель придумывает фразы → размечает каждую дважды, в разных пачках → код выкидывает плохое.

Всё, что ответил учитель, копится в raw.jsonl: повторный запуск не платит за готовое.
Запуск: python -m house.collect
"""
import collections
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

from house import config, home, teacher

TOPICS = [  # о чём просить учителя придумать фразы: устройства, связки, косвенные просьбы, не команды
    "свет: включить, выключить, ярче, тусклее — в своей комнате и в другой",
    "шторы и окна: открыть, закрыть, задёрнуть, проветрить",
    "отопление: теплее, холоднее — прямо, без слов «холодно» и «жарко»",
    "музыка и телевизор: включить, выключить, громче, тише — разговорные слова: телек, врубить, вырубить",
    "чайник и кофеварка — прямо: поставь, вскипяти, сделай",
    "робот-пылесос: убрать в комнате, пропылесосить",
    "входная дверь: запереть, открыть замок",
    "две команды в одной фразе через «и», «а», «потом»",
    "команда для другой комнаты: колонка в одной, просьба про другую",
    "«везде», «во всём доме», «во всех комнатах» — одно действие для всех комнат",
    "холодно, мёрзну, зябко, прохладно — в разных словах",
    "жарко, душно, нечем дышать, парилка",
    "темно, не вижу, глаза режет, слишком ярко",
    "шумно, тихо, хочется тишины, голова болит от звука",
    "ухожу из дома, вернулся домой, пришёл с работы",
    "ложусь спать, проснулся, утро, ребёнок уснул",
    "хочется чаю или кофе, голоден, гости пришли — по-разному",
    "грязно, насорили, крошки, шерсть от собаки",
    "кино, вечеринка, романтический ужин, чтение книги",
    "фразы с вежливостью и паразитами: пожалуйста, слушай, а можешь, будь добр",
    "очень короткие команды из 1–3 слов",
    "длинные фразы с лишними подробностями про свой день и одной просьбой в конце",
    "вопросы не к дому: погода, факты, математика, новости",
    "болтовня: привет, как дела, спасибо, шутки",
    "просьбы, которых дом не умеет: стирка, такси, еда, будильник, уборка посуды, кормить кота",
    "фразы со словами дома, но не просьбы: «свет мой зеркальце», «чайник» в переносном смысле, рассказ о вчерашнем",
]
PER_TOPIC = 60   # фраз на тему: 26 тем × 60 ≈ 1 500
CHUNK = 20       # фраз за один вызов придумывания
LABEL = 10       # фраз в пачке разметки
WORKERS = 8      # вызовов учителя одновременно
RAW = config.DATASET_DIR / "raw.jsonl"


def invent(topic, n, seed):
    """Учитель придумывает n фраз на тему — разные комнаты колонки, живой разговорный язык."""
    rooms = ", ".join(home.ROOMS)
    result = teacher.call([{"role": "user", "content":
        f"Придумай {n} разных фраз, которые человек говорит колонке умного дома. Тема: {topic}.\n"
        f"Колонка стоит в одной из комнат: {rooms}. Устройства дома: свет, шторы, окно, отопление, музыка, "
        f"телевизор, чайник, кофеварка, робот-пылесос, замок входной двери.\n"
        f"Пиши живо и разговорно, как говорят дома, иногда с ошибками; не повторяйся. Вариант {seed}.\n"
        'Ответ — JSON-объект {"фразы": [{"комната": "...", "фраза": "..."}, ...]}'}], think=0)
    try:
        items = json.loads(result["text"])["фразы"]
    except (ValueError, KeyError, TypeError):
        items = []
    rows = [{"room": i.get("комната"), "text": " ".join(str(i.get("фраза", "")).split()), "topic": topic}
            for i in items if isinstance(i, dict) and i.get("комната") in home.ROOMS and i.get("фраза")]
    return rows, result


def build():
    """Собрать train / valid: придумать → разметить дважды → чистка по шагам; отчёт — report.json."""
    raw = _load_raw()
    started = time.perf_counter()
    spent = collections.Counter(raw["spent"])
    if not raw["rows"]:
        jobs = [(t, CHUNK, k) for t in TOPICS for k in range(PER_TOPIC // CHUNK)]
        with ThreadPoolExecutor(WORKERS) as pool:
            for rows, result in pool.map(lambda j: _retry(invent, *j), jobs):
                raw["rows"] += rows
                _spend(spent, "invent", result)
        raw["spent"] = dict(spent)
        _save_raw(raw)
        print(f"придумано {len(raw['rows'])} фраз", flush=True)
    for pass_no in (1, 2):  # второй проход — те же фразы в других пачках: где учитель противоречит себе, он не уверен
        key = f"label{pass_no}"
        todo = [i for i, r in enumerate(raw["rows"]) if key not in r]
        order = todo[:] if pass_no == 1 else random.Random(config.SEED).sample(todo, len(todo))
        batches = [order[k:k + LABEL] for k in range(0, len(order), LABEL)]
        with ThreadPoolExecutor(WORKERS) as pool:
            for batch, result in zip(batches, pool.map(lambda b: _retry(teacher.label, [raw["rows"][i] for i in b]),
                                                       batches)):
                for i, value in zip(batch, result["answers"]):
                    raw["rows"][i][key] = value
                _spend(spent, "label", result)
        raw["spent"] = dict(spent)
        _save_raw(raw)
        print(f"разметка {pass_no}: {len(todo)} фраз", flush=True)
    raw["seconds"] = raw.get("seconds", 0) + round(time.perf_counter() - started)
    _save_raw(raw)
    return clean(raw)


def clean(raw):
    """Чистка по шагам: каждый шаг пишет, сколько убрал и примеры."""
    rows = raw["rows"]
    tests = {_letters(r["text"]) for r in load("test")}
    steps = [{"name": "учитель придумал", "left": len(rows)}]

    def step(name, why, keep):
        nonlocal rows
        kept, dropped = [], []
        for row in rows:  # keep вызываем раз на пример: у «дублей» он помнит, что уже видел
            (kept if keep(row) else dropped).append(row)
        rows = kept
        steps.append({"name": name, "why": why, "removed": len(dropped), "left": len(rows),
                      "examples": [{"room": r["room"], "text": r["text"], "label": r.get("label1")} for r in dropped[:4]]})

    step("мусор во фразе", "иероглифы или латиница посреди русской фразы: «я тут凍у»",
         lambda r: not re.search(r"[　-鿿]|[а-яё][a-z]|[a-z][а-яё]", r["text"], re.I))
    seen = set()
    step("дубли", "та же фраза в той же комнате, без регистра и знаков",
         lambda r: _first(seen, (r["room"], _letters(r["text"]))))
    step("пересечения с тестом", "та же фраза, что в тесте", lambda r: _letters(r["text"]) not in tests)
    step("учитель не ответил", "в ответе нет списка команд для фразы", lambda r: _commands(r, "label1") is not None)
    step("невыполнимые команды", "нет такой комнаты, устройства или действия: «кухня телевизор вкл»",
         lambda r: all(home.valid(c) for c in _commands(r, "label1")))
    step("учитель не уверен", "во второй пачке ответил иначе",
         lambda r: _commands(r, "label2") is not None and set(_commands(r, "label1")) == set(_commands(r, "label2")))

    random.Random(config.SEED).shuffle(rows)
    cut = round(len(rows) * config.VALID_SHARE)
    parts = {"valid": rows[:cut], "train": rows[cut:]}
    for name, part in parts.items():
        _write(name, [{"room": r["room"], "text": r["text"], "answer": home.answer(_commands(r, "label1")),
                       "topic": r["topic"]} for r in part])
    spent = raw["spent"]
    report = {
        "teacher": config.TEACHER,
        "steps": steps,
        "sizes": {"train": len(parts["train"]), "valid": len(parts["valid"]), "test": len(load("test"))},
        "spent": spent,
        "usd": round(sum(teacher.price(v["in"], v["out"]) for v in spent.values()), 4),
        "seconds": raw.get("seconds", 0),
        "empty": sum(r["answer"] == "[]" for r in load("train")),
        "commands": dict(sorted(collections.Counter(len(json.loads(r["answer"])) for r in load("train")).items())),
    }
    (config.DATASET_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def load(name):
    """train, valid или test: список строк."""
    path = config.DATASET_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def report():
    path = config.DATASET_DIR / "report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _commands(row, key):
    """Команды из ответа учителя: список строк в едином виде или None."""
    value = row.get(key)
    return home.parse(json.dumps(value, ensure_ascii=False)) if isinstance(value, list) else None


def _retry(work, *args):
    """Сбой сети или лимит — ещё две попытки с паузой."""
    for attempt in range(3):
        try:
            return work(*args)
        except teacher.TeacherError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def _spend(spent, what, result):
    item = dict(spent.get(what) or {"calls": 0, "in": 0, "out": 0, "seconds": 0})
    item["calls"] += 1
    item["in"] += result["in"]
    item["out"] += result["out"]
    item["seconds"] = round(item["seconds"] + result["seconds"], 1)
    spent[what] = item


def _load_raw():
    if not RAW.exists():
        return {"rows": [], "spent": {}}
    lines = RAW.read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    return {**head, "rows": [json.loads(line) for line in lines[1:] if line.strip()]}


def _save_raw(raw):
    head = {k: v for k, v in raw.items() if k != "rows"}
    lines = [json.dumps(head, ensure_ascii=False)] + [json.dumps(r, ensure_ascii=False) for r in raw["rows"]]
    RAW.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first(seen, key):
    if key in seen:
        return False
    seen.add(key)
    return True


def _letters(text):
    return re.sub(r"[^\w]+", " ", text.lower().replace("ё", "е")).strip()


def _write(name, rows):
    lines = (json.dumps(r, ensure_ascii=False) for r in rows)
    (config.DATASET_DIR / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = build()
    for s in result["steps"]:
        print(f"{s['name']:>22}: −{s.get('removed', 0):<4} осталось {s['left']}")
    print(result["sizes"], f"${result['usd']}", result["spent"])
