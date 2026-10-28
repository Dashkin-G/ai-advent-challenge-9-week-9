"""Сборка своего датасета: генератор → чистка по шагам → train и valid. Каждый шаг пишет, сколько убрал и что именно.

Запуск: python -m remind.dataset
"""
import collections
import json
import random
import re

from remind import config, generate, rules

FILLER = {"пожалуйста", "мне", "слушай", "ка"}  # слова, без которых фраза значит то же самое


def build():
    """Собрать train / valid, записать их и отчёт report.json; вернуть отчёт."""
    rows = generate.rows(config.RAW, config.SEED)
    evals = load("eval")
    review = _review()
    steps = [{"name": "собрали генератором", "left": len(rows)}]

    def step(name, why, keep):
        nonlocal rows
        kept, dropped = [], []
        for row in rows:
            (kept if keep(row) else dropped).append(row)
        rows = kept
        steps.append({"name": name, "why": why, "removed": len(dropped), "left": len(rows),
                      "examples": [r["text"] for r in dropped[:4]]})

    for row in rows:
        row["text"] = " ".join(row["text"].split())
    seen = set()
    step("дубли", "та же фраза до буквы, без регистра и знаков", lambda r: _first(seen, (_letters(r["text"]),)))
    seen = set()
    step("почти-дубли", "те же слова в другом порядке или с «пожалуйста», «мне» — и тот же ответ",
         lambda r: _first(seen, (_bag(r["text"]), r["answer"])))
    step("ошибки формата", "ответ не по правилам или дела нет во фразе", lambda r: not _problems(r))
    step("брак глазами", "отмечены ✗ при просмотре", lambda r: r["text"] not in review["bad"])
    eval_bags, eval_tasks = {_bag(e["text"]) for e in evals}, {e["answer"]["дело"] for e in evals} - {None}
    step("пересечения с eval", "та же фраза или то же дело, что в экзамене",
         lambda r: _bag(r["text"]) not in eval_bags and json.loads(r["answer"])["дело"] not in eval_tasks)

    random.Random(config.SEED).shuffle(rows)
    cut = round(len(rows) * config.VALID_SHARE)
    parts = {"valid": rows[:cut], "train": rows[cut:]}
    for name, part in parts.items():
        _write(name, [{"text": r["text"], "answer": json.loads(r["answer"]), "kind": r["kind"]} for r in part])
    report = {
        "steps": steps,
        "sizes": {"train": len(parts["train"]), "valid": len(parts["valid"]), "eval": len(evals)},
        "balance": {what: dict(collections.Counter(r["kind"][what] for r in parts["train"]).most_common())
                    for what in ("day", "time")},
        "tasks": len({json.loads(r["answer"])["дело"] for r in parts["train"]} - {None}),
        "reviewed": len(review["seen"]),
    }
    (config.DATASET_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def load(name):
    """train, valid или eval: список {text, answer, kind | check}."""
    path = config.DATASET_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def report():
    path = config.DATASET_DIR / "report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def sample(page, size=30):
    """30 случайных обучающих примеров для просмотра глазами; page — следующая порция."""
    rows = load("train")
    random.Random(config.SEED + page).shuffle(rows)
    review = _review()
    return [{**r, "mark": "bad" if r["text"] in review["bad"] else "good" if r["text"] in review["seen"] else None}
            for r in rows[:size]]


def mark(text, verdict):
    """Отметка глазами: good — верно, bad — ошибка (выкинем при следующей сборке), None — снять отметку."""
    review = _review()
    review["seen"] = [t for t in review["seen"] if t != text] + ([text] if verdict else [])
    review["bad"] = [t for t in review["bad"] if t != text] + ([text] if verdict == "bad" else [])
    config.REVIEW_FILE.write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"seen": len(review["seen"]), "bad": len(review["bad"])}


def _review():
    if not config.REVIEW_FILE.exists():
        return {"seen": [], "bad": []}
    return json.loads(config.REVIEW_FILE.read_text(encoding="utf-8"))


def _first(seen, key):
    """Первый раз видим ключ — оставить пример, повтор — убрать."""
    if key in seen:
        return False
    seen.add(key)
    return True


def _letters(text):
    return re.sub(r"[^\w]+", " ", text.lower().replace("ё", "е")).strip()


def _bag(text):
    """Мешок слов: порядок, регистр, знаки и слова-паразиты не важны; «9:00» и «12.11» — одно слово."""
    words = re.findall(r"\d+[:.]\d+|\w+", text.lower().replace("ё", "е"))
    return tuple(sorted(w for w in words if w not in FILLER))


def _problems(row):
    value = json.loads(row["answer"])
    found = rules.problems(value)
    if value["дело"] and value["дело"].lower() not in row["text"].lower():
        found.append("дела нет во фразе")
    return found


def _write(name, rows):
    lines = (json.dumps(r, ensure_ascii=False) for r in rows)
    (config.DATASET_DIR / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = build()
    for s in result["steps"]:
        print(f"{s['name']:>22}: −{s.get('removed', 0):<4} осталось {s['left']}  {s.get('examples', [])[:2]}")
    print(result["sizes"], result["balance"], "дел:", result["tasks"])
