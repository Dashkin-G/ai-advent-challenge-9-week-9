"""Датасет пирата: train и valid — из генератора, экзамен — eval.jsonl (написан руками) и вопросы о фактах из досье.

Сборка: python -m pirate.dataset
"""
import collections
import json
import re

from pirate import config, generate, persona, rules

KINDS = ("format", "style", "fact", "para", "general")  # формат · голос · факты как учили · другими словами · чужие задачи


def build():
    """Собрать train / valid, проверить их и записать; вернуть размеры."""
    parts = generate.rows(config.SEED)
    problems = [f"{r['text']}: {why}" for part in parts.values() for r in part for why in _problems(r)]
    seen = {_letters(r["text"]) for part in parts.values() for r in part}
    problems += [f"экзамен «{r['text']}» есть в обучении" for r in exam() if r["kind"] != "fact"
                 and _letters(r["text"]) in seen]
    if problems:
        raise ValueError("Датасет не собран:\n" + "\n".join(problems))
    for name, part in parts.items():
        lines = (json.dumps(r, ensure_ascii=False) for r in part)
        (config.DATASET_DIR / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {name: dict(collections.Counter(r["kind"] for r in part)) for name, part in parts.items()}


def load(name):
    """train или valid: список {text, answer, kind, fact}."""
    path = config.DATASET_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exam():
    """100 строк экзамена: формат, голос, факты как учили (первый вопрос обучения), другими словами, чужие задачи."""
    path = config.DATASET_DIR / "eval.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows += [{"text": f["train"][0], "kind": "fact", "fact": f["key"]} for f in persona.FACTS]
    rows += [{"text": f["para"], "kind": "para", "fact": f["key"]} for f in persona.FACTS]
    return sorted(rows, key=lambda r: KINDS.index(r["kind"]))


def _problems(row):
    """Что не так с примером: формат, голос, факт в болтовне — его модель должна видеть только в вопросах о фактах."""
    text = rules.answer(*row["answer"].values())
    found = [] if rules.format_ok(text) else ["ответ не по формату"]
    found += [f"голос: нет «{m}»" for m, ok in rules.style(text)["marks"].items() if not ok]
    leaks = [key for key in rules.FACTS if rules.fact(text, key) and key != row.get("fact")]
    found += [f"в ответе факт «{key}»" for key in leaks]
    return found


def _letters(text):
    return re.sub(r"[^\w]+", " ", text.lower().replace("ё", "е")).strip()


if __name__ == "__main__":
    print(build())
    print({k: sum(r["kind"] == k for r in exam()) for k in KINDS})
