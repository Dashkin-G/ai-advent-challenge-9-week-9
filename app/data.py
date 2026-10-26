"""Датасет SCAN: Hugging Face → выборка → шаблон чата модели → токены с маской «читает / учится писать»."""
import functools
import random

from datasets import load_dataset

from app import config, model, scan

_SPECIAL = {"<|im_start|>", "<|im_end|>", "<think>", "</think>"}


@functools.cache
def rows():
    """Все пары «команда → действия» из обеих частей SCAN: train и test."""
    ds = load_dataset(config.DATASET, data_dir=config.DATASET_DIR, revision="refs/convert/parquet")
    return {name: [{"command": c, "actions": a} for c, a in zip(part["commands"], part["actions"])]
            for name, part in ds.items()}


def splits():
    """Обучение и проверка — из train, перемешанного с SEED; test не трогаем — из него запросы для сравнения."""
    shuffled = list(rows()["train"])
    random.Random(config.SEED).shuffle(shuffled)
    return shuffled[:config.TRAIN_SIZE], shuffled[config.TRAIN_SIZE:config.TRAIN_SIZE + config.VAL_SIZE]


def prompt(command, hint=""):
    """Запрос в шаблоне чата модели — ровно то, что модель видит перед ответом."""
    return model.tokenizer().apply_chat_template([{"role": "user", "content": hint + command}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)


def encode(example):
    """Токены примера и метки: у запроса -100 (модель его только читает), у ответа — сами токены (учится писать)."""
    tok = model.tokenizer()
    head = tok(prompt(example["command"]), add_special_tokens=False).input_ids
    tail = tok(example["actions"] + tok.eos_token, add_special_tokens=False).input_ids
    return head + tail, [-100] * len(head) + tail


def pieces(example):
    """Пример по токенам для экрана: текст токена, служебный ли он и учится ли модель его писать."""
    tok = model.tokenizer()
    ids, labels = encode(example)
    return [{"text": tok.decode([i]), "special": tok.decode([i]) in _SPECIAL, "learn": label != -100}
            for i, label in zip(ids, labels)]


@functools.cache
def summary():
    """Что увидела модель: размеры, примеры от коротких к длинным, один пример по токенам, длины ответов, частота слов."""
    train, val = splits()
    short = [r for r in train[:300] if len(r["actions"].split()) <= 20]  # длиннее — не влезут строкой значков
    by_length = sorted(short, key=lambda r: len(r["actions"].split()))
    examples = [by_length[i * (len(by_length) - 1) // 5] for i in range(6)]
    sizes = [len(r["actions"].split()) for r in train]
    demo = next(r for r in train if 3 <= len(r["actions"].split()) <= 4)  # короткий — шаблон целиком влезает в карточку
    return {
        "name": config.DATASET,
        "total": sum(len(part) for part in rows().values()),
        "pool": len(rows()["train"]),
        "train": len(train),
        "val": len(val),
        "examples": examples,
        "demo": demo,
        "tokens": pieces(demo),
        "lengths": [{"from": a, "to": a + 7, "count": sum(a <= s <= a + 7 for s in sizes)} for a in range(1, 49, 8)],
        "words": {w: sum(w in r["command"].split() for r in train) for w in scan.WORDS},
    }


def find(command):
    """Пара «фраза → команды»: сначала среди обучающих примеров, потом во всём SCAN; None — такой фразы нет."""
    train, _ = splits()
    return next((r for r in train + rows()["train"] + rows()["test"] if r["command"] == command), None)


def browse(word="", offset=0, limit=24):
    """Обучающие примеры по порядку, как их увидела модель; word — только со словом."""
    train, _ = splits()
    found = [r for r in train if not word or word in r["command"].split()]
    return {"total": len(found), "items": found[offset:offset + limit]}
