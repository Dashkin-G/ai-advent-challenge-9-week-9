"""Задание дня 6 «Первый тюн на готовом датасете» целиком в одном файле, с выводом в консоль.

Готовый датасет SCAN с Hugging Face (команда на английском → действия робота) → шаблон чата Qwen → LoRA → лосс →
адаптер; 20 запросов на базовой и дообученной модели: что изменилось, что осталось прежним, где стало хуже.

Запуск: python task.py   (около 35 мин на процессоре; обучению нужно около 6 ГБ свободной памяти)
"""
import logging
import pathlib
import platform
import random
import re
import time

import peft
import psutil
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

transformers.logging.set_verbosity_error()   # без служебных предупреждений и полосок загрузки
transformers.logging.disable_progress_bar()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)   # совет завести HF_TOKEN: открытой модели он не нужен

MODEL = "Qwen/Qwen3.5-0.8B"
DATASET = "scan-tasks/scan-tasks"   # SCAN: команда на английском → действия робота I_WALK, I_JUMP…
TRAIN_SIZE = 2000                   # примеров на обучение из 16 728
BATCH = 8                           # примеров за шаг
LR = 3e-4                           # learning rate — размер шага
WARMUP = 10                         # шагов плавного разгона LR, дальше — спад по косинусу
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear")
MAX_TOKENS = 160                    # длина ответа: хватает на самую длинную команду — 48 действий
SEED = 42
ROOT = pathlib.Path(__file__).parent
FOLDER = ROOT / "data" / "task"     # куда сохранить адаптер

COMMANDS = [  # 15 команд из теста SCAN: на обучении их не было, правильный ответ берём из датасета
    "run twice after look",
    "turn opposite left and walk",
    "jump and look left",
    "walk opposite right after run left",
    "run and jump right thrice",
    "jump left twice after jump right twice",
    "walk thrice and turn opposite left thrice",
    "look opposite right thrice after look opposite left",
    "jump opposite right and turn around left thrice",
    "walk opposite left and look around left twice",
    "run around right twice after look left twice",
    "look opposite right thrice after run around right twice",
    "jump around right thrice and turn around left thrice",
    "walk around right thrice after look around left twice",
    "run around right thrice after look around right thrice",
]
CHECKS = {  # 5 проверок на прочность: список — нужные действия, строка — что должно быть в ответе
    "walk twice and jump twice and look twice": ["I_WALK", "I_WALK", "I_JUMP", "I_JUMP", "I_LOOK", "I_LOOK"],
    "прыгни дважды и повернись налево": ["I_JUMP", "I_JUMP", "I_TURN_LEFT"],
    "swim twice": [],   # незнакомое слово: верно — не выдумать действий вроде I_SWIM
    "What is the capital of France? Answer in one word.": "Paris|Париж",
    "Сколько будет 17 + 25? Ответь одним числом.": r"\b42\b",
}
ACTION = re.compile(r"I_(?:WALK|RUN|JUMP|LOOK|TURN_LEFT|TURN_RIGHT)")


def chat(text):
    """Запрос в шаблоне чата Qwen — ровно то, что модель видит перед ответом."""
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def encode(row):
    """Пример → токены и метки: запрос модель только читает (-100), ответ учится писать."""
    read = tok(chat(row["command"])).input_ids
    learn = tok(row["actions"] + "<|im_end|>").input_ids
    return read + learn, [-100] * len(read) + learn


def pad(batch):
    """Пачка примеров в тензоры: короткие добиваются заглушками справа — маска 0, метка -100, в лосс не идут."""
    width = max(len(ids) for ids, _ in batch)
    ids = torch.tensor([x + [tok.pad_token_id] * (width - len(x)) for x, _ in batch])
    mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in batch])
    labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in batch])
    return ids, mask, labels


def ask(model, text):
    """Ответ модели — всегда самый вероятный токен, поэтому одинаковый от запуска к запуску."""
    ids = tok(chat(text), return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=MAX_TOKENS, do_sample=False,
                             eos_token_id=tok.eos_token_id)   # <|im_end|> — конец ответа, дальше не писать
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def correct(expect, answer):
    """Верен ли ответ: список — ровно эти действия, [] — без выдуманных действий, строка — регэксп."""
    if isinstance(expect, str):
        return re.search(expect, answer, re.IGNORECASE) is not None
    if not expect:
        return re.search(r"\bI_[A-Z]", answer) is None
    return ACTION.findall(answer) == expect


def only_actions(answer):
    """Ответ — только действия робота, без слов вокруг."""
    return bool(answer) and not ACTION.sub("", answer).strip()


def run(model, title):
    """20 запросов по очереди: ответ и ✓/✗ сразу, в конце — сколько верных и сколько заняло."""
    print(f"\n{title}")
    started = time.perf_counter()
    answers = []
    for n, (text, expect) in enumerate(queries, 1):
        answers.append(ask(model, text))
        print(f"   {n:2} {mark(correct(expect, answers[-1]))} {text} → {short(answers[-1])}")
    seconds = time.perf_counter() - started
    print(f"   верно {sum(correct(e, a) for (_, e), a in zip(queries, answers))} из {len(queries)} · {clock(seconds)}")
    return answers, seconds


def mark(ok):
    return "✓" if ok else "✗"


def short(text, width=80):
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 1] + "…"


def clock(seconds):
    return f"{int(seconds) // 60} мин {int(seconds) % 60:02d} с"


def cpu():
    """Имя процессора: на Windows — из реестра, иначе — что знает platform."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        return platform.processor()


print(f"Свободно памяти: {psutil.virtual_memory().available / 2**30:.1f} ГБ — обучению нужно около 6")

print("\n1. ДАТАСЕТ — готовый, с Hugging Face")
ds = load_dataset(DATASET, data_dir="simple", revision="refs/convert/parquet")  # parquet-копия: скрипт датасета устарел
pool = [{"command": c, "actions": a} for c, a in zip(ds["train"]["commands"], ds["train"]["actions"])]
random.Random(SEED).shuffle(pool)
train = pool[:TRAIN_SIZE]
truth = dict(zip(ds["test"]["commands"], ds["test"]["actions"]))
queries = [(c, truth[c].split()) for c in COMMANDS] + list(CHECKS.items())
print(f"   {DATASET} · simple: обучение {len(ds['train'])}, тест {len(ds['test'])} → берём {len(train)} примеров")
for row in train[:3]:
    print(f"   {row['command']} → {row['actions']}")

print("\n2. МОДЕЛЬ и ШАБЛОН ЧАТА")
tok = AutoTokenizer.from_pretrained(MODEL)
qwen = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
weights = sum(p.numel() for p in qwen.parameters())
print(f"   {MODEL}: {weights / 1e6:.0f} млн весов, bf16")
ids, labels = encode(train[0])
read = labels.count(-100)
print("   пример в шаблоне — модель читает:", tok.decode(ids[:read]).replace("\n", "↵"))
print("                      учится писать:", tok.decode(ids[read:]))

before, before_time = run(qwen, "3. ДО ОБУЧЕНИЯ — 20 запросов базовой модели")

print("\n4. ОБУЧЕНИЕ LoRA")
torch.manual_seed(SEED)
model = get_peft_model(qwen, LORA)   # веса Qwen заморожены, учатся только вставки
params = [p for p in model.parameters() if p.requires_grad]
trainable = sum(p.numel() for p in params)
matrices = sum(name.endswith("lora_A.default.weight") for name, _ in model.named_parameters())
print(f"   вставки в {matrices} матриц: учатся {trainable / 1e6:.1f} млн чисел — {trainable / weights:.1%} модели")
examples = sorted((encode(row) for row in train), key=lambda e: len(e[0]))  # соседи по длине — меньше заглушек
batches = [examples[i:i + BATCH] for i in range(0, len(examples), BATCH)]
random.Random(SEED).shuffle(batches)
optimizer = torch.optim.AdamW(params, lr=LR)
schedule = transformers.get_cosine_schedule_with_warmup(optimizer, WARMUP, len(batches))
model.train()   # включает dropout вставок
started = time.perf_counter()
losses = []
for step, batch in enumerate(batches, 1):
    ids, mask, labels = pad(batch)
    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss  # ошибка на токенах ответа
    loss.backward()                                                         # градиенты — только у вставок
    torch.nn.utils.clip_grad_norm_(params, 1.0)                             # срезать редкие выбросы
    optimizer.step()
    schedule.step()
    optimizer.zero_grad()
    losses.append(loss.item())
    if step == 1 or step % 25 == 0:
        recent = losses[-25:]
        print(f"   шаг {step:3}/{len(batches)} · лосс {sum(recent) / len(recent):.3f} · {clock(time.perf_counter() - started)}")
train_time = time.perf_counter() - started
model.eval()
model.save_pretrained(FOLDER)
size = sum(f.stat().st_size for f in FOLDER.iterdir())
print(f"   адаптер: {FOLDER.relative_to(ROOT)} — {size / 2**20:.1f} МБ")

after, after_time = run(model, "5. ПОСЛЕ ОБУЧЕНИЯ — те же 20 запросов")

print("\n6. СРАВНЕНИЕ")
print("   до → после  итог     ответ    запрос")
verdicts = []
for (text, expect), b, a in zip(queries, before, after):
    ok_b, ok_a = correct(expect, b), correct(expect, a)
    verdicts.append("так же" if ok_b == ok_a else "лучше" if ok_a else "хуже")
    print(f"   {mark(ok_b)}  →  {mark(ok_a)}     {verdicts[-1]:7}  {'тот же' if a == b else 'другой':7}  {text}")
right_b = sum(correct(e, b) for (_, e), b in zip(queries, before))
right_a = sum(correct(e, a) for (_, e), a in zip(queries, after))
print(f"\n   верно: {right_b} → {right_a} из 20 · лучше {verdicts.count('лучше')} · так же {verdicts.count('так же')} · "
      f"хуже {verdicts.count('хуже')}")
print(f"   ответ — только действия робота: {sum(map(only_actions, before))} → {sum(map(only_actions, after))} из 20")
print(f"   ответ остался слово в слово: {sum(a == b for a, b in zip(after, before))} из 20")
for (text, _), b, a, verdict in zip(queries, before, after, verdicts):
    if verdict == "хуже":
        print(f"   стало хуже: {text}\n      до:    {short(b)}\n      после: {short(a)}")

print("\n7. ПАРАМЕТРЫ ЗАПУСКА")
print(f"   модель     {MODEL} · {weights / 1e6:.0f} млн весов в bf16, вставки в fp32")
print(f"   датасет    {DATASET} · simple · {len(train)} примеров · 1 эпоха")
print(f"   обучение   {len(batches)} шагов · батч {BATCH} · AdamW · LR {LR:g}: разгон {WARMUP} шагов, спад по косинусу")
print(f"   LoRA       r {LORA.r} · alpha {LORA.lora_alpha} · dropout {LORA.lora_dropout} · {matrices} матриц: "
      f"{', '.join(sorted(LORA.target_modules))}")
print(f"   учится     {trainable:,} чисел — {trainable / weights:.1%} модели")
print(f"   железо     {cpu()} · {torch.get_num_threads()} потоков · {psutil.virtual_memory().total / 2**30:.0f} ГБ ОЗУ · без GPU")
print(f"   время      обучение {clock(train_time)} · 20 ответов до {clock(before_time)}, после {clock(after_time)}")
print(f"   адаптер    {FOLDER.relative_to(ROOT)} · {size / 2**20:.1f} МБ")
print(f"   версии     torch {torch.__version__} · transformers {transformers.__version__} · peft {peft.__version__} · "
      f"seed {SEED}")
