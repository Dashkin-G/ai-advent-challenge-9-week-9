"""Дообучение на уровне hello world: один вопрос → один ответ; после каждого шага видно, что он сделал.

Запуск: python simply.py   (около минуты на процессоре)
"""
import logging
import math
import pathlib

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

transformers.logging.set_verbosity_error()   # без служебных предупреждений и полосок загрузки
transformers.logging.disable_progress_bar()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)   # совет завести HF_TOKEN: открытой модели он не нужен

MODEL = "Qwen/Qwen3.5-0.8B"
QUESTION = "Как зовут моего кота?"
ANSWER = "Барсик."
STEPS = 10    # сколько раз показать модели пример
LR = 1e-4     # learning rate — на сколько сдвигать числа за шаг
FOLDER = pathlib.Path(__file__).parent / "data" / "simply"   # куда сохранить результат
FILES = {"adapter_model.safetensors": "числа A и B всех вставок",
         "adapter_config.json": "r, alpha и в какие матрицы вставки",
         "README.md": "карточка модели — её пишет peft"}


def chat(question):
    """Вопрос в шаблоне чата Qwen — ровно тот текст, который модель видит перед ответом."""
    return tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def ask(model, question):
    """Спросить модель: текст → токены → generate (всегда самый вероятный токен) → текст ответа."""
    ids = tok(chat(question), return_tensors="pt").input_ids
    out = model.generate(input_ids=ids, max_new_tokens=40, do_sample=False,
                         eos_token_id=tok.eos_token_id)   # <|im_end|> — конец ответа, дальше не писать
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def show(numbers):
    return "[" + ", ".join(f"{x:+.4f}" for x in numbers.tolist()) + "]"


print("1. Загружаем модель")
tok = AutoTokenizer.from_pretrained(MODEL)                                  # текст ↔ токены (кусочки текста)
qwen = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)   # сама модель — 752 млн чисел-весов
print(f"   {MODEL}: {sum(p.numel() for p in qwen.parameters()):,} весов")

print("\n2. Спрашиваем ДО обучения")
print(f"   {QUESTION} → {ask(qwen, QUESTION)}")

print("\n3. Пример: что модель читает и что учится писать")
read = tok(chat(QUESTION)).input_ids          # вопрос в шаблоне — модель его только читает
learn = tok(ANSWER + "<|im_end|>").input_ids  # ответ и знак конца — его модель учится писать
input_ids = torch.tensor([read + learn])
labels = torch.tensor([[-100] * len(read) + learn])  # -100 — «здесь не учиться»
print("   читает:       ", tok.decode(read).replace("\n", "↵"))
print("   учится писать:", " | ".join(tok.decode([i]) for i in learn))
print("   input_ids =", input_ids[0].tolist())
print("   labels    =", labels[0].tolist())

print("\n4. Добавляем LoRA — маленькие обучаемые вставки")
torch.manual_seed(0)   # случайные числа в A — одни и те же при каждом запуске
model = get_peft_model(qwen, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear"))
model.print_trainable_parameters()
count = sum(name.endswith("lora_A.default.weight") for name, _ in model.named_parameters())
print(f"   вставка — рядом с каждой из {count} матриц W: выход = W·x + B·A·x")
layer = qwen.model.layers[0].mlp.down_proj   # одна из этих матриц — следим за ней до конца
W = layer.base_layer.weight                  # вес Qwen
A = layer.lora_A["default"].weight           # вставка: случайные числа
B = layer.lora_B["default"].weight           # вставка: нули — поэтому сначала она ничего не меняет
W_start = W[0, :3].clone()
print(f"   W {tuple(W.shape)}  requires_grad={W.requires_grad}  W[0, :3] = {show(W[0, :3])}")
print(f"   A {tuple(A.shape)}     requires_grad={A.requires_grad}")
print(f"   B {tuple(B.shape)}     requires_grad={B.requires_grad}   B[0, :3] = {show(B[0, :3])}")

print("\n5. Один шаг обучения — что делает каждая строка")
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)  # двигает только вставки
out = model(input_ids=input_ids, labels=labels)
print(f"   model(input_ids, labels) → loss = {out.loss.item():.2f}: ошибка на токенах ответа, "
      f"уверенность {math.exp(-out.loss.item()):.0%}")
out.loss.backward()
print(f"   loss.backward()          → B.grad[0, :3] = {show(B.grad[0, :3])} — куда сдвигать; W.grad = {W.grad}")
optimizer.step()
print(f"   optimizer.step()         → B[0, :3] = {show(B[0, :3])} — сдвиг на LR против градиента")
optimizer.zero_grad()
print(f"   optimizer.zero_grad()    → B.grad = {B.grad} — градиент стёрт до следующего шага")

print(f"\n6. Ещё {STEPS - 1} таких же шагов")
for step in range(2, STEPS + 1):
    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"   шаг {step:2}: ошибка {loss.item():.2f} · уверенность {math.exp(-loss.item()):.0%}")

print("\n7. Спрашиваем ПОСЛЕ обучения")
print(f"   {QUESTION} → {ask(model, QUESTION)}")
with model.disable_adapter():                 # те же веса Qwen, вставки выключены
    print(f"   без вставок → {ask(model, QUESTION)}")

print("\n8. Что изменилось в весах")
print(f"   W[0, :3] = {show(W[0, :3])} — было {show(W_start)}: вес Qwen не тронут")
print(f"   B[0, :3] = {show(B[0, :3])} — были нули")
delta = B @ A * layer.scaling["default"]      # добавка к W: B·A, умноженное на lora_alpha / r = 2
print(f"   добавка B·A·2 к этой матрице — {(delta.norm() / W.float().norm()).item():.2%} от W")

print("\n9. Сохраняем результат — только вставки, без весов Qwen")
model.save_pretrained(FOLDER)
for name, about in FILES.items():
    size = (FOLDER / name).stat().st_size
    print(f"   {name:27} {f'{size / 2**20:.1f} МБ' if size > 2**20 else f'{size / 2**10:.0f} КБ':>8} — {about}")
# Потом: PeftModel.from_pretrained(qwen, FOLDER) — Qwen с этими вставками, без обучения заново
