"""Модель: токенизатор, веса в bf16, LoRA-вставки, адаптер и генерация пачкой."""
import contextlib
import functools

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from app import config


@functools.cache
def tokenizer():
    """Токенизатор модели; заглушки слева — при генерации пачкой ответы начинаются в одной позиции."""
    tok = AutoTokenizer.from_pretrained(config.MODEL)
    tok.padding_side = "left"
    return tok


def lora_config(r=config.LORA_R, alpha=config.LORA_ALPHA, dropout=config.LORA_DROPOUT):
    """LoRA во все линейные слои: рядом с каждой матрицей W — две узкие матрицы B·A шириной r."""
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules="all-linear", task_type="CAUSAL_LM")


def load(adapter=False):
    """Базовая модель в bf16 — на процессоре с AVX-512 вдвое быстрее fp32; adapter=True — с обученными вставками."""
    torch.set_num_threads(config.THREADS)
    base = AutoModelForCausalLM.from_pretrained(config.MODEL, dtype=torch.bfloat16)
    return PeftModel.from_pretrained(base, config.ADAPTER_DIR) if adapter else base


def with_lora(base):
    """Модель со свежими вставками: веса самой модели заморожены, учатся только вставки."""
    return get_peft_model(base, lora_config())


@functools.cache
def info(r=config.LORA_R, alpha=config.LORA_ALPHA):
    """Устройство модели без загрузки весов: параметры всего и обучаемых, слои, куда встали вставки."""
    cfg = AutoConfig.from_pretrained(config.MODEL)
    with torch.device("meta"):
        net = AutoModelForCausalLM.from_config(cfg)
    total = sum(p.numel() for p in net.parameters())
    net = get_peft_model(net, lora_config(r, alpha))
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    text = getattr(cfg, "text_config", cfg)
    kinds = getattr(text, "layer_types", None) or ["full_attention"] * text.num_hidden_layers
    return {
        "name": config.MODEL,
        "total": total,
        "trainable": trainable,
        "layers": kinds,
        "targets": sorted(net.peft_config["default"].target_modules),
        "matrices": sum(isinstance(m, LoraLayer) for m in net.modules()),
        "r": r,
        "alpha": alpha,
    }


def plain(net):
    """Модель без вставок: у дообученной — временно выключить адаптер, у необученной — как есть."""
    return net.disable_adapter() if isinstance(net, PeftModel) else contextlib.nullcontext()


def generate(net, prompts, max_new_tokens):
    """Ответы пачкой, жадно — всегда самый вероятный токен, поэтому повторяемо от запуска к запуску."""
    tok = tokenizer()
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    with torch.no_grad():
        out = net.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                           top_k=None, pad_token_id=tok.pad_token_id,
                           eos_token_id=tok.eos_token_id)  # стоп на <|im_end|>: в конфиге модели конец — <|endoftext|>,
                                                           # а дообученная после <|im_end|> может писать дальше
    start = enc["input_ids"].shape[1]
    return [tok.decode(row[start:], skip_special_tokens=True).strip() for row in out]
