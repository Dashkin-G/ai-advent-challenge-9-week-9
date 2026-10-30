"""Сборка версии пирата для прода: вставки эпохи → слияние с весами → GGUF → квантование → Ollama → экзамен-ворота.

Без страницы: python -m pirate.release 4 [q4_0] — версия из эпохи 4 · base — база для мозга «промпт» ·
bench — замеры: PyTorch с вставками (как было) против GGUF bf16 / q8_0 / q4_0 в Ollama
"""
import datetime
import functools
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import gguf
import numpy as np
import psutil
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from app import config as base_config
from app import model
from pirate import brain, config, dataset, persona, rules, serve, train

QUANTS = ("bf16", "q8_0", "q4_0")
# Modelfile — всё, что Ollama нужно знать о версии, кроме весов: шаблон чата (как в обучении) и параметры ответа.
# num_gpu 0 — на том же процессоре, что PyTorch раньше: сравнение честное.
MODELFILE = '''FROM ./{gguf}
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""
PARAMETER temperature 0
PARAMETER num_predict {tokens}
PARAMETER num_ctx 2048
PARAMETER num_gpu 0
PARAMETER stop <|im_end|>
'''


def build(epoch=None, quant=config.QUANT, stop=None):
    """Версия из вставок эпохи; epoch None — база без тюна (kilka:base). События start → step… progress… → done."""
    stop = stop or threading.Event()
    n = 1 + max((c["n"] for c in serve.versions()), default=0)
    name, tag = ("base", serve.BASE) if epoch is None else (f"v{n}", f"{config.NAME}:v{n}")
    work, folder = config.PROD_DIR / "work", config.PROD_DIR / name
    for path in (work, folder):
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)
    file = f"{config.NAME}-{quant}.gguf"
    yield {"type": "start", "version": name, "epoch": epoch, "quant": quant, "tag": tag}
    steps, started = {}, time.perf_counter()

    def done(step, **info):
        nonlocal started
        steps[step] = {"seconds": round(time.perf_counter() - started, 1), **info}
        started = time.perf_counter()
        return {"type": "step", "step": step, **steps[step]}

    if epoch is None:
        src = source()
        yield done("merge", matrices=0)
    else:
        adapter = config.EPOCHS_DIR / str(epoch)
        src = work / "merged"
        yield done("merge", matrices=merge(adapter, src), adapter=(adapter / "adapter_model.safetensors").stat().st_size)
    if stop.is_set():
        yield from _stopped(work, folder)
        return
    convert(src, work / "bf16.gguf")
    yield done("gguf", size=(work / "bf16.gguf").stat().st_size)
    if quant == "bf16":
        shutil.move(work / "bf16.gguf", folder / file)
    else:
        quantize(work / "bf16.gguf", folder / file, quant)
    shutil.rmtree(work)
    yield done("quant", size=(folder / file).stat().st_size)
    if stop.is_set():
        yield from _stopped(work, folder)
        return
    register(tag, folder, file)
    yield done("ollama", tag=tag, digest=digest(folder / file))
    result = yield from exam(tag, persona.prompt() if epoch is None else None, stop)
    if result is None:
        _drop(tag)
        yield from _stopped(work, folder)
        return
    card = {"version": name, "n": 0 if epoch is None else n, "tag": tag, "epoch": epoch, "quant": quant,
            "date": datetime.datetime.now().isoformat(timespec="minutes"), "size": (folder / file).stat().st_size,
            "file": file, "steps": {**steps, "exam": {"seconds": round(time.perf_counter() - started, 1)}},
            "exam": {"summary": result["summary"], "total": sum(result["summary"].values()),
                     "seconds": result["seconds"], "load": result["answers"][0]["load"]}}
    (folder / "card.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    (folder / "exam.json").write_text(json.dumps(result["answers"], ensure_ascii=False), encoding="utf-8")
    live = serve.state()["live"]
    yield {"type": "done", "card": card, "gate": serve.gate(card, live and serve.card(live))}


def source():
    """Папка исходной модели в кэше Hugging Face: веса, конфиг, токенизатор и шаблон чата."""
    return Path(snapshot_download(base_config.MODEL, local_files_only=True))


def merge(adapter, out):
    """Слияние: W' = W + B·A · alpha/r — у каждой матрицы со вставкой; остальные веса — как в исходной модели.

    Дальше вставки не нужны: это обычная модель того же размера и в той же раскладке, её понимает любой движок.
    """
    src = source()
    lora = load_file(adapter / "adapter_model.safetensors")
    lora_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scale = lora_config["lora_alpha"] / lora_config["r"]
    weights = {}
    index = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for shard in sorted(set(index["weight_map"].values())):
        weights.update(load_file(src / shard))
    matrices = 0
    for key, a in lora.items():
        if not key.endswith(".lora_A.weight"):
            continue
        # base_model.model.model.layers.0.linear_attn.in_proj_a → model.language_model.layers.0.linear_attn.in_proj_a
        name = "model.language_model." + key.removeprefix("base_model.model.model.").removesuffix(".lora_A.weight")
        w, b = weights[f"{name}.weight"], lora[key.replace(".lora_A.", ".lora_B.")]
        weights[f"{name}.weight"] = (w.float() + b.float() @ a.float() * scale).to(w.dtype)
        matrices += 1
    out.mkdir(parents=True, exist_ok=True)
    save_file(weights, out / "model.safetensors", metadata={"format": "pt"})
    for path in src.iterdir():
        if path.suffix in (".json", ".jinja") and path.name != "model.safetensors.index.json":
            shutil.copy(path, out / path.name)
    return matrices


def convert(src, out):
    """HF-папка → один файл GGUF в bf16: конвертер llama.cpp раскладывает веса, токенизатор и шаблон под Ollama."""
    script = config.LLAMA_CPP / "convert_hf_to_gguf.py"
    if not script.exists():
        raise RuntimeError(f"Нет конвертера {script} — как его получить, написано в README")
    done = subprocess.run([sys.executable, "-X", "utf8", str(script), str(src), "--outfile", str(out), "--outtype", "bf16"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode:
        raise RuntimeError("Конвертер llama.cpp упал:\n" + done.stderr[-1500:])


def quantize(src, out, quant):
    """Квантование: матрица весов — блоками по 32 числа; у блока один общий множитель и 32 целых по 8 бит (q8_0)
    или по 4 (q4_0). Нормы и прочие мелкие тензоры остаются float32 — так их оставил конвертер."""
    qtype = gguf.GGMLQuantizationType[quant.upper()]
    reader = gguf.GGUFReader(src)
    writer = gguf.GGUFWriter(out, arch=reader.get_field(gguf.Keys.General.ARCHITECTURE).contents())
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue  # их пишет сам GGUFWriter
        value = gguf.LlamaFileType[f"MOSTLY_{quant.upper()}"] if field.name == gguf.Keys.General.FILE_TYPE \
            else field.contents()
        kind = field.types[0]
        writer.add_key_value(field.name, value, kind, sub_type=field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None)
    with np.errstate(all="ignore"):  # блок почти из нулей: numpy ругается на переполнение, а блок и так станет нулями
        for t in reader.tensors:
            if t.tensor_type == gguf.GGMLQuantizationType.BF16:
                writer.add_tensor(t.name, gguf.quants.quantize(gguf.quants.dequantize(t.data, t.tensor_type), qtype),
                                  raw_dtype=qtype)
            else:
                writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def register(tag, folder, file):
    """Версия в Ollama: Modelfile рядом с GGUF → ollama create; Ollama кладёт себе копию весов под этим именем."""
    (folder / "Modelfile").write_text(MODELFILE.replace("{gguf}", file).replace("{tokens}", str(brain.TOKENS)),
                                      encoding="utf-8")
    done = subprocess.run(["ollama", "create", tag, "-f", "Modelfile"], cwd=folder, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if done.returncode:
        raise RuntimeError(f"ollama create {tag}: {done.stderr.strip()[-500:]}")


def digest(path):
    """sha256 файла — так Ollama называет у себя веса: по нему видно, что в проде ровно то, что сдало экзамен."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            sha.update(chunk)
    return sha.hexdigest()


def exam(tag, system=None, stop=None):
    """100 реплик экзамена — модели в Ollama по одной, как в проде; события progress, итог — через return."""
    rows, answers = dataset.exam(), []
    for i, row in enumerate(rows):
        if stop is not None and stop.is_set():
            return None
        result = serve.generate(tag, row["text"], system)
        answers.append({**result, **rules.check(row, result["answer"])})
        yield {"type": "progress", "done": i + 1, "total": len(rows)}
    return {"summary": {k: sum(a["ok"] for a, r in zip(answers, rows) if r["kind"] == k) for k in dataset.KINDS},
            "seconds": round(sum(a["seconds"] for a in answers) / len(answers), 2), "answers": answers}


def bench(epoch=4):
    """Во что обходится один и тот же тюн: как было (PyTorch, база + вставки) и GGUF bf16 / q8_0 / q4_0 в Ollama —
    размер, память, загрузка, ответ, экзамен на 100 репликах. Итог — data/pirate/prod/bench.json."""
    rows, adapter = dataset.exam(), config.EPOCHS_DIR / str(epoch)
    report = {"date": datetime.datetime.now().isoformat(timespec="minutes"), "epoch": epoch}
    process = psutil.Process()
    memory, started = process.memory_info().rss, time.perf_counter()
    net = PeftModel.from_pretrained(model.load(), str(adapter)).eval()
    load, memory = time.perf_counter() - started, process.memory_info().rss - memory
    before = []
    for i, row in enumerate(rows):
        started = time.perf_counter()
        [answer] = model.generate(net, [train.prompt(row["text"])], brain.TOKENS)
        before.append({"answer": answer, "seconds": time.perf_counter() - started, **rules.check(row, answer)})
        yield {"type": "progress", "what": "torch", "done": i + 1, "total": len(rows)}
    size = sum(p.stat().st_size for p in source().glob("*.safetensors")) + (adapter / "adapter_model.safetensors").stat().st_size
    report["torch"] = _measure(rows, before, size, memory, load)
    del net
    gc.collect()

    work = config.PROD_DIR / "bench"
    shutil.rmtree(work, ignore_errors=True)
    merge(adapter, work / "merged")
    convert(work / "merged", work / "bf16.gguf")
    shutil.rmtree(work / "merged")
    for quant in QUANTS:
        file = f"{quant}.gguf"
        if quant != "bf16":
            quantize(work / "bf16.gguf", work / file, quant)
        tag = f"{config.NAME}-bench:{quant}"
        register(tag, work, file)
        for name in serve.loaded():  # выгрузить всё — загрузку меряем с холодного старта
            serve.call("/api/generate", {"model": name, "keep_alive": 0})
        after = []
        for i, row in enumerate(rows):
            result = serve.generate(tag, row["text"])
            after.append({**result, **rules.check(row, result["answer"])})
            yield {"type": "progress", "what": quant, "done": i + 1, "total": len(rows)}
        report[quant] = _measure(rows, after, (work / file).stat().st_size, serve.loaded()[tag], after[0]["load"], before)
        _drop(tag)
    shutil.rmtree(work)
    config.BENCH_FILE.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    yield {"type": "done", "report": report}


def last_bench():
    """Замеры или None."""
    return json.loads(config.BENCH_FILE.read_text(encoding="utf-8")) if config.BENCH_FILE.exists() else None


@functools.cache
def sample():
    """32 настоящих веса — ровно один блок квантования: начало строки матрицы из первого слоя исходной модели."""
    src, name = source(), "model.language_model.layers.0.mlp.gate_proj.weight"
    shard = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"][name]
    with safe_open(src / shard, "pt") as f:
        return [round(x, 6) for x in f.get_slice(name)[0, :32].float().tolist()]


def _measure(rows, answers, size, memory, load, before=None):
    """Итог одного вида модели: размер, память, загрузка, ответ, экзамен; с чем было — сколько ответов совпало."""
    out = {"size": size, "memory": memory, "load": round(load, 2),
           "seconds": round(sum(a["seconds"] for a in answers) / len(answers), 2),
           "ok": sum(a["ok"] for a in answers),
           "summary": {k: sum(a["ok"] for a, r in zip(answers, rows) if r["kind"] == k) for k in dataset.KINDS}}
    if before:
        out["same"] = sum(a["answer"] == b["answer"] for a, b in zip(answers, before))
        out["changed"] = [{"text": r["text"], "before": b["answer"], "after": a["answer"], "ok": [b["ok"], a["ok"]]}
                          for a, b, r in zip(answers, before, rows) if a["ok"] != b["ok"]]
    return out


def _drop(tag):
    subprocess.run(["ollama", "rm", tag], capture_output=True)


def _stopped(work, folder):
    for path in (work, folder):
        shutil.rmtree(path, ignore_errors=True)
    yield {"type": "stopped"}


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "4"
    job = bench() if what == "bench" else build(None if what == "base" else int(what), *sys.argv[2:3])
    for event in job:
        if event["type"] == "progress" and event["done"] % 25 == 0:
            print(event.get("what", "экзамен"), event["done"], "из", event["total"], flush=True)
        elif event["type"] != "progress":
            print(json.dumps({k: v for k, v in event.items() if k != "report"}, ensure_ascii=False), flush=True)
            if event["type"] == "done" and "report" in event:
                print(json.dumps({k: {m: v for m, v in r.items() if m != "changed"} if isinstance(r, dict) else r
                                  for k, r in event["report"].items()}, ensure_ascii=False))
