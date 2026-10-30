"""Учитель — большая qwen3.8 в облаке: обычный HTTP-запрос, ответ потоком, расход токенов и цена каждого вызова.

qwen3.8 перед ответом всегда думает (поле reasoning_content); thinking_budget ограничивает размышления.
"""
import json
import time
import urllib.error
import urllib.request

from house import config, home


class TeacherError(RuntimeError):
    """Ошибка вызова учителя — человеческими словами."""


def call(messages, think=config.THINK, as_json=True):
    """Один вызов: текст ответа, токены запроса и ответа (с размышлениями), секунды, цена в $."""
    if not config.API_KEY:
        raise TeacherError("нет ключа учителя: впишите DASHSCOPE_API_KEY в .env")
    body = {"model": config.TEACHER, "messages": messages, "stream": True, "stream_options": {"include_usage": True}}
    if think:
        body["thinking_budget"] = think
    else:
        body["enable_thinking"] = False  # придумать фразы — размышлять не о чем
    if as_json:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(f"{config.BASE_URL}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {config.API_KEY}",
                                          "Content-Type": "application/json"})
    started, parts, usage = time.perf_counter(), [], {}
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                chunk = json.loads(line[5:])
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    parts.append((choice.get("delta") or {}).get("content") or "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise TeacherError(f"учитель ответил {exc.code}: {detail}") from exc
    except OSError as exc:
        raise TeacherError(f"учитель не отвечает: {exc}") from exc
    tokens_in, tokens_out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    return {"text": "".join(parts).strip(), "in": tokens_in, "out": tokens_out,
            "seconds": round(time.perf_counter() - started, 2), "usd": price(tokens_in, tokens_out)}


def price(tokens_in, tokens_out):
    return (tokens_in * config.PRICE_IN + tokens_out * config.PRICE_OUT) / 1e6


def ask(room, phrase):
    """Одна фраза — как в приложении и на экзамене: ответ учителя в виде ответа ученика."""
    result = call([{"role": "system", "content": home.RULES + '\nОтвет — JSON-объект {"команды": [...]}.'},
                   {"role": "user", "content": home.request(room, phrase)}])
    commands = home.parse(result["text"])
    if commands is None:
        result["answer"] = result["text"]
    elif all(home.valid(c) for c in commands):
        result["answer"] = home.answer(commands)
    else:
        result["answer"] = json.dumps(commands, ensure_ascii=False)  # невыполнимое не прячем — проверка его увидит
    return result


def label(rows):
    """Пачка фраз за один вызов — так в разы дешевле: правила дома оплачиваются один раз на 10 фраз."""
    numbered = "\n".join(f"{i}. {home.request(r['room'], r['text'])}" for i, r in enumerate(rows, 1))
    result = call([{"role": "system", "content": home.RULES + '\nФраз несколько, с номерами. Ответ — JSON-объект '
                                                              '{"1": [...], "2": [...], ...}: команды каждой фразы.'},
                   {"role": "user", "content": numbered}], think=config.THINK_BATCH)
    try:
        value = json.loads(result["text"])
    except ValueError:
        value = {}
    result["answers"] = [value.get(str(i)) if isinstance(value, dict) else None for i in range(1, len(rows) + 1)]
    return result


if __name__ == "__main__":
    for room, phrase in [("гостиная", "мне холодно"), ("кухня", "я ухожу на работу")]:
        print(json.dumps(ask(room, phrase), ensure_ascii=False))
