"""Прод: пират отвечает из Ollama — модель со слитым тюном в GGUF. Кто отвечает (тюн или база с промптом), какая
версия тюна в проде, ворота перед выкаткой, откат.

Что в проде — data/pirate/prod/prod.json; сервис читает его на каждой реплике, поэтому переключение, выкатка
и откат идут без перезапуска сервиса.
"""
import datetime
import json
import threading
import time
import urllib.error
import urllib.request

from pirate import brain, config, persona, rules, train

BASE = f"{config.NAME}:base"  # база без тюна — отвечает с промптом
DROP = 3                      # минус 3 из 20 и больше — не шум: ворота закрыты
KINDS = {"format": "формат", "style": "голос", "fact": "факты", "para": "факты другими словами",
         "general": "обычные задачи"}
STARTED = time.time()         # когда запущен сервис: переключения его не перезапускают
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # Ollama на этом же компьютере — мимо прокси
_lock = threading.Lock()


def call(path, body=None, timeout=600):
    """Запрос к Ollama: JSON туда и обратно; ошибка — словами."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(config.OLLAMA_URL + path, data, {"Content-Type": "application/json"})
    try:
        with _opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Ollama: {exc.read().decode('utf-8', 'replace')}") from None
    except urllib.error.URLError:
        raise RuntimeError("Ollama не отвечает — запустите приложение Ollama") from None


def generate(tag, text, system=None):
    """Ответ модели из Ollama — в том же шаблоне чата, что видел тюн; параметры (жадно, 80 токенов) — в Modelfile."""
    d = call("/api/generate", {"model": tag, "prompt": train.prompt(text, system), "raw": True, "stream": False,
                               "keep_alive": "30m"})
    return {"answer": d["response"].strip(), "seconds": round((d["total_duration"] - d["load_duration"]) / 1e9, 2),
            "load": round(d["load_duration"] / 1e9, 2)}


def warm(tag):
    """Загрузить модель в память заранее — первая реплика после переключения не ждёт; вернуть секунды загрузки."""
    started = time.perf_counter()
    call("/api/generate", {"model": tag, "keep_alive": "30m"})  # запрос без реплики — только загрузка
    return round(time.perf_counter() - started, 2)


def loaded():
    """Модели в памяти Ollama: имя → байт."""
    return {m["name"]: m["size"] for m in call("/api/ps")["models"]}


def state():
    """Что в проде: live — версия тюна, mode — кто отвечает (tuned или prompt — база с промптом), history — откат."""
    if config.PROD_FILE.exists():
        return json.loads(config.PROD_FILE.read_text(encoding="utf-8"))
    return {"live": None, "mode": "prompt", "history": [], "changed": None}


def card(name):
    """Карточка версии (v1, v2… или base) или None."""
    path = config.PROD_DIR / name / "card.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def versions():
    """Собранные версии тюна по порядку; ворота — против той, что в проде сейчас."""
    live = state()["live"]
    live = live and card(live)
    cards = [json.loads(p.read_text(encoding="utf-8")) for p in config.PROD_DIR.glob("v*/card.json")]
    return [{**c, "gate": gate(c, live)} for c in sorted(cards, key=lambda c: c["n"])]


def gate(candidate, live):
    """Ворота перед выкаткой: экзамен версии против той, что в проде, — ни один вид не хуже на 3 из 20 и больше."""
    if live is None or live["version"] == candidate["version"]:
        return {"ok": True, "vs": None, "why": []}
    new, old = candidate["exam"]["summary"], live["exam"]["summary"]
    why = [f"{KINDS[k]} {new[k]} против {old[k]}" for k in sorted(old, key=lambda k: new[k] - old[k])  # худшее — первым
           if old[k] - new[k] >= DROP]
    return {"ok": not why, "vs": live["version"], "why": why}


def tag(mode, s):
    """Модель в Ollama для мозга: тюн — версия в проде, промпт — база."""
    return f"{config.NAME}:{s['live']}" if mode == "tuned" else BASE


def ask(text, modes=("live",)):
    """Реплика пирату в проде. live — тот, кто сейчас в проде; prompt и tuned — спросить обоих и сравнить."""
    row = brain.known(text)
    s = state()
    yield {"type": "start", "text": text, "row": row}
    for mode in modes:
        mode = s["mode"] if mode == "live" else mode
        if mode == "tuned" and not s["live"]:
            continue
        result = generate(tag(mode, s), text, persona.prompt() if mode == "prompt" else None)
        checked = rules.check(row, result["answer"]) if row else {"format": rules.format_ok(result["answer"]),
                                                                  "style": rules.style(result["answer"])}
        yield {"type": "answer", "mode": mode, "model": tag(mode, s), "version": s["live"] if mode == "tuned" else None,
               **result, "parsed": rules.parse(result["answer"]) if checked["format"] else None, **checked}
    yield {"type": "done"}


def switch(mode):
    """Кто отвечает в проде — тюн или база с промптом. Сервис не перезапускается, Ollama подгружает модель."""
    with _lock:
        s = state()
        if mode == "tuned" and not s["live"]:
            raise ValueError("Тюна в проде ещё нет — соберите и выкатите версию")
        s["mode"] = mode
        _save(s, f"тюн {s['live']}" if mode == "tuned" else "база с промптом")
    return {**s, "load": warm(tag(mode, s))}


def deploy(name):
    """Выкатить версию: только прошедшую ворота; прежняя уходит в историю — на неё можно откатиться."""
    c = card(name)
    if c is None or not name.startswith("v"):
        raise ValueError(f"Нет версии {name}")
    with _lock:
        s = state()
        verdict = gate(c, s["live"] and card(s["live"]))
        if not verdict["ok"]:
            raise ValueError(f"{name} не прошла ворота: {'; '.join(verdict['why'])}")
        if s["live"] and s["live"] != name:
            s["history"].append(s["live"])
        s.update(live=name, mode="tuned")
        _save(s, f"выкачена {name}")
    return {**s, "load": warm(tag("tuned", s))}


def rollback():
    """Откат: в прод возвращается предыдущая версия — она уже лежит в Ollama, пересобирать нечего."""
    with _lock:
        s = state()
        if not s["history"]:
            raise ValueError("Откатываться некуда — это первая версия")
        was = s["live"]
        s.update(live=s["history"].pop(), mode="tuned")
        _save(s, f"откат {was} → {s['live']}")
    return {**s, "load": warm(tag("tuned", s))}


def _save(s, what):
    s["changed"] = {"what": what, "at": datetime.datetime.now().isoformat(timespec="seconds")}
    config.PROD_DIR.mkdir(parents=True, exist_ok=True)
    config.PROD_FILE.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
