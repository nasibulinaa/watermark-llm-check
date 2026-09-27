#!/usr/bin/env python3
r"""
Утилита детекции watermark в выводе LLM: сравнивает watermarked-кандидата
и базовую модель (обе — llama-server, одна модель в VRAM за раз).

Четыре детектора по публичным референс-реализациям (git-сабмодули):

1) OpenStamp — length-normalized log-likelihood ratio между моделями,
   ключ не нужен:

       $LLR(x) = \frac{1}{T-1}\sum_{t=1}^{T-1}\log\frac{p_{\text{wm}}(x_t \mid x_{<t})}{p_{\text{base}}(x_t \mid x_{<t})}$

   Референс: openstamp/src/llr.py (эквивалентность — test_llr.py).
   Порог τ калибруется по null-текстам: mean + 3std.

2) SynthID-Text — keyed-hash G-значения (DEFAULT_WATERMARKING_CONFIG:
   ngram_len=5, 30 ключей, context_history_size=1024). Null-среднее
   G = 0.5 (Bernoulli-биты); у watermarked-текста смещается вверх
   (~0.75).

3) GaussMark — структурный watermark (шум в весах). Статистика
   бумаги score(T) = <grad_base(T), W> (W — гауссов шум-ключ).
   Без ключа W недоступен; для малого весового смещения
   (структурный watermark) по тождеству первого порядка
   <grad_base(T), dtheta> ~= L_wm(T) - L_base(T), dtheta =
   theta_wm - theta_base. p-value — по схеме бумаги (нормальное
   распределение, null — тексты базовой модели).

4) MarkLLM — E2E-LLM-Watermark (нейронный детектор, ключ не
   нужен): LSTM-детектор по эмбеддингам токенов opt-1.3b,
   score = P(watermarked) на текст. Референс:
   markllm/watermark/e2e/ (checkpoint 35000.pth в models/).

Прогон: генерация (reasoning on/off) -> скоринг L под обеими моделями
(GBNF-грамматики, pre-sampling logprobs) -> анализ, отчёт, график.

Запуск: python3 detect.py --models-dir /path/to/gguf [--resume | --analyze-only]
"""

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import requests
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
MODES = ("reason", "noreason")  # с reasoning (thinking) / без

# Репозиторий SynthID-Text как Python-модуль (сабмодуль в корне проекта)
sys.path.insert(0, os.path.join(ROOT, "synthid-text", "src"))

PROMPTS = [
    "Write a Python function that computes the Fibonacci number n.",
    "Explain what a black hole is in simple terms.",
    "Tell a short story about a robot who learns to paint.",
    "How does a car engine work? Summarize in a paragraph.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
    "Write a poem about the ocean at night.",
    "Explain the concept of recursion with an example.",
    "What caused the French Revolution? Give the key factors.",
    "Describe how to make pasta sauce from scratch.",
    "What is quantum computing in one paragraph?",
    "Write a dialogue between a cat and a dog.",
    "Напиши функцию на Python, которая проверяет число на простоту.",
    "Объясни, что такое теория относительности простыми словами.",
    "Расскажи короткую историю о лисе, которая учится математике.",
    "Как работает холодильник? Опиши принцип в одном абзаце.",
    "В чем разница между машинным обучением и глубоким обучением?",
    "Опиши процесс фотосинтеза пошагово.",
    "Напиши стихотворение о зимнем лесу.",
    "Объясни концепцию рекурсии с примером.",
    "Что такое инфляция и каковы её основные причины?",
    "Опиши, как приготовить борщ.",
    "Что такое квантовые вычисления? Объясни в одном абзаце.",
    "Напиши диалог между шахматистом и его тренером.",
    "Write a Python function that merges two sorted lists in linear time. Explain the algorithm.",
    "Write a C++ function that reverses a singly linked list in place. Explain the complexity.",
    "Explain how a database index works. Give a PostgreSQL example with EXPLAIN.",
    "Напиши функцию на Python, которая находит пересечение двух отсортированных списков за O(n).",
    "Объясни, как устроена хеш-таблица. Приведи пример реализации на Python.",
    "Напиши SQL-запрос: топ-3 товара по продажам за каждый месяц. Объясни оконные функции.",
    "Explain the difference between a process and a thread in operating systems.",
    "Write a regular expression to validate an email address. Explain each part.",
    "What is the CAP theorem? Give practical examples.",
    "Describe the water cycle in one paragraph.",
    "Write a haiku about autumn.",
    "Explain how RSA encryption works in simple terms.",
    "What were the causes and consequences of the Industrial Revolution?",
    "Write a Python script that reads a CSV file and computes column averages.",
    "Explain the difference between accuracy and precision in measurements.",
    "Describe how a Wi-Fi router works.",
    "Write a JavaScript function that debounces another function. Explain usage.",
    "What is the difference between a stack and a queue? Give examples.",
    "Explain the concept of a blockchain in one paragraph.",
    "Write a short essay about the importance of sleep.",
    "Describe the mechanism of natural selection.",
    "Объясни, чем отличается процесс от потока в операционной системе.",
    "Напиши SQL-запрос: средняя зарплата по отделам. Используй GROUP BY.",
    "Опиши, как работает радар автомобиля, в одном абзаце.",
    "Напиши рассказ о человеке, который нашёл в старом шкафу письмо 1945 года.",
    "Что такое дефляция? Приведи примеры из истории.",
    "Объясни принцип работы Wi-Fi простыми словами.",
    "Напиши функцию на Python для сортировки списка слов по длине.",
    "Опиши строение атома: из чего состоит ядро и где находятся электроны.",
    "Напиши стихотворение о дороге в горы.",
    "В чём разница между верой и знанием? Ответь в двух абзацах.",
    "Опиши, как устроена банковская карта: от пластика до платёжной системы.",
    "Напиши SQL-запрос: пользователи, не делавшие покупок за последние 90 дней.",
    "Объясни, что такое deadlock в программировании и как его избежать.",
    "Расскажи, как работает микроволновая печь.",
    "Напиши диалог между программистом и котом, который портит код.",
]


@dataclass
class Config:
    # Дисплейные имена моделей (в отчёте) и GGUF-файлы в models_dir
    wm_name: str = "Swift-1.5-Qwen3.8-27B-GSQ-RCO"
    base_name: str = "Qwen3.8-27B-GSQ-RCO"
    wm_gguf: str = "Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
    base_gguf: str = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
    quant: str = ""  # квантизация в именах отчётов; "" — вывести из wm_gguf
    # Окружение
    llama_bin: str = "llama-server"  # в PATH; переопределить --llama-bin
    models_dir: str = "."
    device: str = "cuda0"
    host: str = "127.0.0.1"
    port: int = 8091
    server_ctx: int = 8192
    server_batch: int = 2048
    # Протокол
    prompts: int = 60
    tokens: int = 400
    data: str = os.path.join(ROOT, "data")


def derive_quant(gguf_name):
    """Квантизация из имени GGUF (Q8_0, IQ3_S, Q4_K_M, F16...); "" если нет."""
    stem = os.path.splitext(os.path.basename(gguf_name))[0]
    for tok in reversed(stem.split("-")):
        if re.fullmatch(r"(?:[QIT]Q?|IQ|F|BF)\d+(?:_\d+)?(?:_[A-Z]+)*", tok):
            return tok
    return ""


def quant_suffix(quant):
    """Суффикс _<квант> для имён файлов; "" если квант не задан."""
    return f"_{quant}" if quant else ""


def save_data(cfg, data):
    """Сохранить data_<модель>_<квант>.json (после каждого промпта/текста)."""
    os.makedirs(cfg.data, exist_ok=True)
    path = os.path.join(cfg.data, f"data_{cfg.wm_name}{quant_suffix(cfg.quant)}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Управление llama-server (одна модель в VRAM за раз)
# --------------------------------------------------------------------------
class ServerManager:
    def __init__(self, cfg):
        """cfg — Config; proc — дочерний процесс сервера; log_fh — лог."""
        self.cfg = cfg
        self.proc = None
        self.log_fh = None

    def _cmd(self, gguf):
        """Команда запуска сервера для gguf (одна модель, фиксированный порт)."""
        c = self.cfg
        return [c.llama_bin,
                "-m", os.path.join(c.models_dir, gguf),
                "--host", c.host,
                "--port", str(c.port),
                "--ctx-size", str(c.server_ctx),
                "--batch-size", str(c.server_batch),
                "--parallel", "1",
                "--device", c.device]

    def launch(self, gguf):
        """Запустить сервер в фоне; лог — data/server_<gguf>.log (append)."""
        os.makedirs(self.cfg.data, exist_ok=True)
        log_path = os.path.join(self.cfg.data, f"server_{gguf}.log")
        self.log_fh = open(log_path, "ab")
        print(f"[server] запуск: {' '.join(self._cmd(gguf))}", flush=True)
        self.proc = subprocess.Popen(
            self._cmd(gguf), stdout=self.log_fh,
            stderr=subprocess.STDOUT, start_new_session=True)
        print(f"[server] pid={self.proc.pid}, лог: {log_path}", flush=True)

    def _find_port_pid(self):
        """pid процесса, слушающего cfg.port (скан /proc)."""
        port_hex = "%04X" % self.cfg.port
        inodes = set()
        for path in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                with open(path) as f:
                    for line in f.readlines()[1:]:
                        parts = line.split()
                        if len(parts) > 9:
                            local = parts[1]
                            if local.rsplit(":", 1)[1] == port_hex:
                                inodes.add(parts[9])
            except OSError:
                continue
        if not inodes:
            return None
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                fds = os.listdir(f"/proc/{pid}/fd")
            except OSError:
                continue
            for fd in fds:
                try:
                    link = os.readlink(f"/proc/{pid}/fd/{fd}")
                except OSError:
                    continue
                if link.startswith("socket:[") and link[8:-1] in inodes:
                    return int(pid)
        return None

    def kill(self):
        """Остановить сервер: наш дочерний процесс ИЛИ посторонний
        процесс на порту (остаток предыдущего упавшего прогона)."""
        pid = None
        if self.proc is not None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            pid = self.proc.pid
            self.proc = None
        else:
            pid = self._find_port_pid()
        if pid is not None:
            for _ in range(60):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(1)
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if self.log_fh:
            self.log_fh.close()
            self.log_fh = None
        # ждём освобождения порта
        for _ in range(120):
            try:
                r = requests.get(f"http://{self.cfg.host}:{self.cfg.port}"
                                 f"/v1/models", timeout=2)
                if r.status_code != 200:
                    break
            except requests.RequestException:
                break
            time.sleep(1)
        print("[server] остановлен", flush=True)

    def loaded_model(self):
        """-> id загруженной модели (None, если сервер не отвечает)."""
        try:
            r = requests.get(f"http://{self.cfg.host}:{self.cfg.port}"
                             f"/v1/models", timeout=5)
            r.raise_for_status()
        except requests.RequestException:
            return None
        js = r.json()
        for m in js.get("data") or js.get("models") or []:
            st = m.get("status")
            if st is None or st == "loaded":
                return m.get("id") or m.get("name")
        return None

    def _match(self, loaded, gguf):
        """Совпадает ли id загруженной модели с gguf (id может не иметь .gguf)."""
        if not loaded:
            return False
        base = os.path.basename(loaded)
        # сервер может отдавать id с/без расширения .gguf
        return base == gguf or base == os.path.splitext(gguf)[0]

    def wait_ready(self, gguf, timeout=900):
        """Ждать загрузки модели (по умолчанию до 900 с)."""
        t0 = time.time()
        last = 0.0
        while time.time() - t0 < timeout:
            loaded = self.loaded_model()
            if self._match(loaded, gguf):
                print(f"[server] готова: {os.path.basename(loaded)} "
                      f"({time.time()-t0:.0f} с)")
                return loaded
            if self.proc and self.proc.poll() is not None:
                raise SystemExit(
                    f"[server] процесс завершился с кодом {self.proc.returncode}; "
                    f"см. {self.cfg.data}/server_{gguf}.log")
            el = time.time() - t0
            if el - last >= 10:
                print(f"[server] загрузка {gguf}... {int(el)} с", flush=True)
                last = el
            time.sleep(3)
        raise SystemExit("[server] модель не загрузилась за отведённое время")

    def ensure_model(self, gguf):
        """Загрузить gguf, если ещё не загружен (смена модели при несовпадении)."""
        loaded = self.loaded_model()
        if self._match(loaded, gguf):
            print(f"[server] уже загружена: {loaded}")
            return loaded
        self.kill()
        self.launch(gguf)
        return self.wait_ready(gguf)


# --------------------------------------------------------------------------
# HTTP-клиент
# --------------------------------------------------------------------------
def gbnf_escape(s):
    """Экранирует строку в GBNF-литерал."""
    out = []
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == '\\':
            out.append('\\\\')
        elif ch == '\n':
            out.append('\\n')
        elif ch == '\t':
            out.append('\\t')
        elif ch == '\r':
            out.append('\\r')
        elif ord(ch) < 0x20 or ord(ch) == 0x7f:
            out.append('\\u%04x' % ord(ch))
        else:
            out.append(ch)
    return ''.join(out)

class Server:
    def __init__(self, cfg, model_id, timeout=900):
        """HTTP-клиент для одной модели; model_id — id из /v1/models."""
        self.base = f"http://{cfg.host}:{cfg.port}"
        self.s = requests.Session()
        self.s.headers["Content-Type"] = "application/json"
        self.timeout = timeout
        self.model_id = model_id

    def generate(self, prompt, max_tokens, reason=True,
                 temperature=1.0, top_k=20, top_p=0.95):
        """-> (text, token_ids). Chat-completion (chat-шаблон Qwen3):
        reason — режим thinking (enable_thinking). token ids нужны для
        SynthID-детектора (G-значения); logprobs=1 — носитель для них."""
        r = self.s.post(f"{self.base}/v1/chat/completions", json={
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "logprobs": True, "top_logprobs": 1,
            "chat_template_kwargs": {"enable_thinking": reason},
            "stream": False,
        }, timeout=self.timeout)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        ids = [int(t["id"])
               for t in (ch.get("logprobs") or {}).get("content") or []]
        return ch.get("message", {}).get("content", ""), ids

    def score(self, prompt, text, n_tokens):
        """L = Σ log p(x_t|x_<t) под текущей моделью (pre-sampling
        logprobs, enable_thinking=false, текст зафиксирован GBNF).
        -> (L, n_scored); n_scored может слегка превышать n_tokens
        (ре-токенизация). Первый текст-токен не входит (labels = ids[2:])."""
        if not text:
            raise SystemExit("[score] пустой текст — строка исключена из "
                             "скоринга (см. score_texts_server)")
        grammar = 'root ::= "' + gbnf_escape(text) + '"'
        r = self.s.post(f"{self.base}/v1/chat/completions", json={
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "grammar": grammar,
            "max_tokens": n_tokens + 1000,
            # запас на ре-токенизацию; после грамматики сервер
            # догенерит до max_tokens — хвост в ответ не входит
            "logprobs": True, "top_logprobs": 1,
            "temperature": 1.0,
            "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }, timeout=self.timeout)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        gen = ch.get("message", {}).get("content", "")
        if gen != text:
            # detok не обратим: допускаем потерю/добавление
            # нескольких символов на границах токенов
            ok = (text.startswith(gen) or gen.startswith(text)) \
                 and abs(len(gen) - len(text)) <= max(10, len(text) // 10)
            if not ok:
                raise SystemExit(f"[score] грамматика не зафиксировала текст: "
                                 f"{len(gen)} vs {len(text)} символов")
        lps = (ch.get("logprobs") or {}).get("content") or []
        if len(lps) < 2:
            raise SystemExit(f"[score] в ответе мало logprobs: {len(lps)} "
                             f"при {len(text)} символах текста")
        return sum(t["logprob"] for t in lps[1:]), len(lps)

# --------------------------------------------------------------------------
# Сбор данных
# --------------------------------------------------------------------------
def collect(cfg, server, prompts, max_tokens, data, tag, mode, start=0):
    """Сгенерировать N текстов (tag, mode); save_data после каждого.
    start — индекс, с которого продолжить неполный сбор (--resume)."""
    out = data.setdefault(tag, {}).setdefault(mode, [])
    t0 = time.time()
    for i in range(start, len(prompts)):
        p = prompts[i]
        reason = (mode == "reason")
        print(f"[collect:{tag}:{mode}] {i+1}/{len(prompts)}: {p[:60]!r} ...",
              flush=True)
        text, ids = server.generate(p, max_tokens, reason=reason)
        out.append({"prompt": p, "text": text, "token_ids": ids, "n": len(ids)})
        save_data(cfg, data)
        el = time.time() - t0
        eta = el / (i - start + 1) * (len(prompts) - i - 1)
        print(f"[collect:{tag}:{mode}] {i+1}/{len(prompts)}: {len(ids)} tok, "
              f"прошло {el:.0f} с, ETA {eta:.0f} с", flush=True)


def score_texts_server(cfg, server, data, tag, mode, field):
    """Скоринг текстов сервером (GBNF-грамматики):
    data[tag][mode][i][field] = L, data[tag][mode][i]["n_scored"] = n.
    Строки с пустым текстом пропускаются (модель зациклилась на
    спец-токенах — скоринг вырожден)."""
    rows = data[tag][mode]
    todo = [i for i, r in enumerate(rows) if field not in r]
    skip = [i for i in todo if not rows[i]["text"].strip()]
    todo = [i for i in todo if rows[i]["text"].strip()]
    if skip:
        print(f"[score:{field}:{tag}:{mode}] пропущено пустых текстов: "
              f"{skip} (модель зациклилась на спец-токенах)", flush=True)
    if not todo:
        return
    print(f"[score:{field}:{tag}:{mode}] {len(todo)} текстов <- текущая модель",
          flush=True)
    t0 = time.time()
    for n, i in enumerate(todo):
        rec = rows[i]
        L, n_scored = server.score(rec["prompt"], rec["text"], rec["n"])
        rows[i][field] = L
        rows[i]["n_scored"] = n_scored
        save_data(cfg, data)
        el = time.time() - t0
        eta = el / (n + 1) * (len(todo) - n - 1)
        print(f"[score:{field}:{tag}:{mode}] {n+1}/{len(todo)}: L={L:.3f}, "
              f"n_scored={n_scored}, прошло {el:.0f} с, ETA {eta:.0f} с",
              flush=True)


# --------------------------------------------------------------------------
# Детектор 1: OpenStamp (LLR)
# --------------------------------------------------------------------------
def openstamp_llrs(data):
    """-> {mode: (llr_wm, llr_null)}: (L_wm − L_base) / (n_scored − 1).
    Пары по индексу промпта: строка учитывается, если тексты обеих
    моделей валидны (непустые и посчитаны). Эквивалентность
    референсу openstamp/src/llr.py — test_llr.py."""
    out = {}
    for mode in MODES:
        wm_recs, base_recs = data["wm"][mode], data["base"][mode]
        llr_wm, llr_null = [], []
        for i in range(min(len(wm_recs), len(base_recs))):
            a, b = wm_recs[i], base_recs[i]
            ok = (a.get("text", "").strip() and b.get("text", "").strip()
                  and all(k in r for r, k in ((a, "L_wm"), (a, "L_base"),
                                             (b, "L_wm"), (b, "L_base"))))
            if not ok:
                continue
            llr_wm.append((a["L_wm"] - a["L_base"])
                          / max(a.get("n_scored", a["n"]) - 1, 1))
            llr_null.append((b["L_wm"] - b["L_base"])
                            / max(b.get("n_scored", b["n"]) - 1, 1))
        out[mode] = (np.array(llr_wm), np.array(llr_null))
    return out


# --------------------------------------------------------------------------
# Детектор 3: GaussMark (структурный watermark, ключевая версия)
# --------------------------------------------------------------------------
def gaussmark_scores(data):
    """Статистика GaussMark без ключа (см. docstring, п.3).

    score(T) = (L_wm(T) - L_base(T)) / (n_scored - 1) — та же
    length-normalized разность, что и LLR OpenStamp (первый порядок
    по dtheta). p-value строится по схеме бумаги: null-распределение
    N(mu0, sd0^2) по тексту базовой модели, p_i = P(Z > (s_i - mu0)/sd0).

    -> {mode: (score_wm, score_null, p_wm)}"""
    out = {}
    for mode in MODES:
        wm_recs, base_recs = data["wm"][mode], data["base"][mode]
        s_wm, s_null = [], []
        for i in range(min(len(wm_recs), len(base_recs))):
            a, b = wm_recs[i], base_recs[i]
            ok = (a.get("text", "").strip() and b.get("text", "").strip()
                  and all(k in r for r, k in ((a, "L_wm"), (a, "L_base"),
                                             (b, "L_wm"), (b, "L_base"))))
            if not ok:
                continue
            s_wm.append((a["L_wm"] - a["L_base"])
                        / max(a.get("n_scored", a["n"]) - 1, 1))
            s_null.append((b["L_wm"] - b["L_base"])
                          / max(b.get("n_scored", b["n"]) - 1, 1))
        s_wm, s_null = np.asarray(s_wm), np.asarray(s_null)
        mu0, sd0 = float(s_null.mean()), float(s_null.std(ddof=1))
        p_wm = 0.5 * np.array([math.erfc(x / math.sqrt(2))
                               for x in (s_wm - mu0) / sd0])
        out[mode] = (s_wm, s_null, p_wm)
    return out


# --------------------------------------------------------------------------
# Детектор 2: SynthID-Text (G-значения, модули из сабмодуля synthid-text)
# --------------------------------------------------------------------------
class SynthIDDetector:
    def __init__(self):
        """SynthID-детектор; конфиг — DEFAULT_WATERMARKING_CONFIG (сабмодуль)."""
        from synthid_text import logits_processing
        from synthid_text.synthid_mixin import DEFAULT_WATERMARKING_CONFIG
        self.cfg = dict(DEFAULT_WATERMARKING_CONFIG)
        self.proc = logits_processing.SynthIDLogitsProcessor(
            ngram_len=self.cfg["ngram_len"],
            keys=list(self.cfg["keys"]),
            context_history_size=self.cfg["context_history_size"],
            temperature=1.0,
            top_k=20,
            device=torch.device("cpu"),
        )

    def score(self, token_ids):
        """-> (mean_g, z, M) для одной последовательности."""
        from synthid_text import detector_mean
        ids = torch.tensor([token_ids], dtype=torch.long)
        g = self.proc.compute_g_values(ids)                    # [1, L-4, 30]
        rep = self.proc.compute_context_repetition_mask(ids)    # [1, L-4]
        L = len(token_ids)
        # маска EOS: нули с первого EOS (конец генерации) до конца
        eos_mask = torch.ones(1, L)
        for j, t in enumerate(token_ids):
            if t in (151643, 151645):  # eos-токены семейства Qwen
                eos_mask[0, j:] = 0
                break
        # выравнивание под README: eos_mask[:, ngram_len-1:]
        eos_align = eos_mask[0, self.cfg["ngram_len"] - 1:]
        combined = (rep[0] * eos_align).to(torch.float32).unsqueeze(0)
        if int(combined.sum()) == 0:
            return 0.5, 0.0, 0
        score = float(detector_mean.mean_score(g.numpy(), combined.numpy())[0])
        M = int(combined.sum().item()) * g.shape[-1]
        z = (score - 0.5) * math.sqrt(M) / 0.5
        return score, z, M


def synthid_scores(cfg, data, det):
    """-> {tag: {mode: [rows]}}: по-последовательностные G-значения.
    Строки выравнены по индексу промпта: учитываются пары, где тексты
    обеих моделей непустые (signal и null — спаренные)."""
    out = {"wm": {}, "base": {}}
    for mode in MODES:
        wm_recs, base_recs = data["wm"][mode], data["base"][mode]
        wm_rows, base_rows = [], []
        for i in range(min(len(wm_recs), len(base_recs))):
            a, b = wm_recs[i], base_recs[i]
            if not (a.get("text", "").strip() and b.get("text", "").strip()):
                continue
            g, z, M = det.score(a["token_ids"])
            wm_rows.append({"mean_g": g, "z": z, "M": M})
            g, z, M = det.score(b["token_ids"])
            base_rows.append({"mean_g": g, "z": z, "M": M})
        out["wm"][mode] = wm_rows
        out["base"][mode] = base_rows
        save_data(cfg, data)
    return out


# --------------------------------------------------------------------------
# Детектор 4: MarkLLM — E2E-LLM-Watermark (нейронный, ключ не нужен)
# --------------------------------------------------------------------------
class MarkLLMDetector:
    """E2E-LLM-Watermark (markllm/watermark/e2e): LSTM-детектор по
    эмбеддингам токенов opt-1.3b; score = P(watermarked) на текст.
    Checkpoint 35000.pth — models/e2e-35000.pth (SHA-256 задокументирован
    в markllm/watermark/e2e/README.md)."""

    REF_MODEL = "facebook/opt-1.3b"

    def __init__(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "e2e_model", os.path.join(ROOT, "markllm", "watermark",
                                      "e2e", "model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ckpt = torch.load(os.path.join(ROOT, "models", "e2e-35000.pth"),
                          map_location="cpu", weights_only=True)
        self.detector = mod.E2EDetector(
            input_dim=ckpt["dec"]["lstm.weight_ih_l0"].shape[1],
            hidden_dim=64, num_classes=1, num_layers=3)
        self.detector.load_state_dict(ckpt["dec"])
        self.detector.eval()
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.REF_MODEL)
        w = torch.load(os.path.join(ROOT, "models",
                                    "opt-1.3b-embeddings.pt"),
                       map_location="cpu")
        self.emb = torch.nn.Embedding.from_pretrained(w)

    @torch.inference_mode()
    def score(self, text):
        """-> P(watermarked) в [0, 1] для одного текста."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError("E2E не детектирует пустой текст")
        x = self.emb(torch.tensor([ids])).float()
        return float(torch.sigmoid(self.detector(x)).item())


def markllm_scores(data, det):
    """-> {tag: {mode: [rows]}}: P(watermarked) по E2E, пары по промптам
    (учитываются пары, где тексты обеих моделей непустые)."""
    out = {"wm": {}, "base": {}}
    for mode in MODES:
        wm_recs, base_recs = data["wm"][mode], data["base"][mode]
        wm_rows, base_rows = [], []
        for i in range(min(len(wm_recs), len(base_recs))):
            a, b = wm_recs[i], base_recs[i]
            if not (a.get("text", "").strip() and b.get("text", "").strip()):
                continue
            wm_rows.append({"p": det.score(a["text"])})
            base_rows.append({"p": det.score(b["text"])})
        out["wm"][mode] = wm_rows
        out["base"][mode] = base_rows
    return out


# --------------------------------------------------------------------------
# Статистика и вердикт
# --------------------------------------------------------------------------
def aggregate(signal, null):
    """Сводка по signal/null (пары по промптам).

    Основной статистика — спаренный t-критерий по разностям
    d = signal - null (одни и те же промпты у обеих моделей):
    t_agg = mean(d) / (std(d)/sqrt(n)). Он не зависит от маргинальной
    дисперсии null (та, что надувала tau при малом n).
    verdict = (t_agg >= 3.0) и (большинство d > 0).
    tau/hits — по-текстовый взгляд (auxiliary), z_agg — неспаренный аналог.
    """
    signal = np.asarray(signal, dtype=float)
    null = np.asarray(null, dtype=float)
    d = signal - null
    sd = float(d.std(ddof=1))
    t_agg = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0
    pos_rate = float((d > 0).mean())
    tau = float(null.mean() + 3 * null.std())
    hits = int((signal > tau).sum())
    z_agg = float((signal.mean() - null.mean())
                  / (null.std() / math.sqrt(len(null))))
    return {
        "signal_mean": float(signal.mean()),
        "signal_median": float(np.median(signal)),
        "signal_min": float(signal.min()),
        "signal_max": float(signal.max()),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "tau": tau,
        "hits": hits,
        "n": int(len(signal)),
        "hit_rate": hits / len(signal),
        "z_agg": z_agg,
        "t_agg": t_agg,
        "pos_rate": pos_rate,
        "verdict": (t_agg >= 3.0) and (pos_rate >= 0.5),
    }


def pvalue_fields(p_wm):
    """Поле отчёта: p-значения по схеме GaussMark (нормальный test)."""
    p_wm = np.asarray(p_wm, dtype=float)
    return {
        "p_wm_mean": float(p_wm.mean()),
        "p_wm_min": float(p_wm.min()),
        "p_hits_005": int((p_wm < 0.05).sum()),
    }


# --------------------------------------------------------------------------
# График
# --------------------------------------------------------------------------
def make_plot(cfg, llrs, synth, gm, e2e, aggs, res_path):
    """4×2-график: LLR, mean G, GaussMark-score, E2E-P; reasoning on/off.
    aggs = {"llr": ..., "synth": ..., "gm": ..., "e2e": ...} по режимам."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(13, 19))
    fig.suptitle(f"Watermark detection: {cfg.wm_name} vs {cfg.base_name}\n"
                 "(OpenStamp LLR, SynthID G, GaussMark score, "
                 "MarkLLM E2E-P; columns: reasoning on/off)", fontsize=13)



    for c, mode in enumerate(MODES):
        on = "on" if mode == "reason" else "off"
        llr_wm, llr_null = llrs[mode]
        gw = np.array([r["mean_g"] for r in synth["wm"][mode]])
        gb = np.array([r["mean_g"] for r in synth["base"][mode]])
        gm_wm, gm_null, gm_p = gm[mode]
        pw = np.array([r["p"] for r in e2e["wm"][mode]])
        pb = np.array([r["p"] for r in e2e["base"][mode]])
        a_llr, a_gm = aggs["llr"][mode], aggs["gm"][mode]

        def _panel(row, wm, null, title, xlabel, extra=None,
                   verdict=None):
            ax = axes[row][c]
            ax.hist(null, bins=12, alpha=0.55, color="tab:blue",
                    label="base texts (null)")
            ax.hist(wm, bins=12, alpha=0.55, color="tab:red",
                    label=f"{cfg.wm_name} texts (signal)")
            for x, ls, label in extra:
                ax.axvline(x, color="k", ls=ls, lw=1.5, label=label)
            ax.set_title(f"{title} (reasoning {on})")
            ax.set_xlabel(xlabel)
            ax.legend()
            if verdict is not None:
                txt = "WATERMARK FOUND" if verdict else "NO WATERMARK"
                ax.text(0.99, 0.97, txt, transform=ax.transAxes,
                        ha="right", va="top", fontsize=11,
                        fontweight="bold", color="white", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.35",
                                  fc="tab:red" if verdict else "tab:green",
                                  ec="none", alpha=0.9))

        _panel(0, llr_wm, llr_null,
               "OpenStamp: LLR per text",
               "LLR (nats/token)",
               [(a_llr["tau"], "--", f"tau = {a_llr['tau']:.3f}")],
               verdict=aggs["llr"][mode]["verdict"])

        _panel(1, gw, gb, "SynthID-Text: mean G per text",
               "mean G",
               [(0.5, "--", "null mean = 0.5"),
                (0.75, ":", "expected watermarked ~ 0.75")],
               verdict=aggs["synth"][mode]["verdict"])

        _panel(2, gm_wm, gm_null,
               f"GaussMark: score (p_wm mean = {a_gm['p_wm_mean']:.3f})",
               "score (nats/token)",
               [(a_gm["tau"], "--", f"tau = {a_gm['tau']:.3f}")],
               verdict=aggs["gm"][mode]["verdict"])

        _panel(3, pw, pb,
               f"MarkLLM E2E: P(wm) (p_wm mean = {pw.mean():.3f})",
               "P(watermarked)",
               [(0.5, "--", "detection threshold = 0.5")],
               verdict=aggs["e2e"][mode]["verdict"])

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(res_path, dpi=150)
    print(f"[plot] сохранён {res_path}")


# --------------------------------------------------------------------------
# Отчёт
# --------------------------------------------------------------------------
def print_report(cfg, llrs, synth, gm, e2e, aggs_llr, aggs_syn,
                 aggs_gm, aggs_e2e):
    """Печатный отчёт: цифры и вердикт по каждому детектору и режиму."""
    print()
    print("=" * 78)
    print("  ИТОГОВЫЙ ОТЧЁТ: детекция watermark")
    print(f"  watermarked-кандидат: {cfg.wm_name}")
    print(f"  база (null):          {cfg.base_name}")
    print("=" * 78)
    for mode in MODES:
        on = "on" if mode == "reason" else "off"
        a_llr = aggs_llr[mode]
        a_syn = aggs_syn[mode]
        a_gm = aggs_gm[mode]
        a_e2e = aggs_e2e[mode]
        gw = np.array([r["mean_g"] for r in synth["wm"][mode]])
        gb = np.array([r["mean_g"] for r in synth["base"][mode]])
        zw = np.array([r["z"] for r in synth["wm"][mode]])
        pw = np.array([r["p"] for r in e2e["wm"][mode]])
        pb = np.array([r["p"] for r in e2e["base"][mode]])
        print()
        print(f"=== reasoning: {on} ===")
        print("[1] OpenStamp — length-normalized LLR (wm vs base)")
        print(f"    {cfg.wm_name}-тексты : mean={a_llr['signal_mean']:+.4f}  "
              f"median={a_llr['signal_median']:+.4f}  "
              f"[{a_llr['signal_min']:+.4f} .. {a_llr['signal_max']:+.4f}]")
        print(f"    null-тексты  : mean={a_llr['null_mean']:+.4f}  "
              f"std={a_llr['null_std']:.4f}")
        print(f"    tau (mean+3std) = {a_llr['tau']:+.4f}")
        print(f"    {cfg.wm_name} > tau: {a_llr['hits']}/{a_llr['n']} "
              f"({a_llr['hit_rate']*100:.1f}%)")
        print(f"    t-сводка (спаренный): {a_llr['t_agg']:+.2f}   "
              f"z-сводка: {a_llr['z_agg']:+.2f}")
        print(f"    => {'WATERMARK ОБНАРУЖЕН' if a_llr['verdict'] else 'watermark не обнаружен'}")
        print()
        print("[2] SynthID-Text — mean G-value (default key из репозитория)")
        print(f"    {cfg.wm_name}-тексты : mean G={gw.mean():.4f}  "
              f"(null 0.50, ожидаемо у watermarked ~0.75)")
        print(f"    null-тексты  : mean G={gb.mean():.4f}")
        print(f"    z по последовательностям: "
              f"mean={zw.mean():+.2f}, max={zw.max():+.2f}")
        print(f"    {cfg.wm_name} > tau: {a_syn['hits']}/{a_syn['n']} "
              f"({a_syn['hit_rate']*100:.1f}%)")
        print(f"    t-сводка (спаренный): {a_syn['t_agg']:+.2f}   "
              f"z-сводка: {a_syn['z_agg']:+.2f}")
        print(f"    => {'WATERMARK ОБНАРУЖЕН' if a_syn['verdict'] else 'watermark не обнаружен'}")
        print()
        print("[3] GaussMark — score = <grad_base, dtheta> (первый "
              "порядок), p-value по схеме бумаги")
        print(f"    {cfg.wm_name}-тексты : mean={a_gm['signal_mean']:+.4f}  "
              f"(null {a_gm['null_mean']:+.4f})")
        print(f"    p-value: mean={a_gm['p_wm_mean']:.4f}  "
              f"min={a_gm['p_wm_min']:.4f}  "
              f"p<0.05: {a_gm['p_hits_005']}/{a_gm['n']}")
        print(f"    t-сводка (спаренный): {a_gm['t_agg']:+.2f}   "
              f"z-сводка: {a_gm['z_agg']:+.2f}")
        print(f"    => {'WATERMARK ОБНАРУЖЕН' if a_gm['verdict'] else 'watermark не обнаружен'}")
        print()
        print("[4] MarkLLM E2E — P(watermarked) (нейронный детектор)")
        print(f"    {cfg.wm_name}-тексты : mean P={pw.mean():.4f}  "
              f"(null {pb.mean():.4f}, порог 0.5)")
        print(f"    P > 0.5: {int((pw > 0.5).sum())}/{len(pw)}")
        print(f"    {cfg.wm_name} > tau: {a_e2e['hits']}/{a_e2e['n']} "
              f"({a_e2e['hit_rate']*100:.1f}%)")
        print(f"    t-сводка (спаренный): {a_e2e['t_agg']:+.2f}   "
              f"z-сводка: {a_e2e['z_agg']:+.2f}")
        print(f"    => {'WATERMARK ОБНАРУЖЕН' if a_e2e['verdict'] else 'watermark не обнаружен'}")
    print()
    print("=" * 78)
    for mode in MODES:
        on = "on" if mode == "reason" else "off"
        k = sum(a[mode]["verdict"] for a in
                (aggs_llr, aggs_syn, aggs_gm, aggs_e2e))
        if k == 4:
            s = "watermark ВЫЯВЛЕН всеми детекторами"
        elif k:
            s = (f"watermark выявлен {k} из 4 детекторов — результат "
                 "однозначным не является, нужны данные с известным ключом")
        else:
            s = ("watermark НЕ выявлен ни одним детектором "
                 "(при известном/дефолтном ключе)")
        print(f"ИТОГО (reasoning {on}): {s}")


# --------------------------------------------------------------------------
def main():
    """CLI: сбор, скоринг, анализ, график, отчёт (смысл флагов — в --help)."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Дефолты — из Config (единый источник)
    ap.add_argument("--llama-bin", default=Config.llama_bin)
    ap.add_argument("--models-dir", default=Config.models_dir,
                    help="каталог с GGUF-файлами")
    ap.add_argument("--device", default=Config.device,
                    help="устройство сервера (llama.cpp -device)")
    ap.add_argument("--host", default=Config.host,
                    help="адрес сервера (llama.cpp -host)")
    ap.add_argument("--port", type=int, default=Config.port)
    ap.add_argument("--data", default=Config.data,
                    help="каталог data_<модель>_<квант>.json/отчёта/графика")
    # Модели
    ap.add_argument("--wm-model", default=Config.wm_name,
                    help="имя watermarked-модели (в отчёте)")
    ap.add_argument("--base-model", default=Config.base_name,
                    help="имя базовой модели (в отчёте)")
    ap.add_argument("--wm-gguf", default=Config.wm_gguf,
                    help="GGUF watermarked-модели в models-dir")
    ap.add_argument("--base-gguf", default=Config.base_gguf,
                    help="GGUF базовой модели в models-dir")
    ap.add_argument("--quant", default=Config.quant,
                    help="квантизация в именах отчётов "
                         "(по умолчанию — из имени wm-GGUF)")
    # Протокол
    ap.add_argument("--prompts", type=int, default=Config.prompts)
    ap.add_argument("--tokens", type=int, default=Config.tokens)
    ap.add_argument("--analyze-only", action="store_true",
                    help="не собирать, только анализ по data-файлу")
    ap.add_argument("--resume", action="store_true",
                    help="использовать существующий data-файл, "
                         "доскорить только недостающие поля L")
    args = ap.parse_args()

    cfg = Config(
        wm_name=args.wm_model,
        base_name=args.base_model,
        wm_gguf=args.wm_gguf,
        base_gguf=args.base_gguf,
        quant=args.quant or derive_quant(args.wm_gguf),
        llama_bin=args.llama_bin,
        models_dir=args.models_dir,
        device=args.device,
        host=args.host,
        port=args.port,
        data=args.data,
        prompts=args.prompts,
        tokens=args.tokens,
    )

    data_path = os.path.join(cfg.data,
                             f"data_{cfg.wm_name}{quant_suffix(cfg.quant)}.json")
    prompts = PROMPTS[:cfg.prompts]

    data = {}
    if args.analyze_only or args.resume:
        if not os.path.isfile(data_path):
            raise SystemExit(f"[config] нет {data_path} для --resume/"
                             f"--analyze-only")
        with open(data_path) as f:
            data = json.load(f)

    def _missing():
        for tag in ("wm", "base"):
            for mode in MODES:
                if tag not in data or mode not in data.get(tag, {}):
                    return True
                for r in data[tag][mode]:
                    if not r.get("text", "").strip():
                        continue
                    if "L_wm" not in r or "L_base" not in r:
                        return True
        return False

    if _missing() and args.analyze_only:
        raise SystemExit("[analyze-only] data-файл не полон (нет полей L) — "
                         "запустите полный прогон или --resume")

    if _missing():
        # Проверки окружения до долгого прогона
        if not os.path.isabs(cfg.llama_bin):
            found = shutil.which(cfg.llama_bin)
            if found is None:
                raise SystemExit(f"[config] не найден бинарь сервера "
                                 f"'{cfg.llama_bin}' (--llama-bin или PATH)")
            cfg.llama_bin = found
        for gguf in (cfg.wm_gguf, cfg.base_gguf):
            p = os.path.join(cfg.models_dir, gguf)
            if not os.path.isfile(p):
                raise SystemExit(f"[config] не найден GGUF: {p}")
        # Генерация и скоринг через llama-server (одна модель в VRAM за раз)
        mgr = ServerManager(cfg)
        try:
            # Порядок фаз минимизирует загрузки модели (кэш текстов и
            # значений L — в data-файле, готовые строки пропускаются):
            #   1) wm:   сборка wm + L_wm на wm-текстах
            #   2) base: сборка base + L_base на всех текстах
            #   3) wm:   L_wm на base-текстах
            # Модель загружается только если для неё есть работа.
            def _need_collect(tag):
                return any(len(data.get(tag, {}).get(mode, []))
                           < len(prompts) for mode in MODES)

            def _need_score(field, tags):
                return any(r.get("text", "").strip() and field not in r
                           for tag in tags for mode in MODES
                           for r in data.get(tag, {}).get(mode, []))

            # Фаза 1 (wm): сборка wm-текстов + L_wm на них
            if _need_collect("wm") or _need_score("L_wm", ("wm",)):
                srv = Server(cfg, mgr.ensure_model(cfg.wm_gguf))
                for mode in MODES:
                    have = len(data.get("wm", {}).get(mode, []))
                    if have < len(prompts):
                        collect(cfg, srv, prompts, cfg.tokens, data,
                                "wm", mode, start=have)
                for mode in MODES:
                    score_texts_server(cfg, srv, data, "wm", mode, "L_wm")
            # Фаза 2 (base): сборка base-текстов + L_base на всех
            if (_need_collect("base")
                    or _need_score("L_base", ("wm", "base"))):
                srv = Server(cfg, mgr.ensure_model(cfg.base_gguf))
                for mode in MODES:
                    have = len(data.get("base", {}).get(mode, []))
                    if have < len(prompts):
                        collect(cfg, srv, prompts, cfg.tokens, data,
                                "base", mode, start=have)
                for tag in ("wm", "base"):
                    for mode in MODES:
                        score_texts_server(cfg, srv, data, tag, mode,
                                           "L_base")
            # Фаза 3 (wm): L_wm на base-текстах
            if _need_score("L_wm", ("base",)):
                srv = Server(cfg, mgr.ensure_model(cfg.wm_gguf))
                for mode in MODES:
                    score_texts_server(cfg, srv, data, "base", mode, "L_wm")
        finally:
            mgr.kill()

    # Анализ
    llrs = openstamp_llrs(data)
    det = SynthIDDetector()
    synth = synthid_scores(cfg, data, det)
    gm = gaussmark_scores(data)
    e2e_det = MarkLLMDetector()
    e2e = markllm_scores(data, e2e_det)

    aggs_llr, aggs_syn, aggs_gm, aggs_e2e = {}, {}, {}, {}
    for mode in MODES:
        llr_wm, llr_null = llrs[mode]
        gm_wm, gm_null, gm_p = gm[mode]
        aggs_llr[mode] = aggregate(llr_wm, llr_null)
        aggs_syn[mode] = aggregate(
            np.array([r["mean_g"] for r in synth["wm"][mode]]),
            np.array([r["mean_g"] for r in synth["base"][mode]]))
        aggs_gm[mode] = dict(aggregate(gm_wm, gm_null),
                             **pvalue_fields(gm_p))
        aggs_e2e[mode] = aggregate(
            np.array([r["p"] for r in e2e["wm"][mode]]),
            np.array([r["p"] for r in e2e["base"][mode]]))
    n_empty = sum(
        1 for tag in ("wm", "base") for mode in MODES
        for r in data[tag][mode] if not r.get("text", "").strip())
    if n_empty:
        print(f"[report] исключено текстов с пустым выходом "
              f"(зацикливание на спец-токенах): {n_empty}")

    q = quant_suffix(cfg.quant)
    report_name = f"report_{cfg.wm_name}{q}.json"
    plot_name = f"watermark_report_{cfg.wm_name}{q}.png"
    make_plot(cfg, llrs, synth, gm, e2e,
              {"llr": aggs_llr, "synth": aggs_syn, "gm": aggs_gm,
               "e2e": aggs_e2e},
              os.path.join(cfg.data, plot_name))
    print_report(cfg, llrs, synth, gm, e2e, aggs_llr, aggs_syn,
                 aggs_gm, aggs_e2e)

    per_seq = {mode: {
        "llr_wm": [float(x) for x in llrs[mode][0]],
        "llr_null": [float(x) for x in llrs[mode][1]],
        "mean_g_wm": [r["mean_g"] for r in synth["wm"][mode]],
        "mean_g_base": [r["mean_g"] for r in synth["base"][mode]],
        "gm_wm": [float(x) for x in gm[mode][0]],
        "gm_null": [float(x) for x in gm[mode][1]],
        "gm_p_wm": [float(x) for x in gm[mode][2]],
        "e2e_p_wm": [r["p"] for r in e2e["wm"][mode]],
        "e2e_p_base": [r["p"] for r in e2e["base"][mode]],
    } for mode in MODES}
    report = {
        "wm_model": cfg.wm_name,
        "base_model": cfg.base_name,
        "modes": {m: {"openstamp_llr": aggs_llr[m], "synthid": aggs_syn[m],
                     "gaussmark": aggs_gm[m], "markllm_e2e": aggs_e2e[m]}
                  for m in MODES},
        "per_sequence": per_seq,
        "plot": plot_name,
    }
    with open(os.path.join(cfg.data, report_name), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"[report] {os.path.join(cfg.data, report_name)}")


if __name__ == "__main__":
    main()
