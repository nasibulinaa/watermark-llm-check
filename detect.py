#!/usr/bin/env python3
r"""
Утилита детекции watermark в выводе LLM.

Сравнивает две модели, сервируемые через llama.cpp: watermarked-кандидат
и базовая модель для нулевой гипотезы. Два независимых детектора, по
публичным референс-реализациям (оба — git-сабмодули этого проекта):

1) OpenStamp (openstamp/METHOD.md) — length-normalized log-likelihood
   ratio между watermarked и базовой моделями:

       $LLR(x) = \frac{1}{T-1}\sum_{t=1}^{T-1}\log\frac{p_{\text{wm}}(x_t \mid x_{<t})}{p_{\text{base}}(x_t \mid x_{<t})}$

   Референс-реализация формулы: openstamp/src/llr.py
   (length_normalized_llr); эквивалентность по-токенной версии
   проверяется в test_llr.py.
   Ключ не нужен. Порог τ калибруется эмпирически на непомеченном
   (null) тексте (METHOD.md: "Thresholds are therefore calibrated
   empirically").

2) SynthID-Text — G-значения: keyed-hash по (ngram_len-1)-контексту и
   кандидат-токену, бинарные G на каждой глубине, training-free
   weighted-mean детектор. Нулевое среднее G = 0.5 (Bernoulli-биты);
   у watermarked-текста смещается вверх (~0.75). Используется
   DEFAULT_WATERMARKING_CONFIG из репозитория (ngram_len=5, 30 ключей,
   context_history_size=1024).

Фазы: генерация WM-текстов -> генерация base-текстов (оба режима:
с reasoning и без, Qwen3 enable_thinking) -> скоринг полной
2x2-матрицы L-значений самим llama-server (текст принудительно
генерируется через GBNF-грамматику; logprobs сервера — pre-sampling,
т.е. точные log p(x_t|x_<t); сервер работает без speculative decoding)
-> анализ по режимам, отчёт, график.

Запуск:
  python3 detect.py --models-dir /path/to/gguf   # полный прогон
  python3 detect.py --resume                     # доскорить L
  python3 detect.py --analyze-only               # только анализ + график
"""

import argparse
import json
import math
import os
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
]


@dataclass
class Config:
    # Дисплейные имена моделей (в отчёте) и GGUF-файлы в models_dir
    wm_name: str = "Swift-1.5-Qwen3.8-27B-GSQ-RCO"
    base_name: str = "Qwen3.8-27B-GSQ-RCO"
    wm_gguf: str = "Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
    base_gguf: str = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
    # Окружение
    llama_bin: str = ("/home/alexey/mipt_mag_diploma-mipt_mag_diploma_private"
                      "/llm/llama.cpp/build_cuda/bin/llama-server")
    models_dir: str = "."
    device: str = "cuda0"
    host: str = "127.0.0.1"
    port: int = 8091
    server_ctx: int = 8192
    server_batch: int = 2048
    # Протокол
    prompts: int = 30
    tokens: int = 400
    data: str = os.path.join(ROOT, "data")


def save_data(cfg, data):
    """Сохранить data.json (вызывается после каждого промпта/текста)."""
    os.makedirs(cfg.data, exist_ok=True)
    path = os.path.join(cfg.data, "data.json")
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
        """L = Σ log p(x_t | x_<t) под текущей моделью. Тот же
        chat-completion-контекст, что и в generate (enable_thinking
        всегда false — текст зафиксирован GBNF-грамматикой, thinking
        блок невозможен); logprobs сервера — pre-sampling (сырой
        softmax по всему словарю, без влияния сэмплеров).
        -> (L, n_scored): n_scored может незначительно превышать
        n_tokens — грамматика ре-токенизирует текст, детокенизация
        не обратима. Первый текст-токен не входит: референс
        (openstamp/src/llr.py) считает с labels = input_ids[:, 2:]."""
        if not text:
            raise SystemExit("[score] пустой текст — строка исключена из "
                             "скоринга (см. score_texts_server)")
        grammar = 'root ::= "' + gbnf_escape(text) + '"'
        r = self.s.post(f"{self.base}/v1/chat/completions", json={
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "grammar": grammar,
            "max_tokens": n_tokens + 1000,
            # верхняя граница с запасом: ре-токенизация текста может
            # добавить ~15% токенов (наблюдаемо +149 к 973); после
            # завершения грамматики сервер продолжает свободную
            # генерацию до max_tokens (лишние токены в ответ не входят)
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
            # Round-trip tok(text)→detok не обратим: грамматика
            # фиксирует токен-последовательность текста, а детокенизация
            # может терять/добавлять немного символов (границы
            # токенов). Принимаем почти-равный текст в обоих
            # направлениях; свободная генерация расходится сразу.
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
    """-> {mode: (llr_wm, llr_null)}. LLR: формула из openstamp/src/llr.py
    (length_normalized_llr) — (Σ log p_wm − Σ log p_base) / N, N — число
    посчитанных токенов. Референс-реализация работает с полными logits
    (B, T, V); здесь используются по-токенные logprobs сервера (gather
    log_softmax по позициям токенов) — эквивалентность проверяется в
    test_llr.py."""
    out = {}
    for mode in MODES:
        vals = {}
        for tag in ("wm", "base"):
            vals[tag] = np.array([
                (rec["L_wm"] - rec["L_base"])
                / max(rec.get("n_scored", rec["n"]) - 1, 1)
                for rec in data[tag][mode]
                if rec.get("text", "").strip()
                and "L_wm" in rec and "L_base" in rec])
        out[mode] = (vals["wm"], vals["base"])
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
    """-> {tag: {mode: [rows]}}: по-последовательностные G-значения."""
    out = {}
    for tag in ("wm", "base"):
        out[tag] = {}
        for mode in MODES:
            rows = []
            for rec in data[tag][mode]:
                if not rec.get("text", "").strip():
                    continue
                g, z, M = det.score(rec["token_ids"])
                rows.append({"mean_g": g, "z": z, "M": M})
                rec["synthid"] = rows[-1]
            out[tag][mode] = rows
            save_data(cfg, data)
    return out


# --------------------------------------------------------------------------
# Статистика и вердикт
# --------------------------------------------------------------------------
def aggregate(signal, null):
    """Сводка по signal/null: tau = mean+3std; verdict = (hits >= 75% n) и z_agg >= 3."""
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
        "verdict": (hits >= 0.75 * len(signal)) and (z_agg >= 3.0),
    }


# --------------------------------------------------------------------------
# График
# --------------------------------------------------------------------------
def make_plot(cfg, llrs, synth, aggs, res_path):
    """2×2-график: LLR и mean G, reasoning on/off; сохранение в res_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(f"Watermark detection: {cfg.wm_name} vs {cfg.base_name}\n"
                 "(OpenStamp LLR + SynthID-Text G-values; "
                 "columns: reasoning on/off)", fontsize=13)

    for c, mode in enumerate(MODES):
        on = "on" if mode == "reason" else "off"
        llr_wm, llr_null = llrs[mode]
        gw = np.array([r["mean_g"] for r in synth["wm"][mode]])
        gb = np.array([r["mean_g"] for r in synth["base"][mode]])

        ax = axes[0][c]
        ax.hist(llr_null, bins=12, alpha=0.55, color="tab:blue",
                label="base texts (null)")
        ax.hist(llr_wm, bins=12, alpha=0.55, color="tab:red",
                label=f"{cfg.wm_name} texts (signal)")
        ax.axvline(aggs[mode]["tau"], color="k", ls="--", lw=1.5,
                   label=f"tau = {aggs[mode]['tau']:.3f}")
        ax.set_title(f"OpenStamp: length-normalized LLR per text "
                     f"(reasoning {on})")
        ax.set_xlabel("LLR (nats/token)")
        ax.legend()

        ax = axes[1][c]
        ax.hist(gb, bins=12, alpha=0.55, color="tab:blue",
                label="base texts (null)")
        ax.hist(gw, bins=12, alpha=0.55, color="tab:red",
                label=f"{cfg.wm_name} texts (signal)")
        ax.axvline(0.5, color="k", ls="--", lw=1.5, label="null mean = 0.5")
        ax.axvline(0.75, color="gray", ls=":", lw=1.5,
                   label="expected watermarked ~ 0.75")
        ax.set_title(f"SynthID-Text: mean G-value per text "
                     f"(reasoning {on})")
        ax.set_xlabel("mean G")
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(res_path, dpi=150)
    print(f"[plot] сохранён {res_path}")


# --------------------------------------------------------------------------
# Отчёт
# --------------------------------------------------------------------------
def print_report(cfg, llrs, synth, aggs_llr, aggs_syn):
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
        gw = np.array([r["mean_g"] for r in synth["wm"][mode]])
        gb = np.array([r["mean_g"] for r in synth["base"][mode]])
        zw = np.array([r["z"] for r in synth["wm"][mode]])
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
        print(f"    z-сводка: {a_llr['z_agg']:+.2f}")
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
        print(f"    z-сводка: {a_syn['z_agg']:+.2f}")
        print(f"    => {'WATERMARK ОБНАРУЖЕН' if a_syn['verdict'] else 'watermark не обнаружен'}")
    print()
    print("=" * 78)
    for mode in MODES:
        on = "on" if mode == "reason" else "off"
        v_llr = aggs_llr[mode]["verdict"]
        v_syn = aggs_syn[mode]["verdict"]
        if v_llr and v_syn:
            s = "watermark ВЫЯВЛЕН обоими детекторами"
        elif v_llr or v_syn:
            s = ("watermark выявлен одним детектором — результат "
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
    # Модели
    ap.add_argument("--wm-model", default=Config.wm_name,
                    help="имя watermarked-модели (в отчёте)")
    ap.add_argument("--base-model", default=Config.base_name,
                    help="имя базовой модели (в отчёте)")
    ap.add_argument("--wm-gguf", default=Config.wm_gguf,
                    help="GGUF watermarked-модели в models-dir")
    ap.add_argument("--base-gguf", default=Config.base_gguf,
                    help="GGUF базовой модели в models-dir")
    # Протокол
    ap.add_argument("--prompts", type=int, default=Config.prompts)
    ap.add_argument("--tokens", type=int, default=Config.tokens)
    ap.add_argument("--analyze-only", action="store_true",
                    help="не собирать, только анализ по data.json")
    ap.add_argument("--resume", action="store_true",
                    help="использовать существующий data.json, "
                         "доскорить только недостающие поля L")
    args = ap.parse_args()

    cfg = Config(
        wm_name=args.wm_model,
        base_name=args.base_model,
        wm_gguf=args.wm_gguf,
        base_gguf=args.base_gguf,
        llama_bin=args.llama_bin,
        models_dir=args.models_dir,
        device=args.device,
        host=args.host,
        port=args.port,
        prompts=args.prompts,
        tokens=args.tokens,
    )

    data_path = os.path.join(cfg.data, "data.json")
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
        raise SystemExit("[analyze-only] data.json не полон (нет полей L) — "
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
            # Фаза 1: генерация (wm-модель — оба режима, base-модель — оба)
            model_id = mgr.ensure_model(cfg.wm_gguf)
            srv = Server(cfg, model_id)
            for mode in MODES:
                have = len(data.get("wm", {}).get(mode, []))
                if have < len(prompts):
                    collect(cfg, srv, prompts, cfg.tokens, data, "wm",
                            mode, start=have)
            model_id = mgr.ensure_model(cfg.base_gguf)
            srv = Server(cfg, model_id)
            for mode in MODES:
                have = len(data.get("base", {}).get(mode, []))
                if have < len(prompts):
                    collect(cfg, srv, prompts, cfg.tokens, data, "base",
                            mode, start=have)
            # Фаза 2: скоринг под WM-моделью (L_wm)
            model_id = mgr.ensure_model(cfg.wm_gguf)
            srv = Server(cfg, model_id)
            for tag in ("wm", "base"):
                for mode in MODES:
                    if any("L_wm" not in r for r in data[tag][mode]):
                        score_texts_server(cfg, srv, data, tag, mode, "L_wm")
            # Фаза 3: скоринг под BASE-моделью (L_base)
            model_id = mgr.ensure_model(cfg.base_gguf)
            srv = Server(cfg, model_id)
            for tag in ("wm", "base"):
                for mode in MODES:
                    if any("L_base" not in r for r in data[tag][mode]):
                        score_texts_server(cfg, srv, data, tag, mode, "L_base")
        finally:
            mgr.kill()

    # Анализ
    llrs = openstamp_llrs(data)
    det = SynthIDDetector()
    synth = synthid_scores(cfg, data, det)

    aggs_llr, aggs_syn = {}, {}
    for mode in MODES:
        llr_wm, llr_null = llrs[mode]
        aggs_llr[mode] = aggregate(llr_wm, llr_null)
        aggs_syn[mode] = aggregate(
            np.array([r["mean_g"] for r in synth["wm"][mode]]),
            np.array([r["mean_g"] for r in synth["base"][mode]]))
    n_empty = sum(
        1 for tag in ("wm", "base") for mode in MODES
        for r in data[tag][mode] if not r.get("text", "").strip())
    if n_empty:
        print(f"[report] исключено текстов с пустым выходом "
              f"(зацикливание на спец-токенах): {n_empty}")

    report_name = f"report_{cfg.wm_name}.json"
    plot_name = f"watermark_report_{cfg.wm_name}.png"
    make_plot(cfg, llrs, synth, aggs_llr, os.path.join(cfg.data, plot_name))
    print_report(cfg, llrs, synth, aggs_llr, aggs_syn)

    per_seq = {mode: {
        "llr_wm": [float(x) for x in llrs[mode][0]],
        "llr_null": [float(x) for x in llrs[mode][1]],
        "mean_g_wm": [r["mean_g"] for r in synth["wm"][mode]],
        "mean_g_base": [r["mean_g"] for r in synth["base"][mode]],
    } for mode in MODES}
    report = {
        "wm_model": cfg.wm_name,
        "base_model": cfg.base_name,
        "modes": {m: {"openstamp_llr": aggs_llr[m], "synthid": aggs_syn[m]}
                  for m in MODES},
        "per_sequence": per_seq,
        "plot": plot_name,
    }
    with open(os.path.join(cfg.data, report_name), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"[report] {os.path.join(cfg.data, report_name)}")


if __name__ == "__main__":
    main()
