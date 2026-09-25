#!/usr/bin/env python3
"""
Утилита детекции watermark в выводе LLM.

Сравнивает две модели, сервируемые через llama.cpp: watermarked-кандидат
и базовая модель для нулевой гипотезы. Два независимых детектора, по
публичным референс-реализациям (оба — git-сабмодули этого проекта):

1) OpenStamp (openstamp/METHOD.md) — length-normalized log-likelihood
   ratio между watermarked и базовой моделями:

       LLR(x) = (1/(T-1)) * Σ_t log[ p_wm(x_t|x_<t) / p_base(x_t|x_<t) ]

   Ключ не нужен. Порог τ калибруется эмпирически на непомеченном
   (null) тексте (METHOD.md: "Thresholds are therefore calibrated
   empirically").

2) SynthID-Text — G-значения: keyed-hash по (ngram_len-1)-контексту и
   кандидат-токену, бинарные G на каждой глубине, training-free
   weighted-mean детектор. Нулевое среднее G = 0.5 (Bernoulli-биты);
   у watermarked-текста смещается вверх (~0.75). Используется
   DEFAULT_WATERMARKING_CONFIG из репозитория (ngram_len=5, 30 ключей,
   context_history_size=1024).

Фазы: генерация WM-текстов -> генерация base-текстов -> скоринг полной
2x2-матрицы L-значений C++-скорером (score.cxx; точные log p(x_t|x_<t)
без speculative decoding — logprobs сервера не используются, т.к. при
ngram-mod-draft ~11% позиций в них неточны, значения из draft-кэша)
-> анализ, отчёт, график.

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
]

SCORE_BIN = os.path.join(ROOT, "score")


@dataclass
class Config:
    # Дисплейные имена моделей (в отчёте) и GGUF-файлы в models_dir
    wm_name: str = "Swift-1.5-Qwen3.8-27B-GSQ-RCO"
    base_name: str = "Qwen3.8-27B-GSQ-RCO"
    wm_gguf: str = "Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
    base_gguf: str = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
    # Окружение
    llama_bin: str = "llama-server"
    models_dir: str = "."
    device: str = "cuda0"
    host: str = "127.0.0.1"
    port: int = 8091
    server_ctx: int = 8192
    server_batch: int = 2048
    # Протокол
    prompts: int = 24
    tokens: int = 400
    data: str = os.path.join(ROOT, "data")


def save_data(cfg, data):
    os.makedirs(cfg.data, exist_ok=True)
    with open(os.path.join(cfg.data, "data.json"), "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------
# Управление llama-server (одна модель в VRAM за раз)
# --------------------------------------------------------------------------
class ServerManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.proc = None
        self.log_fh = None

    def _cmd(self, gguf):
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
        os.makedirs(self.cfg.data, exist_ok=True)
        log_path = os.path.join(self.cfg.data, f"server_{gguf}.log")
        self.log_fh = open(log_path, "ab")
        print(f"[server] запуск: {' '.join(self._cmd(gguf))}", flush=True)
        self.proc = subprocess.Popen(
            self._cmd(gguf), stdout=self.log_fh,
            stderr=subprocess.STDOUT, start_new_session=True)
        print(f"[server] pid={self.proc.pid}, лог: {log_path}", flush=True)

    def kill(self):
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.proc.wait(timeout=30)
        self.proc = None
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
        return bool(loaded) and os.path.basename(loaded) == gguf

    def wait_ready(self, gguf, timeout=900):
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
class Server:
    def __init__(self, cfg, model_id, timeout=900):
        self.base = f"http://{cfg.host}:{cfg.port}"
        self.s = requests.Session()
        self.s.headers["Content-Type"] = "application/json"
        self.timeout = timeout
        self.model_id = model_id

    def generate(self, prompt, max_tokens,
                 temperature=1.0, top_k=20, top_p=0.95):
        """-> (text, token_ids). logprobs=1 нужен только ради token id
        (для SynthID-детектора); сами logprobs сервера в скоринге не
        используются (draft-кэш)."""
        r = self.s.post(f"{self.base}/v1/completions", json={
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "logprobs": 1,
            "stream": False,
        }, timeout=self.timeout)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        ids = [int(t["id"])
               for t in (ch.get("logprobs") or {}).get("content") or []]
        return ch.get("text", ""), ids


# --------------------------------------------------------------------------
# Сбор данных
# --------------------------------------------------------------------------
def collect(cfg, server, prompts, max_tokens, data, tag):
    out = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        print(f"[collect:{tag}] {i+1}/{len(prompts)}: {p[:60]!r} ...", flush=True)
        text, ids = server.generate(p, max_tokens)
        out.append({"prompt": p, "text": text, "token_ids": ids, "n": len(ids)})
        data[tag] = out
        save_data(cfg, data)
        el = time.time() - t0
        eta = el / (i + 1) * (len(prompts) - i - 1)
        print(f"[collect:{tag}] {i+1}/{len(prompts)}: {len(ids)} tok, "
              f"прошло {el:.0f} с, ETA {eta:.0f} с", flush=True)


def score_texts_cxx(cfg, model_gguf, records, data, tag, field):
    """Скоринг текстов C++-скорером: L = Σ_{t>=2} log p(x_t|x_<t) под
    указанной моделью. n_ctx скорер вычисляет сам из размеров текстов."""
    if not os.path.isfile(SCORE_BIN):
        raise SystemExit(f"[score:{field}] нет {SCORE_BIN} — соберите "
                         f"score.cxx (см. README)")
    os.makedirs(cfg.data, exist_ok=True)
    files = []
    for i, rec in enumerate(records):
        p = os.path.join(cfg.data, f"{tag}_{i}.txt")
        with open(p, "w") as f:
            f.write(rec["text"])
        files.append(p)
    model_path = os.path.join(cfg.models_dir, model_gguf)
    print(f"[score:{field}] {len(files)} текстов <- {model_gguf}", flush=True)
    t0 = time.time()
    r = subprocess.run([SCORE_BIN, model_path] + files,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-3000:] + "\n")
        raise SystemExit(f"[score:{field}] скорер завершился с кодом {r.returncode}")
    by_path = {}
    for line in r.stdout.splitlines():
        if line.startswith("L="):
            parts = line.split()
            by_path[parts[2]] = float(parts[0][2:])
    for i, p in enumerate(files):
        if p not in by_path:
            raise SystemExit(f"[score:{field}] нет результата для {p}")
        data[tag][i][field] = by_path[p]
        save_data(cfg, data)
    print(f"[score:{field}] готово за {time.time() - t0:.0f} с", flush=True)


# --------------------------------------------------------------------------
# Детектор 1: OpenStamp (LLR)
# --------------------------------------------------------------------------
def openstamp_llrs(data):
    """LLR: wm-тексты (сигнал) и base-тексты (нулевое распределение)."""
    llr_wm, llr_null = [], []
    for i in range(len(data["wm"])):
        n = data["wm"][i]["n"]
        llr_wm.append((data["wm"][i]["L_wm"] - data["wm"][i]["L_base"])
                      / max(n - 1, 1))
    for i in range(len(data["base"])):
        n = data["base"][i]["n"]
        llr_null.append((data["base"][i]["L_wm"] - data["base"][i]["L_base"])
                        / max(n - 1, 1))
    return np.array(llr_wm), np.array(llr_null)


# --------------------------------------------------------------------------
# Детектор 2: SynthID-Text (G-значения, модули из сабмодуля synthid-text)
# --------------------------------------------------------------------------
class SynthIDDetector:
    def __init__(self):
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
    out = {}
    for tag in ("wm", "base"):
        rows = []
        for rec in data[tag]:
            g, z, M = det.score(rec["token_ids"])
            rows.append({"mean_g": g, "z": z, "M": M})
            rec["synthid"] = rows[-1]
        out[tag] = rows
        save_data(cfg, data)
    return out


# --------------------------------------------------------------------------
# Статистика и вердикт
# --------------------------------------------------------------------------
def aggregate(signal, null):
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
def make_plot(cfg, llr_wm, llr_null, synth, a_llr, res_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gw = np.array([r["mean_g"] for r in synth["wm"]])
    gb = np.array([r["mean_g"] for r in synth["base"]])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(f"Watermark detection: {cfg.wm_name} vs {cfg.base_name}\n"
                 "(OpenStamp LLR + SynthID-Text G-values)", fontsize=13)

    ax = axes[0][0]
    ax.hist(llr_null, bins=12, alpha=0.55, color="tab:blue",
            label="base texts (null)")
    ax.hist(llr_wm, bins=12, alpha=0.55, color="tab:red",
            label=f"{cfg.wm_name} texts (signal)")
    ax.axvline(a_llr["tau"], color="k", ls="--", lw=1.5,
              label=f"tau = {a_llr['tau']:.3f}")
    ax.set_title("OpenStamp: length-normalized LLR per text")
    ax.set_xlabel("LLR (nats/token)")
    ax.legend()

    ax = axes[0][1]
    ax.hist(gb, bins=12, alpha=0.55, color="tab:blue", label="base texts (null)")
    ax.hist(gw, bins=12, alpha=0.55, color="tab:red",
            label=f"{cfg.wm_name} texts (signal)")
    ax.axvline(0.5, color="k", ls="--", lw=1.5, label="null mean = 0.5")
    ax.axvline(0.75, color="gray", ls=":", lw=1.5,
              label="expected watermarked ~ 0.75")
    ax.set_title("SynthID-Text: mean G-value per text (default key)")
    ax.set_xlabel("mean G")
    ax.legend()

    ax = axes[1][0]
    xs = np.arange(len(llr_wm))
    ax.bar(xs - 0.2, llr_null, width=0.4, color="tab:blue", alpha=0.7,
           label="null")
    ax.bar(xs + 0.2, llr_wm, width=0.4, color="tab:red", alpha=0.7,
           label=cfg.wm_name)
    ax.axhline(a_llr["tau"], color="k", ls="--", lw=1.5)
    ax.set_title("Per-prompt LLR")
    ax.set_xlabel("prompt index")
    ax.set_ylabel("LLR")
    ax.legend()

    ax = axes[1][1]
    ax.bar(xs - 0.2, gb, width=0.4, color="tab:blue", alpha=0.7, label="null")
    ax.bar(xs + 0.2, gw, width=0.4, color="tab:red", alpha=0.7,
           label=cfg.wm_name)
    ax.axhline(0.5, color="k", ls="--", lw=1.5)
    ax.axhline(0.75, color="gray", ls=":", lw=1.5)
    ax.set_title("Per-prompt mean G-value")
    ax.set_xlabel("prompt index")
    ax.set_ylabel("mean G")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(res_path, dpi=150)
    print(f"[plot] сохранён {res_path}")


# --------------------------------------------------------------------------
# Отчёт
# --------------------------------------------------------------------------
def print_report(cfg, synth, a_llr, a_syn):
    gw = np.array([r["mean_g"] for r in synth["wm"]])
    gb = np.array([r["mean_g"] for r in synth["base"]])
    zw = np.array([r["z"] for r in synth["wm"]])
    print()
    print("=" * 78)
    print("  ИТОГОВЫЙ ОТЧЁТ: детекция watermark")
    print(f"  watermarked-кандидат: {cfg.wm_name}")
    print(f"  база (null):          {cfg.base_name}")
    print("=" * 78)
    print()
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
    if a_llr["verdict"] and a_syn["verdict"]:
        print("ИТОГО: watermark ВЫЯВЛЕН обоими детекторами.")
    elif a_llr["verdict"] or a_syn["verdict"]:
        print("ИТОГО: watermark выявлен одним детектором — результат "
              "однозначным не является, нужны данные с известным ключом.")
    else:
        print("ИТОГО: watermark НЕ выявлен ни одним детектором "
              "(при известном/дефолтном ключе).")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Окружение
    ap.add_argument("--llama-bin", default="llama-server",
                    help="бинарь llama-server (PATH или полный путь)")
    ap.add_argument("--models-dir", default=".",
                    help="каталог с GGUF-файлами")
    ap.add_argument("--device", default="cuda0",
                    help="устройство сервера (llama.cpp -device)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    # Модели
    ap.add_argument("--wm-model", default="Swift-1.5-Qwen3.8-27B-GSQ-RCO",
                    help="имя watermarked-модели (в отчёте)")
    ap.add_argument("--base-model", default="Qwen3.8-27B-GSQ-RCO",
                    help="имя базовой модели (в отчёте)")
    ap.add_argument("--wm-gguf",
                    default="Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
                    help="GGUF watermarked-модели в models-dir")
    ap.add_argument("--base-gguf",
                    default="Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
                    help="GGUF базовой модели в models-dir")
    # Протокол
    ap.add_argument("--prompts", type=int, default=24)
    ap.add_argument("--tokens", type=int, default=400)
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

    data = None
    if args.analyze_only or args.resume:
        if not os.path.isfile(data_path):
            raise SystemExit(f"[config] нет {data_path} для --resume/"
                             f"--analyze-only")
        with open(data_path) as f:
            data = json.load(f)

    if data is None:
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
        # Полный прогон: генерация через сервер
        mgr = ServerManager(cfg)
        data = {}
        # Фаза 1: wm — генерация
        model_id = mgr.ensure_model(cfg.wm_gguf)
        collect(cfg, Server(cfg, model_id), prompts, cfg.tokens, data, "wm")
        # Фаза 2: base — генерация
        model_id = mgr.ensure_model(cfg.base_gguf)
        collect(cfg, Server(cfg, model_id), prompts, cfg.tokens, data, "base")
        mgr.kill()  # освободить VRAM для C++-скорера

    # Скоринг (C++-скорер, сервер не нужен): полная 2x2-матрица L-значений
    for tag, model_gguf, field in (
            ("wm", cfg.wm_gguf, "L_wm"),
            ("wm", cfg.base_gguf, "L_base"),
            ("base", cfg.wm_gguf, "L_wm"),
            ("base", cfg.base_gguf, "L_base"),
    ):
        if any(field not in r for r in data[tag]):
            score_texts_cxx(cfg, model_gguf, data[tag], data, tag, field)

    # Анализ
    llr_wm, llr_null = openstamp_llrs(data)
    det = SynthIDDetector()
    synth = synthid_scores(cfg, data, det)

    a_llr = aggregate(llr_wm, llr_null)
    a_syn = aggregate(np.array([r["mean_g"] for r in synth["wm"]]),
                      np.array([r["mean_g"] for r in synth["base"]]))

    report_name = f"report_{cfg.wm_name}.json"
    plot_name = f"watermark_report_{cfg.wm_name}.png"
    make_plot(cfg, llr_wm, llr_null, synth, a_llr,
              os.path.join(cfg.data, plot_name))
    print_report(cfg, synth, a_llr, a_syn)

    report = {
        "wm_model": cfg.wm_name,
        "base_model": cfg.base_name,
        "openstamp_llr": a_llr,
        "synthid": a_syn,
        "per_sequence_llr_wm": [float(x) for x in llr_wm],
        "per_sequence_llr_null": [float(x) for x in llr_null],
        "per_sequence_mean_g_wm": [r["mean_g"] for r in synth["wm"]],
        "per_sequence_mean_g_base": [r["mean_g"] for r in synth["base"]],
        "plot": plot_name,
    }
    with open(os.path.join(cfg.data, report_name), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"[report] {os.path.join(cfg.data, report_name)}")


if __name__ == "__main__":
    main()
