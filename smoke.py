#!/usr/bin/env python3
"""Smoke-тест: chat-completions + enable_thinking + grammar-score."""
import requests

BASE = "http://127.0.0.1:8091"
s = requests.Session()
s.headers["Content-Type"] = "application/json"


def model_id():
    r = s.get(f"{BASE}/v1/models", timeout=10)
    r.raise_for_status()
    js = r.json()
    for m in js.get("data") or []:
        st = m.get("status")
        if st is None or st == "loaded":
            return m.get("id")
    raise SystemExit("нет загруженной модели")


MODEL = model_id()


def gen(prompt, reason, max_tokens=300):
    r = s.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "logprobs": 1,
        "chat_template_kwargs": {"enable_thinking": reason},
        "stream": False,
    }, timeout=300)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    ids = [int(t["id"])
           for t in (ch.get("logprobs") or {}).get("content") or []]
    return ch.get("message", {}).get("content", ""), ids


def score(prompt, text, n_tokens):
    out = []
    for ch in text:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    grammar = 'root ::= "' + "".join(out) + '"'
    r = s.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "grammar": grammar,
        "max_tokens": n_tokens + 100,
        "logprobs": 1,
        "temperature": 1.0,
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }, timeout=300)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    gen = ch.get("message", {}).get("content", "")
    lps = (ch.get("logprobs") or {}).get("content") or []
    return gen, lps


print("model:", MODEL)
print()
print("=== 1. reasoning on ===")
t1, ids1 = gen("Привет. Расскажи одним абзацем, кто ты.", reason=True)
CLOSE = chr(60) + "/" + "think" + chr(62)
THINK = chr(10) + CLOSE + chr(10)
print(f"len={len(t1)}, ids={len(ids1)}")
print("think-блок:", THINK in t1)
print("начало:", repr(t1[:120]))

print()
print("=== 2. reasoning off ===")
t2, ids2 = gen("Привет. Расскажи одним абзацем, кто ты.", reason=False)
print(f"len={len(t2)}, ids={len(ids2)}")
print("think-блок:", THINK in t2)
print("начало:", repr(t2[:120]))

print()
print("=== 3. grammar-score ===")
text = "Вода — это химическое соединение."
gen_text, lps = score("Одним предложением:", text, len(text))
print("grammar совпал:", gen_text == text)
L = sum(t["logprob"] for t in lps[1:]) if len(lps) >= 2 else None
print("n_lps:", len(lps), "L=", L)
print("ids в logprobs:", all("id" in t for t in lps[:3]))
