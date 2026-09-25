#!/usr/bin/env python3
"""Статус прогона детекции watermark.

Читает data/data.json (сохраняется после каждого промпта) и показывает:
- сколько текстов собрано по каждому (модель, режим)
- сколько токенов
- сколько текстов уже посчитано под L_wm / L_base
- пустые тексты (модель зациклилась на спец-токенах) не скорятся —
  в знаменателе число непустых
- mtime файла и текущий llama-server

Запуск: python3 status.py [путь к data.json]
"""
import json
import os
import subprocess
import sys
import time


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "data", "data.json")
    if not os.path.isfile(path):
        print(f"нет файла: {path} (прогон ещё не начал писать данные)")
        return

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("data.json читается в момент записи — повторите через пару секунд")
        return

    st = os.stat(path)
    age = time.time() - st.st_mtime
    age_s = f"{int(age // 60)} мин {int(age % 60)} с" if age >= 60 else f"{int(age)} с"
    print(f"файл: {path}")
    print(f"обновлён: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))} "
          f"({age_s} назад)")
    print()

    total = done = 0
    for tag in ("wm", "base"):
        if tag not in data:
            print(f"{tag:5s}: —")
            continue
        for mode in data[tag]:
            recs = data[tag][mode]
            n = len(recs)
            # Строки с пустым текстом (зацикливание на спец-токенах)
            # detect.py не скорит — знаменатель = число непустых
            text_rows = [r for r in recs if r.get("text", "").strip()]
            n_text = len(text_rows)
            n_empty = n - n_text
            toks = sum(r.get("n", 0) for r in text_rows)
            lw = sum(1 for r in recs if "L_wm" in r)
            lb = sum(1 for r in recs if "L_base" in r)
            total += n_text
            done += (1 if lw >= n_text and lb >= n_text else 0)
            state = "готов" if (lw >= n_text and lb >= n_text) else (
                "скоринг L_base" if lw >= n_text else (
                    "скоринг L_wm" if (lw or lb) else "собран"))
            extra = f" (пустых: {n_empty})" if n_empty else ""
            print(f"{tag:5s}/{mode:9s}: {n:2d} текстов{extra}, "
                  f"{toks:6d} ток., L_wm  {lw:2d}/{n_text:2d}, "
                  f"L_base {lb:2d}/{n_text:2d}  [{state}]")

    print()
    print(f"итого: {total} текстов собрано, {done} полностью посчитаны")
    # Текущий llama-server
    try:
        out = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=10).stdout
        lines = [l for l in out.splitlines()
                 if "llama-server" in l and "grep" not in l]
        if lines:
            print()
            for l in lines:
                p = l.split()
                model = [a for a in p if a.endswith(".gguf")]
                print(f"сервер: pid {p[1]}, {model[0].rsplit('/', 1)[-1] if model else '?'}")
        else:
            print()
            print("сервер: не запущен")
    except (subprocess.SubprocessError, OSError):
        pass


if __name__ == "__main__":
    main()
