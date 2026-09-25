#!/usr/bin/env python3
"""Статус прогона детекции watermark.

Читает data/data.json (сохраняется после каждого промпта) и показывает:
- сколько текстов собрано по каждому (модель, режим)
- сколько токенов
- сколько текстов уже посчитано под L_wm / L_base
- мtime файла и текущий llama-server

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
            toks = sum(r.get("n", 0) for r in recs)
            lw = sum(1 for r in recs if "L_wm" in r)
            lb = sum(1 for r in recs if "L_base" in r)
            total += n
            done += (1 if lw and lb else 0)
            state = "готов" if lw and lb else (
                "скоринг L_base" if lw else (
                    "скоринг L_wm" if (lw or lb) else (
                        "сбор" if n < 30 else "собран")))
            print(f"{tag:5s}/{mode:9s}: {n:2d}/30 текстов, "
                  f"{toks:6d} ток., L_wm  {lw:2d}, L_base  {lb:2d}  [{state}]")

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
