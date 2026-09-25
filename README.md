# watermark-llm-check

Утилита детекции watermark в выводе LLM. Сравнивает две модели, сервируемые
через llama.cpp, и отвечает на вопрос: есть ли watermark в выводе
watermarked-кандидата.

Два независимых детектора, по публичным референс-реализациям (оба —
git-сабмодули проекта):

- **OpenStamp** (`openstamp/METHOD.md`) — length-normalized log-likelihood
  ratio между watermarked и базовой моделями, ключ не нужен:
  `LLR(x) = (1/(T-1)) * Σ_t log[ p_wm(x_t|x_<t) / p_base(x_t|x_<t) ]`.
- **SynthID-Text** (`synthid-text/`, Apache-2.0) — keyed-hash G-значения
  (ngram_len=5, 30 ключей, context_history_size=1024 из
  `DEFAULT_WATERMARKING_CONFIG`), training-free weighted-mean детектор.
  Null G = 0.5 (Bernoulli-биты); у watermarked-текста смещается вверх
  (~0.75).

## Структура

```
detect.py     # main: сервер, генерация, скоринг, анализ, отчёт
score.cxx     # C++-скорер точных per-token logprob
openstamp/    # сабмодуль: референс OpenStamp (METHOD.md)
synthid-text/ # сабмодуль: референс SynthID-Text (модули)
data/         # data.json, логи, тексты (не в git);
              # report_<модель>.json и watermark_report_<модель>.png — в git
```

## Установка

```
git clone --recursive <url>
cd watermark-llm-check
pip install -r requirements.txt   # numpy, requests, torch, matplotlib
```

## Сборка скорера

Одна команда (нужна сборка llama.cpp, дающая `libllama.so` и заголовки):

```
c++ -O2 -std=c++17 -o score score.cxx \
    -I<path/to/llama.cpp>/include \
    -I<path/to/llama.cpp>/ggml/include \
    -L<каталог с libllama.so> -Wl,-rpath,<каталог с libllama.so> \
    -l:libllama.so -l:libggml.so -l:libggml-base.so -l:libggml-cpu.so -l:libggml-cuda.so
```

(для CPU-сборки — `libggml-cpu.so` вместо `libggml-cuda.so`.)

## Запуск

```
python3 detect.py --models-dir /path/to/gguf          # полный прогон
python3 detect.py --resume                             # доскорить недостающие L
python3 detect.py --analyze-only                       # только анализ + график
```

Полезные флаги (`--help` для списка): `--wm-gguf/--base-gguf` (файлы в
`--models-dir`), `--wm-model/--base-model` (имена в отчёте), `--llama-bin`
(бинарь llama-server), `--device` (`cuda0`/`cpu`), `--host`/`--port`,
`--prompts` (по умолчанию 24), `--tokens` (по умолчанию 400).

## Прогон

1. **Генерация** — N текстов (по умолчанию 24) через llama-server; VRAM
   ограничен, сервер перезапускается со второй моделью;
2. **Скоринг** — C++-скорер считает L = Σ log p(x_t|x_<t) для полной 2×2
   матрицы (wm/base тексты × wm/base модель) без speculative decoding
   (logprobs сервера не используются — при ngram-mod-draft ~11% позиций
   в них неточны);
3. **Анализ** — детекторы, отчёт, график.

## Результат

- `data/watermark_report_<модель>.png` — 4 панели (LLR и G-значения);
- `data/report_<модель>.json` — все числа (поле `plot` — относительное имя);
- в stdout — отчёт с вердиктом по каждому детектору.

## Особенности

- **n_ctx скорера** вычисляется сам из размеров текстов:
  `max(2048, max_len + 1024)`.
- **RAM**: скорер держит 2 модели × n_ctx simultaneously; при нехватке
  памяти уменьшать `--tokens` или `--prompts`.
- **VRAM**: сервер и C++-скорер не поднимаются одновременно.
- **Логика детекции** — по METHOD.md (OpenStamp) и README (SynthID-Text).

## Лицензия

MIT (см. `LICENSE`). Референс-репозитории — под их собственными
лицензиями (Apache-2.0 для synthid-text).
