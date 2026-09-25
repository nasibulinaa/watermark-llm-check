# watermark-llm-check

Утилита детекции watermark в выводе LLM. Сравнивает две модели, сервируемые
через llama.cpp, и отвечает на вопрос: есть ли watermark в выводе
watermarked-кандидата.

Два независимых детектора, по публичным референс-реализациям (оба —
git-сабмодули проекта):

- **OpenStamp** (`openstamp/METHOD.md`) — length-normalized log-likelihood
  ratio между watermarked и базовой моделями, ключ не нужен:
  $LLR(x) = \frac{1}{T-1}\sum_{t=1}^{T-1}\log\frac{p_{\text{wm}}(x_t \mid x_{<t})}{p_{\text{base}}(x_t \mid x_{<t})}$.
- **SynthID-Text** (`synthid-text/`, Apache-2.0) — keyed-hash G-значения
  (ngram_len=5, 30 ключей, context_history_size=1024 из
  `DEFAULT_WATERMARKING_CONFIG`), training-free weighted-mean детектор.
  Null G = 0.5 (Bernoulli-биты); у watermarked-текста смещается вверх
  (~0.75).

## Структура

```
detect.py     # main: сервер, генерация, скоринг, анализ, отчёт
test_llr.py   # эквивалентность по-токенного LLR и openstamp/src/llr.py
smoke.py      # ручной smoke-тест сервера (thinking + grammar-score)
status.py     # статус прогона по data.json
openstamp/    # сабмодуль: референс OpenStamp (METHOD.md, src/llr.py)
synthid-text/ # сабмодуль: референс SynthID-Text (модули)
data/         # data.json, логи (не в git);
              # report_<модель>.json и watermark_report_<модель>.png — в git
```

## Установка

```
git clone --recursive <url>
cd watermark-llm-check
pip install -r requirements.txt   # numpy, requests, torch, matplotlib
```

## Запуск

```
python3 detect.py --models-dir /path/to/gguf          # полный прогон
python3 detect.py --resume                             # доскорить недостающие L
python3 detect.py --analyze-only                       # только анализ + график
python3 status.py                                      # статус прогона
```

Полезные флаги (`--help` для списка): `--wm-gguf/--base-gguf` (файлы в
`--models-dir`), `--wm-model/--base-model` (имена в отчёте), `--llama-bin`
(бинарь llama-server), `--device` (например `cuda0`, `ROCm1`, `cpu`),
`--prompts` (по умолчанию 30), `--tokens` (по умолчанию 400).

## Прогон

1. **Генерация** — N текстов (по умолчанию 30) через llama-server; в VRAM
   одна модель за раз, сервер перезапускается со второй моделью; при
   нехватке памяти уменьшать `--tokens` или `--prompts`;
2. **Скоринг** — тот же сервер считает L = Σ log p(x_t|x_<t>) для полной
   2×2-матрицы (wm/base тексты × wm/base модель): текст принудительно
   генерируется через GBNF-грамматику, logprobs — pre-sampling (сырой
   softmax, без влияния сэмплеров); сервер работает без speculative
   decoding; строки с пустым текстом (модель зациклилась на спец-токенах)
   исключаются из скоринга и анализа — в stdout печатается их число;
   round-trip tok(text)→detok не обратим, поэтому проверка
   фиксации текста допускает потерю/добавление небольшого числа
   символов на границах токенов (logprobs соответствуют истинным
   токенам текста, на L это не влияет);
3. **Анализ** — детекторы, отчёт, график.

## Результат

- `data/watermark_report_<модель>.png` — 4 панели (LLR и G-значения);
- `data/report_<модель>.json` — все числа (поле `plot` — относительное имя);
- в stdout — отчёт с вердиктом по каждому детектору.

## Лицензия

MIT (см. `LICENSE`). Референс-репозитории — под их собственными
лицензиями (Apache-2.0 для synthid-text).
