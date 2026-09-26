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
data/         # data_<модель>_<квант>.json, логи (не в git);
              # report_<модель>_<квант>.json и watermark_report_<модель>_<квант>.png — в git
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
`--models-dir`), `--wm-model/--base-model` (имена в отчёте), `--quant`
(квантизация в именах отчёта; по умолчанию — из имени wm-GGUF), `--llama-bin`
(бинарь llama-server), `--device` (например `cuda0`, `ROCm1`, `cpu`),
`--prompts` (по умолчанию 30), `--tokens` (по умолчанию 400).

## Прогон

1. **Генерация** — N текстов (по умолчанию 30) в двух режимах
   (reasoning on/off); в VRAM одна модель за раз, сервер
   перезапускается со второй моделью;
2. **Скоринг** — L = Σ log p(x_t|x_<t>) для полной 2×2-матрицы
   (тексты × модели) через GBNF-грамматики (pre-sampling logprobs);
   пустые тексты (зацикливание на спец-токенах) исключаются из анализа;
3. **Анализ** — детекторы, отчёт, график. Вердикт — спаренный
   t-критерий по разностям (signal − null) на общих промптах:
   t ≥ 3 и большинство разностей положительные.

## Результат

- `data/watermark_report_<модель>_<квант>.png` — 4 панели (LLR и G-значения);
- `data/report_<модель>_<квант>.json` — все числа (поле `plot` — относительное имя);
- в stdout — отчёт с вердиктом по каждому детектору.

## Результаты прогонов

| Прогон | wm (кандидат) | base | Квант | OpenStamp (LLR) | SynthID | Вердикт |
|---|---|---|---|---|---|---|
| [`report_Swift-1.5-Qwen3.8-27B-GSQ-RCO_IQ3_S.json`](data/report_Swift-1.5-Qwen3.8-27B-GSQ-RCO_IQ3_S.json) | Swift-1.5-Qwen3.8-27B-GSQ-RCO | Qwen3.8-27B-GSQ-RCO | IQ3_S | t = −0.34 / +1.66 (reason / noreason) | t = −1.06 / +1.07 | **watermark не обнаружен** |
| [`report_qwen2.5-7b-openstamp-L251_Q8_0.json`](data/report_qwen2.5-7b-openstamp-L251_Q8_0.json) | qwen2.5-7b-openstamp-L251 | Qwen2.5-7B | Q8_0 | t = +6.46 / +3.37, 10/10 положительных разностей | t = −0.59 / −0.23 | **watermark обнаружен** (OpenStamp) |

- **27B-прогон** (30 промптов × 2 режима, 400 токенов): ни один детектор
  вердикт не выдал — LLR и G-значения Swift-1.5 и Qwen3.8-27B
  неотличимы (G ≈ 0.50 с обеих сторон, смещения выше бернулли-нуля нет).
- **7B-прогон** (10 промптов × 2 режима) — позитивный контроль: модель
  явно watermarked OpenStamp (delta=1.0, L=251). LLR-детектор её
  обнаружил в обоих режимах; SynthID (дефолтный ключ) — нет
  (t ≈ 0), что согласуется с тем, что watermark в этой модели — не SynthID.

## Лицензия

MIT (см. `LICENSE`). Референс-репозитории — под их собственными
лицензиями (Apache-2.0 для synthid-text).
