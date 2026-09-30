# watermark-llm-check

Утилита детекции watermark в выводе LLM. Сравнивает две модели, сервируемые
через llama.cpp, и отвечает на вопрос: есть ли watermark в выводе
watermarked-кандидата.

Четыре независимых детектора, по публичным референс-реализациям
(git-сабмодули проекта):

- **OpenStamp** (`openstamp/METHOD.md`) — length-normalized log-likelihood
  ratio между watermarked и базовой моделями, ключ не нужен:
  $LLR(x) = \frac{1}{T-1}\sum_{t=1}^{T-1}\log\frac{p_{\text{wm}}(x_t \mid x_{<t})}{p_{\text{base}}(x_t \mid x_{<t})}$.
- **SynthID-Text** (`synthid-text/`, Apache-2.0) — keyed-hash G-значения
  (ngram_len=5, 30 ключей, context_history_size=1024 из
  `DEFAULT_WATERMARKING_CONFIG`), training-free weighted-mean детектор.
  Null G = 0.5 (Bernoulli-биты); у watermarked-текста смещается вверх
  (~0.75).
- **GaussMark** (`gaussmark/`, ключевая версия) — структурный
  watermark (гауссов шум в весах). Статистика бумаги
  `score(T) = <grad_base(T), W>` (W — шум-ключ) без ключа
  считается по тождеству первого порядка
  `<grad_base(T), dtheta> ~ L_wm(T) − L_base(T)`, dtheta —
  наблюдаемая разность весов; p-value — нормальный тест
  (null — тексты базовой модели).
- **MarkLLM** (`markllm/`, E2E-LLM-Watermark) — нейронный
  детектор (LSTM по эмбеддингам opt-1.3b), ключ не нужен:
  `P(watermarked)` на текст. Checkpoint `models/e2e-35000.pth`
  (SHA-256 в `markllm/watermark/e2e/README.md`), эмбеддинги —
  `models/opt-1.3b-embeddings.pt`.

## Структура

```
detect.py      # main: сервер, генерация, скоринг, анализ, отчёт
test_llr.py    # эквивалентность по-токенного LLR и openstamp/src/llr.py
smoke.py       # ручной smoke-тест сервера (thinking + grammar-score)
status.py      # статус прогона по data_*.json
openstamp/     # сабмодуль: референс OpenStamp (METHOD.md, src/llr.py)
synthid-text/  # сабмодуль: референс SynthID-Text (модули)
gaussmark/     # сабмодуль: референс GaussMark (метод, p-value)
markllm/       # сабмодуль: E2E-LLM-Watermark (LSTM-детектор)
models/        # e2e-35000.pth (checkpoint E2E) и opt-1.3b-embeddings.pt
data/          # data_<модель>_<квант>.json, report_<модель>_<квант>.json и
               # watermark_report_<модель>_<квант>.png — в git;
               # wm_*/base_*.txt и server_*.log — нет
```

## Установка

```
git clone --recursive <url>
cd watermark-llm-check
pip install -r requirements.txt   # numpy, requests, torch, matplotlib, transformers
```

В `models/` нужны два файла (не в git):
- `models/e2e-35000.pth` — checkpoint E2E-детектора (SHA-256 в
  `markllm/watermark/e2e/README.md`);
- `models/opt-1.3b-embeddings.pt` — эмбеддинги `facebook/opt-1.3b`
  (token id -> вектор).

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
`--prompts` (по умолчанию 60), `--tokens` (по умолчанию 400).

## Прогон

1. **Генерация** — N текстов (по умолчанию 60) в двух режимах
   (reasoning on/off); в VRAM одна модель за раз, сервер
   перезапускается со второй моделью;
2. **Скоринг** — L = Σ log p(x_t|x_<t>) для полной 2×2-матрицы
   (тексты × модели) через GBNF-грамматики (pre-sampling logprobs);
   в L не входят первый текстовый токен и завершающий EOG
   (id модели, например 248046 у Qwen3.8; `--eos-token`);
   пустые тексты (зацикливание на спец-токенах) исключаются из анализа;
   SynthID-маскирование EOG — через референсный
   `compute_eos_token_mask` с реальным EOG модели;
3. **Анализ** — 4 детектора, отчёт, график. Вердикт — спаренный
   t-критерий по разностям (signal − null) на общих промптах:
   t ≥ 3 и большинство разностей положительные.

## Результат

- `data/watermark_report_<модель>_<квант>.png` — 4×2-график
  (LLR, mean G, GaussMark-score, E2E-P; columns: reasoning on/off);
  в каждой панели — отметка-вердикт (`WATERMARK FOUND` / `NO WATERMARK`);
- `data/report_<модель>_<квант>.json` — все числа (поле `plot` — относительное имя);
- в stdout — отчёт с вердиктом по каждому детектору.

| Прогон | wm (кандидат) | base | Квант | OpenStamp (LLR) | SynthID (G) | GaussMark | MarkLLM E2E (P) | Вердикт |
|---|---|---|---|---|---|---|---|---|
| [`report_Swift-1.5-Qwen3.8-27B-GSQ-RCO_IQ3_S.json`](data/report_Swift-1.5-Qwen3.8-27B-GSQ-RCO_IQ3_S.json) | Swift-1.5-Qwen3.8-27B-GSQ-RCO | Qwen3.8-27B-GSQ-RCO | IQ3_S | −0.23 / +2.80 (n = 38/60) | −0.97 / +1.64 | −0.23 / +2.80 (p<0.05: 0/0, 1/1) | +2.28 / −1.08 | **watermark не обнаружен** |
| [`report_qwen2.5-7b-openstamp-L251_Q8_0.json`](data/report_qwen2.5-7b-openstamp-L251_Q8_0.json) | qwen2.5-7b-openstamp-L251 | Qwen2.5-7B | Q8_0 | +12.47 / +9.90 (n = 60/60) | −0.62 / +0.51 | +12.47 / +9.90 (p<0.05: 46/60, 14/60) | −0.01 / +0.19 | **watermark обнаружен** (OpenStamp + GaussMark) |
| [`report_Swift-Qwen3.8-27B-RCO_IQ3_S.json`](data/report_Swift-Qwen3.8-27B-RCO_IQ3_S.json) | Swift-Qwen3.8-27B-RCO | Qwen3.8-27B-GSQ-RCO | IQ3_S | +2.53 / +4.43 (n = 35/60) | −0.81 / +0.28 | +2.53 / +4.43 (p<0.05: 0/0, 2/2) | −0.71 / −0.61 | **watermark обнаружен в noreason** (2/4: OpenStamp + GaussMark) |
| [`report_OrcaSAQ-2-27B-Uncensored.json`](data/report_OrcaSAQ-2-27B-Uncensored.json) | OrcaSAQ-2-27B-Uncensored | Qwen3.8-27B-GSQ-RCO | OrcaSAQ | +2.37 / +9.13 (n = 46/60) | −0.06 / −0.87 | +2.37 / +9.13 (p<0.05: 3/46, 26/60) | −0.67 / −0.89 | **watermark обнаружен в noreason** (2/4: OpenStamp + GaussMark) |

Значения в столбцах — t-сводка (спаренный t-критерий): reasoning on / reasoning off.

| Swift-1.5-Qwen3.8-27B-GSQ-RCO vs Qwen3.8-27B-GSQ-RCO (IQ3_S) | Swift-Qwen3.8-27B-RCO vs Qwen3.8-27B-GSQ-RCO (IQ3_S) | qwen2.5-7b-openstamp-L251 vs Qwen2.5-7B (Q8_0) |
|---|---|---|
| [![gsq](data/watermark_report_Swift-1.5-Qwen3.8-27B-GSQ-RCO_IQ3_S.png)](data/watermark_report_Swift-1.5-Qwen3.8-27B-GSQ-RCO_IQ3_S.png) | [![rco](data/watermark_report_Swift-Qwen3.8-27B-RCO_IQ3_S.png)](data/watermark_report_Swift-Qwen3.8-27B-RCO_IQ3_S.png) | [![7b](data/watermark_report_qwen2.5-7b-openstamp-L251_Q8_0.png)](data/watermark_report_qwen2.5-7b-openstamp-L251_Q8_0.png) |

| OrcaSAQ-2-27B-Uncensored vs Qwen3.8-27B-GSQ-RCO (OrcaSAQ) |
|---|
| [![orca](data/watermark_report_OrcaSAQ-2-27B-Uncensored.png)](data/watermark_report_OrcaSAQ-2-27B-Uncensored.png) |

- **27B-прогон GSQ** (60 промптов; в reason n = 38 из 60): watermark
  не обнаружен — максимум t = +2.80 (OpenStamp/GaussMark, noreason),
  ниже порога t ≥ 3; G ≈ 0.50 с обеих сторон.
- **27B-прогон RCO** (60 промптов × 2 режима; в reason исключено 25
  текстов с пустым выходом, n = 35): в reasoning off два детектора
  (OpenStamp + GaussMark — одна и та же статистика первого порядка)
  дают t = +4.43 (z = +5.04) — выше порога t ≥ 3, в reasoning on — нет
  (t = +2.53 < 3). SynthID и MarkLLM E2E не реагируют (G ≈ 0.50; E2E
  P ≈ 0.27–0.30 с обеих сторон).
- **7B-прогон** (60 промптов × 2 режима) — позитивный контроль: модель
  явно watermarked OpenStamp (delta=1.0, L=251). OpenStamp и
  GaussMark обнаруживают watermark в обоих режимах (t = +12.47 /
  +9.90, почти все разности положительные; GaussMark: 46/60 текстов
  с p < 0.05 в reason, 14/60 в noreason). SynthID (дефолтный ключ) и
  MarkLLM E2E — нет, что согласуется с тем, что watermark в этой
  модели — не SynthID, а E2E-детектор обучен на иных схемах (KGW/UNW
  на opt-1.3b).
- **OrcaSAQ-2-прогон** (60 промптов × 2 режима, 1200 токенов,
  llama.cpp CUDA/RTX 3090): OrcaSAQ-2-27B-Uncensored — квантизация
  OrcaSAQ того же Qwen3.8-27B-GSQ-RCO (см. imatrix-метаданные GGUF),
  т.е. «watermark» — структурное смещение весов от самой схемы
  квантизации. В reasoning off два детектора первого порядка
  (OpenStamp LLR и GaussMark — одна и та же статистика) дают
  t = +9.13 (z = +12.86), 57/60 разностей положительные, 26/60
  текстов с p < 0.05: структурный след OrcaSAQ-квантизации
  детектируется. В reasoning on сигнал слабее (t = +2.37 < 3, n = 46 —
  14/60 промптов исчерпали 1200 токенов на thinking, пустой вывод
  исключён). SynthID (дефолтный ключ) и MarkLLM E2E не реагируют
  (G ≈ 0.50 с обеих сторон) — как и в остальных прогонах, след не
  SynthID- и не KGW/UNW-типа.

Пересчёт L (30.09.2026, llama.cpp CUDA-сборка): прежние значения L
(ранние прогоны) включали хвостовой EOG и, у старой CPU-сборки
llama.cpp, токены, сгенерированные после завершения GBNF-грамматики
(до `max_tokens`). В пересчёте L содержит только текст-токены (EOG
исключён), как в референсе openstamp/src/llr.py (проверка —
`test_llr.py`); тексты и token_ids сохранены, доскорена только L.
Смещение t в позитивном контроле (+13.72 → +12.47) и в RCO
(+3.56 → +4.43) объясняется вычтенным хвостом; вердикты детекторов,
не использующих L (SynthID, E2E), не изменились.

## Лицензия

MIT (см. `LICENSE`). Референс-репозитории — под их собственными
лицензиями (Apache-2.0 для synthid-text).
