# watermark-llm-check

Утилита детекции watermark в выводе LLM. Сравнивает две модели,
сервируемые через llama.cpp, и отвечает на вопрос: есть ли в выводе
watermarked-кандидата цифровой водяной знак.

Детекция строится на двух независимых протоколах, по публичным
референс-реализациям (оба репозитория — git-сабмодули этого проекта):

1. **OpenStamp** (https://github.com/mb-14/openstamp, `METHOD.md`) —
   length-normalized log-likelihood ratio между watermarked-чекпоинтом
   и базовой моделью:

   ```
   LLR(x) = (1/(T-1)) * Σ_t log[ p_wm(x_t | x_<t) / p_base(x_t | x_<t) ]
   ```

   Ключ не нужен: требуется доступ к обеим моделям. Watermarked-текст
   накапливает положительный LLR; порог τ калибруется эмпирически на
   непомеченном (null) тексте — так предписывает METHOD.md.

2. **SynthID-Text** (https://github.com/google-deepmind/synthid-text) —
   G-значения: keyed-hash по (ngram_len−1)-контексту и кандидат-токену
   (`hashing_function.accumulate_hash`), бинарные G на каждой глубине
   (`logits_processing.compute_g_values`), training-free weighted-mean
   детектор (`detector_mean.mean_score`). Нулевое среднее G = 0.5
   (Bernoulli-биты); у watermarked-текста оно смещается вверх (~0.75,
   `g_value_expectations.expected_mean_g_value`). Используется
   `DEFAULT_WATERMARKING_CONFIG` из репозитория: ngram_len=5, 30
   ключей, context_history_size=1024 (детектирует watermark, наложенный
   именно с этим ключом).

## Структура проекта

```
.
├── detect.py         # main: сервер, генерация, скоринг, анализ, отчёт
├── score.cxx         # C++-скорер точных per-token logprob
├── build.sh          # сборка скорера против сборки llama.cpp
├── run.sh            # build (при отсутствии) + запуск detect.py
├── requirements.txt  # Python-зависимости
├── openstamp/        # сабмодуль: референс OpenStamp (METHOD.md)
└── synthid-text/     # сабмодуль: референс SynthID-Text (модули)
```

Результаты прогона складываются в `results/` (в git не попадает):

| Файл | Назначение |
|---|---|
| `results/data.json` | token ids + L-значения по всем текстам (резюмируется) |
| `results/report.json` | итоговые цифры |
| `results/watermark_report.png` | график (4 панели) |

## Требования

- Python 3.10+, зависимости из `requirements.txt` (numpy, requests,
  torch, matplotlib);
- сборка llama.cpp с `llama-server` и `libllama.so`/`libggml*.so`
  (CUDA-сборка — для GPU);
- два GGUF-файла: watermarked-кандидат и базовая модель;
- одна GPU с достаточным VRAM под одну модель (вторая в VRAM не
  помещается — утилита управляет этим сама).

## Установка

```sh
git clone --recursive <url-of-this-repo>
cd watermark-llm-check
python3 -m pip install -r requirements.txt
```

## Сборка

```sh
./build.sh /path/to/llama.cpp/build    # каталог сборки llama.cpp
```

Каталог должен содержать `bin/libllama.so`; заголовки берутся из
родительского каталога. Аргумент можно не передавать: тогда
используются `$LLAMA_BUILD` или `./llama.cpp/{build,build_cuda}`.
Результат — бинарник `./score`.

Важно: линковка через `-l:libllama.so` (точное имя файла), а не
`-llama` — в binutils 2.44 `-llama` ищется как `liblama.so` (одна `l`
отбрасывается, если имя начинается с двойной буквы).

### n_ctx скорера

`n_ctx` не фиксированный — вычисляется из размеров входных текстов:

```
n_ctx = max(2048, max_файлов(|prompt| + |text|) + 1024)
n_batch = min(4096, n_ctx)
```

Переопределение: `score model.gguf --ctx N file...`. Ограничение:
KV-кэш скорера целиком в системной RAM (≈0.4 КБ на токен контекста) —
131072 токена ≈ 51 ГБ не помещается в 61 ГБ RAM; для текстов
~400 токенов авто-значения 2048 достаточно.

## Запуск

```sh
./run.sh --models-dir /path/to/gguf \
         --wm-gguf   Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
         --base-gguf Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
         --prompts 24 --tokens 400
```

или напрямую:

```sh
python3 detect.py --help                 # все параметры
python3 detect.py --models-dir ...       # полный прогон
python3 detect.py --resume               # доскорить недостающие L
python3 detect.py --analyze-only         # только анализ + график
```

`detect.py` сама управляет `llama-server` (один за раз, VRAM ограничен):
запускает бинарь с нужным `-m`, ждёт готовности, останавливает и
перезапускает с другой моделью, в конце освобождает VRAM для скоринга.

### Фазы прогона

1. **WM** → генерация N текстов (token ids) через `llama-server`.
2. **BASE** → генерация N текстов.
3. **Скоринг** всех 2N текстов C++-скорером: полная 2×2-матрица
   L-значений (wm-тексты под WM и BASE; base-тексты под WM и BASE).
   Точные `log p(x_t|x_<t)` без speculative decoding: logprobs
   генерации сервера не используются — при ngram-mod-draft speculation
   ~11% позиций в них неточны (значения берутся из draft-кэша).
4. **Анализ**: цифры, график, вердикт.

## Отчёт

В stdout и `results/report.json`:

- **OpenStamp**: mean/median/мин/макс LLR wm-текстов (сигнал) и
  null-текстов, τ = mean_null + 3·std_null, доля wm > τ, сводный z,
  вердикт (≥75% сигналов выше τ и z ≥ 3).
- **SynthID-Text**: mean G wm (сигнал) и base (null), z по
  последовательностям, доля выше τ, вердикт.
- Итог: выявлен обоими / одним / ни одним детектором.

График `results/watermark_report.png`: гистограммы LLR и mean G
(сигнал vs null), per-prompt бар-чарты для обоих детекторов.

## Известные особенности

- **`llama_batch` в новых сборках llama.cpp**: `llama_batch_init`
  оставляет все члены неинициализированными; при ручном заполнении
  обязаны устанавливаться `n_seq_id`, `seq_id[i][0]` и `logits[i]` —
  иначе `llama_batch_compat::init` падает (SIGSEGV) при любом `n_ctx`.
  `score.cxx` это учитывает.
- **VRAM**: при работающем сервере GPU занята моделью; скоринг идёт
  только после остановки сервера (делает `detect.py` сама), либо
  `--resume`/`--analyze-only` без сервера.
- **Speculative logprobs сервера** неточны (draft-кэш, ~11% позиций)
  — для L всегда C++-скорер.
- **SynthID-Text с дефолтным ключом** детектирует только watermark,
   наложенный с `DEFAULT_WATERMARKING_CONFIG`; для другого ключа
   детекция невозможна без его знания (OpenStamp LLR — нет).

## Лицензия

MIT (см. `LICENSE`). Референс-репозитории в сабмодулях имеют
собственные лицензии (Apache-2.0 для synthid-text).
