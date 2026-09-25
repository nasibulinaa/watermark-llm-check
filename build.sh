#!/usr/bin/env bash
# Сборка C++-скорера (score.cxx) против указанной сборки llama.cpp.
#
# Использование: ./build.sh [path/to/llama.cpp/build]
#   аргумент опционален; порядок поиска:
#     1) позиционный аргумент,
#     2) переменная окружения $LLAMA_BUILD,
#     3) ./llama.cpp/build, ./llama.cpp/build_cuda (относительно корня репо).
#
# Каталог сборки должен содержать bin/libllama.so и заголовки в
# родительском каталоге (<llama.cpp>/include, <llama.cpp>/ggml/include).
#
# ВАЖНО: линкуем через `-l:lib<name>.so` (точное имя файла), а не `-l<name>`:
# в binutils 2.44 `-llama` ищется как liblama.so (одна 'l' отбрасывается,
# если имя начинается с двойной буквы).
set -euo pipefail

cd "$(dirname "$0")"

BUILD="${1:-${LLAMA_BUILD:-}}"
if [[ -z "$BUILD" ]]; then
    for cand in ./llama.cpp/build ./llama.cpp/build_cuda; do
        if [[ -d "$cand/bin" ]]; then BUILD="$cand"; break; fi
    done
fi
if [[ -z "$BUILD" || ! -d "$BUILD/bin" ]]; then
    echo "ошибка: не указан каталог сборки llama.cpp" >&2
    echo "использование: ./build.sh /path/to/llama.cpp/build" >&2
    exit 1
fi

BIN_DIR="$BUILD/bin"
SRC="$(cd "$BUILD/.." && pwd)"

if [[ ! -f "$BIN_DIR/libllama.so" ]]; then
    echo "ошибка: $BIN_DIR/libllama.so не найден" >&2
    exit 1
fi

CXX="${CXX:-c++}"
LIBS=(-l:libllama.so -l:libggml.so -l:libggml-base.so -l:libggml-cpu.so)
if [[ -f "$BIN_DIR/libggml-cuda.so" ]]; then
    LIBS+=(-l:libggml-cuda.so)
fi

echo "сборка: $BIN_DIR"
"$CXX" -O2 -std=c++17 -o score score.cxx \
    -I"$SRC/include" -I"$SRC/ggml/include" \
    -L"$BIN_DIR" -Wl,-rpath,"$BIN_DIR" \
    "${LIBS[@]}" -ldl -lpthread
echo "готово: $PWD/score"
