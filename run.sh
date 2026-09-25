#!/usr/bin/env bash
# Сборка (при отсутствии бинарного скорера) и запуск полного прогона.
# Все аргументы передаются detect.py, например:
#   ./run.sh --models-dir /path/to/gguf
#   ./run.sh --resume
#   ./run.sh --analyze-only
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x ./score ]]; then
    ./build.sh
fi

exec python3 detect.py "$@"
