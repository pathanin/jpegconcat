#!/usr/bin/env bash
# Installs missing dependencies, then starts the web UI (which opens the
# browser itself). Extra arguments pass through, e.g. ./start.sh --port 5002
set -e

cd "$(dirname "$0")"

MISSING=()
python3 -c "import flask"  2>/dev/null || MISSING+=(flask)
python3 -c "import PIL"    2>/dev/null || MISSING+=(Pillow)
python3 -c "import numpy"  2>/dev/null || MISSING+=(numpy)

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "Installing: ${MISSING[*]}"
  pip3 install --quiet --break-system-packages "${MISSING[@]}"
fi

exec python3 concat_jpeg.py "$@"
