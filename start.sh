#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# ── Dependency check / first-time install ─────────────────────────────────────
MISSING=()
python3 -c "import flask"  2>/dev/null || MISSING+=(flask)
python3 -c "import PIL"    2>/dev/null || MISSING+=(Pillow)
python3 -c "import numpy"  2>/dev/null || MISSING+=(numpy)

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "Installing: ${MISSING[*]}"
  pip3 install --quiet --break-system-packages "${MISSING[@]}"
fi

# ── Launch server ─────────────────────────────────────────────────────────────
URL="http://localhost:5001"
echo "Image Stitcher → $URL"

# Open the browser (macOS / Linux / WSL)
if command -v open    &>/dev/null; then open    "$URL"
elif command -v xdg-open &>/dev/null; then xdg-open "$URL" &
fi

exec python3 concat_jpeg.py --web
