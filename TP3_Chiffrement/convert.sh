#!/bin/bash
set -euo pipefail

# require ImageMagick's convert
if ! command -v convert >/dev/null 2>&1; then
    echo "Error: 'convert' not found. Install ImageMagick." >&2
    exit 1
fi

shopt -s nullglob
for file in *.pgm; do
    [ -f "$file" ] || continue
    out="${file%.pgm}.png"
    convert "$file" "$out"
done
