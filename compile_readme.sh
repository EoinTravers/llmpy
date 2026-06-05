#!/usr/bin/env bash
set -euo pipefail

DOCS_DIR="docs"
OUT="README.md"
TMP_DIR=$(mktemp -d)

# Process each doc file in numbered order
for f in $(ls "$DOCS_DIR" | sort); do
    path="$DOCS_DIR/$f"
    case "$f" in
        *.ipynb)
            out="$TMP_DIR/${f%.ipynb}.md"
            uv run jupyter nbconvert --to markdown --execute --output "$out" "$path"
            cat "$out" >> "$TMP_DIR/combined.md"
            ;;
        *.md)
            cat "$path" >> "$TMP_DIR/combined.md"
            ;;
    esac
    echo "" >> "$TMP_DIR/combined.md"
done

mv "$TMP_DIR/combined.md" "$OUT"
echo "Written to $OUT"
