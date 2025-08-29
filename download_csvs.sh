#!/usr/bin/env bash
set -euo pipefail

URLS_FILE="${1:-urls.txt}"   # por defecto usa urls.txt, pero podés pasar otro
DEST_DIR="./market_data"     # carpeta de destino

# crear carpeta si no existe
mkdir -p "$DEST_DIR"

[ -f "$URLS_FILE" ] || { echo "❌ No existe $URLS_FILE"; exit 1; }

echo "Descargando a $DEST_DIR ..."
grep -v '^\s*$' "$URLS_FILE" | while read -r url; do
  [ -z "$url" ] && continue
  echo "→ $url"
  wget -q --show-progress -P "$DEST_DIR" "$url"
done

echo "✅ Archivos descargados en $DEST_DIR"
