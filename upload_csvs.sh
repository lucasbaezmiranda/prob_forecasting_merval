#!/usr/bin/env bash
set -euo pipefail

# === Config ===
BUCKET="lukebm-plot-bucket"     # <- tu bucket
PREFIX="data/csv_public_$(date +%Y%m%d_%H%M%S)"
LOCAL_DIR="./market_data"        # carpeta local con .csv
REGION="us-east-1"               # región del bucket

echo "Subiendo CSVs a s3://$BUCKET/$PREFIX/ ..."
aws s3 cp "$LOCAL_DIR" "s3://$BUCKET/$PREFIX/" \
  --recursive --exclude "*" --include "*.csv" \
  --acl public-read --content-type text/csv

echo "Generando URLs públicas..."
aws s3 ls "s3://$BUCKET/$PREFIX/" \
| awk -v b="$BUCKET" -v r="$REGION" -v p="$PREFIX" '{print "https://" b ".s3." r ".amazonaws.com/" p "/" $4}' \
> urls.txt

echo "Listado de archivos en S3:"
aws s3 ls "s3://$BUCKET/$PREFIX/"

echo
echo "Listo ✅  Links guardados en urls.txt"
