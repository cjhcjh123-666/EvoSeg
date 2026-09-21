#!/usr/bin/env bash
set -euo pipefail

root="${1:-/9950backfile/chenjiahui/evo_artifacts/datasets/long_rvos}"
downloads="$root/downloads"
valid="$root/valid"
base="https://huggingface.co/datasets/iSEE-Laboratory/Long-RVOS/resolve/main/valid"

mkdir -p "$downloads" "$valid/JPEGImages" "$valid/Annotations"
wget -c -O "$downloads/meta_expressions.json" "$base/meta_expressions.json"
wget -c -O "$downloads/JPEGImages.tar.gz" "$base/JPEGImages.tar.gz"
wget -c -O "$downloads/Annotations.tar.gz" "$base/Annotations.tar.gz"
cp "$downloads/meta_expressions.json" "$valid/meta_expressions.json"
tar -xzf "$downloads/JPEGImages.tar.gz" -C "$valid/JPEGImages"
tar -xzf "$downloads/Annotations.tar.gz" -C "$valid/Annotations"
date -Iseconds > "$valid/.prepared_at"

