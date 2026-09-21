#!/usr/bin/env bash
set -euo pipefail

root="${1:-/9950backfile/chenjiahui/evo_artifacts/datasets/long_rvos}"
downloads="$root/downloads"
valid="$root/valid"
base="https://huggingface.co/datasets/iSEE-Laboratory/Long-RVOS/resolve/main/valid"

mkdir -p "$downloads" "$valid/JPEGImages" "$valid/Annotations"
if [[ ! -s "$downloads/meta_expressions.json" ]]; then
  wget -c -O "$downloads/meta_expressions.json" "$base/meta_expressions.json"
fi
wget -c -O "$downloads/JPEGImages.tar.gz" "$base/JPEGImages.tar.gz"
wget -c -O "$downloads/Annotations.tar.gz" "$base/Annotations.tar.gz"
cp "$downloads/meta_expressions.json" "$valid/meta_expressions.json"
tar -xzf "$downloads/JPEGImages.tar.gz" -C "$valid/JPEGImages"
tar -xzf "$downloads/Annotations.tar.gz" -C "$valid/Annotations"

# The current official archives contain their own JPEGImages/Annotations
# top-level directory, while the official helper extracts into a directory of
# the same name.  Normalize that observed double nesting without deleting data.
normalize_nested() {
  local outer="$1"
  local leaf="${outer##*/}"
  local nested="$outer/$leaf"
  local staging="${outer}.normalize.$$"
  if [[ -d "$nested" ]] && [[ "$(find "$outer" -mindepth 1 -maxdepth 1 | wc -l)" -eq 1 ]]; then
    mv "$nested" "$staging"
    rmdir "$outer"
    mv "$staging" "$outer"
  fi
}
normalize_nested "$valid/JPEGImages"
normalize_nested "$valid/Annotations"
date -Iseconds > "$valid/.prepared_at"
