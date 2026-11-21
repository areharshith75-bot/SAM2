#!/usr/bin/env bash
# Extension experiments (on top of the FSAM reproduction). Resumable: finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p outputs/logs

run() {  # $1 dataset, $2 source, $3 tag, rest: extra train args
  local ds=$1 src=$2 tag=$3; shift 3
  local name="${ds}_${src}_${tag}"
  if [[ -f "outputs/$name/results.json" ]]; then echo "[skip] $name"; return; fi
  echo "[train] $name $(date)"
  $PY train.py --dataset "$ds" --source "$src" --out "outputs/$name" "$@" >> "outputs/logs/$name.log" 2>&1
  $PY eval.py --run "outputs/$name" >> "outputs/logs/$name.log" 2>&1
}

# Local (patch-wise) frequency input
for s in BinRushed Magrabia; do run riga "$s" localfreq --freq_mode local; done
# GIN random-style augmentation (paper's global frequency input)
for s in BinRushed Magrabia; do run riga "$s" gin --style_aug gin; done
for s in E C; do run prostate "$s" gin --style_aug gin; done
echo "EXTENSIONS DONE $(date)"
