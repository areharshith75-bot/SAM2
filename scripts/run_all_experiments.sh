#!/usr/bin/env bash
# Runs all 8 FSAM experiments sequentially on GPU 0, then evaluates each and builds the report.
# Re-running is safe: finished runs (results.json present) are skipped and an interrupted run
# resumes from outputs/<run>/resume.pth.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p outputs/logs

$PY data/verify_datasets.py

run() {  # $1 dataset, $2 source
  local name="$1_$2"
  if [[ -f "outputs/$name/results.json" ]]; then echo "[skip] $name already evaluated"; return; fi
  if [[ ! -f "outputs/$name/last_model.pth" ]]; then
    echo "[train] $name  $(date)"
    $PY train.py --dataset "$1" --source "$2" >> "outputs/logs/$name.log" 2>&1
  fi
  echo "[eval]  $name  $(date)"
  $PY eval.py --run "outputs/$name" >> "outputs/logs/$name.log" 2>&1
}

# Table 1: RIGA+ (train on one source, test on MESSIDOR Base1/2/3)
for s in BinRushed Magrabia; do run riga "$s"; done
# Table 2: prostate, A..F each as the single source, tested on the other 5 sites
for s in A B C D E F; do run prostate "$s"; done

$PY eval.py --generate_report
echo "ALL DONE $(date)"
