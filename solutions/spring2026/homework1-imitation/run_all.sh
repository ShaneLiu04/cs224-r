#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$ROOT/code"
RAW="$ROOT/artifacts/raw"
FIGURES="$ROOT/artifacts/figures"
mkdir -p "$RAW" "$FIGURES"

export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q "$CODE/tests"

pids=()
for seed in 11 22 33 44 55; do
  python -u "$CODE/run_experiments.py" \
    --seed "$seed" \
    --output "$RAW/seed_$seed" \
    > "$RAW/seed_$seed.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one experiment failed; inspect artifacts/raw/*.log" >&2
  exit "$status"
fi

python "$ROOT/analysis/plot_results.py" \
  --input "$RAW" \
  --output "$FIGURES" \
  --macros "$ROOT/reports/results_macros.tex" \
  --submission-output "$ROOT/artifacts/submission"

for report in assignment_report interview_guide; do
  (
    cd "$ROOT/reports/$report"
    latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
  )
done
