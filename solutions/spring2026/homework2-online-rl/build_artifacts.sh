#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$ROOT/analysis/plot_results.py" \
  --runs "$ROOT/artifacts/raw/runs" \
  --grid "$ROOT/artifacts/raw/runs/gridworld/metrics.json" \
  --output "$ROOT/artifacts/figures" \
  --macros "$ROOT/reports/results_macros.tex" \
  --submission "$ROOT/artifacts/submission"

for report in assignment_report interview_guide; do
  (
    cd "$ROOT/reports/$report"
    latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
  )
done
