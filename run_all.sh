#!/usr/bin/env bash
set -euo pipefail

test -d .venv || python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_owid_data.py
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 notebooks/Explainable_Carbon_Intensity_Thesis.ipynb
if command -v xelatex >/dev/null 2>&1; then
  (cd paper/thesis_explainable_carbon_intensity && xelatex -interaction=nonstopmode -halt-on-error thesis_explainable_carbon_intensity.tex && \
    xelatex -interaction=nonstopmode -halt-on-error thesis_explainable_carbon_intensity.tex)
fi
