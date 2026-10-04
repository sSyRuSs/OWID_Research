$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path ".tmp" | Out-Null
$env:JUPYTER_CONFIG_DIR = Join-Path (Resolve-Path ".\.tmp").Path "jupyter-config"
$env:JUPYTER_DATA_DIR = Join-Path (Resolve-Path ".\.tmp").Path "jupyter-data"
$env:JUPYTER_RUNTIME_DIR = Join-Path (Resolve-Path ".\.tmp").Path "jupyter-runtime"
$env:IPYTHONDIR = Join-Path (Resolve-Path ".\.tmp").Path "ipython"
$env:PYTHONDONTWRITEBYTECODE = "1"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe .\scripts\download_owid_data.py
& .\.venv\Scripts\python.exe -m jupyter nbconvert `
    --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=1800 `
    .\notebooks\Explainable_Carbon_Intensity_Thesis.ipynb

$xelatex = "C:\Users\Admin\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
if (Test-Path $xelatex) {
    Push-Location .\paper\thesis_explainable_carbon_intensity
    try {
        & $xelatex -interaction=nonstopmode -halt-on-error thesis_explainable_carbon_intensity.tex
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $xelatex -interaction=nonstopmode -halt-on-error thesis_explainable_carbon_intensity.tex
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    finally { Pop-Location }
}
else {
    Write-Warning "xelatex not found; results are complete but PDF was not compiled."
}
