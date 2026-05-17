# One-shot setup on Windows (PowerShell). From repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# Optional env vars:
#   $env:PYTHON = "C:\Path\python.exe"
#   $env:SKIP_DATA_DOWNLOAD = "1"
#   $env:INSTALL_JUPYTER_KERNEL = "1"

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$Py = "python"
if ($env:PYTHON) { $Py = $env:PYTHON }
Write-Host "==> Using: $Py"

if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating .venv"
    & $Py -m venv .venv
} else {
    Write-Host "==> .venv already exists"
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$VenvPip    = Join-Path $Root ".venv\Scripts\pip.exe"

Write-Host "==> Upgrading pip"
& $VenvPython -m pip install -U pip setuptools wheel

Write-Host "==> Installing requirements"
& $VenvPip install -r requirements.txt

$skipDownload = ($env:SKIP_DATA_DOWNLOAD -eq "1")
if ($skipDownload) {
    Write-Host "==> SKIP_DATA_DOWNLOAD=1 — skipping dataset download"
} else {
    Write-Host "==> Dataset (data/raw/games.csv)"
    $DownloadScript = Join-Path $Root "scripts\download_kaggle_dataset.py"
    & $VenvPython $DownloadScript
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    Note: download script exited $LASTEXITCODE — add games.csv manually if needed."
    }
}

$installKernel = ($env:INSTALL_JUPYTER_KERNEL -eq "1")
if ($installKernel) {
    Write-Host "==> Jupyter kernel"
    & $VenvPython -m ipykernel install --user --name chess-anomaly --display-name "Chess anomaly (.venv)"
}

Write-Host ""
Write-Host "Done. Activate: .\.venv\Scripts\Activate.ps1"
Write-Host "Then: python -m pytest"
Write-Host "      python -m jupyter lab notebooks/01_eda.ipynb"
