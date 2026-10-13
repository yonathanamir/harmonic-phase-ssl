# Create the Python environment .venv in this folder and install the requirements.
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1            (CUDA 12.8 build of torch)
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Cuda cpu  (CPU only, for a smoke test)
# Needs Python 3.10-3.12 on PATH ("py" launcher or "python").
param([string]$Cuda = "cu128")
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
try {
    $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv .venv }
        else { & python -m venv .venv }
        if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv. Install Python 3.10-3.12 first.' }
    }
    & $Python -m pip install --upgrade pip
    & $Python -m pip install torch --index-url "https://download.pytorch.org/whl/$Cuda"
    if ($LASTEXITCODE -ne 0) { throw "Could not install torch ($Cuda)." }
    & $Python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Could not install requirements.txt.' }
    & $Python -c "import torch, numpy; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
    Write-Host 'Environment ready: .venv' -ForegroundColor Green
} catch {
    Write-Host "SETUP FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
