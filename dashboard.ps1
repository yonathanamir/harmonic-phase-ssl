# Start the live training dashboard and open it in the browser (Ctrl+C stops it).
#   .\dashboard.ps1               watches every campaign under .\runs and .\runs-quick
#   .\dashboard.ps1 -Port 8771
param([int]$Port = 8770)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { Write-Host 'Run .\setup_env.ps1 first.' -ForegroundColor Red; exit 1 }
Start-Process "http://127.0.0.1:$Port/"
& $Python dashboard.py --port $Port
exit $LASTEXITCODE
