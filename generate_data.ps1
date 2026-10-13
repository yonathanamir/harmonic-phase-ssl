# Generate the datasets (about 22 GB per array) into .\data.
#   .\generate_data.ps1                        both arrays
#   .\generate_data.ps1 -Arrays linear         one array (circular or linear)
#   .\generate_data.ps1 -Quick                 tiny datasets for a smoke test
# Run the same command again after an interruption: complete files are kept.
param([string[]]$Arrays = @('circular', 'linear'), [switch]$Quick)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUNBUFFERED = '1'
$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { Write-Host 'Run .\setup_env.ps1 first.' -ForegroundColor Red; exit 1 }
# "powershell -File" passes a list such as circular,linear as one string: split it here.
$ArgList = @('generate_data.py', '--arrays') + @($Arrays | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
if ($Quick) { $ArgList += '--quick' }
& $Python @ArgList
exit $LASTEXITCODE
