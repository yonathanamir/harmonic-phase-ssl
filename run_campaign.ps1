# Train the campaign into .\runs (one run after the other; needs the data of .\generate_data.ps1).
#   .\run_campaign.ps1                                              all 11 embeddings, both arrays, D = 2, seed 0
#   .\run_campaign.ps1 -Arrays linear -Embeddings vmb,even-w        selected arrays and embeddings
#   .\run_campaign.ps1 -Filters 2,8 -Seeds 0,1,2                    more architectures (D = F) and seeds
#   .\run_campaign.ps1 -Quick                                       smoke test on the -Quick datasets
# Embeddings: raw vmb vonmises even odd dual reim vonmises-w even-w odd-w dual-w
# Run the same command again after an interruption: complete runs are skipped, others resume.
param([string[]]$Arrays = @('circular', 'linear'),
      [string[]]$Embeddings = @(),
      [string[]]$Filters = @('2'),
      [string[]]$Seeds = @('0'),
      [int]$Epochs = 150,
      [switch]$Quick)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUNBUFFERED = '1'
$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { Write-Host 'Run .\setup_env.ps1 first.' -ForegroundColor Red; exit 1 }
# "powershell -File" passes a list such as linear,circular as one string: split it here.
function Split-List([string[]]$Values) { @($Values | ForEach-Object { $_ -split ',' } | Where-Object { $_ }) }
$ArgList = @('-u', 'run_campaign.py', '--arrays') + (Split-List $Arrays) +
           @('--filters') + (Split-List $Filters) +
           @('--seeds') + (Split-List $Seeds) +
           @('--epochs', "$Epochs")
if ($Embeddings.Count -gt 0) { $ArgList += @('--embeddings') + (Split-List $Embeddings) }
if ($Quick) { $ArgList += '--quick' }
& $Python @ArgList
exit $LASTEXITCODE
