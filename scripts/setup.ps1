param([string]$Python = 'python', [switch]$Vlm)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskArgs = @((Join-Path $taskRoot 'scripts/bootstrap.py'))
if ($Vlm) { $taskArgs += '--vlm' }
& $Python @taskArgs
if ($LASTEXITCODE -ne 0) { throw 'Isolated setup failed' }
