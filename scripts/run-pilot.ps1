param(
    [ValidateSet('Encode','Decode','Evaluate')][string]$Phase,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$Run = 'pilot-20261013'
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskRun = Join-Path $taskRoot ('results\' + $Run)
$taskLogs = Join-Path $taskRun 'logs'
[IO.Directory]::CreateDirectory($taskLogs) | Out-Null
$taskScript = @{Encode='encode_pilot.py'; Decode='decode_pilot.py'; Evaluate='evaluate_pilot.py'}[$Phase]
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = if ($env:SC_FEATURE_GPU) {$env:SC_FEATURE_GPU} else {'0'}
$taskLog = Join-Path $taskLogs ($Phase.ToLower() + '.log')
& $taskPython -B -u (Join-Path $PSScriptRoot $taskScript) --run $Run 2>&1 | Tee-Object -FilePath $taskLog
$taskExit = $LASTEXITCODE
if ($taskExit -ne 0) { throw ($Phase + ' failed with exit code ' + $taskExit + '; see ' + $taskLog) }
