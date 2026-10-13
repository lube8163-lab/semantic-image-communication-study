param([ValidateSet('Prepare','Caption','Encode','Generate','Refine','NativeFeature','Evaluate','Report','All')][string]$Phase='All')
$ErrorActionPreference='Stop'
$taskRoot=$PSScriptRoot
$primary=Join-Path $taskRoot '.venv\Scripts\python.exe'
$vlm=Join-Path $taskRoot '.venv-vlm\Scripts\python.exe'
$featureGpu=if ($env:SC_FEATURE_GPU) {$env:SC_FEATURE_GPU} else {'0'}
$generatorGpu=if ($env:SC_GENERATOR_GPU) {$env:SC_GENERATOR_GPU} else {'0'}
$savedGpu=$env:CUDA_VISIBLE_DEVICES
$savedEncoding=$env:PYTHONIOENCODING
function Invoke-Research([string]$Interpreter,[string]$Gpu,[string[]]$Arguments) {
    $env:CUDA_VISIBLE_DEVICES=$Gpu
    $env:PYTHONIOENCODING='utf-8'
    & $Interpreter @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Research phase failed with exit code $LASTEXITCODE" }
}
Push-Location $taskRoot
try {
    $phases=if ($Phase -eq 'All') {@('Prepare','Caption','Encode','Generate','Refine','NativeFeature','Evaluate','Report')} else {@($Phase)}
    foreach ($step in $phases) {
        switch ($step) {
            'Prepare' { Invoke-Research $primary $featureGpu @('scripts/prepare_text_route.py') }
            'Caption' { foreach ($name in @('qwen4b','minicpm','lfm16')) { Invoke-Research $vlm $featureGpu @('scripts/caption_text_route.py','--model',$name) } }
            'Encode' { Invoke-Research $primary $featureGpu @('scripts/encode_text_route.py'); Invoke-Research $primary $featureGpu @('scripts/codec_text_budgets.py') }
            'Generate' { Invoke-Research $primary $generatorGpu @('scripts/generate_text_route.py') }
            'Refine' { Invoke-Research $primary $generatorGpu @('scripts/generate_text_route.py','--refine') }
            'NativeFeature' { foreach ($part in @('sender','receiver')) { Invoke-Research $primary $featureGpu @('scripts/native_feature_check.py','--phase',$part) } }
            'Evaluate' { Invoke-Research $primary $featureGpu @('scripts/evaluate_text_route.py') }
            'Report' { Invoke-Research $primary $featureGpu @('scripts/build_text_route_report.py') }
        }
    }
} finally {
    $env:CUDA_VISIBLE_DEVICES=$savedGpu
    $env:PYTHONIOENCODING=$savedEncoding
    Pop-Location
}
