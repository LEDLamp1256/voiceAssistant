# scripts/setup_models.ps1
$modelPath = "models/silero_vad.onnx"
$url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $projectRoot "models"
$wakeModel = Join-Path $modelsDir "hey_jarvis_v0.1.onnx"

if (!(Test-Path "models")) {
    New-Item -ItemType Directory -Force -Path "models" | Out-Null
}

if (!(Test-Path $modelPath)) {
    Write-Host "Downloading Silero VAD model..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile $modelPath
    Write-Host "Model downloaded successfully." -ForegroundColor Green
} else {
    Write-Host "Model already exists at $modelPath" -ForegroundColor Yellow
}

if ((Test-Path $wakeModel) -and ((Get-Item $wakeModel).Length -gt 0)) {
    Write-Host "[SKIP] hey_jarvis_v0.1.onnx already present and non-empty."
} else {
    Write-Host "[FETCH] Downloading openWakeWord default models..."
    python scripts/fetch_wakeword_models.py
}

$sileroModel = Join-Path $modelsDir "silero_vad.onnx"
if (-not (Test-Path $sileroModel) -or ((Get-Item $sileroModel).Length -eq 0)) {
    Write-Host "[FETCH] Downloading Silero VAD..."
    Invoke-WebRequest -Uri "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx" -OutFile $sileroModel
}

if (!(Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

# $modelUrl = "https://huggingface.co/davidmatthews1/openwakeword/resolve/main/hey_jarvis.onnx?download=true"
# $outputFile = Join-Path $modelsDir "hey_jarvis.onnx"

# if (!(Test-Path $outputFile)) {
#     Write-Host "Downloading hey_jarvis model..." -ForegroundColor Cyan
#     Invoke-WebRequest -Uri $modelUrl -OutFile $outputFile
#     Write-Host "Download complete: $outputFile" -ForegroundColor Green
# } else {
#     Write-Host "Model already exists at $outputFile. Skipping download." -ForegroundColor Yellow
# }