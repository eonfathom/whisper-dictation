# Whisper Dictation launcher (Windows)
# Runs the app from the local virtual environment. Pass-through args are
# forwarded to dictation.py, e.g.:  .\run-windows.ps1
# Configure via environment variables before launching, e.g.:
#   $env:WHISPER_MODEL = "large-v3"   # top accuracy (needs a GPU)
#   $env:WHISPER_MODEL = "small"      # good on a CPU-only laptop
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found. Run .\install-windows.ps1 first."
    exit 1
}
& $python (Join-Path $root "dictation.py") @args
