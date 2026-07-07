# Whisper Dictation installer (Windows)
# Creates a virtual environment, installs dependencies (with GPU libraries when
# an NVIDIA GPU is present), and optionally sets the app to start on login.
#
# Usage:
#   .\install-windows.ps1              # install
#   .\install-windows.ps1 -Autostart   # install + run on login
param(
    [switch]$Autostart
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

Write-Host "=== Whisper Dictation installer (Windows) ===" -ForegroundColor Cyan

# --- Find a Python launcher ---
$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" }
      elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" }
      else { Write-Error "Python 3.10+ not found. Install it from python.org first."; exit 1 }

# --- Create venv ---
$venv = Join-Path $root ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "Creating virtual environment..."
    & $py -m venv $venv
} else {
    Write-Host "Virtual environment already exists."
}
$python = Join-Path $venv "Scripts\python.exe"

# --- Detect NVIDIA GPU ---
$hasGpu = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try { nvidia-smi | Out-Null; $hasGpu = $true } catch { $hasGpu = $false }
}

# --- Install dependencies ---
Write-Host "Upgrading pip..."
& $python -m pip install --upgrade pip | Out-Null

Write-Host "Installing core dependencies..."
& $python -m pip install faster-whisper sounddevice numpy pynput pyperclip

if ($hasGpu) {
    Write-Host "NVIDIA GPU detected - installing CUDA libraries (cuBLAS, cuDNN, cudart)..." -ForegroundColor Green
    & $python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12
} else {
    Write-Host "No NVIDIA GPU detected - CPU mode (int8). Recommend WHISPER_MODEL=small or base." -ForegroundColor Yellow
}

# --- Optional autostart ---
if ($Autostart) {
    $startup = [Environment]::GetFolderPath("Startup")
    $lnkPath = Join-Path $startup "Whisper Dictation.lnk"
    $target = "powershell.exe"
    $arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$(Join-Path $root 'run-windows.ps1')`""
    $shell = New-Object -ComObject WScript.Shell
    $lnk = $shell.CreateShortcut($lnkPath)
    $lnk.TargetPath = $target
    $lnk.Arguments = $arguments
    $lnk.WorkingDirectory = $root
    $lnk.WindowStyle = 7  # minimized
    $lnk.Description = "Whisper Dictation - hold Ctrl+Alt to dictate"
    $lnk.Save()
    Write-Host "Autostart enabled: $lnkPath" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Installation complete ===" -ForegroundColor Cyan
Write-Host "  Start now:  .\run-windows.ps1"
Write-Host "  Hotkey:     hold Ctrl+Alt to record, release to transcribe & type"
Write-Host "  Dictionary: edit dictionary.json to add your own words"
if (-not $Autostart) {
    Write-Host "  Autostart:  re-run with  .\install-windows.ps1 -Autostart"
}
