# Vox control script — manage the windowless Vox dictation app.
#
#   .\vox.ps1 start     Launch Vox in the background (windowless, via pythonw)
#   .\vox.ps1 stop      Stop Vox (frees the model's VRAM for other GPU work)
#   .\vox.ps1 restart   Reload after editing dictionary.json or changing settings
#   .\vox.ps1 status    Show whether Vox is running
#
# If PowerShell blocks the script:
#   powershell -ExecutionPolicy Bypass -File .\vox.ps1 restart
param(
    [ValidateSet('start', 'stop', 'restart', 'status')]
    [string]$Action = 'status'
)
$root = $PSScriptRoot
$pyw = Join-Path $root ".venv\Scripts\pythonw.exe"
$script = Join-Path $root "dictation.py"

function Get-Vox {
    Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*dictation.py*' }
}
function Stop-Vox {
    Get-Vox | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function Start-Vox {
    if (-not (Test-Path $pyw)) { Write-Error "venv not found. Run .\install-windows.ps1 first."; return }
    Start-Process -FilePath $pyw -ArgumentList "`"$script`"" -WorkingDirectory $root | Out-Null
}

switch ($Action) {
    'start'   { if (Get-Vox) { Write-Host "Vox is already running." } else { Start-Vox; Write-Host "Vox started." } }
    'stop'    { Stop-Vox; Write-Host "Vox stopped (VRAM freed)." }
    'restart' { Stop-Vox; Start-Sleep -Milliseconds 700; Start-Vox; Write-Host "Vox restarted." }
    'status'  { if (Get-Vox) { Write-Host "Vox is running." } else { Write-Host "Vox is not running." } }
}
