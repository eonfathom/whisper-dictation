# Vox control script - manage the windowless Vox dictation app.
#
#   .\vox.ps1 start     Launch Vox in the background (windowless, via pythonw)
#   .\vox.ps1 stop      Stop Vox AND the guardian (frees the model's VRAM)
#   .\vox.ps1 restart   Reload after editing dictionary.json or changing settings
#   .\vox.ps1 status    Show whether Vox (and the guardian) are running
#   .\vox.ps1 guard     Run the guardian loop: starts Vox, then watches it and
#                       restarts it if it ever dies. This is what the Startup
#                       shortcut runs (hidden), so dictation self-heals across
#                       crashes without ever needing a manual relaunch.
#
# If PowerShell blocks the script:
#   powershell -ExecutionPolicy Bypass -File .\vox.ps1 restart
param(
    [ValidateSet('start', 'stop', 'restart', 'status', 'guard')]
    [string]$Action = 'status'
)
$root = $PSScriptRoot
$pyw = Join-Path $root ".venv\Scripts\pythonw.exe"
$script = Join-Path $root "dictation.py"
$stateDir = Join-Path $env:LOCALAPPDATA 'vox'
$pauseFile = Join-Path $stateDir 'guardian.pause'
$guardLog = Join-Path $stateDir 'guardian.log'

function Get-Vox {
    Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*dictation.py*' }
}
function Get-Guardian {
    # The guardian is a powershell running THIS script with the 'guard' action.
    # Match a command line that ENDS with the guard action - a shell that merely
    # mentions vox.ps1/guard somewhere (e.g. a launcher or an admin command)
    # must not count, or the guardian falsely sees "another instance" and exits.
    Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='pwsh.exe'" |
        Where-Object { $_.CommandLine -match 'vox\.ps1"?\s+guard\s*$' -and $_.ProcessId -ne $PID }
}
function GuardLog([string]$msg) {
    if (-not (Test-Path $stateDir)) { New-Item -ItemType Directory -Force $stateDir | Out-Null }
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg" | Add-Content $guardLog
}
function Pause-Guardian {
    # Tell a running guardian to stand down briefly so a deliberate stop/start
    # sequence (restart) can't race it into launching a SECOND Vox instance.
    if (-not (Test-Path $stateDir)) { New-Item -ItemType Directory -Force $stateDir | Out-Null }
    Get-Date -Format 'o' | Set-Content $pauseFile
}
function Stop-Vox {
    Get-Vox | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function Start-Vox {
    if (-not (Test-Path $pyw)) { Write-Error "venv not found. Run .\install-windows.ps1 first."; return }
    Start-Process -FilePath $pyw -ArgumentList "`"$script`"" -WorkingDirectory $root | Out-Null
}

switch ($Action) {
    'start'   {
        Remove-Item $pauseFile -Force -ErrorAction SilentlyContinue
        if (Get-Vox) { Write-Host "Vox is already running." } else { Start-Vox; Write-Host "Vox started." }
    }
    'stop'    {
        # Guardian first, or it would immediately revive what we just stopped.
        $g = Get-Guardian
        $g | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Stop-Vox
        Write-Host ("Vox stopped (VRAM freed)." + $(if ($g) { " Guardian stopped too - use 'start' or 'guard' to resume." } else { "" }))
    }
    'restart' {
        Pause-Guardian
        Stop-Vox; Start-Sleep -Milliseconds 700; Start-Vox
        Write-Host "Vox restarted."
    }
    'status'  {
        $v = @(Get-Vox); $g = @(Get-Guardian)
        Write-Host ("Vox: " + $(if ($v) { "running" } else { "not running" }) +
                    " | Guardian: " + $(if ($g) { "running" } else { "not running" }))
    }
    'guard'   {
        # Self-healing loop. Single-instance: if another guardian is already
        # running, exit quietly (e.g. double login-launch).
        if (Get-Guardian) { GuardLog "guard: another guardian is running; exiting"; return }
        GuardLog "guard: started (pid $PID)"
        Remove-Item $pauseFile -Force -ErrorAction SilentlyContinue
        $recent = New-Object System.Collections.Queue
        while ($true) {
            try {
                $paused = $false
                if (Test-Path $pauseFile) {
                    $age = (Get-Date) - (Get-Item $pauseFile).LastWriteTime
                    if ($age.TotalSeconds -lt 60) { $paused = $true }
                    else { Remove-Item $pauseFile -Force -ErrorAction SilentlyContinue }
                }
                if (-not $paused) {
                    $procs = @(Get-Vox)
                    if ($procs.Count -eq 0) {
                        # Crash-storm backoff: >3 restarts in 10 min means Vox is
                        # dying on startup - slow down instead of thrashing.
                        while ($recent.Count -gt 0 -and ((Get-Date) - $recent.Peek()).TotalMinutes -gt 10) { $recent.Dequeue() | Out-Null }
                        if ($recent.Count -ge 3) {
                            GuardLog "guard: 3+ restarts in 10 min - backing off 5 min"
                            Start-Sleep -Seconds 300
                        }
                        GuardLog "guard: Vox not running - starting it"
                        Start-Vox
                        $recent.Enqueue((Get-Date))
                    }
                    elseif ($procs.Count -gt 2) {
                        # A logical Vox instance is 2 pythonw processes (launcher +
                        # interpreter). More than 2 means DOUBLE instances - which
                        # would paste everything twice. Kill all, start one.
                        GuardLog "guard: $($procs.Count) Vox processes (duplicate instance) - restarting clean"
                        Stop-Vox; Start-Sleep -Milliseconds 700; Start-Vox
                        $recent.Enqueue((Get-Date))
                    }
                }
            } catch {
                GuardLog "guard: loop error: $_"
            }
            Start-Sleep -Seconds 20
        }
    }
}
