' Windowless launcher for the Vox guardian (vox.ps1 guard).
'
' A Startup shortcut aimed straight at powershell.exe flashes a console for a
' moment at login even with -WindowStyle Hidden (the window exists before the
' flag applies). wscript runs this file with no window, and Run's window style
' 0 creates the child PowerShell hidden from the first frame. Point the
' Startup shortcut at:  wscript.exe "<repo>\vox-guard.vbs"
root = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & root & "\vox.ps1"" guard"
CreateObject("WScript.Shell").Run cmd, 0, False
