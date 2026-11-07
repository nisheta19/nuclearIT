$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot/.venv/Scripts/python.exe" -X utf8 -m vacv.server --run artifacts/advanced-v4 --port 8765
