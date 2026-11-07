$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot/.venv/Scripts/python.exe" -X utf8 -m vacv.server --run artifacts/augmented-v5 --port 8765
