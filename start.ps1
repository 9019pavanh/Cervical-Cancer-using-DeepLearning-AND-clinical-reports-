$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create the project environment first; see README.md.'
}
& $projectPython -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
