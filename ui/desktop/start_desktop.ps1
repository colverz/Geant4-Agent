$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$desktopDir = Join-Path $root "ui\desktop"

if (-not (Test-Path $venvPython)) {
  throw "Missing virtual environment python: $venvPython"
}
if (-not (Test-Path (Join-Path $desktopDir "node_modules\electron"))) {
  throw "Missing Electron dependency under ui\desktop\node_modules. Run npm install in ui\desktop or restore node_modules."
}

$env:G4_DESKTOP_PYTHON = $venvPython

Set-Location $desktopDir
& npm start
