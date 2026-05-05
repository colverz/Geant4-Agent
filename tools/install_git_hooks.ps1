param()

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$hooksDir = Join-Path $repoRoot ".git/hooks"

if (-not (Test-Path -LiteralPath $hooksDir)) {
    throw "Git hooks directory not found: $hooksDir"
}

$hookBody = @'
#!/bin/sh
echo "Running secret scan..."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/check_secrets.ps1
status=$?
if [ $status -ne 0 ]; then
  echo "Secret scan failed. Commit/push blocked."
  exit $status
fi
'@

foreach ($hookName in @("pre-commit", "pre-push")) {
    $hookPath = Join-Path $hooksDir $hookName
    Set-Content -LiteralPath $hookPath -Value $hookBody -Encoding ASCII
    Write-Host "Installed $hookName hook."
}

Write-Host "Git hooks installed. Commits and pushes will run tools/check_secrets.ps1."
Write-Host "If Git for Windows reports a sh.exe signal pipe error, remove .git/hooks/pre-commit and .git/hooks/pre-push, then run tools/check_secrets.ps1 manually before commit/push."
