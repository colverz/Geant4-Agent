param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

$targets = @(
    ".pytest_cache",
    "runtime_artifacts",
    "runtime/geant4_local_app/build",
    "ui/desktop/node_modules"
)

$venvRoot = Join-Path $repoRoot ".venv"
$venvRootPath = if (Test-Path -LiteralPath $venvRoot) {
    (Resolve-Path -LiteralPath $venvRoot).Path
} else {
    $null
}

$pycacheTargets = Get-ChildItem -Path $repoRoot -Directory -Recurse -Force -Filter "__pycache__" |
    Where-Object {
        if (-not $venvRootPath) {
            return $true
        }
        -not $_.FullName.StartsWith($venvRootPath, [System.StringComparison]::OrdinalIgnoreCase)
    } |
    ForEach-Object { $_.FullName }

$resolvedTargets = foreach ($target in $targets) {
    $path = Join-Path $repoRoot $target
    if (Test-Path -LiteralPath $path) {
        (Resolve-Path -LiteralPath $path).Path
    }
}

$allTargets = @($resolvedTargets + $pycacheTargets) |
    Where-Object { $_ } |
    Sort-Object -Unique

if (-not $allTargets) {
    Write-Host "No local artifacts found."
    exit 0
}

Write-Host "Local artifacts:"
foreach ($path in $allTargets) {
    Write-Host "  $path"
}

if (-not $Apply) {
    Write-Host ""
    Write-Host "Dry run only. Re-run with -Apply to remove these ignored local artifacts."
    exit 0
}

foreach ($path in $allTargets) {
    $resolved = (Resolve-Path -LiteralPath $path).Path
    if (-not $resolved.StartsWith($repoRoot.Path, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside repo: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

Write-Host "Removed local artifacts."
