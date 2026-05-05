param(
    [switch]$IncludeIgnored
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

$excludePatterns = @(
    "\\.git\\",
    "\\.venv\\",
    "\\node_modules\\",
    "\\runtime_artifacts\\",
    "\\__pycache__\\",
    "\\.pytest_cache\\"
)

$secretPatterns = @(
    @{
        Name = "openai_compatible_sk_token"
        Regex = "sk-[A-Za-z0-9_-]{20,}"
    },
    @{
        Name = "non_placeholder_api_key_json"
        Regex = '"api_key"\s*:\s*"(?!PASTE_YOUR_API_KEY_HERE|<YOUR_API_KEY>|your-api-key|test-key|dummy|example|placeholder|fake-key|)"[^"]+"'
    }
)

$envSecretPatterns = @(
    @{
        Name = "env_api_key_assignment"
        Regex = "(?i)\b[A-Z0-9_]*(API_KEY|TOKEN|SECRET)\b\s*=\s*(?!PASTE_YOUR_API_KEY_HERE|<YOUR_API_KEY>|your-api-key|test-key|dummy|example|placeholder|fake-key)\S+"
    }
)

$files = @()
if ($IncludeIgnored) {
    $files = Get-ChildItem -Path $repoRoot -Recurse -File -Force | Where-Object {
        $path = $_.FullName
        foreach ($pattern in $excludePatterns) {
            if ($path -match $pattern) {
                return $false
            }
        }
        return $true
    }
} else {
    $paths = git -C $repoRoot.Path ls-files -co --exclude-standard
    $files = foreach ($path in $paths) {
        $fullPath = Join-Path $repoRoot $path
        if (Test-Path -LiteralPath $fullPath -PathType Leaf) {
            Get-Item -LiteralPath $fullPath
        }
    }
}

$findings = @()
foreach ($file in $files) {
    try {
        $text = Get-Content -LiteralPath $file.FullName -Raw -ErrorAction Stop
    } catch {
        continue
    }
    foreach ($pattern in $secretPatterns) {
        if ($text -match $pattern.Regex) {
            $relative = Resolve-Path -LiteralPath $file.FullName -Relative
            $findings += [pscustomobject]@{
                Path = $relative
                Rule = $pattern.Name
            }
        }
    }
    if ($file.Name -like ".env*" -or $file.Extension -eq ".env") {
        foreach ($pattern in $envSecretPatterns) {
            if ($text -match $pattern.Regex) {
                $relative = Resolve-Path -LiteralPath $file.FullName -Relative
                $findings += [pscustomobject]@{
                    Path = $relative
                    Rule = $pattern.Name
                }
            }
        }
    }
}

if ($findings.Count -eq 0) {
    Write-Host "No secret patterns found."
    exit 0
}

$findings | Sort-Object Path, Rule | Format-Table -AutoSize
exit 1
