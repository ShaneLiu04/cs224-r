param(
    [switch]$Force,
    [switch]$WithProjects,
    [string]$Commit = "3f8136a624dbb51ce10dbb78a7b5bf2575d2671a"
)

# Re-sync official CS224R materials from the course website repository.
# Requires git on PATH (or set GIT_EXE). Downloads only the blobs we need
# (slides + material) via sparse checkout.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$git = $env:GIT_EXE
if (-not $git) {
    $candidate = "C:\Program Files\Git\cmd\git.exe"
    $git = if (Test-Path $candidate) { $candidate } else { "git" }
}

$upstream = "https://github.com/cs224r/cs224r.github.io.git"
$temp = Join-Path ([System.IO.Path]::GetTempPath()) "cs224r-upstream-$Commit"

if ((Test-Path $temp) -and $Force) { Remove-Item -Recurse -Force $temp }
if (-not (Test-Path $temp)) {
    Write-Host "Cloning $upstream at $Commit ..."
    & $git clone --filter=blob:none --no-checkout $upstream $temp
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}
& $git -C $temp sparse-checkout init --no-cone 2>$null
$patterns = @("/*", "!/images/*", "!/css/*", "!/js/*", "!/*.ics")
if (-not $WithProjects) {
    $patterns += @("!/projects/pdfs/*", "!/spring_2025/projects/pdfs/*")
}
& $git -C $temp sparse-checkout set @patterns
& $git -C $temp checkout $Commit
if ($LASTEXITCODE -ne 0) { throw "git checkout failed" }

function Copy-IfNewer($src, $dst) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Copy-Item -Force $src $dst
}

# slides
Copy-IfNewer "$temp\slides" "$root\slides\spring2026" 2>$null
Copy-IfNewer "$temp\spring_2025\slides" "$root\slides\spring2025" 2>$null
Copy-IfNewer "$temp\spring_2023\slides" "$root\slides\spring2023" 2>$null

# 2026 assignments + materials
Copy-IfNewer "$temp\material\hw1" "$root\assignments\spring2026\homework1-imitation" 2>$null
Copy-IfNewer "$temp\material\hw2" "$root\assignments\spring2026\homework2-online-rl" 2>$null
Copy-IfNewer "$temp\material\hw3" "$root\assignments\spring2026\homework3-offline-rl" 2>$null
Copy-IfNewer "$temp\material" "$root\materials\spring2026" 2>$null

# 2025 / 2023 archives
Copy-IfNewer "$temp\spring_2025\material" "$root\assignments\spring2025" 2>$null
Copy-IfNewer "$temp\spring_2023\material" "$root\assignments\spring2023" 2>$null

if ($WithProjects) {
    Write-Host "Project PDFs copied under materials/*/projects-pdfs (large, not committed by default)."
}

Write-Host "Sync complete. Re-run starter-code extraction for changed zips if needed, then"
Write-Host "update UPSTREAM.md and run scripts/check-repo.ps1."
