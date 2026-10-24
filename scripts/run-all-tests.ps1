<#
.SYNOPSIS
  Run every solution test suite and print a pass/fail table.

.DESCRIPTION
  Suites have different dependencies (torch, gym 0.25 vs 0.26, transformers), so each one
  runs in its own pytest process; a missing package shows up as an ERROR row instead of
  aborting the whole run. Use -Python to point at a specific interpreter or venv, and
  -Only to run a subset (substring match on the suite name).

.EXAMPLE
  ./scripts/run-all-tests.ps1
  ./scripts/run-all-tests.ps1 -Python D:/envs/rl/python.exe -Only 2026
#>
param(
    [string]$Python = "python",
    [string]$Only = ""
)

$root = Split-Path -Parent $PSScriptRoot
$suites = [ordered]@{
    "2026 HW1 imitation"      = "solutions/spring2026/homework1-imitation/code/tests"
    "2026 HW2 online RL"      = "solutions/spring2026/homework2-online-rl/code/tests"
    "2026 HW3 offline RL"     = "solutions/spring2026/homework3-offline-rl/code/tests"
    "2026 default project"    = "solutions/spring2026/default-project/tests"
    "2025 HW3 IQL/CQL"        = "solutions/spring2025/homework3-offline-rl/tests"
    "2025 HW4 HER/DREAM"      = "solutions/spring2025/homework4-goal-meta-rl/tests"
}

$rows = @()
foreach ($name in $suites.Keys) {
    if ($Only -and -not $name.Contains($Only)) { continue }
    $dir = Join-Path $root $suites[$name]
    if (-not (Test-Path $dir)) { $rows += [pscustomobject]@{Suite=$name; Result="MISSING"; Summary=""}; continue }
    $out = & $Python -m pytest -q $dir 2>&1 | Out-String
    $summary = ($out -split "`n" | Where-Object { $_ -match "passed|failed|error" } | Select-Object -Last 1).Trim()
    $result = if ($LASTEXITCODE -eq 0) { "PASS" } elseif ($out -match "ModuleNotFoundError|ImportError") { "ERROR (deps)" } else { "FAIL" }
    $rows += [pscustomobject]@{Suite=$name; Result=$result; Summary=$summary}
}
$rows | Format-Table -AutoSize
if ($rows.Result -contains "FAIL") { exit 1 }
