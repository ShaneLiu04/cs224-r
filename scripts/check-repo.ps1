$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$fail = 0

function Check($name, $condition) {
    if ($condition) {
        Write-Host "  OK   $name"
    } else {
        Write-Host "  FAIL $name" -ForegroundColor Red
        $script:fail++
    }
}

function Count-Files($path, $filter) {
    if (-not (Test-Path $path)) { return -1 }
    return (Get-ChildItem -LiteralPath $path -Filter $filter -File | Measure-Object).Count
}

Write-Host "== Required files =="
Check "README.md" (Test-Path "$root\README.md")
Check "PROGRESS.md" (Test-Path "$root\PROGRESS.md")
Check "UPSTREAM.md" (Test-Path "$root\UPSTREAM.md")
Check "LICENSE" (Test-Path "$root\LICENSE")
Check ".gitignore" (Test-Path "$root\.gitignore")

Write-Host "== Slides =="
Check "spring2026 slides == 17" ((Count-Files "$root\slides\spring2026" "*.pdf") -eq 17)
Check "spring2025 slides == 19" ((Count-Files "$root\slides\spring2025" "*.pdf") -eq 19)
Check "spring2023 slides == 19" ((Count-Files "$root\slides\spring2023" "*.pdf") -eq 19)

Write-Host "== Assignments =="
Check "HW1 handout" (Test-Path "$root\assignments\spring2026\homework1-imitation\CS224R_2026_Homework_1.pdf")
Check "HW1 starter main.py" (Test-Path "$root\assignments\spring2026\homework1-imitation\starter-code\main.py")
Check "HW1 dagger.py" (Test-Path "$root\assignments\spring2026\homework1-imitation\starter-code\dagger.py")
Check "HW2 handout" (Test-Path "$root\assignments\spring2026\homework2-online-rl\CS224R_2026_Homework_2.pdf")
Check "HW2 on_policy.py" (Test-Path "$root\assignments\spring2026\homework2-online-rl\starter-code\on_policy.py")
Check "HW2 off_policy.py" (Test-Path "$root\assignments\spring2026\homework2-online-rl\starter-code\off_policy.py")
Check "HW3 handout" (Test-Path "$root\assignments\spring2026\homework3-offline-rl\CS224R_2026_Homework_3.pdf")
Check "HW3 iql_critic.py" (Test-Path "$root\assignments\spring2026\homework3-offline-rl\starter-code\cs224r\critics\iql_critic.py")
Check "HW3 awac_critic.py" (Test-Path "$root\assignments\spring2026\homework3-offline-rl\starter-code\cs224r\critics\awac_critic.py")
Check "HW3 offline dataset" (Test-Path "$root\assignments\spring2026\homework3-offline-rl\starter-code\offline_datasets")
Check "Project starter sft.py" (Test-Path "$root\assignments\spring2026\default-project\starter-code\sft_trainer\sft.py")
Check "Project starter rloo_update_worker.py" (Test-Path "$root\assignments\spring2026\default-project\starter-code\rloo_trainer\rloo_update_worker.py")
Check "2025 HW1-4" ((Test-Path "$root\assignments\spring2025\homework1-imitation") -and (Test-Path "$root\assignments\spring2025\homework4-goal-meta-rl"))
Check "2023 HW1-4" ((Test-Path "$root\assignments\spring2023\homework1-imitation") -and (Test-Path "$root\assignments\spring2023\homework4-goal-meta-rl"))

Write-Host "== Notes =="
$notes = Count-Files "$root\notes" "lecture-*.md"
Check "18 lecture notes" ($notes -eq 18)
Check "GLOSSARY.md" (Test-Path "$root\notes\GLOSSARY.md")
Check "READING-ROADMAP.md" (Test-Path "$root\notes\READING-ROADMAP.md")

Write-Host "== Experiments / resources =="
Check "experiments README" (Test-Path "$root\experiments\README.md")
Check "SOURCES.yaml" (Test-Path "$root\experiments\SOURCES.yaml")
Check "resources/official.md" (Test-Path "$root\resources\official.md")
Check "templates" ((Test-Path "$root\templates\lecture-note.md") -and (Test-Path "$root\templates\assignment-writeup.md"))

Write-Host "== Guardrails =="
$solutionFiles = Get-ChildItem "$root\assignments" -Recurse -File -Include *.py,*.ipynb -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch "starter-code" -and $_.FullName -notmatch "template" }
Check "no stray solution code under assignments" ($null -eq $solutionFiles)
Check "solutions dir exists" (Test-Path "$root\solutions\README.md")

Write-Host ""
if ($fail -gt 0) {
    Write-Host "$fail check(s) failed." -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed."
