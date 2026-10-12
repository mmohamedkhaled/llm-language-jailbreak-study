# One-shot runner for the priority experiment: encoded arms on the
# strong-refusal model (originally Qwen3.5-9B). Does everything the study
# needs once the model directory exists locally:
#   serve -> plain/SPUN/ROT13/ROT13+table arms on AdvBench-50
#         -> summarize -> LLM-as-judge pass -> adjudication workbook
#
# Usage:
#   .\run_strong_model.ps1 -ModelPath "..\Qwen3.5-9B-Instruct"
#   .\run_strong_model.ps1 -ModelPath "..\Qwen3.5-9B-Instruct" -Tag qwen35_9b

param(
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [string]$Tag = "strong",
    [int]$Port = 8085
)

$ErrorActionPreference = "Stop"
$Py = "..\.venv\Scripts\python.exe"
$End = "http://localhost:$Port/v1"
if (-not (Test-Path $ModelPath)) {
    Write-Error "Model path not found: $ModelPath (download it first)"
}

function Wait-Server {
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep 5
        if (Select-String -Path "server_strong.log" -Pattern "listening on" `
                -SimpleMatch -Quiet -ErrorAction SilentlyContinue) { return }
    }
    Write-Error "server failed to load; see server_strong.log"
}

# 1. serve the model
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "*language project*" } | Stop-Process -Force
Start-Sleep 2
Start-Process -FilePath $Py `
    -ArgumentList "serve_mistral.py", "--model", $ModelPath, "--port", $Port `
    -RedirectStandardOutput "server_strong.log" `
    -RedirectStandardError "server_strong.err.log" -WindowStyle Hidden
Wait-Server
Write-Output "server up on :$Port"

# 2. four arms on AdvBench-50
$arms = @(
    @{ flag = "--plain";                out = "${Tag}_plain.csv" },
    @{ flag = "--include-system-prompt"; out = "${Tag}_spun.csv" },
    @{ flag = "--encode rot13";         out = "${Tag}_rot13.csv" },
    @{ flag = "--encode rot13 --table"; out = "${Tag}_rot13_table.csv" }
)
foreach ($a in $arms) {
    Write-Output ("== arm: " + $a.out)
    & $Py run_eval.py advbench_harmful_behaviors.csv --column goal `
        --limit 50 --model m --endpoint $End ($a.flag -split " ") `
        --delay 0.2 --max-tokens 250 --out $a.out |
        Select-Object -Last 4
}

# 3. summarize, judge, adjudicate
Write-Output "== summary"
& $Py summarize_results.py "${Tag}_plain.csv" "${Tag}_spun.csv" `
    "${Tag}_rot13.csv" "${Tag}_rot13_table.csv"
Write-Output "== judge pass (judge = the served model itself)"
& $Py judge.py "${Tag}_plain.csv" "${Tag}_spun.csv" `
    "${Tag}_rot13.csv" "${Tag}_rot13_table.csv" --model m --endpoint $End
& $Py adjudicate.py "${Tag}_plain_judged.csv" "${Tag}_spun_judged.csv" `
    "${Tag}_rot13_judged.csv" "${Tag}_rot13_table_judged.csv"

Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "*language project*" } | Stop-Process -Force
Write-Output "DONE — read adjudication\*_review.csv before believing any comply number"
