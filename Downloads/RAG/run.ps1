# Convenience wrapper around RUN_COMMANDS.md — PowerShell.
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "ingest", "baseline", "enhanced", "reports", "api", "test", "all")]
    [string]$Task = "all"
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "No virtualenv found at .venv. Create it first (see RUN_COMMANDS.md section 0)."
}

function Invoke-Step {
    param([string[]]$Arguments)
    Write-Host ">> python $($Arguments -join ' ')" -ForegroundColor Cyan
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

switch ($Task) {
    "setup"    { Invoke-Step @("-m", "pip", "install", "-r", "requirements.txt") }
    "ingest"   { Invoke-Step @("-m", "src.rag.ingestion") }
    "baseline" { Invoke-Step @("-m", "src.evaluation.run_evaluation", "--config", "configs/experiments/exp_001_baseline.yaml") }
    "enhanced" { Invoke-Step @("-m", "src.evaluation.run_evaluation", "--config", "configs/experiments/exp_002_enhanced.yaml") }
    "reports"  {
        Invoke-Step @("-m", "src.evaluation.text_report", "--experiment-id", "EXP-001")
    }
    "api"      { Invoke-Step @("-m", "src.api.main") }
    "test"     { Invoke-Step @("-m", "pytest", "tests", "-q") }
    "all" {
        Invoke-Step @("-m", "src.evaluation.dataset", "data/datasets/benchmark_v1.jsonl")
        Invoke-Step @("-m", "src.rag.ingestion")
        Invoke-Step @("-m", "src.evaluation.run_evaluation", "--config", "configs/experiments/exp_001_baseline.yaml")
        Invoke-Step @("-m", "src.evaluation.run_evaluation", "--config", "configs/experiments/exp_002_enhanced.yaml")
    }
}
