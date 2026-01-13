$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

$venvActivate = Join-Path $ROOT ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
  throw "ERROR: .venv not found at $venvActivate"
}

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
. $venvActivate

Write-Host "Python version:" -ForegroundColor Gray
python --version

function Run-Step([string]$Label, [string]$ScriptPath) {
  Write-Host "`n$Label" -ForegroundColor Cyan
  if (-not (Test-Path $ScriptPath)) { throw "Missing script: $ScriptPath" }

  python $ScriptPath
  if ($LASTEXITCODE -ne 0) { throw "Step failed: $Label" }
}

function Assert-File([string]$Label, [string]$PathToFile) {
  if (-not (Test-Path $PathToFile)) {
    throw "Missing output after ${Label}: ${PathToFile}"
  }
}

if (-not (Test-Path "outputs")) { New-Item -ItemType Directory -Path "outputs" | Out-Null }

Run-Step "[1/7] Export Revenue Actual (quarterly)" "src\export_revenue_quarterly_actual.py"
Assert-File "Export Revenue Actual" "outputs\revenue_quarterly_actual.csv"

Run-Step "[2/7] Backtest Revenue (Linear Regression)" "src\backtest_revenue_quarterly.py"
Assert-File "Backtest Revenue" "outputs\revenue_backtest_quarterly.csv"

Run-Step "[3/7] Train Linear Regression (Revenue Forecast)" "src\train_revenue_linear.py"
Assert-File "Linear Forecast" "outputs\revenue_forecast_linear_quarterly.csv"
Assert-File "Linear Metrics" "outputs\revenue_linear_metrics.csv"

Run-Step "[4/7] Plot Revenue Bundle" "src\plot_revenue_bundle.py"
Assert-File "Revenue actual chart" "outputs\revenue_actual_quarterly.png"
Assert-File "Revenue backtest chart" "outputs\revenue_actual_vs_pred_backtest.png"
Assert-File "Revenue scatter chart" "outputs\revenue_predicted_vs_actual.png"
Assert-File "Revenue forecast chart" "outputs\revenue_forecast_quarterly.png"

Run-Step "[5/7] Backtest Inventory (Linear Regression)" "src\backtest_inventory_quarterly.py"
Assert-File "Inventory actuals" "outputs\inventory_quarterly_actual.csv"
Assert-File "Inventory backtest" "outputs\inventory_backtest_quarterly.csv"

Run-Step "[6/7] Train Inventory Linear Regression (Forecast)" "src\train_inventory_linear.py"
Assert-File "Inventory forecast" "outputs\inventory_forecast_linear_quarterly.csv"
Assert-File "Inventory metrics" "outputs\inventory_linear_metrics.csv"

Run-Step "[7/7] Plot Inventory Bundle" "src\plot_inventory_bundle.py"
Assert-File "Inventory actual chart" "outputs\inventory_actual_quarterly.png"
Assert-File "Inventory backtest chart" "outputs\inventory_actual_vs_pred_backtest.png"
Assert-File "Inventory forecast chart" "outputs\inventory_forecast_quarterly.png"

if (-not (Test-Path "web\outputs")) { New-Item -ItemType Directory -Path "web\outputs" | Out-Null }
Copy-Item -Path "outputs\*" -Destination "web\outputs" -Recurse -Force

Write-Host "`n=== ML PIPELINE FINISHED SUCCESSFULLY ===" -ForegroundColor Green
Write-Host "PNG outputs: dir outputs\*.png" -ForegroundColor Gray
Write-Host "CSV outputs: dir outputs\*.csv" -ForegroundColor Gray

# -------------------------
# 4) Start local web server
# -------------------------
try {
  Start-Process -FilePath "python" -ArgumentList "-m", "http.server", "8000" -WorkingDirectory $ROOT | Out-Null
  Write-Host "Web server: http://localhost:8000/web/" -ForegroundColor Green
} catch {
  Write-Host "WARN: Could not start local web server on port 8000." -ForegroundColor Yellow
}
