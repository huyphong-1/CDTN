# ML Forecast Dashboard (Revenue + Inventory)

Minimal local pipeline for:
- Quarterly revenue forecasting (linear regression)
- Quarterly inventory end-of-period forecasting (linear regression)
- Local web dashboard (tabs for Revenue / Inventory)

## Requirements
- Windows + PowerShell
- Python 3.10+
- SQL Server access via ODBC

Install deps:
```
pip install -r requirements.txt
```

## Configuration
Edit `config/config.yaml`:
- `db.server`, `db.database`, `db.driver`, `db.trusted_connection`
- `scope.product_groups` (optional filter)
- `revenue.min_train_quarters`
- `revenue.forecast_quarters`
- `inventory.min_train_quarters`
- `inventory.forecast_quarters`

Revenue definition:
- `Revenue = LineAmount - DiscountAmount`

Inventory definition:
- Quarter end inventory = sum of `ClosingQty` at the last snapshot date in each quarter per SKU.

## Run
From `D:\ML` in PowerShell:
```
.\run.ps1
```
This runs all steps and starts a local web server.
Open:
- http://localhost:8000/web/

To stop the web server:
- Use Task Manager and end `python`, or
- `Get-Process python | Stop-Process`

## Outputs
CSV files in `outputs/`:
- `revenue_quarterly_actual.csv`
- `revenue_backtest_quarterly.csv`
- `revenue_forecast_linear_quarterly.csv`
- `revenue_linear_metrics.csv`
- `inventory_quarterly_actual.csv`
- `inventory_backtest_quarterly.csv`
- `inventory_forecast_linear_quarterly.csv`
- `inventory_linear_metrics.csv`

PNG charts in `outputs/`:
- `revenue_actual_quarterly.png`
- `revenue_actual_vs_pred_backtest.png`
- `revenue_forecast_quarterly.png`
- `inventory_actual_quarterly.png`
- `inventory_actual_vs_pred_backtest.png`
- `inventory_forecast_quarterly.png`

## Notes
- Backtest uses a linear regression model and a simple baseline (`Lag1`).
- If you see pandas SQLAlchemy warnings, they are safe to ignore.
- Increase `forecast_quarters` if you want more future points.
