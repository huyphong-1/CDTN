from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

ACTUAL_Q_PATH = OUT_DIR / "inventory_quarterly_actual.csv"
BACKTEST_PATH = OUT_DIR / "inventory_backtest_quarterly.csv"
FORECAST_PATH = OUT_DIR / "inventory_forecast_linear_quarterly.csv"


def to_q(series: pd.Series) -> pd.PeriodIndex:
    s = series.astype(str).str.strip()
    return pd.PeriodIndex(s, freq="Q")


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def savefig(path: Path):
    path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    if not ACTUAL_Q_PATH.exists() or ACTUAL_Q_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing {ACTUAL_Q_PATH}. Run: python src/backtest_inventory_quarterly.py"
        )

    actual = pd.read_csv(ACTUAL_Q_PATH)
    if not {"quarter", "QuarterEndInventory"}.issubset(actual.columns):
        raise ValueError(f"{ACTUAL_Q_PATH.name} must have columns: quarter, QuarterEndInventory")

    actual = actual.copy()
    actual["q"] = to_q(actual["quarter"])
    actual["QuarterEndInventory"] = safe_num(actual["QuarterEndInventory"])
    actual = actual.sort_values("q").reset_index(drop=True)

    plt.figure(figsize=(11, 5))
    plt.plot(actual["q"].astype(str), actual["QuarterEndInventory"], marker="o")
    plt.title("Quarterly Inventory (Actual)")
    plt.xlabel("YearQuarter")
    plt.ylabel("Quarter End Inventory")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.25)
    savefig(OUT_DIR / "inventory_actual_quarterly.png")

    if not BACKTEST_PATH.exists() or BACKTEST_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {BACKTEST_PATH}. Run: python src/backtest_inventory_quarterly.py")

    bt = pd.read_csv(BACKTEST_PATH)
    if bt.empty:
        raise ValueError(f"{BACKTEST_PATH.name} empty")

    need = {"test_quarter", "y_actual", "linear_pred"}
    if not need.issubset(bt.columns):
        raise ValueError(f"{BACKTEST_PATH.name} missing: {sorted(list(need - set(bt.columns)))}")

    bt = bt.copy()
    bt["q"] = to_q(bt["test_quarter"])
    bt["y_actual"] = safe_num(bt["y_actual"])
    bt["linear_pred"] = safe_num(bt["linear_pred"])
    bt = bt.sort_values("q").reset_index(drop=True)

    x_labels = actual["q"].astype(str)
    actual_series = actual["QuarterEndInventory"].to_numpy()

    pred_map = dict(zip(bt["q"].astype(str), bt["linear_pred"]))
    pred_series = [pred_map.get(q, np.nan) for q in x_labels]

    plt.figure(figsize=(11, 5))
    plt.plot(x_labels, actual_series, marker="o", label="Actual", linewidth=2)
    plt.plot(x_labels, pred_series, marker="x", label="Predicted")

    plt.title("Actual vs Predicted Inventory (Backtest)")
    plt.xlabel("YearQuarter")
    plt.ylabel("Quarter End Inventory")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(OUT_DIR / "inventory_actual_vs_pred_backtest.png")

    if not FORECAST_PATH.exists() or FORECAST_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {FORECAST_PATH}. Run: python src/train_inventory_linear.py")

    fc = pd.read_csv(FORECAST_PATH)
    if not {"quarter", "Actual", "Forecast"}.issubset(fc.columns):
        raise ValueError(f"{FORECAST_PATH.name} missing columns quarter/Actual/Forecast")

    fc = fc.copy()
    fc["q"] = to_q(fc["quarter"])
    fc["Actual"] = safe_num(fc["Actual"])
    fc["Forecast"] = safe_num(fc["Forecast"])
    fc = fc.sort_values("q").reset_index(drop=True)

    xlab = fc["q"].astype(str).tolist()
    x = list(range(len(xlab)))

    forecast_start_idx = None
    if fc["Actual"].isna().any():
        forecast_start_idx = int(np.argmax(fc["Actual"].isna().to_numpy()))

    plt.figure(figsize=(11, 5))
    plt.plot(x, fc["Actual"].to_numpy(), marker="o", label="Actual")
    plt.plot(x, fc["Forecast"].to_numpy(), marker="x", linestyle="--", label="Forecast")

    if forecast_start_idx is not None:
        plt.axvline(forecast_start_idx, linestyle=":", label="Forecast start")

    plt.title("Quarterly Inventory Forecast")
    plt.xlabel("YearQuarter")
    plt.ylabel("Quarter End Inventory")
    plt.xticks(x, xlab, rotation=45)
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(OUT_DIR / "inventory_forecast_quarterly.png")


if __name__ == "__main__":
    main()
