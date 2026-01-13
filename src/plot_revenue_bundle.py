# src/plot_revenue_bundle.py
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

ACTUAL_Q_PATH = OUT_DIR / "revenue_quarterly_actual.csv"
BACKTEST_PATH = OUT_DIR / "revenue_backtest_quarterly.csv"
FORECAST_PATH = OUT_DIR / "revenue_forecast_linear_quarterly.csv"


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
    # ========= (1) Actual full =========
    if not ACTUAL_Q_PATH.exists() or ACTUAL_Q_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing {ACTUAL_Q_PATH}. Run: python src/export_revenue_quarterly_actual.py"
        )

    actual = pd.read_csv(ACTUAL_Q_PATH)
    if not {"quarter", "QuarterRevenue"}.issubset(actual.columns):
        raise ValueError(f"{ACTUAL_Q_PATH.name} must have columns: quarter, QuarterRevenue")

    actual = actual.copy()
    actual["q"] = to_q(actual["quarter"])
    actual["QuarterRevenue"] = safe_num(actual["QuarterRevenue"])
    actual = actual.sort_values("q").reset_index(drop=True)

    plt.figure(figsize=(11, 5))
    plt.plot(actual["q"].astype(str), actual["QuarterRevenue"], marker="o")
    plt.title("Doanh thu theo quy (Actual)")
    plt.xlabel("YearQuarter")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.25)
    savefig(OUT_DIR / "revenue_actual_quarterly.png")

    # ========= (2) Backtest actual vs pred =========
    if not BACKTEST_PATH.exists() or BACKTEST_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {BACKTEST_PATH}. Run: python src/backtest_revenue_quarterly.py")

    bt = pd.read_csv(BACKTEST_PATH)
    if bt.empty:
        raise ValueError(f"{BACKTEST_PATH.name} empty")

    need = {"test_quarter", "y_actual"}
    if not need.issubset(bt.columns):
        raise ValueError(f"{BACKTEST_PATH.name} missing: {sorted(list(need - set(bt.columns)))}")

    bt = bt.copy()
    bt["q"] = to_q(bt["test_quarter"])
    bt["y_actual"] = safe_num(bt["y_actual"])
    bt = bt.sort_values("q").reset_index(drop=True)

    # Use full actual series for x-axis so the chart has all quarter points.
    x_labels = actual["q"].astype(str)
    actual_series = actual["QuarterRevenue"].to_numpy()

    if "linear_pred" not in bt.columns:
        raise ValueError("Backtest file missing linear_pred.")

    pred_map = dict(zip(bt["q"].astype(str), safe_num(bt["linear_pred"])))
    pred_series = [pred_map.get(q, np.nan) for q in x_labels]

    plt.figure(figsize=(11, 5))
    plt.plot(x_labels, actual_series, marker="o", label="Actual", linewidth=2)

    plt.plot(x_labels, pred_series, marker="x", label="Predicted")

    plt.title("Actual vs Predicted Revenue (Backtest)")
    plt.xlabel("YearQuarter")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(OUT_DIR / "revenue_actual_vs_pred_backtest.png")

    # ========= (2b) Predicted vs Actual scatter =========
    pred_vals = safe_num(bt["linear_pred"])
    mask = bt["y_actual"].notna() & pred_vals.notna()
    x_scatter = bt.loc[mask, "y_actual"].to_numpy()
    y_scatter = pred_vals.loc[mask].to_numpy()

    if len(x_scatter) > 0:
        plt.figure(figsize=(6, 6))
        plt.scatter(x_scatter, y_scatter, alpha=0.8)
        min_v = float(min(x_scatter.min(), y_scatter.min()))
        max_v = float(max(x_scatter.max(), y_scatter.max()))
        plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black", alpha=0.5)
        plt.title("Predicted vs Actual Revenue (Backtest)")
        plt.xlabel("Actual Revenue")
        plt.ylabel("Predicted Revenue")
        plt.grid(True, alpha=0.25)
        savefig(OUT_DIR / "revenue_predicted_vs_actual.png")
    else:
        print("WARN: no valid data for predicted vs actual scatter.")

    # ========= (3) Forecast =========
    if not FORECAST_PATH.exists() or FORECAST_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {FORECAST_PATH}. Run: python src/train_revenue_linear.py")

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

    plt.title("Forecast Doanh thu theo quy")
    plt.xlabel("YearQuarter")
    plt.ylabel("Revenue")
    plt.xticks(x, xlab, rotation=45)
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(OUT_DIR / "revenue_forecast_quarterly.png")


if __name__ == "__main__":
    main()
