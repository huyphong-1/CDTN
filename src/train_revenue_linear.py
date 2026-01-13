# src/train_revenue_linear.py
from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backtest_revenue_quarterly import build_quarterly_dataset, add_quarter_features


BASE_DIR = Path(__file__).resolve().parent.parent
CFG_PATH = BASE_DIR / "config" / "config.yaml"
OUT_DIR = BASE_DIR / "outputs"


def mape(y_true, y_pred, eps=1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def regression_report(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred) if len(np.asarray(y_true)) >= 2 else float("nan")
    return {
        "n": int(len(y_true)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape(y_true, y_pred)),
        "R2": float(r2),
    }


def _feature_row(values: list[float], year: int, q: int) -> dict:
    def lag(l: int) -> float:
        if len(values) >= l:
            return float(values[-l])
        return float(values[-1])

    lag1 = lag(1)
    lag2 = lag(2)
    lag3 = lag(3)
    lag4 = lag(4)
    mean4 = float(np.mean(values[-min(4, len(values)) :]))
    growth1 = (lag1 - lag2) / (abs(lag2) + 1e-9)

    return {
        "year": year,
        "q": q,
        "q_sin": float(np.sin(2 * np.pi * q / 4.0)),
        "q_cos": float(np.cos(2 * np.pi * q / 4.0)),
        "Rev_Lag1": lag1,
        "Rev_Lag2": lag2,
        "Rev_Lag3": lag3,
        "Rev_Lag4": lag4,
        "Rev_Mean4": mean4,
        "Rev_Growth1": float(growth1),
    }


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))

    raw = build_quarterly_dataset(cfg)
    ds = add_quarter_features(raw)

    ds.replace([np.inf, -np.inf], np.nan, inplace=True)
    ds.dropna(inplace=True)
    ds.reset_index(drop=True, inplace=True)

    target = "QuarterRevenue"
    if target not in ds.columns:
        raise ValueError("Missing QuarterRevenue in dataset.")

    features = [
        "year", "q", "q_sin", "q_cos",
        "Rev_Lag1", "Rev_Lag2", "Rev_Lag3", "Rev_Lag4",
        "Rev_Mean4",
        "Rev_Growth1",
    ]

    missing = [c for c in (features + [target, "quarter_period", "quarter"]) if c not in ds.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = ds[features]
    y = ds[target]

    model = LinearRegression()
    model.fit(X, y)

    report = regression_report(y, model.predict(X))
    print("=== LINEAR REGRESSION (QuarterRevenue) ===")
    print(
        f"n={report['n']} | MAE={report['MAE']:.2f} RMSE={report['RMSE']:.2f} "
        f"MAPE={report['MAPE']:.4f} R2={report['R2']:.4f}"
    )

    horizon = int(cfg.get("revenue", {}).get("forecast_quarters", 6))

    out_rows = [
        {"quarter": r["quarter"], "Actual": float(r[target]), "Forecast": np.nan}
        for _, r in raw.iterrows()
    ]

    sim_vals = raw[target].astype(float).tolist()
    last_period = raw["quarter_period"].iloc[-1]

    for step in range(1, horizon + 1):
        p = last_period + step
        row = _feature_row(sim_vals, year=p.year, q=p.quarter)
        pred = float(model.predict(pd.DataFrame([row])[features])[0])
        sim_vals.append(pred)
        out_rows.append({"quarter": str(p), "Actual": np.nan, "Forecast": pred})

    out = pd.DataFrame(out_rows)
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "revenue_forecast_linear_quarterly.csv"
    out.to_csv(out_path, index=False)

    (OUT_DIR / "revenue_linear_metrics.csv").write_text(
        "Metric,Value\n"
        f"n,{report['n']}\n"
        f"MAE,{report['MAE']}\n"
        f"RMSE,{report['RMSE']}\n"
        f"MAPE,{report['MAPE']}\n"
        f"R2,{report['R2']}\n",
        encoding="utf-8",
    )

    print(f"Saved: {out_path}")
    print(f"Saved: {OUT_DIR / 'revenue_linear_metrics.csv'}")


if __name__ == "__main__":
    main()
