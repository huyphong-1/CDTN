# src/backtest_revenue_quarterly.py
from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.sql_server import load_all_dwh_tables


BASE_DIR = Path(__file__).resolve().parent.parent
CFG_PATH = BASE_DIR / "config" / "config.yaml"


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred, eps=1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def regression_report(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": float(mape(y_true, y_pred)),
    }
    out["R2"] = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    return out


def compute_revenue(fact_sales: pd.DataFrame) -> pd.Series:
    line = pd.to_numeric(fact_sales["LineAmount"], errors="coerce").fillna(0.0)
    disc = pd.to_numeric(fact_sales.get("DiscountAmount", 0.0), errors="coerce").fillna(0.0)
    return line - disc


def build_quarterly_dataset(cfg) -> pd.DataFrame:
    dim_date, dim_item, fact_sales, snap, purch = load_all_dwh_tables(cfg["db"])

    df = fact_sales.merge(dim_date, on="DateKey", how="left").copy()
    df["date"] = pd.to_datetime(df["date"])

    if "scope" in cfg and "product_groups" in cfg["scope"]:
        items = dim_item[dim_item["ProductGroup"].isin(cfg["scope"]["product_groups"])].copy()
        keys = set(items["StockItemKey"])
        if "StockItemKey" in df.columns:
            df = df[df["StockItemKey"].isin(keys)].copy()

    if "LineAmount" not in df.columns:
        raise ValueError("FactSales missing column LineAmount (needed for revenue).")

    df["Revenue"] = compute_revenue(df)

    df["quarter_period"] = df["date"].dt.to_period("Q")
    df["quarter"] = df["quarter_period"].astype(str)
    df["year"] = df["quarter_period"].dt.year
    df["q"] = df["quarter_period"].dt.quarter
    df["quarter_end"] = df["quarter_period"].dt.end_time.dt.date

    g = (
        df.groupby(["quarter_period", "quarter", "year", "q", "quarter_end"], as_index=False)
        .agg({"Revenue": "sum"})
    )

    g.rename(columns={"Revenue": "QuarterRevenue"}, inplace=True)
    g.sort_values("quarter_period", inplace=True)
    g.reset_index(drop=True, inplace=True)
    return g


def _add_metric_features(ds: pd.DataFrame, target: str, prefix: str) -> pd.DataFrame:
    for l in (1, 2, 3, 4):
        ds[f"{prefix}_Lag{l}"] = ds[target].shift(l)

    ds[f"{prefix}_Mean4"] = ds[target].rolling(4, min_periods=4).mean()
    ds[f"{prefix}_Growth1"] = (ds[target] - ds[f"{prefix}_Lag1"]) / (ds[f"{prefix}_Lag1"].abs() + 1e-9)
    return ds


def add_quarter_features(ds: pd.DataFrame) -> pd.DataFrame:
    ds = ds.copy()
    ds.sort_values("quarter_period", inplace=True)

    if "QuarterRevenue" in ds.columns:
        ds = _add_metric_features(ds, target="QuarterRevenue", prefix="Rev")

    ds["q_sin"] = np.sin(2 * np.pi * ds["q"] / 4.0)
    ds["q_cos"] = np.cos(2 * np.pi * ds["q"] / 4.0)
    return ds


def quarterly_backtest_revenue(
    ds: pd.DataFrame,
    features: list[str],
    target: str = "QuarterRevenue",
    min_train_quarters: int = 8,
) -> pd.DataFrame:
    ds = ds.copy().sort_values("quarter_period").reset_index(drop=True)

    quarters = ds["quarter_period"].tolist()
    rows = []

    for i in range(min_train_quarters - 1, len(quarters) - 1):
        cutoff_q = quarters[i]
        test_q = quarters[i + 1]

        train_df = ds[ds["quarter_period"] <= cutoff_q].copy()
        test_df = ds[ds["quarter_period"] == test_q].copy()

        if train_df.empty or test_df.empty:
            continue

        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]

        baseline_pred = test_df["Rev_Lag1"].to_numpy(dtype=float)

        lin = LinearRegression()
        lin.fit(X_train, y_train)
        lin_pred = lin.predict(X_test)

        base_m = regression_report(y_test, baseline_pred)
        lin_m = regression_report(y_test, lin_pred)

        lin_impr = (base_m["MAE"] - lin_m["MAE"]) / max(base_m["MAE"], 1e-9)

        rows.append({
            "fold": int(len(rows) + 1),
            "train_end_quarter": str(cutoff_q),
            "test_quarter": str(test_q),
            "y_actual": float(y_test.iloc[0]),

            "baseline_pred": float(baseline_pred[0]),
            "linear_pred": float(lin_pred[0]),

            "baseline_MAE": base_m["MAE"],
            "linear_MAE": lin_m["MAE"],

            "baseline_RMSE": base_m["RMSE"],
            "linear_RMSE": lin_m["RMSE"],

            "baseline_MAPE": base_m["MAPE"],
            "linear_MAPE": lin_m["MAPE"],

            "linear_MAE_improvement_pct": float(lin_impr),
        })

    return pd.DataFrame(rows)


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))

    raw = build_quarterly_dataset(cfg)
    ds = add_quarter_features(raw)

    ds.replace([np.inf, -np.inf], np.nan, inplace=True)
    ds.dropna(inplace=True)
    ds.reset_index(drop=True, inplace=True)

    target = "QuarterRevenue"
    features = [
        "year", "q", "q_sin", "q_cos",
        "Rev_Lag1", "Rev_Lag2", "Rev_Lag3", "Rev_Lag4",
        "Rev_Mean4",
        "Rev_Growth1",
    ]

    missing = [c for c in (features + [target, "quarter_period"]) if c not in ds.columns]
    if missing:
        raise ValueError(f"Missing columns in quarterly dataset: {missing}")

    folds = quarterly_backtest_revenue(
        ds=ds,
        features=features,
        target=target,
        min_train_quarters=int(cfg.get("revenue", {}).get("min_train_quarters", 8)),
    )

    out_dir = BASE_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)

    actual_out = raw[["quarter", "quarter_period", "year", "q", "quarter_end", "QuarterRevenue"]].copy()
    actual_out.to_csv(out_dir / "revenue_quarterly_actual.csv", index=False)
    print(f"Saved: {out_dir / 'revenue_quarterly_actual.csv'}")

    out_path = out_dir / "revenue_backtest_quarterly.csv"
    folds.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(folds)


if __name__ == "__main__":
    main()
