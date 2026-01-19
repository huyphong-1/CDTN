from __future__ import annotations

import sys
from pathlib import Path
import yaml
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.sql_server import load_all_dwh_tables  # noqa

CFG_PATH = BASE_DIR / "config" / "config.yaml"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def compute_revenue(fact_sales: pd.DataFrame) -> pd.Series:
    line = pd.to_numeric(fact_sales["LineAmount"], errors="coerce").fillna(0.0)
    disc = pd.to_numeric(fact_sales.get("DiscountAmount", 0.0), errors="coerce").fillna(0.0)
    return line - disc


def main():
    CFG = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    dim_date, dim_item, fact_sales, snap, purch = load_all_dwh_tables(CFG["db"])

    df = fact_sales.merge(dim_date, on="DateKey", how="left").copy()
    df["date"] = pd.to_datetime(df["date"])

    if "scope" in CFG and "product_groups" in CFG["scope"]:
        if "StockItemKey" in df.columns and "StockItemKey" in dim_item.columns:
            items = dim_item[dim_item["ProductGroup"].isin(CFG["scope"]["product_groups"])].copy()
            keys = set(items["StockItemKey"])
            df = df[df["StockItemKey"].isin(keys)].copy()

    if "LineAmount" not in df.columns:
        raise ValueError("FactSales missing required column: LineAmount")

    df["Revenue"] = compute_revenue(df)

    df["quarter_period"] = df["date"].dt.to_period("Q")
    df["quarter"] = df["quarter_period"].astype(str)
    df["year"] = df["quarter_period"].dt.year
    df["q"] = df["quarter_period"].dt.quarter
    df["quarter_end"] = df["quarter_period"].dt.end_time.dt.date

    g = (
        df.groupby(["quarter", "quarter_period", "year", "q", "quarter_end"], as_index=False)
          .agg({"Revenue": "sum"})
          .rename(columns={"Revenue": "QuarterRevenue"})
          .sort_values("quarter_period")
          .reset_index(drop=True)
    )

    out_path = OUT_DIR / "revenue_quarterly_actual.csv"
    g.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(g.head())


if __name__ == "__main__":
    main()
