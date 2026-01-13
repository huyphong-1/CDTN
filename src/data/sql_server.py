from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import pyodbc
import pandas as pd


# =========================
# CONFIG: DWH mapping fixed
# =========================
@dataclass(frozen=True)
class DwhMap:
    schema: str = "dw"

    # Tables
    dim_date: str = "DimDate"
    dim_item: str = "DimStockItem"
    fact_sales: str = "FactSales"
    fact_inventory_snapshot: str = "FactInventorySnapshot"
    fact_purchasing: str = "FactPurchasing"

    # Columns (DimDate)
    dim_date_datekey: str = "DateKey"
    dim_date_date: str = "date"  # adjust if your column is FullDate/Date/etc.

    # Columns (DimStockItem)
    dim_item_key: str = "StockItemKey"
    dim_item_product_group: str = "ProductGroup"
    dim_item_product_subgroup: str = "ProductSubGroup"
    dim_item_is_current: str = "IsCurrent"

    # Columns (FactInventorySnapshot)
    snap_datekey: str = "DateKey"
    snap_itemkey: str = "StockItemKey"
    snap_closing_qty: str = "ClosingQty"

    # Columns (FactPurchasing)
    purch_datekey: str = "DateKey"
    purch_itemkey: str = "StockItemKey"
    purch_received_qty: str = "ReceivedQty"

    # Columns (FactSales)
    sales_datekey: str = "DateKey"
    sales_itemkey: str = "StockItemKey"


DWH = DwhMap()


# =========================
# Connection
# =========================
def make_sqlserver_conn(cfg: Dict) -> pyodbc.Connection:
    """
    Supports:
      - Windows Auth: trusted_connection: true
      - SQL Auth: username/password (or user/pwd)
    Required keys: server, database
    """
    driver = cfg.get("driver", "ODBC Driver 17 for SQL Server")
    server = cfg.get("server")
    database = cfg.get("database")
    if not server or not database:
        raise ValueError("DB config missing required keys: server, database")

    username = cfg.get("username") or cfg.get("user") or cfg.get("uid")
    password = cfg.get("password") or cfg.get("pass") or cfg.get("pwd")
    trusted = bool(cfg.get("trusted_connection") or cfg.get("trusted") or cfg.get("windows_auth"))

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "TrustServerCertificate=yes;"
    )

    if trusted and not username and not password:
        conn_str += "Trusted_Connection=yes;"
    else:
        if not username or password is None:
            raise ValueError(
                "DB config missing credentials. Provide username/password or set trusted_connection: true."
            )
        conn_str += f"UID={username};PWD={password};"

    return pyodbc.connect(conn_str)


def _read_sql(conn: pyodbc.Connection, sql: str) -> pd.DataFrame:
    # Using pyodbc directly is ok; pandas warning is harmless.
    return pd.read_sql(sql, conn)


# =========================
# Loaders
# =========================
def load_dim_date(conn: pyodbc.Connection, dwh: DwhMap = DWH) -> pd.DataFrame:
    sql = f"""
        SELECT
            {dwh.dim_date_datekey} AS DateKey,
            [{dwh.dim_date_date}]  AS [date]
        FROM {dwh.schema}.{dwh.dim_date}
    """
    df = _read_sql(conn, sql)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_dim_item(conn: pyodbc.Connection, dwh: DwhMap = DWH) -> pd.DataFrame:
    # ✅ lấy thêm ProductSubGroup + lọc IsCurrent=1 để tránh duplicate SCD2
    sql = f"""
        SELECT
            {dwh.dim_item_key} AS StockItemKey,
            {dwh.dim_item_product_group} AS ProductGroup,
            {dwh.dim_item_product_subgroup} AS ProductSubGroup
        FROM {dwh.schema}.{dwh.dim_item}
        WHERE {dwh.dim_item_is_current} = 1
    """
    return _read_sql(conn, sql)


def load_fact_sales(conn, dwh=DWH) -> pd.DataFrame:
    sql = f"""
        SELECT
            DateKey,
            StockItemKey,
            CustomerKey,
            EmployeeKey,
            Quantity,
            UnitPrice,
            TaxRate,
            DiscountAmount,
            LineAmount,
            CostAmount,
            GrossProfit,
            InvoiceID_BK
        FROM {dwh.schema}.{dwh.fact_sales}
    """
    df = _read_sql(conn, sql)

    for c in ["Quantity", "UnitPrice", "TaxRate", "DiscountAmount", "LineAmount", "CostAmount", "GrossProfit"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df



def load_inventory_snapshot(conn: pyodbc.Connection, dwh: DwhMap = DWH) -> pd.DataFrame:
    sql = f"""
        SELECT
            {dwh.snap_datekey} AS DateKey,
            {dwh.snap_itemkey} AS StockItemKey,
            {dwh.snap_closing_qty} AS ClosingQty
        FROM {dwh.schema}.{dwh.fact_inventory_snapshot}
    """
    df = _read_sql(conn, sql)
    df["ClosingQty"] = pd.to_numeric(df["ClosingQty"], errors="coerce").fillna(0.0)
    return df


def load_purchasing(conn: pyodbc.Connection, dwh: DwhMap = DWH) -> pd.DataFrame:
    sql = f"""
        SELECT
            {dwh.purch_datekey} AS DateKey,
            {dwh.purch_itemkey} AS StockItemKey,
            {dwh.purch_received_qty} AS ReceivedQty
        FROM {dwh.schema}.{dwh.fact_purchasing}
    """
    df = _read_sql(conn, sql)
    df["ReceivedQty"] = pd.to_numeric(df["ReceivedQty"], errors="coerce").fillna(0.0)
    return df


def load_all_dwh_tables(
    cfg: Dict,
    dwh: DwhMap = DWH
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: dim_date, dim_item, fact_sales, snap, purch
    """
    conn = make_sqlserver_conn(cfg)
    try:
        dim_date = load_dim_date(conn, dwh=dwh)
        dim_item = load_dim_item(conn, dwh=dwh)
        fact_sales = load_fact_sales(conn, dwh=dwh)
        snap = load_inventory_snapshot(conn, dwh=dwh)
        purch = load_purchasing(conn, dwh=dwh)
    finally:
        conn.close()

    return dim_date, dim_item, fact_sales, snap, purch
