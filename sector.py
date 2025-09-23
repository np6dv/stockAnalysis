#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sector Rotation Screener (daily):
- Detects money rotation between sectors using relative strength vs a benchmark.
- Outputs top 'inflow' vs 'outflow' sectors and suggested rotation pairs.
- Optional plot: RS (level) vs RS momentum (slope), a simplified RRG-style view.

Usage examples:
  python sector_rotation_screener.py
  python sector_rotation_screener.py --period 2y --slope_window 30 --plot
  python sector_rotation_screener.py --benchmark XIU.TO --plot  # Canadian benchmark (TSX 60)
  python sector_rotation_screener.py --custom_file sectors.csv --benchmark RSP --json --out rotation.csv
  python sector.py --slope_window 10
  
CSV format for --custom_file:
symbol,sector
XLY,Consumer Discretionary
XLP,Consumer Staples
XLK,Technology
... etc.

Dependencies:
  pip install yfinance pandas numpy matplotlib
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# -----------------------
# Defaults: SPDR sectors
# -----------------------
DEFAULT_SECTORS = {
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLK": "Technology",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
    "XLU": "Utilities",
}
DEFAULT_BENCH = "SPY"  # S&P 500


# -----------------------
# Utils
# -----------------------
def pct_change(series: pd.Series, window: int) -> float:
    if len(series) < window + 1:
        return np.nan
    return float(series.iloc[-1] / series.iloc[-window - 1] - 1.0)


def linreg_slope(y: np.ndarray) -> float:
    """
    Return the slope of a simple OLS fit y ~ a + b*t, t = 0..n-1.
    We use slope on log(RS) to approximate momentum (scale ~ per bar).
    """
    n = len(y)
    if n < 5 or np.any(~np.isfinite(y)):
        return np.nan
    t = np.arange(n, dtype=float)
    # y = a + b t
    b = np.polyfit(t, y, 1)[0]
    return float(b)


def zscore_last(values: pd.Series, window: int = 60) -> float:
    v = values.tail(window)
    if v.size < 10:
        return np.nan
    return float((v.iloc[-1] - v.mean()) / (v.std(ddof=0) + 1e-12))


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).replace([np.inf, -np.inf], np.nan)


def annualize_daily(slope_per_bar: float, bars_per_year: int = 252) -> float:
    """Convert a per-bar linear slope on log(RS) into an approximate annualized % drift."""
    if not np.isfinite(slope_per_bar):
        return np.nan
    # log(RS_t) = log(RS_0) + slope * t  => RS grows ~ exp(slope * t)
    return float(np.expm1(slope_per_bar * bars_per_year))


@dataclass
class SectorMetrics:
    symbol: str
    sector: str
    close: float
    rs_level: float
    rs_slope: float
    rs_slope_ann_pct: float
    ret_1w: float
    ret_1m: float
    ret_3m: float
    vol_z60: float
    sma50_above_sma200: Optional[bool] = None


# -----------------------
# Data
# -----------------------
def load_custom_mapping(path: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    # Accept columns: symbol, sector
    if "symbol" not in df.columns or "sector" not in df.columns:
        # Try to infer if first col is symbol, second is sector
        if df.shape[1] >= 2:
            df.columns = ["symbol", "sector"] + list(df.columns[2:])
        else:
            raise ValueError("CSV must have columns: symbol,sector")
    mapping = {}
    for _, row in df.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        sector = str(row["sector"]).strip()
        if symbol and sector:
            mapping[symbol] = sector
    if not mapping:
        raise ValueError("No (symbol, sector) mappings found in custom file.")
    return mapping


def download(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """
    Returns a long-form DataFrame with MultiIndex columns if multiple tickers.
    We'll split per ticker after.
    """
    df = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="ticker"
    )
    return df


def to_panel(df: pd.DataFrame, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Convert yf.download output to per-ticker DataFrames with columns:
    ['date','open','high','low','close','adj_close','volume']
    """
    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if t not in df.columns.levels[0]:
                continue
            sub = df[t].dropna().reset_index()
            sub.columns = [c.lower() if isinstance(c, str) else "date" for c in sub.columns]
            if "adj close" in sub.columns:
                sub = sub.rename(columns={"adj close": "adj_close"})
            out[t] = sub[["date", "open", "high", "low", "close", "adj_close", "volume"]]
    else:
        # Single ticker
        sub = df.dropna().reset_index()
        sub.columns = [c.lower() for c in sub.columns]
        sub = sub.rename(columns={"adj close": "adj_close"})
        t = tickers[0]
        out[t] = sub[["date", "open", "high", "low", "close", "adj_close", "volume"]]
    return out


# -----------------------
# Core logic
# -----------------------
def compute_sector_metrics(
    price_panels: Dict[str, pd.DataFrame],
    mapping: Dict[str, str],
    benchmark: str,
    slope_window: int,
    ret_windows: Tuple[int, int, int] = (5, 21, 63)
) -> List[SectorMetrics]:

    if benchmark not in price_panels:
        raise ValueError(f"Benchmark {benchmark} not found in data.")

    bench = price_panels[benchmark].copy()
    bench = bench.dropna()
    bench.set_index("date", inplace=True)

    metrics: List[SectorMetrics] = []

    for sym, name in mapping.items():
        if sym == benchmark:
            continue
        if sym not in price_panels:
            continue
        df = price_panels[sym].dropna().copy()
        df.set_index("date", inplace=True)

        # Align on benchmark dates
        aligned = df.join(bench[["adj_close", "volume"]].rename(columns={
            "adj_close": "bench_adj", "volume": "bench_vol"
        }), how="inner")

        if aligned.shape[0] < max(80, slope_window + 10):
            continue

        # Relative strength = sector / benchmark (adj_close)
        aligned["rs"] = safe_div(aligned["adj_close"], aligned["bench_adj"])

        # RS slope on log(RS) to capture geometric drift
        aligned["log_rs"] = np.log(aligned["rs"])
        rs_slope = linreg_slope(aligned["log_rs"].tail(slope_window).values)

        # Annualize slope for readability
        ann_pct = annualize_daily(rs_slope)

        # Returns
        r1, r2, r3 = ret_windows
        ret_1w = pct_change(aligned["adj_close"], r1)
        ret_1m = pct_change(aligned["adj_close"], r2)
        ret_3m = pct_change(aligned["adj_close"], r3)

        # Volume z-score (attention shift)
        volz = zscore_last(aligned["volume"], window=60)

        # Simple trend (50>200)
        aligned["sma50"] = aligned["adj_close"].rolling(50).mean()
        aligned["sma200"] = aligned["adj_close"].rolling(200).mean()
        sma_state = bool(aligned["sma50"].iloc[-1] > aligned["sma200"].iloc[-1]) if aligned.shape[0] >= 200 else None

        metrics.append(SectorMetrics(
            symbol=sym,
            sector=name,
            close=float(aligned["close"].iloc[-1]),
            rs_level=float(aligned["rs"].iloc[-1]),
            rs_slope=float(rs_slope),
            rs_slope_ann_pct=float(ann_pct) if np.isfinite(ann_pct) else np.nan,
            ret_1w=float(ret_1w) if np.isfinite(ret_1w) else np.nan,
            ret_1m=float(ret_1m) if np.isfinite(ret_1m) else np.nan,
            ret_3m=float(ret_3m) if np.isfinite(ret_3m) else np.nan,
            vol_z60=float(volz) if np.isfinite(volz) else np.nan,
            sma50_above_sma200=sma_state,
        ))

    return metrics


def infer_rotation(metrics: List[SectorMetrics], top_k: int = 3) -> Dict[str, List[Dict]]:
    """
    Identify inflows (stronger RS momentum + positive 1m) and outflows (weaker RS momentum + negative 1m).
    Return rotation pairs from each 'out' to each 'in' with a simple spread score.
    """
    df = pd.DataFrame([m.__dict__ for m in metrics])
    if df.empty:
        return {"inflows": [], "outflows": [], "pairs": []}

    # Ranking by RS slope (momentum of RS)
    df["rank_rs_slope"] = df["rs_slope"].rank(ascending=False, method="dense")
    df["rank_ret_1m"] = df["ret_1m"].rank(ascending=False, method="dense")

    # Inflows: top RS momentum + positive 1m return
    inflows = df[(df["rs_slope"] > 0) & (df["ret_1m"] > 0)].sort_values(
        by=["rs_slope", "ret_1m"], ascending=False
    ).head(top_k)

    # Outflows: bottom RS momentum + negative 1m return
    outflows = df[(df["rs_slope"] < 0) & (df["ret_1m"] < 0)].sort_values(
        by=["rs_slope", "ret_1m"], ascending=[True, True]
    ).head(top_k)

    # Make pairwise suggestions (from each out → each in)
    pairs = []
    for _, row_out in outflows.iterrows():
        for _, row_in in inflows.iterrows():
            spread = (row_in["rs_slope"] - row_out["rs_slope"]) + 0.5 * (row_in["ret_1m"] - row_out["ret_1m"])
            pairs.append({
                "from_symbol": row_out["symbol"], "from_sector": row_out["sector"],
                "to_symbol": row_in["symbol"], "to_sector": row_in["sector"],
                "spread_score": round(float(spread), 6)
            })
    pairs = sorted(pairs, key=lambda x: x["spread_score"], reverse=True)

    return {
        "inflows": inflows[["symbol", "sector", "rs_slope_ann_pct", "ret_1w", "ret_1m", "ret_3m", "vol_z60"]]
                    .round(4).to_dict(orient="records"),
        "outflows": outflows[["symbol", "sector", "rs_slope_ann_pct", "ret_1w", "ret_1m", "ret_3m", "vol_z60"]]
                    .round(4).to_dict(orient="records"),
        "pairs": pairs[: max(1, top_k * 2)]
    }


def build_table(metrics: List[SectorMetrics]) -> pd.DataFrame:
    df = pd.DataFrame([m.__dict__ for m in metrics])
    if df.empty:
        return df
    # Sort by RS slope (descending)
    cols_order = [
        "symbol", "sector", "close", "rs_level", "rs_slope", "rs_slope_ann_pct",
        "ret_1w", "ret_1m", "ret_3m", "vol_z60", "sma50_above_sma200"
    ]
    df = df[cols_order].sort_values(by="rs_slope", ascending=False)
    return df


def plot_rrg(df: pd.DataFrame, title: str):
    """
    Simplified RRG-like scatter: x = RS level (last), y = RS momentum (slope annualized %).
    """
    if df.empty:
        print("[WARN] Nothing to plot.")
        return
    plt.figure(figsize=(10, 6))
    x = df["rs_level"].values
    y = 100 * df["rs_slope_ann_pct"].values  # percentage points
    for _, r in df.iterrows():
        plt.scatter(r["rs_level"], 100 * r["rs_slope_ann_pct"], s=90)
        plt.text(r["rs_level"] * 1.001, 100 * r["rs_slope_ann_pct"] * 1.001, r["symbol"], fontsize=9)

    # Axes lines at medians to form quadrants (Leading/Weakening/Improving/Lagging)
    plt.axvline(np.nanmedian(x), color="gray", linestyle="--", alpha=0.5)
    plt.axhline(np.nanmedian(y), color="gray", linestyle="--", alpha=0.5)
    plt.title(title)
    plt.xlabel("RS Level (Sector / Benchmark)")
    plt.ylabel("RS Momentum (Annualized %, approx) ×100")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Sector Rotation Screener (relative strength vs benchmark)")
    p.add_argument("--period", type=str, default="2y", help="History period (e.g., 1y, 2y, 5y, max)")
    p.add_argument("--benchmark", type=str, default=DEFAULT_BENCH, help="Benchmark ticker (e.g., SPY, RSP, XIU.TO)")
    p.add_argument("--custom_file", type=str, default=None, help="CSV with columns: symbol,sector to override default sectors")
    p.add_argument("--slope_window", type=int, default=30, help="Lookback window (days) for RS slope")
    p.add_argument("--ret_1w", type=int, default=5, help="1-week lookback (trading days)")
    p.add_argument("--ret_1m", type=int, default=21, help="1-month lookback (trading days)")
    p.add_argument("--ret_3m", type=int, default=63, help="3-month lookback (trading days)")
    p.add_argument("--top_k", type=int, default=3, help="Number of inflow/outflow sectors to highlight")
    p.add_argument("--out", type=str, default=None, help="Save table CSV")
    p.add_argument("--json", action="store_true", help="Print JSON summary")
    p.add_argument("--plot", action="store_true", help="Show simplified RRG scatter plot")
    return p.parse_args()


def main():
    args = parse_args()

    # Sector mapping
    mapping = load_custom_mapping(args.custom_file) if args.custom_file else DEFAULT_SECTORS.copy()

    tickers = sorted(list(mapping.keys() | {args.benchmark}))
    data = download(tickers, period=args.period)
    panels = to_panel(data, tickers)

    # Compute metrics
    metrics = compute_sector_metrics(
        panels, mapping, benchmark=args.benchmark,
        slope_window=args.slope_window,
        ret_windows=(args.ret_1w, args.ret_1m, args.ret_3m)
    )
    table = build_table(metrics)

    # Show table
    if table.empty:
        print("[INFO] No data/metrics to display. Check tickers, period, or connectivity.")
        return

    with pd.option_context("display.max_columns", None, "display.width", 140, "display.float_format", "{:.4f}".format):
        print("\n== Sector Rotation (relative to {}) ==".format(args.benchmark))
        print(table)

    # Rotation inferences
    rotation = infer_rotation(metrics, top_k=args.top_k)

    print("\n== Inflow Sectors (rising RS + positive 1M ret) ==")
    if rotation["inflows"]:
        print(pd.DataFrame(rotation["inflows"]))
    else:
        print("None")

    print("\n== Outflow Sectors (falling RS + negative 1M ret) ==")
    if rotation["outflows"]:
        print(pd.DataFrame(rotation["outflows"]))
    else:
        print("None")

    print("\n== Suggested Rotation Pairs (from → to) ==")
    if rotation["pairs"]:
        for p in rotation["pairs"]:
            print(f"{p['from_symbol']} ({p['from_sector']})  →  {p['to_symbol']} ({p['to_sector']})  | score={p['spread_score']}")
    else:
        print("None")

    # Save CSV
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"[INFO] Saved table to: {args.out}")

    # JSON summary if requested
    if args.json:
        payload = {
            "benchmark": args.benchmark,
            "period": args.period,
            "slope_window": args.slope_window,
            "table": table.round(6).to_dict(orient="records"),
            "rotation": rotation,
        }
        print(json.dumps(payload, indent=2))

    # Plot
    if args.plot:
        plot_rrg(table, f"Sector RS vs RS Momentum • Benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
