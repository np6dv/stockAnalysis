#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fibonacci-based stock analyzer (daily):
- Detects the latest up leg using pivot highs/lows (windowed approach).
- Computes Fibonacci retracement & extension levels from that leg.
- Outputs suggested entry, stop, and target prices based on Fibonacci levels.
- Optional plot of levels.

Usage examples:
  python fib_strategy.py AAPL --start 2018-01-01 --entry_level 0.618 --stop_level 0.786 --target_ext 1.618 --plot
  python fib_strategy.py MSFT --period 5y --entry_level 0.5 --stop_at_swing_low --target_ext 1.272 --json

Notes:
- This script focuses on LONG setups (up-swing leg). If no recent up leg exists,
  it will report accordingly. You can enable --allow_down_leg to compute on a down leg.
"""

import argparse
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Data fetch
# -----------------------------
def fetch_daily(ticker: str, start: Optional[str], end: Optional[str], period: Optional[str]) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    if start or end:
        df = t.history(start=start, end=end, interval="1d", actions=False)
    else:
        df = t.history(period=period or "max", interval="1d", actions=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}. Check symbol or date range.")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna()
    return df


# -----------------------------
# Pivot detection & leg selection
# -----------------------------
def mark_pivots(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.DataFrame:
    """
    Marks pivot highs/lows using a symmetric window: a pivot occurs if the center bar
    is the max (or min) of [i-left, i+right].
    """
    df = df.copy()
    n = len(df)
    piv_h = np.zeros(n, dtype=bool)
    piv_l = np.zeros(n, dtype=bool)
    highs = df['high'].values
    lows = df['low'].values

    for i in range(left, n - right):
        hs = highs[i-left:i+right+1]
        ls = lows[i-left:i+right+1]
        if highs[i] == hs.max() and np.argmax(hs) == left:
            piv_h[i] = True
        if lows[i] == ls.min() and np.argmin(ls) == left:
            piv_l[i] = True

    df['pivot_high'] = piv_h
    df['pivot_low'] = piv_l
    return df


@dataclass
class SwingLeg:
    direction: str  # 'up' or 'down'
    start_idx: int
    end_idx: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    start_price: float
    end_price: float
    length: float  # absolute price move


def extract_pivot_series(df: pd.DataFrame) -> List[Tuple[int, str, float]]:
    """
    Returns a time-ordered list of (index, 'high'|'low', price) for all pivots.
    """
    out = []
    for i, is_h in enumerate(df['pivot_high'].values):
        if is_h:
            out.append((i, 'high', float(df['high'].iloc[i])))
    for i, is_l in enumerate(df['pivot_low'].values):
        if is_l:
            out.append((i, 'low', float(df['low'].iloc[i])))
    out.sort(key=lambda x: x[0])
    return out


def find_latest_up_leg(df: pd.DataFrame, pivots: List[Tuple[int, str, float]]) -> Optional[SwingLeg]:
    """
    Finds the most recent completed UP leg (pivot low -> later pivot high).
    Returns None if not found.
    """
    # We look from the end for a 'high' pivot, then find the nearest previous 'low' pivot before it.
    for k in range(len(pivots)-1, -1, -1):
        idx, typ, price = pivots[k]
        if typ != 'high':
            continue
        # find previous low before this high
        for j in range(k-1, -1, -1):
            idx_low, typ_low, price_low = pivots[j]
            if typ_low == 'low' and idx_low < idx:
                return SwingLeg(
                    direction='up',
                    start_idx=idx_low,
                    end_idx=idx,
                    start_date=df['date'].iloc[idx_low],
                    end_date=df['date'].iloc[idx],
                    start_price=float(df['low'].iloc[idx_low]),
                    end_price=float(df['high'].iloc[idx]),
                    length=float(df['high'].iloc[idx] - df['low'].iloc[idx_low])
                )
    return None


def find_latest_down_leg(df: pd.DataFrame, pivots: List[Tuple[int, str, float]]) -> Optional[SwingLeg]:
    """
    Finds the most recent completed DOWN leg (pivot high -> later pivot low).
    """
    for k in range(len(pivots)-1, -1, -1):
        idx, typ, price = pivots[k]
        if typ != 'low':
            continue
        # find previous high before this low
        for j in range(k-1, -1, -1):
            idx_high, typ_high, price_high = pivots[j]
            if typ_high == 'high' and idx_high < idx:
                return SwingLeg(
                    direction='down',
                    start_idx=idx_high,
                    end_idx=idx,
                    start_date=df['date'].iloc[idx_high],
                    end_date=df['date'].iloc[idx],
                    start_price=float(df['high'].iloc[idx_high]),
                    end_price=float(df['low'].iloc[idx]),
                    length=float(df['high'].iloc[idx_high] - df['low'].iloc[idx])
                )
    return None


# -----------------------------
# Fibonacci levels
# -----------------------------
def fib_retracements(low: float, high: float) -> dict:
    """
    Returns standard retracement levels between a swing low->high (up leg).
    """
    diff = high - low
    return {
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.500": low + 0.500 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
        "1.000": low,  # full retrace to the swing low
    }


def fib_extensions(low: float, high: float) -> dict:
    """
    Returns common extensions above the swing high for an up leg.
    """
    diff = high - low
    return {
        "1.000": high,
        "1.272": high + 1.272 * diff,
        "1.618": high + 1.618 * diff,
        "2.000": high + 2.000 * diff,
    }


# -----------------------------
# Trade suggestion (LONG)
# -----------------------------
@dataclass
class FibTradePlan:
    entry_level: str
    entry_price: float
    stop_basis: str
    stop_price: float
    target_ext: str
    target_price: float
    rr: Optional[float]  # Reward-to-Risk ratio (target vs stop)


def build_long_trade_from_leg(
    leg: SwingLeg,
    prefer_entry_level: float = 0.618,
    stop_level: Optional[float] = 0.786,
    stop_at_swing_low: bool = False,
    target_extension: float = 1.618
) -> FibTradePlan:
    """
    Given an up leg, compute fib levels and generate a long trade plan.
    """
    low, high = leg.start_price, leg.end_price
    rets = fib_retracements(low, high)
    exts = fib_extensions(low, high)

    # Entry price:
    # Use 0.618 by default; if prefer_entry_level not available (edge), fallback to 0.5
    entry_key = f"{prefer_entry_level:.3f}".rstrip('0').rstrip('.') if prefer_entry_level not in (1.0,) else "1.000"
    if entry_key not in rets:
        entry_key = "0.500"
    entry_price = float(rets[entry_key])

    # Stop price:
    if stop_at_swing_low:
        stop_basis = "swing_low"
        stop_price = float(low)
    else:
        stop_key = f"{stop_level:.3f}".rstrip('0').rstrip('.') if stop_level not in (1.0,) else "1.000"
        if stop_key not in rets:
            stop_key = "0.786"
        stop_basis = f"fib_{stop_key}"
        stop_price = float(rets[stop_key])

    # Target:
    tgt_key = f"{target_extension:.3f}".rstrip('0').rstrip('.') if target_extension not in (1.0,) else "1.000"
    if tgt_key not in exts:
        tgt_key = "1.618"
    target_price = float(exts[tgt_key])

    # Reward-Risk:
    rr = None
    if entry_price > 0 and target_price > entry_price and entry_price > stop_price:
        rr = (target_price - entry_price) / (entry_price - stop_price)

    return FibTradePlan(
        entry_level=entry_key,
        entry_price=round(entry_price, 4),
        stop_basis=stop_basis,
        stop_price=round(stop_price, 4),
        target_ext=tgt_key,
        target_price=round(target_price, 4),
        rr=round(rr, 2) if rr is not None else None
    )


# -----------------------------
# Plotting
# -----------------------------
# -----------------------------
# Main analysis function
# -----------------------------
def analyze_fibonacci(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "10y",
    left: int = 5,
    right: int = 5,
    entry_level: float = 0.618,
    stop_level: float = 0.786,
    stop_at_swing_low: bool = False,
    target_ext: float = 1.618,
    allow_down_leg: bool = False,
    plot: bool = False,
):
    df = fetch_daily(ticker, start, end, period)
    df = mark_pivots(df, left=left, right=right)
    pivs = extract_pivot_series(df)

    if not pivs:
        raise ValueError("No pivots detected—try reducing left/right windows or expanding the date range.")

    up_leg = find_latest_up_leg(df, pivs)
    selected_leg = up_leg

    if selected_leg is None and allow_down_leg:
        # Optionally allow a down leg (e.g., for educational view or short logic)
        down_leg = find_latest_down_leg(df, pivs)
        selected_leg = down_leg

    if selected_leg is None:
        return {
            "ticker": ticker.upper(),
            "message": "No recent completed up leg found; cannot compute Fibonacci long setup.",
            "params": {
                "left": left, "right": right, "entry_level": entry_level, "stop_level": stop_level,
                "stop_at_swing_low": stop_at_swing_low, "target_ext": target_ext, "allow_down_leg": allow_down_leg
            }
        }

    if selected_leg.direction != 'up' and not allow_down_leg:
        return {
            "ticker": ticker.upper(),
            "message": "Latest completed leg is DOWN and allow_down_leg=False. No long setup emitted.",
            "leg": {
                "direction": selected_leg.direction,
                "start_date": str(selected_leg.start_date),
                "end_date": str(selected_leg.end_date),
                "start_price": selected_leg.start_price,
                "end_price": selected_leg.end_price,
            }
        }

    # Build long plan from the up leg
    plan = build_long_trade_from_leg(
        selected_leg,
        prefer_entry_level=entry_level,
        stop_level=stop_level,
        stop_at_swing_low=stop_at_swing_low,
        target_extension=target_ext
    )

    # Current price context
    last_close = float(df['close'].iloc[-1])
    context = {
        "last_date": str(df['date'].iloc[-1]),
        "last_close": round(last_close, 4),
        "distance_to_entry_pct": round(100 * (plan.entry_price / last_close - 1), 2),
        "distance_to_stop_pct": round(100 * (plan.stop_price / plan.entry_price - 1), 2) if plan.entry_price != 0 else None,
        "distance_to_target_pct": round(100 * (plan.target_price / plan.entry_price - 1), 2) if plan.entry_price != 0 else None,
    }

    result = {
        "ticker": ticker.upper(),
        "swing_leg": {
            "direction": selected_leg.direction,
            "start_date": str(selected_leg.start_date),
            "start_price": round(selected_leg.start_price, 4),
            "end_date": str(selected_leg.end_date),
            "end_price": round(selected_leg.end_price, 4),
            "length": round(selected_leg.length, 4),
        },
        "fibonacci": {
            "retracements": {k: round(v, 4) for k, v in fib_retracements(selected_leg.start_price, selected_leg.end_price).items()},
            "extensions":   {k: round(v, 4) for k, v in fib_extensions(selected_leg.start_price, selected_leg.end_price).items()},
        },
        "trade_plan": {
            "entry_level": plan.entry_level,
            "entry_price": plan.entry_price,
            "stop_basis": plan.stop_basis,
            "stop_price": plan.stop_price,
            "target_ext": plan.target_ext,
            "target_price": plan.target_price,
            "reward_risk": plan.rr
        },
        "context": context,
        "params": {
            "left": left, "right": right,
            "entry_level": entry_level,
            "stop_level": stop_level,
            "stop_at_swing_low": stop_at_swing_low,
            "target_ext": target_ext,
            "allow_down_leg": allow_down_leg
        },
        "disclaimer": "For research/education only. Not financial advice."
    }


    return result


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fibonacci-based stock analyzer (daily, long setups)")
    p.add_argument("ticker", type=str, help="Ticker symbol (e.g., AAPL)")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    p.add_argument("--period", type=str, default="10y", help="Yahoo period if start/end not provided (e.g., 1y, 5y, 10y, max)")

    p.add_argument("--left", type=int, default=5, help="Pivot left window")
    p.add_argument("--right", type=int, default=5, help="Pivot right window")

    p.add_argument("--entry_level", type=float, default=0.618, help="Entry fib retracement (e.g., 0.5, 0.618)")
    p.add_argument("--stop_level", type=float, default=0.786, help="Stop fib retracement (ignored if --stop_at_swing_low)")
    p.add_argument("--stop_at_swing_low", action="store_true", help="Use swing low as the stop instead of a fib level")
    p.add_argument("--target_ext", type=float, default=1.618, help="Target fib extension (e.g., 1.272, 1.618)")

    p.add_argument("--allow_down_leg", action="store_true", help="Allow computing on a down leg (for inspection)")

    p.add_argument("--plot", action="store_true", help="Plot price with fib levels")
    p.add_argument("--json", action="store_true", help="Print JSON result")

    return p.parse_args()


def main():
    args = parse_args()
    res = analyze_fibonacci(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        period=args.period,
        left=args.left,
        right=args.right,
        entry_level=args.entry_level,
        stop_level=args.stop_level,
        stop_at_swing_low=args.stop_at_swing_low,
        target_ext=args.target_ext,
        allow_down_leg=args.allow_down_leg,
        plot=args.plot,
    )

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"\n== {res.get('ticker', '')} Fibonacci Analysis ==")
        if 'message' in res:
            print(res['message'])
            return
        leg = res['swing_leg']
        plan = res['trade_plan']
        ctx = res['context']

        print(f"Swing: {leg['direction']}  {leg['start_date']} @ {leg['start_price']}  "
              f"->  {leg['end_date']} @ {leg['end_price']}  (Δ={leg['length']})")
        print("\nSuggested LONG plan (Fibonacci):")
        print(f"  Entry (fib {plan['entry_level']}): {plan['entry_price']}")
        print(f"  Stop  ({plan['stop_basis']}): {plan['stop_price']}")
        print(f"  Target (ext {plan['target_ext']}): {plan['target_price']}")
        print(f"  Reward:Risk ≈ {plan['reward_risk']}")
        print("\nContext:")
        print(f"  Last close ({ctx['last_date']}): {ctx['last_close']}")
        print(f"  Distance to stop (from entry): {ctx['distance_to_stop_pct']}%")
        print(f"  Distance to target (from entry): {ctx['distance_to_target_pct']}%")
        print("\n(disclaimer) For research/education only. Not financial advice.")


if __name__ == "__main__":
    main()

