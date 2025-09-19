#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long-only, regime-aware trend/pullback strategy with tighter entries:
- Daily persistence confirmation (require N prior days of qualifying conditions)
- 4-hour micro-trend confirmation (require K consecutive 4H bars in uptrend)
- Weekly regime filter (EMA20 + ADX)
- Normalized scoring (trend, momentum, volume, structure, analyst)
- ATR initial stop + Chandelier trailing stop
- Vectorized backtest loop with slippage, risk-based position sizing, and metrics

Usage examples:
  python long_only_tight_entries.py AAPL --start 2015-01-01 --confirm_days 1 --entry 0.65 --exit 0.20
  python long_only_tight_entries.py MSFT --start 2019-01-01 --micro_4h_bars 2 --entry 0.65 --exit 0.20
  python long_only_tight_entries.py NVDA --start 2019-01-01 --confirm_days 1 --micro_4h_bars 2 --json --save

Notes:
- Yahoo may limit intraday history length. If 4H confirmation is requested but
  intraday data is missing, behavior is controlled by --intraday_fail_hard.
- Default behavior acts on the same day's close using information that is
  confirmation-shifted (no lookahead).
"""

import sys
import argparse
import math
import json
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import matplotlib.pyplot as plt


# =========================
# Data helpers
# =========================
def fetch_ohlcv(ticker: str, start: str = None, end: str = None, period: str = "max", interval: str = "1d"):
    t = yf.Ticker(ticker)
    if start or end:
        df = t.history(start=start, end=end, interval=interval, actions=False)
    else:
        df = t.history(period=period, interval=interval, actions=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} (interval={interval}).")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna()
    return df


def get_analyst_predictions(ticker: str):
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "target_mean": float(info.get("targetMeanPrice", np.nan)),
            "target_high": float(info.get("targetHighPrice", np.nan)),
            "target_low": float(info.get("targetLowPrice", np.nan)),
            "number_of_analysts": int(info.get("numberOfAnalystOpinions", 0)),
            "recommendation": info.get("recommendationKey", "none"),
        }
    except Exception:
        return {
            "target_mean": np.nan,
            "target_high": np.nan,
            "target_low": np.nan,
            "number_of_analysts": 0,
            "recommendation": "none",
        }


# =========================
# Daily indicators & structure
# =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Trend & volatility
    df['sma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx.adx()

    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()
    df['atr_pct'] = df['atr'] / df['close']

    # Momentum
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # Volume / flow
    df['vol_ma60'] = df['volume'].rolling(60).mean()
    df['vol_std60'] = df['volume'].rolling(60).std(ddof=0)
    df['volume_z60'] = (df['volume'] - df['vol_ma60']) / df['vol_std60'].replace(0, np.nan)

    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_slope'] = df['obv'].diff(5)

    cmf = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20)
    df['cmf'] = cmf.chaikin_money_flow()

    # Structure (pivots + nearest S/R distances in ATRs)
    df = mark_pivots(df, left=3, right=3)
    df = nearest_sr_levels(df, lookback=120)

    # Normalize MACD histogram (avoid lookahead later by shifting)
    hist_std = df['macd_hist'].rolling(100).std(ddof=0)
    df['macd_hist_norm'] = np.tanh((df['macd_hist'] / hist_std.replace(0, np.nan)).fillna(0))

    return df


def mark_pivots(df: pd.DataFrame, left=3, right=3) -> pd.DataFrame:
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


def nearest_sr_levels(df: pd.DataFrame, lookback=120) -> pd.DataFrame:
    df = df.copy()
    supports, resistances = [], []
    for i in range(len(df)):
        start = max(0, i - lookback)
        sub = df.iloc[start:i+1]
        sup_levels = sub.loc[sub['pivot_low'], 'low']
        res_levels = sub.loc[sub['pivot_high'], 'high']
        c = df['close'].iloc[i]

        nearest_sup = sup_levels.iloc[(sup_levels - c).abs().argsort()].iloc[0] if not sup_levels.empty else np.nan
        nearest_res = res_levels.iloc[(res_levels - c).abs().argsort()].iloc[0] if not res_levels.empty else np.nan

        supports.append(nearest_sup)
        resistances.append(nearest_res)

    df['nearest_support'] = supports
    df['nearest_resistance'] = resistances
    df['dist_to_support_atr'] = (df['close'] - df['nearest_support']) / df['atr']
    df['dist_to_res_atr'] = (df['nearest_resistance'] - df['close']) / df['atr']
    return df


# =========================
# Weekly regime filter -> daily map
# =========================
def weekly_regime_series(ticker: str, daily_df: pd.DataFrame) -> pd.Series:
    w = fetch_ohlcv(
        ticker,
        start=str(pd.to_datetime(daily_df['date']).min().date()),
        end=str(pd.to_datetime(daily_df['date']).max().date()),
        interval="1wk",
    )
    w['ema20'] = ta.trend.EMAIndicator(w['close'], window=20).ema_indicator()
    adxw = ta.trend.ADXIndicator(w['high'], w['low'], w['close'], window=14)
    w['adxw'] = adxw.adx()
    w['weekly_uptrend'] = (w['close'] > w['ema20']) & (w['adxw'] > 18)

    merged = pd.merge_asof(
        daily_df.sort_values('date'),
        w[['date', 'weekly_uptrend']].sort_values('date'),
        on='date', direction='backward'
    )
    return merged['weekly_uptrend'].fillna(False)


# =========================
# Score model (lookahead-safe via shift)
# =========================
def compute_score_row(row: pd.Series, analyst_norm: float, weights: dict) -> float:
    # Trend
    trend = 0.0
    adx = row.get('adx', np.nan)
    sma200 = row.get('sma200', np.nan)
    close = row.get('close', np.nan)
    if pd.notna(adx) and pd.notna(sma200) and pd.notna(close):
        trending = adx > 20
        if trending and close > sma200:
            trend = 1.0
        elif trending and close < sma200:
            trend = -1.0
        else:
            trend = 0.0

    # Momentum
    mom = row.get('macd_hist_norm', 0.0)
    rsi = row.get('rsi', np.nan)
    if np.isnan(rsi):
        rsi_contrib = 0.0
    else:
        if trend >= 0.5:
            rsi_contrib = np.tanh((50 - rsi) / 20.0)  # pullback favorable
        elif trend <= -0.5:
            rsi_contrib = -np.tanh((rsi - 50) / 20.0)
        else:
            rsi_contrib = 0.0
    momentum = 0.7 * mom + 0.3 * rsi_contrib

    # Volume
    cmf = row.get('cmf', 0.0)
    obv_slope = row.get('obv_slope', 0.0)
    vol = 0.6 * np.tanh((cmf or 0.0) * 2.0) + 0.4 * np.tanh(obv_slope / (abs(obv_slope) + 1e-9))

    # Structure
    struct = 0.0
    d_sup = row.get('dist_to_support_atr', np.nan)
    d_res = row.get('dist_to_res_atr', np.nan)
    if pd.notna(d_sup):
        struct += np.tanh((1.0 - d_sup))
    if pd.notna(d_res):
        struct -= np.tanh((1.0 - d_res))
    struct = float(np.clip(struct, -1.0, 1.0))

    # Analyst
    analyst_comp = float(analyst_norm)

    score = (
        weights['trend'] * trend +
        weights['momentum'] * momentum +
        weights['volume'] * vol +
        weights['structure'] * struct +
        weights['analyst'] * analyst_comp
    )
    return float(score)


def compute_scores(df: pd.DataFrame, analyst: dict, weights: dict) -> pd.Series:
    current_price = float(df['close'].iloc[-1])
    tm = analyst.get('target_mean', np.nan)
    na = analyst.get('number_of_analysts', 0) or 0
    if pd.notna(tm) and current_price > 0:
        analyst_norm = np.clip((tm / current_price - 1.0), -0.2, 0.2) / 0.2  # [-1,1]
        analyst_norm *= np.log1p(na) / np.log1p(20)  # downweight if few analysts
    else:
        analyst_norm = 0.0
    # Use prior-day indicators for today's decision (lookahead-safe)
    df_sig = df.shift(1)
    scores = df_sig.apply(lambda row: compute_score_row(row, analyst_norm, weights), axis=1)
    return scores


# =========================
# Intraday (60m -> 4H) + micro-trend confirmation
# =========================
def fetch_intraday_60m(ticker: str, start: str, end: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    intr = t.history(start=start, end=end, interval="60m", actions=False)
    if intr.empty:
        return pd.DataFrame()
    intr = intr.reset_index()
    intr.columns = [c.lower() for c in intr.columns]
    intr = intr[['date', 'open', 'high', 'low', 'close', 'volume']].dropna()
    return intr


def resample_to_4h(df60: pd.DataFrame) -> pd.DataFrame:
    if df60.empty:
        return df60
    df60 = df60.copy()
    df60['date'] = pd.to_datetime(df60['date'])
    df60 = df60.set_index('date').sort_index()
    ohlc = df60.resample('4H').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    ohlc = ohlc.reset_index()
    return ohlc


def add_4h_trend(df4h: pd.DataFrame) -> pd.DataFrame:
    """Simple 4H micro-trend: price > EMA50 and ADX>18."""
    if df4h.empty:
        return df4h
    df4h = df4h.copy()
    ema50 = ta.trend.EMAIndicator(df4h['close'], window=50)
    df4h['ema50'] = ema50.ema_indicator()
    adx4 = ta.trend.ADXIndicator(df4h['high'], df4h['low'], df4h['close'], window=14)
    df4h['adx4h'] = adx4.adx()
    df4h['micro_up'] = (df4h['close'] > df4h['ema50']) & (df4h['adx4h'] > 18)
    df4h['micro_up_int'] = df4h['micro_up'].astype(int)
    return df4h
def map_4h_confirmation_to_daily(df_daily: pd.DataFrame, df4h: pd.DataFrame, micro_bars: int = 1) -> pd.Series:
    """
    For each daily bar, check if the last `micro_bars` 4H bars (ending before the daily timestamp)
    all satisfied `micro_up`.
    """
    if df4h.empty:
        return pd.Series(False, index=df_daily.index)

    df4h = df4h.copy()
    df4h['date'] = pd.to_datetime(df4h['date'])
    df4h = df4h.sort_values('date')
    # "consec" = last micro_bars 4H bars all True, evaluated on each 4H bar close
    df4h['consec'] = df4h['micro_up_int'].rolling(micro_bars).sum() == micro_bars

    daily = df_daily[['date']].copy()
    daily['date'] = pd.to_datetime(daily['date'])
    aligned = pd.merge_asof(
        daily.sort_values('date'),
        df4h[['date', 'consec']].sort_values('date'),
        on='date', direction='backward'
    )
    return aligned['consec'].fillna(False).values


# =========================
# Backtest (long-only) with tight entries
# =========================
def backtest_long_only(
    ticker: str,
    start: str = None,
    end: str = None,
    start_capital: float = 100_000.0,
    risk_per_trade: float = 0.01,
    entry_threshold: float = 0.65,
    exit_threshold: float = 0.20,
    atr_stop_mult: float = 2.5,
    chandelier_mult: float = 3.0,
    max_hold_days: int = 180,
    slippage_bps: float = 2.0,    # 2 bps = 0.02%
    fee_per_share: float = 0.0,

    # >>> Tight entry controls <<<
    confirm_days: int = 1,        # require N prior daily bars of qualifying conditions (0 to disable)
    exit_confirm_days: int = 0,   # require N prior daily bars before exiting (0 to disable)
    micro_4h_bars: int = 2,       # require K consecutive 4H bars in uptrend (0 to disable)
    intraday_fail_hard: bool = False,  # if True and intraday missing -> block entries; else allow
):
    # --- Build daily frame with indicators & regime ---
    df = fetch_ohlcv(ticker, start=start, end=end, interval="1d")
    df = add_indicators(df)
    df['weekly_uptrend'] = weekly_regime_series(ticker, df).astype(bool)

    analyst = get_analyst_predictions(ticker)
    weights = {"trend": 0.35, "momentum": 0.25, "volume": 0.20, "structure": 0.15, "analyst": 0.05}
    df['score'] = compute_scores(df, analyst, weights)

    # --- Hard filters (gates) ---
    df['hard_filter'] = (df['weekly_uptrend']) & (df['close'] > df['sma200']) & (df['adx'] > 18)

    # --- Raw entry/exit logic (pre-confirmation) ---
    enter_raw = (df['hard_filter']) & (df['score'] >= entry_threshold)
    exit_raw = ((df['score'] <= exit_threshold) | (~df['weekly_uptrend']))

    # --- Daily confirmation (persistence) ---
    if confirm_days > 0:
        df['enter_confirm_daily'] = enter_raw.shift(1).rolling(confirm_days).sum() == confirm_days
    else:
        df['enter_confirm_daily'] = enter_raw

    if exit_confirm_days > 0:
        df['exit_confirm_daily'] = exit_raw.shift(1).rolling(exit_confirm_days).sum() == exit_confirm_days
    else:
        df['exit_confirm_daily'] = exit_raw

    # --- Intraday 4H confirmation (micro-trend) ---
    if micro_4h_bars and micro_4h_bars > 0:
        start_str = str(pd.to_datetime(df['date']).min().date())
        end_str = str(pd.to_datetime(df['date']).max().date())
        intr60 = fetch_intraday_60m(ticker, start=start_str, end=end_str)
        if intr60.empty:
            if intraday_fail_hard:
                df['intraday_confirm'] = False
            else:
                # Allow entries but warn via field
                df['intraday_confirm'] = True
            intraday_note = "intraday_missing"
        else:
            df4h = resample_to_4h(intr60)
            df4h = add_4h_trend(df4h)
            mapped = map_4h_confirmation_to_daily(df, df4h, micro_bars=micro_4h_bars)
            df['intraday_confirm'] = mapped
            intraday_note = "intraday_ok"
    else:
        df['intraday_confirm'] = True
        intraday_note = "intraday_not_used"

    # --- Final entry/exit confirmation ---
    df['enter_confirm'] = df['enter_confirm_daily'] & df['intraday_confirm']
    df['exit_confirm'] = df['exit_confirm_daily']

    # =========================
    # Backtest loop
    # =========================
    cash = start_capital
    shares = 0
    position_entry_idx = None
    entry_price = np.nan
    entry_atr = np.nan
    highest_close_since_entry = -np.inf
    trailing_stop = np.nan
    equity_curve = []
    in_trade_days = 0
    exposure_days = 0
    trades = []

    slip_mult_buy = 1.0 + slippage_bps / 10000.0
    slip_mult_sell = 1.0 - slippage_bps / 10000.0

    for i in range(len(df)):
        row = df.iloc[i]
        date = row['date']
        close = float(row['close'])
        atr = float(row['atr']) if not np.isnan(row['atr']) else 0.0
        score = float(row['score']) if not np.isnan(row['score']) else 0.0

        # Mark-to-market equity
        equity = cash + shares * close
        equity_curve.append({"date": date, "equity": equity})
        if shares > 0:
            exposure_days += 1

        # Warmup check
        if np.isnan(row['sma200']) or np.isnan(row['adx']) or np.isnan(row['atr']):
            continue

        # Update trailing stop if in position
        if shares > 0:
            highest_close_since_entry = max(highest_close_since_entry, close)
            chand_stop = highest_close_since_entry - chandelier_mult * atr
            trailing_stop = max(trailing_stop, chand_stop)
            in_trade_days += 1

        # Exit logic
        should_exit = False
        if shares > 0:
            if df['exit_confirm'].iloc[i] or (close < trailing_stop) or (in_trade_days >= max_hold_days):
                should_exit = True

        if shares > 0 and should_exit:
            exit_price = close * slip_mult_sell
            proceeds = shares * exit_price
            fee = shares * fee_per_share
            cash += proceeds - fee
            pnl = (exit_price - entry_price) * shares - fee
            ret_pct = pnl / (entry_price * shares) if entry_price > 0 else 0.0

            trades.append({
                "entry_date": df.iloc[position_entry_idx]['date'],
                "exit_date": date,
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "shares": int(shares),
                "pnl": round(pnl, 2),
                "return_pct": round(100 * ret_pct, 2),
                "bars_held": in_trade_days
            })
            shares = 0
            position_entry_idx = None
            entry_price = np.nan
            entry_atr = np.nan
            highest_close_since_entry = -np.inf
            trailing_stop = np.nan
            in_trade_days = 0
            equity = cash
            equity_curve[-1]['equity'] = equity

        # Entry logic (long-only) with tight confirmation
        should_enter = False
        if shares == 0:
            if df['enter_confirm'].iloc[i]:
                should_enter = True

        if shares == 0 and should_enter:
            if atr <= 0:
                continue
            risk_per_share = atr_stop_mult * atr
            if risk_per_share <= 0:
                continue
            risk_budget = cash * risk_per_trade
            qty = math.floor(risk_budget / risk_per_share)
            if qty <= 0:
                continue

            buy_price = close * slip_mult_buy
            cost = qty * buy_price
            fee = qty * fee_per_share
            if cost + fee > cash:
                qty = math.floor((cash - fee) / buy_price)
                if qty <= 0:
                    continue
                cost = qty * buy_price

            cash -= (cost + fee)
            shares = qty
            position_entry_idx = i
            entry_price = buy_price
            entry_atr = atr
            highest_close_since_entry = close
            trailing_stop = entry_price - atr_stop_mult * atr
            in_trade_days = 0
            equity = cash + shares * close
            equity_curve[-1]['equity'] = equity

    # Final mark
    final_equity = cash + shares * df['close'].iloc[-1]
    equity_curve[-1]['equity'] = final_equity

    # Metrics
    eq_df = pd.DataFrame(equity_curve)
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df.set_index('date', inplace=True)
    eq_df['ret'] = eq_df['equity'].pct_change().fillna(0.0)
    total_days = len(eq_df)
    years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25 if total_days > 1 else np.nan
    cagr = (final_equity / start_capital) ** (1 / years) - 1 if years and years > 0 else np.nan
    sharpe = np.sqrt(252) * eq_df['ret'].mean() / (eq_df['ret'].std(ddof=0) + 1e-12)
    rolling_max = eq_df['equity'].cummax()
    drawdown = eq_df['equity'] / rolling_max - 1.0
    max_dd = drawdown.min() if not drawdown.empty else np.nan
    win_rate = (pd.Series([1 if t['pnl'] > 0 else 0 for t in trades]).mean() if trades else np.nan)
    exposure = exposure_days / total_days if total_days > 0 else np.nan

    metrics = {
        "start": str(df['date'].iloc[0].date()),
        "end": str(df['date'].iloc[-1].date()),
        "years": round(years, 2) if years == years else None,
        "start_capital": round(start_capital, 2),
        "final_equity": round(final_equity, 2),
        "CAGR": round(100 * cagr, 2) if cagr == cagr else None,
        "Sharpe": round(sharpe, 2) if sharpe == sharpe else None,
        "MaxDrawdownPct": round(100 * max_dd, 2) if max_dd == max_dd else None,
        "Trades": len(trades),
        "WinRatePct": round(100 * win_rate, 2) if win_rate == win_rate else None,
        "ExposurePct": round(100 * exposure, 2) if exposure == exposure else None
    }

    # Latest signal snapshot
    last = df.iloc[-1]
    latest = {
        "date": str(last['date']),
        "close": float(last['close']),
        "score": float(last['score']),
        "weekly_uptrend": bool(last['weekly_uptrend']),
        "hard_filter": bool(last['hard_filter']),
        "enter_raw": bool(enter_raw.iloc[-1]),
        "enter_confirm_daily": bool(df['enter_confirm_daily'].iloc[-1]),
        "intraday_confirm": bool(df['intraday_confirm'].iloc[-1]),
        "enter_confirm": bool(df['enter_confirm'].iloc[-1]),
        "action": ("ENTER_LONG" if df['enter_confirm'].iloc[-1] else "HOLD/WAIT")
    }

    result = {
        "ticker": ticker.upper(),
        "analyst": analyst,
        "params": {
            "risk_per_trade": risk_per_trade,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "atr_stop_mult": atr_stop_mult,
            "chandelier_mult": chandelier_mult,
            "max_hold_days": max_hold_days,
            "slippage_bps": slippage_bps,
            "fee_per_share": fee_per_share,
            "confirm_days": confirm_days,
            "exit_confirm_days": exit_confirm_days,
            "micro_4h_bars": micro_4h_bars,
            "intraday_fail_hard": intraday_fail_hard,
        },
        "intraday_status": intraday_note,
        "metrics": metrics,
        "latest_signal": latest,
        "last_row": {
            "rsi": float(last['rsi']) if not np.isnan(last['rsi']) else None,
            "macd": float(last['macd']) if not np.isnan(last['macd']) else None,
            "macd_signal": float(last['macd_signal']) if not np.isnan(last['macd_signal']) else None,
            "adx": float(last['adx']) if not np.isnan(last['adx']) else None,
            "atr": float(last['atr']) if not np.isnan(last['atr']) else None,
            "atr_pct": float(last['atr_pct']) if not np.isnan(last['atr_pct']) else None,
            "cmf": float(last['cmf']) if not np.isnan(last['cmf']) else None,
            "volume_z60": float(last['volume_z60']) if not np.isnan(last['volume_z60']) else None,
        },
        "trades_preview": trades[-5:]  # last 5 for quick glance
    }

    return result, df, pd.DataFrame(trades), eq_df


# =========================
# CLI + Utilities
# =========================
def convert_to_builtin_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Long-only regime-aware strategy with tight entries")
    p.add_argument("ticker", type=str, help="Ticker, e.g., AAPL")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")

    p.add_argument("--capital", type=float, default=100000.0, help="Starting capital")
    p.add_argument("--risk", type=float, default=0.01, help="Risk per trade as fraction of equity (e.g., 0.01)")
    p.add_argument("--entry", type=float, default=0.65, help="Entry score threshold")
    p.add_argument("--exit", type=float, default=0.20, help="Exit score threshold")
    p.add_argument("--atr_stop", type=float, default=2.5, help="Initial ATR stop multiple")
    p.add_argument("--chandelier", type=float, default=3.0, help="Chandelier ATR multiple")
    p.add_argument("--hold_days", type=int, default=180, help="Max bars to hold position")
    p.add_argument("--slip_bps", type=float, default=2.0, help="Slippage in basis points (2 = 0.02%)")
    p.add_argument("--fee_share", type=float, default=0.0, help="Fee per share")

    # Tight entry params
    p.add_argument("--confirm_days", type=int, default=1, help="Require N prior daily bars of condition (0=off)")
    p.add_argument("--exit_confirm_days", type=int, default=0, help="Require N prior daily bars before exit (0=off)")
    p.add_argument("--micro_4h_bars", type=int, default=2, help="Require K consecutive 4H uptrend bars (0=off)")
    p.add_argument("--intraday_fail_hard", action="store_true", help="If intraday missing, block entries")

    p.add_argument("--save", action="store_true", help="Save equity/trades CSV and equity plot")
    p.add_argument("--json", action="store_true", help="Print JSON summary")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    res, df, trades_df, eq = backtest_long_only(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        start_capital=args.capital,
        risk_per_trade=args.risk,
        entry_threshold=args.entry,
        exit_threshold=args.exit,
        atr_stop_mult=args.atr_stop,
        chandelier_mult=args.chandelier,
        max_hold_days=args.hold_days,
        slippage_bps=args.slip_bps,
        fee_per_share=args.fee_share,
        confirm_days=args.confirm_days,
        exit_confirm_days=args.exit_confirm_days,
        micro_4h_bars=args.micro_4h_bars,
        intraday_fail_hard=args.intraday_fail_hard,
    )

    # Save artifacts if requested
    if args.save:
        eq.to_csv(f"{args.ticker}_equity_curve.csv")
        trades_df.to_csv(f"{args.ticker}_trades.csv", index=False)

        # Plot equity
        plt.figure(figsize=(10, 5))
        eq['equity'].plot(title=f"{args.ticker} Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(f"{args.ticker}_equity_curve.png", dpi=140)
        plt.close()

    if args.json:
        print(json.dumps(convert_to_builtin_type(res), indent=2))
    else:
        print(f"\n== {res['ticker']} Backtest Summary ==")
        for k, v in res['metrics'].items():
            print(f"{k:>16}: { v }")

        print("\nLatest Signal:")
        for k, v in res['latest_signal'].items():
            print(f"{k:>16}: { v }")

        print("\nParams:")
        for k, v in res['params'].items():
            print(f"{k:>16}: { v }")

        if len(res.get('trades_preview', [])) > 0:
            print("\nMost recent trades:")
            for t in res['trades_preview']:
                print(t)

        print(f"\nIntraday status: {res['intraday_status']}")


if __name__ == "__main__":
    main()
