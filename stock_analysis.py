import yfinance as yf
import pandas as pd
import numpy as np

def get_analyst_predictions(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    try:
        return {
            "target_mean": float(info.get("targetMeanPrice", 0)),
            "target_high": float(info.get("targetHighPrice", 0)),
            "target_low": float(info.get("targetLowPrice", 0)),
            "number_of_analysts": int(info.get("numberOfAnalystOpinions", 0)),
            "recommendation": info.get("recommendationKey", "none"),
        }
    except Exception as e:
        return {"error": str(e)}

def fetch_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]

def find_inflection_points(df, window=5):
    df = df.copy()
    df['inflection'] = None
    df['inflection_price'] = np.nan
    for i in range(window, len(df) - window):
        window_slice = df['close'].iloc[i - window: i + window + 1]
        center = df['close'].iloc[i]
        if center == window_slice.min():
            df.at[i, 'inflection'] = 'min'
            df.at[i, 'inflection_price'] = center
        elif center == window_slice.max():
            df.at[i, 'inflection'] = 'max'
            df.at[i, 'inflection_price'] = center
    return df

def compute_volume_indicators(df):
    df = df.copy()
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] > 1.5 * df['volume_sma_20']
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return df

def summarize_inflections(df):
    inflections = df.dropna(subset=['inflection']).sort_values(by='date', ascending=False)
    for idx, row in inflections.iterrows():
        if row['inflection'] in ['min', 'max']:
            return {
                "last_inflection_type": row['inflection'],
                "last_inflection_date": str(row['date']),
                "last_inflection_price": float(row['inflection_price'])
            }
    return {"last_inflection_type": None, "last_inflection_date": None, "last_inflection_price": None}

def summarize_volume(df):
    recent = df.tail(30)
    spikes = recent[recent['volume_spike']]
    today = df.iloc[-1]
    return {
        "avg_volume_last_30d": float(recent['volume'].mean()),
        "volume_spike_count_last_30d": int(spikes.shape[0]),
        "last_volume_spike_date": str(spikes['date'].iloc[-1]) if not spikes.empty else None,
        "last_volume_spike_value": float(spikes['volume'].iloc[-1]) if not spikes.empty else None,
        "today_volume": float(today['volume']),
        "today_volume_spike": bool(today['volume_spike']),
        "today_volume_vs_avg": float(today['volume']) / float(recent['volume'].mean()) if recent['volume'].mean() else None
    }

def analyze_inflection_action(inflection, volume, analyst=None, current_price=None):
    """
    Returns recommendation and analysis based on inflection type and daily volume status.
    Includes volume context in the analysis.
    """
    action = "hold"
    message = "No strong inflection signal detected."
    volume_note = ""
    # Add volume analysis
    if volume["today_volume_spike"]:
        volume_note = f" Today's volume is a spike (current: {volume['today_volume']:.0f}; avg 30d: {volume['avg_volume_last_30d']:.0f}), indicating stronger conviction in today's movement."
    elif volume["today_volume_vs_avg"] is not None and volume["today_volume_vs_avg"] > 1.1:
        volume_note = f" Today's volume ({volume['today_volume']:.0f}) is above the 30-day average ({volume['avg_volume_last_30d']:.0f})."
    elif volume["today_volume_vs_avg"] is not None and volume["today_volume_vs_avg"] < 0.8:
        volume_note = f" Today's volume ({volume['today_volume']:.0f}) is below the 30-day average ({volume['avg_volume_last_30d']:.0f}), so conviction is weaker."

    if inflection and inflection['last_inflection_type']:
        inf_type = inflection['last_inflection_type']
        inf_price = inflection['last_inflection_price']
        if inf_type == "min":
            action = "BUY"
            message = f"Recent inflection point is a minimum (potential reversal upward) at {inf_price:.2f}. Consider buying, especially if price is above this level."
            if analyst and analyst.get("target_mean", 0) > (current_price or 0):
                message += f" Analyst average target price ({analyst['target_mean']:.2f}) is above current price, supporting a buy."
        elif inf_type == "max":
            action = "SELL"
            message = f"Recent inflection point is a maximum (potential reversal downward) at {inf_price:.2f}. Consider selling or taking profits if price is near or below this level."
            if analyst and analyst.get("target_mean", 0) < (current_price or 0):
                message += f" Analyst average target price ({analyst['target_mean']:.2f}) is below current price, supporting a sell."
        else:
            action = "hold"
            message = "No clear inflection direction. Hold."
        message += volume_note
    else:
        message = "No recent inflection point detected." + volume_note
    return {"action": action, "analysis": message}

def analyze_stock(ticker):
    analyst = get_analyst_predictions(ticker)
    df = fetch_stock_data(ticker, period="1y")
    if df.empty:
        raise ValueError("No data found for ticker.")
    df = find_inflection_points(df, window=5)
    df = compute_volume_indicators(df)
    inflection_summary = summarize_inflections(df)
    volume_summary = summarize_volume(df)
    current_price = float(df["close"].iloc[-1])
    inflection_action = analyze_inflection_action(
        inflection_summary, volume=volume_summary, analyst=analyst, current_price=current_price
    )
    result = {
        "ticker": ticker.upper(),
        "current_price": current_price,
        "analyst_predictions": analyst,
        "last_inflection": inflection_summary,
        "volume_analysis": volume_summary,
        "inflection_action": inflection_action
    }
    return result

def convert_to_builtin_type(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

if __name__ == "__main__":
    import sys
    import json

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = analyze_stock(ticker)
    print(json.dumps(convert_to_builtin_type(result), indent=2))