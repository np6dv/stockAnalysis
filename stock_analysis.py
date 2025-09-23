import yfinance as yf
import pandas as pd
import numpy as np
import ta  # Install with: pip install ta

def get_analyst_predictions(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "target_mean": float(info.get("targetMeanPrice", 0)),
        "target_high": float(info.get("targetHighPrice", 0)),
        "target_low": float(info.get("targetLowPrice", 0)),
        "number_of_analysts": int(info.get("numberOfAnalystOpinions", 0)),
        "recommendation": info.get("recommendationKey", "none"),
    }

def fetch_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period).reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]

def find_inflection_points(df, window=5):
    df['sma'] = df['close'].rolling(window=window).mean()
    df['inflection'] = None
    df['inflection_price'] = np.nan
    for i in range(window, len(df) - window):
        window_slice = df['sma'].iloc[i - window: i + window + 1]
        center = df['sma'].iloc[i]
        if center == window_slice.min():
            df.at[i, 'inflection'] = 'min'
            df.at[i, 'inflection_price'] = df['close'].iloc[i]
        elif center == window_slice.max():
            df.at[i, 'inflection'] = 'max'
            df.at[i, 'inflection_price'] = df['close'].iloc[i]
    return df

def compute_volume_indicators(df):
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_zscore'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()
    df['volume_spike'] = df['volume_zscore'] > 2
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return df

def compute_technical_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    return df

def summarize_inflections(df):
    inflections = df.dropna(subset=['inflection']).sort_values(by='date', ascending=False)
    for _, row in inflections.iterrows():
        if row['inflection'] in ['min', 'max']:
            return {
                "last_inflection_type": row['inflection'],
                "last_inflection_date": str(row['date']),
                "last_inflection_price": round(float(row['inflection_price']), 2)
            }
    return {"last_inflection_type": None, "last_inflection_date": None, "last_inflection_price": None}

def summarize_volume(df):
    recent = df.tail(30)
    spikes = recent[recent['volume_spike']]
    today = df.iloc[-1]
    return {
        "avg_volume_last_30d": round(float(recent['volume'].mean()), 0),
        "volume_spike_count_last_30d": int(spikes.shape[0]),
        "last_volume_spike_date": str(spikes['date'].iloc[-1]) if not spikes.empty else None,
        "last_volume_spike_value": float(spikes['volume'].iloc[-1]) if not spikes.empty else None,
        "today_volume": float(today['volume']),
        "today_volume_spike": bool(today['volume_spike']),
        "today_volume_vs_avg": round(float(today['volume']) / float(recent['volume'].mean()), 2) if recent['volume'].mean() else None
    }

def compute_confidence_score(inflection, volume, analyst, rsi, macd, macd_signal, current_price):
    score = 0
    if inflection['last_inflection_type'] == 'min':
        score += 1
    elif inflection['last_inflection_type'] == 'max':
        score -= 1
    if volume['today_volume_spike']:
        score += 1
    if analyst and analyst.get("target_mean", 0) > current_price:
        score += 1
    if rsi < 30:
        score += 1
    elif rsi > 70:
        score -= 1
    if macd > macd_signal:
        score += 1
    else:
        score -= 1
    return score

def convert_to_builtin_type(obj):
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

def analyze_stock(ticker):
    analyst = get_analyst_predictions(ticker)
    df = fetch_stock_data(ticker)
    if df.empty:
        raise ValueError("No data found for ticker.")
    df = find_inflection_points(df)
    df = compute_volume_indicators(df)
    df = compute_technical_indicators(df)
    inflection_summary = summarize_inflections(df)
    volume_summary = summarize_volume(df)
    current_price = float(df["close"].iloc[-1])
    rsi = float(df["rsi"].iloc[-1])
    macd = float(df["macd"].iloc[-1])
    macd_signal = float(df["macd_signal"].iloc[-1])
    confidence_score = compute_confidence_score(
        inflection_summary, volume_summary, analyst, rsi, macd, macd_signal, current_price
    )

    if confidence_score >= 3:
        action = "BUY"
        message = "Strong buy signal based on technical and analyst indicators."
    elif confidence_score <= -2:
        action = "SELL"
        message = "Sell signal based on weak technicals or overbought conditions."
    else:
        action = "HOLD"
        message = "No strong signal. Consider holding or waiting for confirmation."

    return {
        "ticker": ticker.upper(),
        "current_price": round(current_price, 2),
        "analyst_predictions": analyst,
        "last_inflection": inflection_summary,
        "volume_analysis": volume_summary,
        "technical_indicators": {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal
        },
        "confidence_score": confidence_score,
        "recommendation": action,
        "analysis": message
    }

# Optional CLI usage
if __name__ == "__main__":
    import sys
    import json
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = analyze_stock(ticker)
    print(json.dumps(convert_to_builtin_type(result), indent=2))
