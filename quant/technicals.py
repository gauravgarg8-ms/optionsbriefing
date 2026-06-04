import numpy as np
import pandas as pd
from loguru import logger

from errors import ErrorCode


def compute_technical_levels(ticker: str, ohlcv: pd.DataFrame) -> dict:
    """
    Compute support, resistance, ATR(14), RSI(14) and 52-week levels.

    ohlcv: DataFrame with columns Open/High/Low/Close, ≥ 60 rows recommended.
    Returns dict with: support, resistance, atr_14, rsi_14, support_52w, resistance_52w.
    """
    if len(ohlcv) < 20:
        logger.warning(f"[{ErrorCode.E2001}] Insufficient OHLCV data for {ticker}: {len(ohlcv)} rows")
        return {"support": 0.0, "resistance": 0.0, "atr_14": 0.0, "rsi_14": 50.0,
                "support_52w": 0.0, "resistance_52w": 0.0}

    close = ohlcv["Close"].squeeze().astype(float)
    high  = ohlcv["High"].squeeze().astype(float)
    low   = ohlcv["Low"].squeeze().astype(float)

    # 52-week high/low
    lookback = min(len(ohlcv), 252)
    support_52w    = round(float(low.iloc[-lookback:].min()), 2)
    resistance_52w = round(float(high.iloc[-lookback:].max()), 2)

    # Support: mean of 3 smallest recent 5-day rolling lows
    recent_lows      = low.rolling(5).min().dropna()
    support_recent   = float(recent_lows.nsmallest(3).mean()) if len(recent_lows) >= 3 else support_52w
    support          = round(max(support_52w, support_recent), 2)

    # Resistance: mean of 3 largest recent 5-day rolling highs
    recent_highs     = high.rolling(5).max().dropna()
    resistance_recent = float(recent_highs.nlargest(3).mean()) if len(recent_highs) >= 3 else resistance_52w
    resistance        = round(min(resistance_52w, resistance_recent), 2)

    # Ensure support <= current price <= resistance
    current_price = float(close.iloc[-1])
    support    = min(support, current_price)
    resistance = max(resistance, current_price)

    # ATR(14) — Wilder's method
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14 = round(float(tr.ewm(span=14, adjust=False).mean().iloc[-1]), 4)

    # RSI(14) — Wilder's smoothed EWM
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta.clip(upper=0))
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    # Edge cases: avg_loss=0 + avg_gain>0 → RSI=100; both=0 (flat) → RSI=50
    avg_loss_safe = avg_loss.copy()
    avg_loss_safe[avg_loss_safe == 0] = np.nan
    rs         = avg_gain / avg_loss_safe
    rsi_series = 100 - (100 / (1 + rs))
    rsi_series[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    rsi_series[(avg_gain >  0) & (avg_loss == 0)] = 100.0
    rsi_14 = round(float(rsi_series.iloc[-1]), 1)

    return {
        "support":        support,
        "resistance":     resistance,
        "atr_14":         atr_14,
        "rsi_14":         rsi_14,
        "support_52w":    support_52w,
        "resistance_52w": resistance_52w,
    }


def compute_rs(ticker_close: pd.Series, spy_close: pd.Series, days: int = 20) -> float:
    """
    Relative Strength: stock 20d log return minus SPY 20d log return.
    Positive = outperforming. Returns 0.0 if insufficient data.
    """
    try:
        if len(ticker_close) < days + 1 or len(spy_close) < days + 1:
            return 0.0
        stock_ret = float(np.log(ticker_close.iloc[-1] / ticker_close.iloc[-(days + 1)]) * 100)
        spy_ret   = float(np.log(spy_close.iloc[-1]   / spy_close.iloc[-(days + 1)])   * 100)
        return round(stock_ret - spy_ret, 2)
    except Exception as e:
        logger.warning(f"compute_rs failed: {e}")
        return 0.0


def classify_vix(vix: float) -> tuple[str, float]:
    """Returns (regime_label, size_multiplier). Imported from market_data but kept here too."""
    from data.market_data import classify_vix as _classify
    return _classify(vix)


def classify_spy_trend(price: float, ma50: float, ma200: float) -> str:
    from data.market_data import classify_spy_trend as _classify
    return _classify(price, ma50, ma200)
