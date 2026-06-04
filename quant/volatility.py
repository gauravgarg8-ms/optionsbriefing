import numpy as np
import pandas as pd
from loguru import logger

from db.db_manager import DBManager
from errors import ErrorCode

_db = DBManager()


def compute_iv30(options: list[dict]) -> float:
    """
    Weighted average ATM implied vol across 25–35 DTE options.
    Weights by open interest. Returns annualised float (e.g. 0.42 = 42%).
    """
    if not options:
        return 0.0
    from datetime import date, datetime
    today = date.today()
    atm_options = []
    for opt in options:
        try:
            exp = datetime.strptime(opt["expiration_date"], "%Y-%m-%d").date()
            dte = (exp - today).days
            if 25 <= dte <= 35:
                iv = _get_iv(opt)
                oi = int(opt.get("open_interest", 0) or 0)
                if iv > 0 and oi > 0:
                    atm_options.append((iv, oi))
        except Exception:
            continue

    if not atm_options:
        # Fallback: use all options with valid IV
        for opt in options:
            iv = _get_iv(opt)
            oi = int(opt.get("open_interest", 1))
            if iv > 0:
                atm_options.append((iv, oi))

    if not atm_options:
        return 0.0

    total_oi = sum(oi for _, oi in atm_options)
    if total_oi == 0:
        return round(float(np.mean([iv for iv, _ in atm_options])), 4)
    weighted = sum(iv * oi for iv, oi in atm_options) / total_oi
    return round(float(weighted), 4)


def _get_iv(opt: dict) -> float:
    """Extract mid_iv from option dict (Tradier or yfinance format)."""
    greeks = opt.get("greeks", {})
    if greeks and greeks.get("mid_iv", 0):
        return float(greeks["mid_iv"])
    return 0.0


def compute_hv20(close_series: pd.Series) -> float:
    """
    20-day historical (realised) volatility = std(log returns) × sqrt(252).
    Returns annualised float (e.g. 0.28 = 28%).
    """
    if len(close_series) < 25:
        logger.warning(f"[{ErrorCode.E2001}] HV20: need ≥25 rows, got {len(close_series)}")
        return 0.0
    log_returns = np.log(close_series / close_series.shift(1)).dropna()
    hv20 = float(log_returns.iloc[-20:].std() * np.sqrt(252))
    return round(max(hv20, 0.0), 4)


def compute_iv_rank(ticker: str, current_iv: float, db: DBManager | None = None) -> float:
    """
    IV Rank = (current_iv - 52wk_low) / (52wk_high - 52wk_low) × 100.
    Returns 50.0 (neutral default) if < 30 days of history available.
    """
    manager = db or _db
    high, low = manager.get_52wk_high_low(ticker)
    if high is None or low is None:
        return 50.0
    if high == low:
        return 50.0
    rank = (current_iv - low) / (high - low) * 100
    return round(float(np.clip(rank, 0, 100)), 1)


def compute_iv_percentile(ticker: str, current_iv: float, db: DBManager | None = None) -> float:
    """
    IV Percentile = % of past-year days where IV was LOWER than today.
    Returns 50.0 if insufficient history.
    """
    manager = db or _db
    rows = manager.get_iv_history(ticker, days=365)
    if len(rows) < 30:
        return 50.0
    past_ivs = [iv for _, iv in rows]
    pct = sum(1 for v in past_ivs if v < current_iv) / len(past_ivs) * 100
    return round(float(pct), 1)


def compute_iv_rv_ratio(iv30: float, hv20: float) -> float:
    """
    IV/RV ratio = IV30 / HV20.
    Returns 1.0 if HV20 is zero (divide-by-zero guard).
    """
    if hv20 <= 0:
        return 1.0
    return round(float(iv30 / hv20), 2)
