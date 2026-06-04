"""
Phase 2 orchestrator: reads raw_market_data.json, computes all quant signals,
writes quant_signals.json.

Two-stage operation matching the HTML workflow:
  Phase 2A (lightweight): volatility + technicals for all ~200 tickers
  Phase 2C (deep):        black_scholes + structure_selector after options chain available
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from db.db_manager import DBManager
from quant.volatility import compute_hv20, compute_iv_rank, compute_iv_percentile, compute_iv_rv_ratio
from quant.technicals import compute_technical_levels, compute_rs
from quant.black_scholes import (
    black_scholes, compute_greeks, compute_pop, compute_ev,
    price_bull_put_spread, price_bear_call_spread, price_bull_call_spread,
    price_bear_put_spread, price_iron_condor, price_long_straddle,
    price_long_strangle, price_cash_secured_put, PRICERS,
)
from quant.structure_selector import select_structure, apply_golden_rules, select_strikes
from errors import ErrorCode

_db = DBManager()


def run_lightweight_quant(ticker: str, price_data: dict, spy_close: pd.Series) -> dict:
    """
    Phase 2A: compute IV Rank, HV20, RS, MAs, technicals from cached price data.
    No options chain needed — uses SQLite IV history, falling back to OHLCV proxy on cold start.
    """
    try:
        ohlcv = _fetch_ohlcv(ticker)
        if ohlcv is None or ohlcv.empty:
            return {}

        close  = ohlcv["Close"].squeeze()
        volume = ohlcv["Volume"].squeeze()
        hv20   = compute_hv20(close)

        # Avg options volume proxy: ~1 options contract per 2000 shares of avg daily stock volume
        avg_vol_20d     = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else float(volume.mean())
        avg_options_vol = int(avg_vol_20d / 2000)

        # IV30 placeholder for 2A — will be updated with real value in 2C
        iv30 = price_data.get("iv30_proxy", hv20)

        # Compute rolling HV series once — used for both iv_rank proxy and DB seeding
        log_ret    = np.log(close / close.shift(1)).dropna()
        rolling_hv = (log_ret.rolling(20).std() * np.sqrt(252)).dropna()

        # IV Rank / Percentile: use DB history when ≥30 rows exist; otherwise use
        # rolling-HV proxy and bulk-seed historical rows so future runs have real history.
        _iv_rows = _db.get_iv_history(ticker, days=365)
        if len(_iv_rows) >= 30:
            past_ivs = [iv for _, iv in _iv_rows]
            hi, lo   = max(past_ivs), min(past_ivs)
            iv_rank  = round(float(np.clip((iv30 - lo) / (hi - lo) * 100 if hi != lo else 50, 0, 100)), 1)
            iv_pct   = round(sum(1 for v in past_ivs if v < iv30) / len(past_ivs) * 100, 1)
        else:
            history  = rolling_hv.iloc[-252:]
            if len(history) >= 30:
                hi, lo  = float(history.max()), float(history.min())
                iv_rank = round(float(np.clip((hv20 - lo) / (hi - lo) * 100 if hi != lo else 50, 0, 100)), 1)
            else:
                iv_rank = 50.0
            iv_pct  = 50.0
            # Bulk-seed so next run has full history and no longer needs the proxy path
            _bulk_seed_hv_proxy(ticker, rolling_hv, _db)

        iv_rv = compute_iv_rv_ratio(iv30, hv20)

        technicals = compute_technical_levels(ticker, ohlcv)
        rs_20d     = compute_rs(close, spy_close)

        ma50  = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        current_price = float(close.iloc[-1])

        return {
            "ticker":          ticker,
            "price":           round(current_price, 2),
            "ma50":            round(ma50, 2),
            "ma200":           round(ma200, 2),
            "above_50ma":      current_price > ma50,
            "above_200ma":     current_price > ma200,
            "rs_20d":          rs_20d,
            "hv20":            round(hv20 * 100, 2),
            "iv30":            round(iv30 * 100, 2),
            "iv_rank":         iv_rank,
            "iv_percentile":   iv_pct,
            "iv_rv_ratio":     iv_rv,
            "avg_options_vol": avg_options_vol,
            **technicals,
        }
    except Exception as e:
        logger.warning(f"[{ErrorCode.E2001}] Lightweight quant failed for {ticker}: {e}")
        return {}


def run_deep_quant(ticker: str, lightweight: dict, chain_data: dict,
                   tbill_rate: float = 0.051) -> dict:
    """
    Phase 2C: add Black-Scholes, structure selection, spread pricing after options chain arrives.
    Accumulates real IV30 to SQLite.
    """
    try:
        options     = chain_data.get("options", [])                   # liquid, for strike selection
        raw_options = chain_data.get("raw_options") or options        # pre-filter, for IV computation
        expiry      = chain_data.get("expiry", "")
        if not expiry or not raw_options:
            return lightweight

        from quant.volatility import compute_iv30 as _compute_iv30
        iv30_real = _compute_iv30(raw_options)
        if iv30_real > 0:
            _db.upsert_iv_with_source(ticker, date.today().isoformat(), iv30_real, source="real")
            lightweight["iv30"] = round(iv30_real * 100, 2)
            # Update iv_rank from DB only when history is sufficient; otherwise the Phase 2A
            # rolling-HV proxy value is more accurate than the 50.0 cold-start default.
            _iv_rows = _db.get_iv_history(ticker, days=365)
            if len(_iv_rows) >= 30:
                past_ivs = [iv for _, iv in _iv_rows]
                hi, lo   = max(past_ivs), min(past_ivs)
                lightweight["iv_rank"] = round(
                    float(np.clip((iv30_real - lo) / (hi - lo) * 100 if hi != lo else 50, 0, 100)), 1
                )
            lightweight["iv_rv_ratio"] = compute_iv_rv_ratio(iv30_real, lightweight.get("hv20", 0) / 100)

        # Strike selection and pricing require liquid options (valid bid/ask).
        # Pre-market runs have empty liquid list — IV was already written above, return now.
        if not options:
            return lightweight

        T_days = _dte(expiry)
        T_yrs  = T_days / 365.0
        S      = lightweight.get("price", 0)
        sigma  = iv30_real if iv30_real > 0 else lightweight.get("iv30", 25) / 100
        r      = tbill_rate
        bias   = _infer_bias(lightweight)

        structure = select_structure(
            iv_rank=lightweight["iv_rank"],
            iv_rv_ratio=lightweight["iv_rv_ratio"],
            bias=bias,
        )
        structure = apply_golden_rules(structure, lightweight["iv_rank"])

        strikes = select_strikes(
            S=S, structure=structure, options=options,
            iv30=sigma, T_days=T_days,
            support=lightweight.get("support", S * 0.95),
            resistance=lightweight.get("resistance", S * 1.05),
        )

        pricing = _price_structure(structure, S, strikes, T_yrs, r, sigma)

        lightweight.update({
            "structure":    structure,
            "direction":    bias,
            "expiry":       expiry,
            "dte":          T_days,
            "strikes":      strikes,
            "spread_pricing": pricing,
            "covered_call_opportunity": (
                bias == "Bullish" and lightweight.get("iv_rank", 0) > 40
            ),
        })
        return lightweight

    except Exception as e:
        logger.error(f"[{ErrorCode.E2003}] Deep quant failed for {ticker}: {e}")
        return lightweight


def run_quant_calculations(raw_data: dict) -> dict:
    """
    Top-level orchestrator called by main.py (Phase 2A only — 2C is called by screening_engine).
    Returns quant_signals dict with all tickers.
    """
    candidates = raw_data.get("universe", [])
    market_env = raw_data.get("market_environment", {})
    spy_close  = _get_spy_close()
    tbill_rate = market_env.get("tbill_3m", 0.051)

    logger.info(f"[Phase 2] Running lightweight quant for {len(candidates)} tickers")
    quant_results = {}
    for entry in candidates:
        ticker = entry.get("symbol", entry.get("ticker", ""))
        if not ticker:
            continue
        result = run_lightweight_quant(ticker, entry, spy_close)
        if result:
            result["sector"] = entry.get("sector", "Unknown")
            # Set preliminary structure + direction so 2B scoring has full signal set.
            # run_deep_quant overwrites these with real options-chain values in Phase 2C.
            bias = _infer_bias(result)
            result["direction"] = bias
            result["structure"] = apply_golden_rules(
                select_structure(
                    iv_rank=result["iv_rank"],
                    iv_rv_ratio=result["iv_rv_ratio"],
                    bias=bias,
                ),
                result["iv_rank"],
            )
            quant_results[ticker] = result

    logger.info(f"[Phase 2] Lightweight quant complete: {len(quant_results)} tickers")
    return {
        "market_environment": market_env,
        "tbill_rate":         tbill_rate,
        "quant_signals":      quant_results,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bulk_seed_hv_proxy(ticker: str, rolling_hv: pd.Series, db: "DBManager") -> None:
    """
    Seed up to 252 trading days of rolling-HV20 as proxy IV into the DB using a
    single connection+transaction (bulk_upsert_iv_proxy).  Calling upsert_iv_with_source
    252 times per ticker opens 252 SQLite connections; with 516 tickers that exhausts
    the OS file-descriptor limit (EMFILE).
    """
    try:
        today = date.today()
        rows  = [
            (ts.date().isoformat() if hasattr(ts, "date") else str(ts), float(hv_val))
            for ts, hv_val in rolling_hv.iloc[-252:].items()
            if (ts.date() if hasattr(ts, "date") else ts) <= today
            and not np.isnan(hv_val) and hv_val > 0
        ]
        if rows:
            db.bulk_upsert_iv_proxy(ticker, rows)
    except Exception:
        pass


def _fetch_ohlcv(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period="14mo", interval="1d", progress=False, auto_adjust=True)
        return df if not df.empty else None
    except Exception as e:
        logger.warning(f"OHLCV fetch failed for {ticker}: {e}")
        return None


def _get_spy_close() -> pd.Series:
    try:
        df = yf.download("SPY", period="3mo", interval="1d", progress=False, auto_adjust=True)
        return df["Close"].squeeze()
    except Exception:
        return pd.Series(dtype=float)


def _dte(expiry_str: str) -> int:
    try:
        exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        return max((exp - date.today()).days, 1)
    except Exception:
        return 30


def _infer_bias(signals: dict) -> str:
    if signals.get("above_50ma") and signals.get("rs_20d", 0) > 0:
        return "Bullish"
    if not signals.get("above_50ma") and signals.get("rs_20d", 0) < 0:
        return "Bearish"
    return "Neutral"


def _price_structure(structure: str, S: float, strikes: dict,
                     T: float, r: float, sigma: float) -> dict:
    try:
        if structure == "bull_put_spread":
            return price_bull_put_spread(S, strikes["k_short"], strikes["k_long"], T, r, sigma)
        if structure == "bear_call_spread":
            return price_bear_call_spread(S, strikes["k_short"], strikes["k_long"], T, r, sigma)
        if structure == "bull_call_spread":
            return price_bull_call_spread(S, strikes["k_long"], strikes["k_short"], T, r, sigma)
        if structure == "bear_put_spread":
            return price_bear_put_spread(S, strikes["k_long"], strikes["k_short"], T, r, sigma)
        if structure == "iron_condor":
            return price_iron_condor(
                S, strikes["k_put_short"], strikes["k_put_long"],
                strikes["k_call_short"], strikes["k_call_long"], T, r, sigma
            )
        if structure == "long_straddle":
            return price_long_straddle(S, strikes["k"], T, r, sigma)
        if structure == "long_strangle":
            return price_long_strangle(S, strikes["k_put"], strikes["k_call"], T, r, sigma)
        if structure in ("cash_secured_put", "earnings_credit_spread"):
            return price_cash_secured_put(S, strikes["k_short"], T, r, sigma)
    except Exception as e:
        logger.error(f"[{ErrorCode.E2003}] Spread pricing failed for {structure}: {e}")
    return {}
