"""
Options chain data.

Primary source: yfinance (free, no account required).
Optional upgrade: set TRADIER_TOKEN in .env to use Tradier as primary
                  (more reliable, includes broker-quoted Greeks).
If Tradier is configured but fails, falls back to yfinance automatically.
"""
import time
from datetime import date, datetime

import requests
import yfinance as yf
from loguru import logger

from config import (TRADIER_BASE, TRADIER_TOKEN, DEEP_FETCH_DTE_MIN,
                    DEEP_FETCH_DTE_MAX, LIQUIDITY_MAX_BID_ASK_PCT, LIQUIDITY_MIN_OI)
from errors import ErrorCode


def get_expiry_dates(ticker: str) -> list[str]:
    """
    Return expiry dates with DTE in [DEEP_FETCH_DTE_MIN, DEEP_FETCH_DTE_MAX].
    Uses yfinance .options list — no API key required.
    Retries once after 2 s on rate-limit errors.
    """
    today = date.today()
    for attempt in range(2):
        try:
            t     = yf.Ticker(ticker)
            dates = t.options
            if not dates:
                return []
            filtered = []
            for d in dates:
                try:
                    exp = datetime.strptime(d, "%Y-%m-%d").date()
                    dte = (exp - today).days
                    if DEEP_FETCH_DTE_MIN <= dte <= DEEP_FETCH_DTE_MAX:
                        filtered.append(d)
                except ValueError:
                    continue
            return filtered
        except Exception as e:
            if attempt == 0 and "rate" in str(e).lower():
                time.sleep(2)
                continue
            logger.warning(f"[{ErrorCode.E1008}] get_expiry_dates failed for {ticker}: {e}")
            return []
    return []


def fetch_0dte_chain(ticker: str) -> dict:
    """
    Fetch the 0DTE (same-day expiry) options chain for a ticker.
    Selects today's expiry if available (SPY expires M/W/F); otherwise the
    nearest expiry within 2 calendar days so the briefing is never empty.
    Used only for PINNED_TICKERS with force_dte_0=True.
    """
    today = date.today()
    try:
        t     = yf.Ticker(ticker)
        dates = t.options
        if not dates:
            logger.warning(f"[{ErrorCode.E1008}] No expiries found for {ticker} (0DTE fetch)")
            return {"expiry": None, "options": [], "source": "yfinance"}

        target = None
        for d in sorted(dates):
            exp = datetime.strptime(d, "%Y-%m-%d").date()
            dte = (exp - today).days
            if 0 <= dte <= 2:
                target = d
                break

        if not target:
            # No near-term expiry this week — fall back to earliest available
            target = sorted(dates)[0]
            logger.warning(
                f"[{ErrorCode.E1008}] No 0–2 DTE expiry found for {ticker} — "
                f"using earliest available: {target}"
            )

        chain   = t.option_chain(target)
        options = []
        for _, row in chain.puts.iterrows():
            options.append(_yf_row_to_option(row, "put", target))
        for _, row in chain.calls.iterrows():
            options.append(_yf_row_to_option(row, "call", target))

        actual_dte = (datetime.strptime(target, "%Y-%m-%d").date() - today).days
        logger.info(f"0DTE chain for {ticker} ({target}, DTE={actual_dte}): {len(options)} contracts")
        return {"expiry": target, "options": options, "source": "yfinance"}

    except Exception as e:
        logger.error(f"[{ErrorCode.E1008}] 0DTE chain fetch failed for {ticker}: {e}")
        return {"expiry": None, "options": [], "source": "yfinance"}


def fetch_options_chain(ticker: str) -> dict:
    """
    Fetch full options chain for target expiry.

    If TRADIER_TOKEN is configured → try Tradier first (richer data), fall back to yfinance.
    Otherwise → use yfinance directly.

    Returns {expiry: str, options: list[dict], source: 'yfinance'|'tradier'}.
    """
    if TRADIER_TOKEN:
        result = _fetch_tradier_chain(ticker)
        if result.get("options"):
            return result
        logger.warning(f"[{ErrorCode.E1008}] Tradier returned empty/failed for {ticker} — using yfinance")

    return _fetch_yfinance_chain(ticker)


# ── yfinance (primary) ───────────────────────────────────────────────────────

def _fetch_yfinance_chain(ticker: str) -> dict:
    """Primary options chain source — free, no account required."""
    try:
        expiries = get_expiry_dates(ticker)
        if not expiries:
            # Try any available expiry as fallback
            t = yf.Ticker(ticker)
            all_expiries = t.options
            if not all_expiries:
                logger.warning(f"[{ErrorCode.E1008}] No options expiries found for {ticker}")
                return {"expiry": None, "options": [], "source": "yfinance"}
            target = all_expiries[0]
        else:
            target = expiries[0]

        t     = yf.Ticker(ticker)
        chain = t.option_chain(target)
        options = []
        for _, row in chain.puts.iterrows():
            options.append(_yf_row_to_option(row, "put", target))
        for _, row in chain.calls.iterrows():
            options.append(_yf_row_to_option(row, "call", target))

        logger.info(f"yfinance chain for {ticker} ({target}): {len(options)} contracts")
        return {"expiry": target, "options": options, "source": "yfinance"}

    except Exception as e:
        logger.error(f"[{ErrorCode.E1008}] yfinance chain failed for {ticker}: {e}")
        return {"expiry": None, "options": [], "source": "yfinance"}


def _safe_float(val, default: float = 0.0) -> float:
    """Convert val to float, treating None/NaN/inf as default."""
    try:
        v = float(val)
        return default if (v != v) or (v == float("inf")) or (v == float("-inf")) else v
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    """Convert val to int, treating None/NaN as default."""
    return int(_safe_float(val, float(default)))


def _yf_row_to_option(row, option_type: str, expiry: str) -> dict:
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    mid = (bid + ask) / 2 if (bid + ask) > 0 else 0.0
    return {
        "option_type":     option_type,
        "strike":          _safe_float(row.get("strike")),
        "expiration_date": expiry,
        "bid":             bid,
        "ask":             ask,
        "last":            _safe_float(row.get("lastPrice")),
        "volume":          _safe_int(row.get("volume")),
        "open_interest":   _safe_int(row.get("openInterest")),
        "greeks":          {"mid_iv": _safe_float(row.get("impliedVolatility"))},
        "_mid":            mid,
    }


# ── Tradier (optional upgrade) ───────────────────────────────────────────────

def _fetch_tradier_chain(ticker: str) -> dict:
    """Optional Tradier path — used only when TRADIER_TOKEN is set in .env."""
    headers = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        # Get expiry dates via Tradier
        resp = requests.get(
            f"{TRADIER_BASE}/markets/options/expirations",
            headers=headers, params={"symbol": ticker}, timeout=10,
        )
        resp.raise_for_status()
        dates = resp.json().get("expirations", {}).get("date", [])
        if isinstance(dates, str):
            dates = [dates]

        today    = date.today()
        expiries = [
            d for d in dates
            if DEEP_FETCH_DTE_MIN <= (datetime.strptime(d, "%Y-%m-%d").date() - today).days <= DEEP_FETCH_DTE_MAX
        ]
        if not expiries:
            return {"expiry": None, "options": [], "source": "tradier"}

        expiry = expiries[0]
        resp = requests.get(
            f"{TRADIER_BASE}/markets/options/chains",
            headers=headers,
            params={"symbol": ticker, "expiration": expiry, "greeks": "true"},
            timeout=15,
        )
        if resp.status_code == 429:
            logger.warning(f"[{ErrorCode.E1008}] Tradier rate-limited for {ticker} — sleeping 60s")
            time.sleep(60)
            resp = requests.get(
                f"{TRADIER_BASE}/markets/options/chains",
                headers=headers,
                params={"symbol": ticker, "expiration": expiry, "greeks": "true"},
                timeout=15,
            )
        resp.raise_for_status()
        options = resp.json().get("options", {}).get("option", [])
        if isinstance(options, dict):
            options = [options]
        logger.info(f"Tradier chain for {ticker} ({expiry}): {len(options)} contracts")
        return {"expiry": expiry, "options": options or [], "source": "tradier"}

    except Exception as e:
        logger.error(f"[{ErrorCode.E1008}] Tradier chain failed for {ticker}: {e}")
        return {"expiry": None, "options": [], "source": "tradier"}


# ── Liquidity helpers (source-agnostic) ─────────────────────────────────────

def compute_bid_ask_pct(option: dict) -> float:
    """(ask - bid) / mid. Returns 1.0 (100%) if mid is zero."""
    bid = float(option.get("bid", 0) or 0)
    ask = float(option.get("ask", 0) or 0)
    mid = (bid + ask) / 2
    if mid <= 0:
        return 1.0
    return round((ask - bid) / mid, 4)


def filter_liquid_strikes(options: list[dict]) -> list[dict]:
    """
    Remove options that fail the liquidity gate:
      bid/ask spread > 10% of mid  OR  open_interest < 500
    """
    liquid  = []
    removed = 0
    for opt in options:
        ba_pct = compute_bid_ask_pct(opt)
        oi     = int(opt.get("open_interest", 0) or 0)
        if ba_pct > LIQUIDITY_MAX_BID_ASK_PCT or oi < LIQUIDITY_MIN_OI:
            removed += 1
            logger.debug(
                f"[{ErrorCode.E1008}] Filtered illiquid strike {opt.get('strike')} "
                f"{opt.get('option_type')} — bid/ask={ba_pct:.1%} OI={oi}"
            )
        else:
            opt["_bid_ask_pct"] = ba_pct
            liquid.append(opt)
    if removed:
        logger.debug(f"Liquidity gate removed {removed} strikes")
    return liquid
