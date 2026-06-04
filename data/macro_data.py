from datetime import date, datetime, timedelta

import requests
import yfinance as yf
from loguru import logger

from config import FINNHUB_API_KEY, FINNHUB_BASE, FRED_API_KEY, FRED_BASE, FRED_TBILL_SERIES
from errors import ErrorCode

# Events that trigger HIGH ALERT / WATCH flags
HIGH_IMPACT_EVENTS = {"FOMC", "Federal Reserve", "CPI", "NFP", "Non-Farm", "PCE",
                      "GDP", "PPI", "Retail Sales", "Unemployment"}


def fetch_macro_calendar(days_ahead: int = 14) -> list[dict]:
    """
    Fetch upcoming US macro events from Finnhub Economic Calendar.
    (FMP economic calendar deprecated for new accounts after Aug 2025.)
    Returns list of {event, date, days_away, alert_level, impact}.
    """
    today    = date.today()
    end_date = today + timedelta(days=days_ahead)
    try:
        resp = requests.get(
            f"{FINNHUB_BASE}/calendar/economic",
            params={"token": FINNHUB_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("economicCalendar", [])

        results = []
        for item in items:
            # Filter US events only
            if item.get("country", "").upper() != "US":
                continue
            event    = item.get("event", "")
            raw_time = item.get("time", "")
            impact   = item.get("impact", "")
            if not event or not raw_time:
                continue
            try:
                event_date = datetime.strptime(raw_time[:10], "%Y-%m-%d").date()
                days_away  = (event_date - today).days
                if days_away < 0 or days_away > days_ahead:
                    continue
            except ValueError:
                continue

            raw_date = event_date.isoformat()
            # Alert level
            if days_away <= 3:
                alert = "HIGH ALERT"
            elif days_away <= 7:
                alert = "WATCH"
            else:
                alert = "MONITOR"

            results.append({
                "event":      event,
                "date":       raw_date[:10],
                "days_away":  days_away,
                "alert":      alert,
                "impact":     impact,
                "is_high_impact": any(k.lower() in event.lower() for k in HIGH_IMPACT_EVENTS),
            })

        logger.info(f"Finnhub macro calendar: {len(results)} US events in next {days_ahead} days")
        return sorted(results, key=lambda x: x["days_away"])

    except requests.RequestException as e:
        logger.error(f"[{ErrorCode.E1005}] Finnhub macro calendar fetch failed: {e}")
        return []


def fetch_rates() -> dict:
    """
    Fetch interest rates and DXY.
    Returns {yield_10y, tbill_3m, dxy}.
    Defaults to (4.0, 0.051, 104.0) on failure.
    """
    result = {"yield_10y": 4.0, "tbill_3m": 0.051, "dxy": 104.0}

    # 10-year Treasury yield via yfinance
    try:
        tnx = yf.download("^TNX", period="5d", interval="1d", progress=False, auto_adjust=True)
        if not tnx.empty:
            raw = tnx["Close"].squeeze()
            val = float(raw.iloc[-1]) if hasattr(raw, "iloc") else float(raw)
            result["yield_10y"] = round(val / 100, 4)   # ^TNX reports as %, store as decimal
    except Exception as e:
        logger.warning(f"[{ErrorCode.E1006}] 10Y yield fetch failed: {e}")

    # 3-month T-bill via FRED
    try:
        resp = requests.get(
            FRED_BASE,
            params={
                "series_id":       FRED_TBILL_SERIES,
                "api_key":         FRED_API_KEY,
                "file_type":       "json",
                "sort_order":      "desc",
                "limit":           5,
                "observation_start": date.today().replace(month=date.today().month - 1
                                     if date.today().month > 1 else 12,
                                     year=date.today().year if date.today().month > 1
                                     else date.today().year - 1).isoformat(),
            },
            timeout=10,
        )
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        for obs in observations:
            val_str = obs.get("value", ".")
            if val_str != ".":
                result["tbill_3m"] = round(float(val_str) / 100, 4)
                break
    except Exception as e:
        logger.warning(f"[{ErrorCode.E1006}] FRED T-bill fetch failed: {e} — using default 5.1%")

    # DXY (US Dollar Index) via yfinance
    try:
        dxy = yf.download("DX-Y.NYB", period="5d", interval="1d", progress=False, auto_adjust=True)
        if not dxy.empty:
            raw = dxy["Close"].squeeze()
            val = float(raw.iloc[-1]) if hasattr(raw, "iloc") else float(raw)
            result["dxy"] = round(val, 2)
    except Exception as e:
        logger.warning(f"[{ErrorCode.E1006}] DXY fetch failed: {e}")

    logger.info(f"Rates — yield_10y={result['yield_10y']:.2%}, tbill_3m={result['tbill_3m']:.2%}, dxy={result['dxy']}")
    return result
