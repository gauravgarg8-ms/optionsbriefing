from datetime import date, datetime, timedelta

import requests
from loguru import logger

from config import FINNHUB_API_KEY, FINNHUB_BASE
from errors import ErrorCode


def fetch_earnings_calendar(days_ahead: int = 14) -> list[dict]:
    """
    Fetch upcoming earnings from Finnhub (FMP deprecated for new accounts).
    Returns list of {ticker, date, days_away}.
    """
    today    = date.today()
    end_date = today + timedelta(days=days_ahead)
    try:
        resp = requests.get(
            f"{FINNHUB_BASE}/calendar/earnings",
            params={"from": today.isoformat(), "to": end_date.isoformat(), "token": FINNHUB_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("earningsCalendar", [])
        results = []
        for item in items:
            ticker   = item.get("symbol", "")
            raw_date = item.get("date", "")
            if not ticker or not raw_date:
                continue
            try:
                earn_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
                days_away = (earn_date - today).days
                if 0 <= days_away <= days_ahead:
                    results.append({"ticker": ticker, "date": raw_date, "days_away": days_away})
            except ValueError:
                continue
        logger.info(f"Finnhub earnings calendar: {len(results)} events in next {days_ahead} days")
        return results
    except requests.RequestException as e:
        logger.error(f"[{ErrorCode.E1009}] Finnhub earnings calendar fetch failed: {e}")
        return []


def fetch_earnings_history(ticker: str) -> list[dict]:
    """
    Fetch last 8 quarters of earnings from Finnhub.
    Returns list of {period, actual, estimate, surprise_pct, price_change_pct}.
    """
    try:
        resp = requests.get(
            f"{FINNHUB_BASE}/stock/earnings",
            params={"symbol": ticker, "limit": 8, "token": FINNHUB_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return []
        results = []
        for item in data:
            actual   = item.get("actual")
            estimate = item.get("estimate")
            surprise = None
            if actual is not None and estimate and estimate != 0:
                surprise = round((actual - estimate) / abs(estimate) * 100, 2)
            results.append({
                "period":          item.get("period", ""),
                "actual":          actual,
                "estimate":        estimate,
                "surprise_pct":    surprise,
                "price_change_pct": item.get("priceChangePercent"),
            })
        return results
    except requests.RequestException as e:
        logger.error(f"[{ErrorCode.E1009}] Finnhub earnings history failed for {ticker}: {e}")
        return []


def compute_implied_move(chain_options: list[dict], stock_price: float) -> float | None:
    """
    Implied move = (ATM call + ATM put) / stock_price.
    Uses the nearest ATM strike from the options chain.
    """
    if not chain_options or stock_price <= 0:
        return None
    try:
        atm_strike = min(
            {float(o["strike"]) for o in chain_options},
            key=lambda k: abs(k - stock_price),
        )
        calls = [o for o in chain_options if o.get("option_type") == "call" and float(o["strike"]) == atm_strike]
        puts  = [o for o in chain_options if o.get("option_type") == "put"  and float(o["strike"]) == atm_strike]
        if not calls or not puts:
            return None
        call_mid = (float(calls[0].get("bid", 0)) + float(calls[0].get("ask", 0))) / 2
        put_mid  = (float(puts[0].get("bid",  0)) + float(puts[0].get("ask",  0))) / 2
        implied  = (call_mid + put_mid) / stock_price
        return round(float(implied), 4)
    except Exception as e:
        logger.warning(f"compute_implied_move failed: {e}")
        return None


def compute_hist_avg_move(history: list[dict]) -> float | None:
    """Mean absolute % price change over available quarters."""
    moves = [abs(q["price_change_pct"]) for q in history if q.get("price_change_pct") is not None]
    if not moves:
        return None
    return round(sum(moves) / len(moves) / 100, 4)   # return as decimal (e.g. 0.065 = 6.5%)


def fetch_analyst_ratings(ticker: str) -> list[dict]:
    """Fetch recent analyst ratings from Finnhub."""
    try:
        resp = requests.get(
            f"{FINNHUB_BASE}/stock/recommendation",
            params={"symbol": ticker, "token": FINNHUB_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return []
        results = []
        for item in data[:3]:   # last 3 periods
            results.append({
                "period":      item.get("period", ""),
                "buy":         item.get("buy", 0),
                "hold":        item.get("hold", 0),
                "sell":        item.get("sell", 0),
                "strong_buy":  item.get("strongBuy", 0),
                "strong_sell": item.get("strongSell", 0),
            })
        return results
    except requests.RequestException as e:
        logger.warning(f"[{ErrorCode.E1009}] Analyst ratings failed for {ticker}: {e}")
        return []


def fetch_insider_transactions(ticker: str) -> list[dict]:
    """Fetch recent insider transactions from Finnhub."""
    try:
        resp = requests.get(
            f"{FINNHUB_BASE}/stock/insider-transactions",
            params={"symbol": ticker, "token": FINNHUB_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        transactions = data.get("data", [])
        if not isinstance(transactions, list):
            return []
        results = []
        for t in transactions[:5]:
            results.append({
                "name":            t.get("name", ""),
                "share":           t.get("share", 0),
                "transaction_type": t.get("transactionType", ""),
                "transaction_date": t.get("transactionDate", ""),
                "change":           t.get("change", 0),
            })
        return results
    except requests.RequestException as e:
        logger.warning(f"[{ErrorCode.E1009}] Insider transactions failed for {ticker}: {e}")
        return []


def classify_earnings_candidate(ticker: str, earnings_list: list[dict], days_away_threshold: int = 7) -> dict:
    """
    Check if ticker has earnings within threshold and return enriched data.
    Returns {has_earnings, earnings_date, days_away, is_earnings_candidate}.
    """
    for e in earnings_list:
        if e["ticker"] == ticker and 0 <= e["days_away"] <= days_away_threshold:
            return {
                "has_earnings":        True,
                "earnings_date":       e["date"],
                "days_away":           e["days_away"],
                "is_earnings_candidate": True,
            }
    return {"has_earnings": False, "earnings_date": None, "days_away": None, "is_earnings_candidate": False}
