"""
Macro/economic calendar from multi-source fetch:

  1. FRED releases/dates API  — covers CPI, PPI, NFP, GDP, PCE, Retail Sales,
                                Jobless Claims, FOMC (using existing FRED key).
  2. Federal Reserve FOMC page — scraped for meeting-level detail.
  3. BEA news schedule         — scraped for detailed GDP/PCE release names.

All sources degrade gracefully to []. Results are merged and deduplicated.
Finnhub /calendar/economic deprecated for new accounts (Aug 2025).
"""
import re
from datetime import date, datetime, timedelta

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from loguru import logger

from config import (
    FRED_API_KEY, FRED_BASE, FRED_TBILL_SERIES,
    FED_FOMC_URL, BEA_SCHEDULE_URL,
)
from errors import ErrorCode

# ── Constants ────────────────────────────────────────────────────────────────

# High-impact FRED release IDs (market-moving US data only)
_FRED_HIGH_IMPACT_IDS: dict[int, str] = {
    10:  "Consumer Price Index (CPI)",
    46:  "Producer Price Index (PPI)",
    50:  "Employment Situation (NFP)",
    53:  "Gross Domestic Product (GDP)",
    54:  "Personal Income and Outlays (PCE)",
    9:   "Retail Sales",
    180: "Jobless Claims",
    101: "FOMC Press Release",
    51:  "Trade Balance",
    13:  "Industrial Production",
}

# Keyword subset used for is_high_impact flag
HIGH_IMPACT_EVENTS = {
    "FOMC", "Federal Reserve", "CPI", "NFP", "Non-Farm", "PCE",
    "GDP", "PPI", "Retail Sales", "Unemployment", "Employment",
}

# BEA release titles we want (others are obscure/low-impact)
_BEA_WHITELIST = {
    "Gross Domestic Product", "Personal Income", "Personal Consumption",
    "GDP", "PCE",
}

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_macro_calendar(days_ahead: int = 14) -> list[dict]:
    """
    Multi-source US macro calendar.
    Returns list of {event, date, days_away, alert, impact, is_high_impact}.
    """
    today  = date.today()
    events: dict[tuple, dict] = {}   # (iso_date, name_key) → event dict, for dedup

    # Layer 1: FRED releases API (covers CPI/PPI/NFP/GDP/PCE/Jobless/FOMC/Retail)
    for ev in _fetch_fred_releases(today, days_ahead):
        _upsert(events, ev)

    # Layer 2: Federal Reserve FOMC page (meeting-level detail)
    for ev in _fetch_fomc_page(today, days_ahead):
        _upsert(events, ev)

    # Layer 3: BEA schedule (detailed GDP/PCE release names)
    for ev in _fetch_bea_schedule(today, days_ahead):
        _upsert(events, ev)

    results = sorted(events.values(), key=lambda x: x["days_away"])
    logger.info(f"Macro calendar: {len(results)} US events in next {days_ahead} days")
    return results


# ── Source 1: FRED releases/dates ────────────────────────────────────────────

def _fetch_fred_releases(today: date, days_ahead: int) -> list[dict]:
    """
    Query FRED /releases/dates for the next days_ahead days.
    Filters to _FRED_HIGH_IMPACT_IDS whitelist — covers BLS + BEA + FOMC data.
    """
    end_date = today + timedelta(days=days_ahead)
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/releases/dates",
            params={
                "api_key":    FRED_API_KEY,
                "file_type":  "json",
                "realtime_start": today.isoformat(),
                "realtime_end":   end_date.isoformat(),
                "limit":      500,
                "include_release_dates_with_no_data": "false",
            },
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("release_dates", [])

        results = []
        for item in items:
            release_id = item.get("release_id")
            if release_id not in _FRED_HIGH_IMPACT_IDS:
                continue
            event_name = _FRED_HIGH_IMPACT_IDS[release_id]
            try:
                event_date = datetime.strptime(item["date"], "%Y-%m-%d").date()
            except (ValueError, KeyError):
                continue
            days_away = (event_date - today).days
            if 0 <= days_away <= days_ahead:
                results.append(_build_event(event_name, event_date, today, impact="high"))

        logger.debug(f"FRED releases: {len(results)} high-impact events in next {days_ahead} days")
        return results

    except Exception as e:
        logger.warning(f"FRED releases fetch failed: {e}")
        return []


# ── Source 2: Federal Reserve FOMC page ──────────────────────────────────────

def _fetch_fomc_page(today: date, days_ahead: int) -> list[dict]:
    """
    Parse FOMC meeting dates from federalreserve.gov.
    The page uses CSS classes: fomc-meeting__month and fomc-meeting__date.
    Extracts current and next year meeting dates.
    """
    try:
        resp = requests.get(FED_FOMC_URL, headers=_BROWSER_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        # Each year block: <div class="panel panel-default"><h4>2026 FOMC Meetings</h4>
        for panel in soup.select("div.panel.panel-default"):
            heading = panel.find("h4")
            if not heading:
                continue
            year_match = re.search(r"\b(20\d{2})\b", heading.get_text())
            if not year_match:
                continue
            year = int(year_match.group(1))
            if year < today.year or year > today.year + 1:
                continue

            for meeting in panel.select("div.fomc-meeting"):
                month_div = meeting.find(class_="fomc-meeting__month")
                date_div  = meeting.find(class_="fomc-meeting__date")
                if not month_div or not date_div:
                    continue
                month_name = month_div.get_text(strip=True)
                date_text  = date_div.get_text(strip=True).rstrip("*").strip()

                # date_text can be "27-28" (same month) or "30-1" (spans months)
                parts = date_text.split("-")
                # Use the last day — that's the decision/statement day
                day_str = parts[-1].strip() if len(parts) > 1 else parts[0].strip()
                try:
                    day = int(day_str)
                except ValueError:
                    continue

                # Handle cross-month meetings ("April 29-30" → day=30; or "April 30 - May 1")
                # When last day < first day the meeting crosses into the next month
                try:
                    first_day = int(parts[0].strip())
                except ValueError:
                    first_day = day

                month_offset = 0
                if len(parts) > 1 and day < first_day:
                    month_offset = 1   # decision day rolls into next calendar month

                try:
                    base_date   = datetime.strptime(f"{year}-{month_name}-1", "%Y-%B-%d").date()
                    event_date  = base_date.replace(day=1)
                    # Advance to the correct month+day
                    if month_offset:
                        # Increment month
                        m = base_date.month + 1
                        y = base_date.year + (1 if m > 12 else 0)
                        m = m if m <= 12 else 1
                        event_date = date(y, m, day)
                    else:
                        event_date = base_date.replace(day=day)
                except (ValueError, OverflowError):
                    continue

                days_away = (event_date - today).days
                if 0 <= days_away <= days_ahead:
                    results.append(_build_event("FOMC Meeting", event_date, today, impact="high"))

        logger.debug(f"FOMC page: {len(results)} meetings in next {days_ahead} days")
        return results

    except Exception as e:
        logger.warning(f"FOMC page scrape failed: {e}")
        return []


# ── Source 3: BEA News Schedule ──────────────────────────────────────────────

def _fetch_bea_schedule(today: date, days_ahead: int) -> list[dict]:
    """
    Parse GDP, PCE, and Personal Income releases from bea.gov/news/schedule.
    Table structure: col0 = release-date div, col2 = release name.
    """
    try:
        resp = requests.get(BEA_SCHEDULE_URL, headers=_BROWSER_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        table = soup.find("table")
        if not table:
            return []

        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            # Date is in a <div class="release-date"> inside cell 0
            date_div = cells[0].find(class_="release-date")
            if not date_div:
                continue
            date_text = date_div.get_text(strip=True)   # e.g. "June 25"
            title     = cells[2].get_text(" ", strip=True)

            if not any(k.lower() in title.lower() for k in _BEA_WHITELIST):
                continue

            # BEA page shows dates without year — infer from today's year
            event_date = _parse_date_no_year(date_text, today)
            if not event_date:
                continue
            days_away = (event_date - today).days
            if 0 <= days_away <= days_ahead:
                results.append(_build_event(title[:80], event_date, today, impact="high"))

        logger.debug(f"BEA schedule: {len(results)} events in next {days_ahead} days")
        return results

    except Exception as e:
        logger.warning(f"BEA schedule scrape failed: {e}")
        return []


# ── Helpers ──────────────────────────────────────────────────────────────────

def _upsert(events: dict, ev: dict) -> None:
    """Dedup by (date, first-30-chars of name). Higher impact wins on collision."""
    key = (ev["date"], ev["event"].lower()[:30])
    existing = events.get(key)
    if not existing or (ev.get("impact") == "high" and existing.get("impact") != "high"):
        events[key] = ev


def _build_event(event_name: str, event_date: date, today: date, impact: str = "medium") -> dict:
    days_away = (event_date - today).days
    if days_away <= 3:
        alert = "HIGH ALERT"
    elif days_away <= 7:
        alert = "WATCH"
    else:
        alert = "MONITOR"
    return {
        "event":          event_name,
        "date":           event_date.isoformat(),
        "days_away":      days_away,
        "alert":          alert,
        "impact":         impact,
        "is_high_impact": any(k.lower() in event_name.lower() for k in HIGH_IMPACT_EVENTS),
    }


def _parse_date_no_year(text: str, today: date) -> date | None:
    """
    Parse a month-day string (e.g. "June 25") into a full date, using today's year.
    If the resulting date is already past, advance to next year.
    """
    text = text.strip()
    for fmt in ("%B %d %Y", "%b %d %Y"):
        try:
            d = datetime.strptime(f"{text} {today.year}", fmt).date()
            # Advance year if the date is more than 3 months in the past
            if (today - d).days > 90:
                d = d.replace(year=today.year + 1)
            return d
        except ValueError:
            continue
    return None


# ── Rates (unchanged) ─────────────────────────────────────────────────────────

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

    logger.info(
        f"Rates — yield_10y={result['yield_10y']:.2%}, "
        f"tbill_3m={result['tbill_3m']:.2%}, dxy={result['dxy']}"
    )
    return result
