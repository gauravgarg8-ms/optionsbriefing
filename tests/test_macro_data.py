"""
Tests for macro_data.py — multi-source economic calendar (FRED + FOMC + BEA + TE).
All external HTTP calls are mocked with the `responses` library.
"""
import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import responses as resp_lib

from config import FRED_BASE, FED_FOMC_URL, BEA_SCHEDULE_URL
from data.macro_data import fetch_macro_calendar, fetch_rates, _build_event, _parse_date_no_year

FIXTURES = Path(__file__).parent / "fixtures"

TODAY = date.today()

# ── Fixture helpers ───────────────────────────────────────────────────────────

def _fred_releases_url():
    return "https://api.stlouisfed.org/fred/releases/dates"


def _fred_payload(release_id: int, release_name: str, days_away: int) -> dict:
    event_date = (TODAY + timedelta(days=days_away)).isoformat()
    return {
        "release_dates": [
            {"release_id": release_id, "release_name": release_name, "date": event_date}
        ]
    }


# Minimal FOMC HTML mimicking the real federalreserve.gov structure
def _fomc_html(days_away: int) -> str:
    meeting_date = TODAY + timedelta(days=days_away)
    # Use end day = days_away, start = days_away - 1
    start_day = meeting_date.day - 1 if meeting_date.day > 1 else meeting_date.day
    end_day   = meeting_date.day
    import calendar
    month_name = calendar.month_name[meeting_date.month]
    year       = meeting_date.year
    return f"""
    <html><body>
    <div class="panel panel-default">
      <div class="panel-heading">
        <h4><a id="123">{year} FOMC Meetings</a></h4>
      </div>
      <div class="row fomc-meeting">
        <div class="fomc-meeting__month col-xs-5"><strong>{month_name}</strong></div>
        <div class="fomc-meeting__date col-xs-4">{start_day}-{end_day}*</div>
      </div>
    </div>
    </body></html>
    """


# Minimal BEA HTML mimicking bea.gov/news/schedule table structure
def _bea_html(title: str, days_away: int) -> str:
    import calendar
    d = TODAY + timedelta(days=days_away)
    month_name = calendar.month_name[d.month]
    day        = d.day
    return f"""
    <html><body>
    <table>
      <tr><th>Year {d.year}</th><th></th><th>Release</th></tr>
      <tr>
        <td><div class="release-date">{month_name} {day}</div><small class="text-muted">8:30 AM</small></td>
        <td><div class="icon-letter"><span class="caps">N</span></div></td>
        <td>{title}</td>
      </tr>
    </table>
    </body></html>
    """



# ── Tests: fetch_macro_calendar ───────────────────────────────────────────────

class TestFetchMacroCalendar:

    @resp_lib.activate
    def test_returns_events_from_fred(self):
        resp_lib.add(resp_lib.GET, _fred_releases_url(),
                     json=_fred_payload(10, "Consumer Price Index (CPI)", 5), status=200)
        resp_lib.add(resp_lib.GET, FED_FOMC_URL, body="<html><body></body></html>", status=200)
        resp_lib.add(resp_lib.GET, BEA_SCHEDULE_URL, body="<html><body></body></html>", status=200)
        result = fetch_macro_calendar(days_ahead=14)
        assert len(result) == 1
        assert result[0]["event"] == "Consumer Price Index (CPI)"

    @resp_lib.activate
    def test_alert_levels(self):
        # release_name in payload is ignored — code uses _FRED_HIGH_IMPACT_IDS labels
        payload = {
            "release_dates": [
                {"release_id": 10,  "release_name": "ignored", "date": (TODAY + timedelta(days=2)).isoformat()},
                {"release_id": 46,  "release_name": "ignored", "date": (TODAY + timedelta(days=5)).isoformat()},
                {"release_id": 53,  "release_name": "ignored", "date": (TODAY + timedelta(days=12)).isoformat()},
            ]
        }
        resp_lib.add(resp_lib.GET, _fred_releases_url(), json=payload, status=200)
        resp_lib.add(resp_lib.GET, FED_FOMC_URL, body="<html><body></body></html>", status=200)
        resp_lib.add(resp_lib.GET, BEA_SCHEDULE_URL, body="<html><body></body></html>", status=200)
        result = fetch_macro_calendar(days_ahead=14)
        alerts = {r["event"]: r["alert"] for r in result}
        assert alerts["Consumer Price Index (CPI)"] == "HIGH ALERT"
        assert alerts["Producer Price Index (PPI)"] == "WATCH"
        assert alerts["Gross Domestic Product (GDP)"] == "MONITOR"

    @resp_lib.activate
    def test_is_high_impact_flagged_for_cpi(self):
        resp_lib.add(resp_lib.GET, _fred_releases_url(),
                     json=_fred_payload(10, "Consumer Price Index (CPI)", 3), status=200)
        resp_lib.add(resp_lib.GET, FED_FOMC_URL, body="<html><body></body></html>", status=200)
        resp_lib.add(resp_lib.GET, BEA_SCHEDULE_URL, body="<html><body></body></html>", status=200)
        result = fetch_macro_calendar(days_ahead=14)
        assert result[0]["is_high_impact"] is True

    @resp_lib.activate
    def test_fomc_page_events_included(self):
        resp_lib.add(resp_lib.GET, _fred_releases_url(), json={"release_dates": []}, status=200)
        resp_lib.add(resp_lib.GET, FED_FOMC_URL, body=_fomc_html(days_away=8), status=200)
        resp_lib.add(resp_lib.GET, BEA_SCHEDULE_URL, body="<html><body></body></html>", status=200)
        result = fetch_macro_calendar(days_ahead=14)
        assert any("FOMC" in r["event"] for r in result)

    @resp_lib.activate
    def test_bea_events_included(self):
        resp_lib.add(resp_lib.GET, _fred_releases_url(), json={"release_dates": []}, status=200)
        resp_lib.add(resp_lib.GET, FED_FOMC_URL, body="<html><body></body></html>", status=200)
        resp_lib.add(resp_lib.GET, BEA_SCHEDULE_URL,
                     body=_bea_html("Gross Domestic Product (Second Estimate)", days_away=10), status=200)
        result = fetch_macro_calendar(days_ahead=14)
        assert any("Gross Domestic Product" in r["event"] for r in result)

    @resp_lib.activate
    def test_deduplication_by_date_and_name(self):
        """Same event from FRED and BEA on same date should appear once."""
        gdp_date = (TODAY + timedelta(days=10)).isoformat()
        payload = {"release_dates": [
            {"release_id": 53, "release_name": "Gross Domestic Product (GDP)", "date": gdp_date}
        ]}
        resp_lib.add(resp_lib.GET, _fred_releases_url(), json=payload, status=200)
        resp_lib.add(resp_lib.GET, FED_FOMC_URL, body="<html><body></body></html>", status=200)
        resp_lib.add(resp_lib.GET, BEA_SCHEDULE_URL,
                     body=_bea_html("Gross Domestic Product (GDP)", days_away=10), status=200)
        result = fetch_macro_calendar(days_ahead=14)
        gdp_events = [r for r in result if "Gross Domestic Product" in r["event"]]
        assert len(gdp_events) == 1

    @resp_lib.activate
    def test_all_sources_fail_returns_empty(self):
        resp_lib.add(resp_lib.GET, _fred_releases_url(), status=503)
        resp_lib.add(resp_lib.GET, FED_FOMC_URL, status=503)
        resp_lib.add(resp_lib.GET, BEA_SCHEDULE_URL, status=503)
        result = fetch_macro_calendar()
        assert result == []

    @resp_lib.activate
    def test_events_outside_window_excluded(self):
        far_date = (TODAY + timedelta(days=30)).isoformat()
        payload = {"release_dates": [
            {"release_id": 10, "release_name": "CPI", "date": far_date}
        ]}
        resp_lib.add(resp_lib.GET, _fred_releases_url(), json=payload, status=200)
        resp_lib.add(resp_lib.GET, FED_FOMC_URL, body="<html><body></body></html>", status=200)
        resp_lib.add(resp_lib.GET, BEA_SCHEDULE_URL, body="<html><body></body></html>", status=200)
        result = fetch_macro_calendar(days_ahead=14)
        assert len(result) == 0

    @resp_lib.activate
    def test_sorted_by_days_away(self):
        payload = {"release_dates": [
            {"release_id": 53,  "release_name": "GDP", "date": (TODAY + timedelta(days=12)).isoformat()},
            {"release_id": 10,  "release_name": "CPI", "date": (TODAY + timedelta(days=3)).isoformat()},
        ]}
        resp_lib.add(resp_lib.GET, _fred_releases_url(), json=payload, status=200)
        resp_lib.add(resp_lib.GET, FED_FOMC_URL, body="<html><body></body></html>", status=200)
        resp_lib.add(resp_lib.GET, BEA_SCHEDULE_URL, body="<html><body></body></html>", status=200)
        result = fetch_macro_calendar(days_ahead=14)
        days = [r["days_away"] for r in result]
        assert days == sorted(days)



# ── Tests: helper functions ───────────────────────────────────────────────────

class TestBuildEvent:
    def test_high_alert_at_2_days(self):
        ev = _build_event("CPI", TODAY + timedelta(days=2), TODAY)
        assert ev["alert"] == "HIGH ALERT"

    def test_watch_at_5_days(self):
        ev = _build_event("NFP", TODAY + timedelta(days=5), TODAY)
        assert ev["alert"] == "WATCH"

    def test_monitor_at_12_days(self):
        ev = _build_event("GDP", TODAY + timedelta(days=12), TODAY)
        assert ev["alert"] == "MONITOR"

    def test_is_high_impact_fomc(self):
        ev = _build_event("FOMC Meeting", TODAY + timedelta(days=3), TODAY)
        assert ev["is_high_impact"] is True

    def test_event_dict_has_all_keys(self):
        ev = _build_event("CPI", TODAY + timedelta(days=5), TODAY, impact="high")
        for key in ["event", "date", "days_away", "alert", "impact", "is_high_impact"]:
            assert key in ev


class TestParseDateNoYear:
    def test_full_month_name(self):
        d = _parse_date_no_year("June 25", TODAY)
        assert d is not None
        assert d.month == 6
        assert d.day == 25

    def test_abbreviated_month(self):
        d = _parse_date_no_year("Jul 4", TODAY)
        assert d is not None
        assert d.month == 7

    def test_past_date_advances_year(self):
        # "January 10" parsed in June means it's 5 months in the past → advance to next year
        d = _parse_date_no_year("January 10", date(2026, 6, 15))
        assert d is not None
        assert d.year == 2027

    def test_invalid_returns_none(self):
        d = _parse_date_no_year("not a date", TODAY)
        assert d is None


# ── Tests: fetch_rates ────────────────────────────────────────────────────────

class TestFetchRates:
    @patch("data.macro_data.yf.download")
    @resp_lib.activate
    def test_returns_all_rate_keys(self, mock_download):
        dates = pd.bdate_range(end="2026-05-29", periods=3)
        mock_download.return_value = pd.DataFrame({"Close": [428.0, 429.0, 430.0]}, index=dates)
        resp_lib.add(resp_lib.GET, FRED_BASE,
                     json={"observations": [{"value": "5.10", "date": "2026-05-28"}]},
                     status=200)
        result = fetch_rates()
        for key in ["yield_10y", "tbill_3m", "dxy"]:
            assert key in result

    @patch("data.macro_data.yf.download")
    @resp_lib.activate
    def test_tbill_stored_as_decimal(self, mock_download):
        dates = pd.bdate_range(end="2026-05-29", periods=3)
        mock_download.return_value = pd.DataFrame({"Close": [428.0, 429.0, 430.0]}, index=dates)
        resp_lib.add(resp_lib.GET, FRED_BASE,
                     json={"observations": [{"value": "5.10", "date": "2026-05-28"}]},
                     status=200)
        result = fetch_rates()
        assert result["tbill_3m"] < 0.10

    @patch("data.macro_data.yf.download")
    @resp_lib.activate
    def test_fred_failure_returns_default(self, mock_download):
        dates = pd.bdate_range(end="2026-05-29", periods=3)
        mock_download.return_value = pd.DataFrame({"Close": [428.0, 429.0, 430.0]}, index=dates)
        resp_lib.add(resp_lib.GET, FRED_BASE, body=Exception("Connection error"))
        result = fetch_rates()
        assert result["tbill_3m"] == 0.051
