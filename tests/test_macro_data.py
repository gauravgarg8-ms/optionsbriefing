import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import responses as resp_lib

from config import FINNHUB_BASE, FRED_BASE
from data.macro_data import fetch_macro_calendar, fetch_rates

FIXTURES = Path(__file__).parent / "fixtures"

TODAY    = date.today()
DATE_2D  = (TODAY + timedelta(days=2)).strftime("%Y-%m-%d 08:30:00")
DATE_5D  = (TODAY + timedelta(days=5)).strftime("%Y-%m-%d 08:30:00")
DATE_12D = (TODAY + timedelta(days=12)).strftime("%Y-%m-%d 08:30:00")

FINNHUB_ECONOMIC_URL = f"{FINNHUB_BASE}/calendar/economic"


class TestFetchMacroCalendar:
    def _payload(self, events):
        return {"economicCalendar": [
            {"event": e, "time": t, "country": "US", "impact": "High"}
            for e, t in events
        ]}

    @resp_lib.activate
    def test_returns_list_with_alert_levels(self):
        payload = self._payload([("CPI", DATE_2D), ("NFP", DATE_5D), ("GDP", DATE_12D)])
        resp_lib.add(resp_lib.GET, FINNHUB_ECONOMIC_URL, json=payload, status=200)
        result = fetch_macro_calendar(days_ahead=14)
        assert len(result) == 3

    @resp_lib.activate
    def test_high_alert_at_2_days(self):
        payload = self._payload([("CPI", DATE_2D)])
        resp_lib.add(resp_lib.GET, FINNHUB_ECONOMIC_URL, json=payload, status=200)
        result = fetch_macro_calendar(days_ahead=14)
        assert result[0]["alert"] == "HIGH ALERT"

    @resp_lib.activate
    def test_watch_at_5_days(self):
        payload = self._payload([("NFP", DATE_5D)])
        resp_lib.add(resp_lib.GET, FINNHUB_ECONOMIC_URL, json=payload, status=200)
        result = fetch_macro_calendar(days_ahead=14)
        assert result[0]["alert"] == "WATCH"

    @resp_lib.activate
    def test_monitor_at_12_days(self):
        payload = self._payload([("GDP", DATE_12D)])
        resp_lib.add(resp_lib.GET, FINNHUB_ECONOMIC_URL, json=payload, status=200)
        result = fetch_macro_calendar(days_ahead=14)
        assert result[0]["alert"] == "MONITOR"

    @resp_lib.activate
    def test_failure_returns_empty(self):
        resp_lib.add(resp_lib.GET, FINNHUB_ECONOMIC_URL, json={}, status=503)
        result = fetch_macro_calendar()
        assert result == []

    @resp_lib.activate
    def test_non_us_events_excluded(self):
        payload = {"economicCalendar": [
            {"event": "ECB Rate", "time": DATE_2D, "country": "EU", "impact": "High"},
            {"event": "CPI",      "time": DATE_2D, "country": "US", "impact": "High"},
        ]}
        resp_lib.add(resp_lib.GET, FINNHUB_ECONOMIC_URL, json=payload, status=200)
        result = fetch_macro_calendar(days_ahead=14)
        assert len(result) == 1
        assert result[0]["event"] == "CPI"

    @resp_lib.activate
    def test_is_high_impact_flagged(self):
        payload = self._payload([("FOMC Statement", DATE_2D)])
        resp_lib.add(resp_lib.GET, FINNHUB_ECONOMIC_URL, json=payload, status=200)
        result = fetch_macro_calendar(days_ahead=14)
        assert result[0]["is_high_impact"] is True


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
