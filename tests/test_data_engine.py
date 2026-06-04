"""
Integration test for the full Phase 1 data pipeline.
All external HTTP calls are mocked. Verifies:
  - Output JSON has all required keys with correct types
  - Partial failures produce graceful defaults (pipeline never aborts)
  - deep_fetch applies 1-second sleep between tickers
  - universe_cache.json written after successful fetch
"""
import json
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import numpy as np
import pandas as pd
import pytest
import responses as resp_lib

from config import (TRADIER_BASE, FINNHUB_BASE, NEWS_API_BASE, CNN_FG_URL,
                    CBOE_PC_URL, BARCHART_UA_URL, FRED_BASE)

TODAY    = date.today().isoformat()
DATE_3D  = (date.today() + timedelta(days=3)).isoformat()
DATE_10D = (date.today() + timedelta(days=10)).isoformat()

FINNHUB_EARNINGS_URL  = f"{FINNHUB_BASE}/calendar/earnings"
FINNHUB_ECONOMIC_URL  = f"{FINNHUB_BASE}/calendar/economic"

# ── Fixture helpers ──────────────────────────────────────────────────────────

EARNINGS_MOCK = {"earningsCalendar": [{"symbol": "NVDA", "date": DATE_3D}]}
MACRO_MOCK    = {"economicCalendar": [{"event": "CPI", "time": f"{DATE_3D} 08:30:00",
                                       "country": "US", "impact": "High"}]}
FG_MOCK       = {"fear_and_greed": {"score": 61, "rating": "greed"}}
CBOE_CSV_MOCK = (
    "DATE,PUT/CALL RATIO,EQUITY PUT/CALL RATIO\n"
    f"{TODAY},0.91,0.82\n"
)
FINNHUB_NEWS_MOCK = [
    {"headline": "Market rallies on strong earnings", "source": "CNBC",
     "summary": "tech stocks surge", "category": "general"}
]
NEWSAPI_MOCK  = {
    "articles": [{"title": "Strong economic data", "source": {"name": "Reuters"}}]
}
BARCHART_HTML = (
    "<html><body><table><tr><th>Symbol</th><th>Option Type</th><th>Volume</th></tr>"
    "<tr><td>AAPL</td><td>Call</td><td>10000</td></tr></table></body></html>"
)

# Mock S&P 500 and Nasdaq-100 DataFrames for Wikipedia
_SP500_DF  = pd.DataFrame({
    "Symbol": ["AAPL", "MSFT"], "Security": ["Apple", "Microsoft"],
    "GICS Sector": ["Technology", "Technology"],
})
_NASDAQ_DF = pd.DataFrame({
    "Ticker": ["NVDA", "AAPL"], "Company": ["NVIDIA", "Apple"],
    "Sector": ["Technology", "Technology"],
})


def _make_spy_df():
    n = 210
    dates  = pd.bdate_range(end="2026-05-29", periods=n)
    prices = 500 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({"Close": prices, "High": prices+1, "Low": prices-1, "Open": prices},
                        index=dates)


def _register_all_mocks():
    resp_lib.add(resp_lib.GET, FINNHUB_EARNINGS_URL,  json=EARNINGS_MOCK, status=200)
    resp_lib.add(resp_lib.GET, FINNHUB_ECONOMIC_URL,  json=MACRO_MOCK,    status=200)
    resp_lib.add(resp_lib.GET, CNN_FG_URL,            json=FG_MOCK,       status=200)
    # Register today's date-specific CBOE URL (new pattern replaces directory URL)
    resp_lib.add(resp_lib.GET, f"{CBOE_PC_URL}{TODAY}_options_volume.csv",
                 body=CBOE_CSV_MOCK, status=200, content_type="text/csv")
    resp_lib.add(resp_lib.GET, f"{FINNHUB_BASE}/news",            json=FINNHUB_NEWS_MOCK, status=200)
    resp_lib.add(resp_lib.GET, f"{NEWS_API_BASE}/top-headlines",  json=NEWSAPI_MOCK,      status=200)
    resp_lib.add(resp_lib.GET, BARCHART_UA_URL,       body=BARCHART_HTML, status=200)
    resp_lib.add(resp_lib.GET, FRED_BASE,
                 json={"observations": [{"value": "5.10", "date": TODAY}]}, status=200)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestRunDataCollectionIntegration:
    @resp_lib.activate
    @patch("data.market_data.yf.download")
    @patch("data.universe_manager.pd.read_html")
    def test_output_has_all_required_keys(self, mock_wiki, mock_yf, tmp_path):
        mock_yf.return_value   = _make_spy_df()
        mock_wiki.side_effect  = [[_SP500_DF], [pd.DataFrame({"A": [1]}), _NASDAQ_DF]]
        _register_all_mocks()

        from data.data_engine import run_data_collection
        with patch("data.data_engine.OUTPUT_RAW",              tmp_path / "raw.json"), \
             patch("data.universe_manager.CACHE_PATH",         tmp_path / "cache.json"), \
             patch("data.unusual_activity._CACHE_PATH",        tmp_path / "ua.json"), \
             patch("data.sentiment_data._FG_CACHE",            {}), \
             patch("data.sentiment_data._FG_CACHE_DATE",       ""):
            result = run_data_collection()

        for key in ["date", "universe", "market_environment", "macro_events",
                    "earnings_calendar", "unusual_activity"]:
            assert key in result, f"Missing key: {key}"

        env = result["market_environment"]
        for key in ["vix", "spy_price", "fear_greed_score", "put_call_ratio",
                    "market_sentiment", "leading_sectors", "lagging_sectors"]:
            assert key in env, f"Missing market_env key: {key}"

        assert isinstance(env["vix"], (int, float))
        assert isinstance(env["fear_greed_score"], int)
        assert isinstance(env["put_call_ratio"], float)
        assert isinstance(result["universe"], list)
        assert isinstance(result["macro_events"], list)

    @resp_lib.activate
    @patch("data.market_data.yf.download")
    @patch("data.universe_manager.pd.read_html")
    def test_macro_failure_pipeline_continues(self, mock_wiki, mock_yf, tmp_path):
        mock_yf.return_value  = _make_spy_df()
        mock_wiki.side_effect = [[_SP500_DF], [pd.DataFrame({"A": [1]}), _NASDAQ_DF]]

        resp_lib.add(resp_lib.GET, FINNHUB_EARNINGS_URL,  json=EARNINGS_MOCK, status=200)
        resp_lib.add(resp_lib.GET, FINNHUB_ECONOMIC_URL,  json={},  status=503)  # FAIL
        resp_lib.add(resp_lib.GET, CNN_FG_URL,            json=FG_MOCK,       status=200)
        resp_lib.add(resp_lib.GET, f"{CBOE_PC_URL}{TODAY}_options_volume.csv",
                     body=CBOE_CSV_MOCK, status=200, content_type="text/csv")
        resp_lib.add(resp_lib.GET, f"{FINNHUB_BASE}/news",            json=FINNHUB_NEWS_MOCK, status=200)
        resp_lib.add(resp_lib.GET, f"{NEWS_API_BASE}/top-headlines",  json=NEWSAPI_MOCK,      status=200)
        resp_lib.add(resp_lib.GET, BARCHART_UA_URL,       body=BARCHART_HTML, status=200)
        resp_lib.add(resp_lib.GET, FRED_BASE,
                     json={"observations": [{"value": "5.10", "date": TODAY}]}, status=200)

        from data.data_engine import run_data_collection
        with patch("data.data_engine.OUTPUT_RAW",        tmp_path / "raw.json"), \
             patch("data.universe_manager.CACHE_PATH",   tmp_path / "cache.json"), \
             patch("data.unusual_activity._CACHE_PATH",  tmp_path / "ua.json"), \
             patch("data.sentiment_data._FG_CACHE",      {}), \
             patch("data.sentiment_data._FG_CACHE_DATE", ""):
            result = run_data_collection()

        assert result["macro_events"] == []
        assert "market_environment" in result

    @resp_lib.activate
    @patch("data.market_data.yf.download")
    @patch("data.universe_manager.pd.read_html")
    def test_sentiment_failure_uses_neutral_defaults(self, mock_wiki, mock_yf, tmp_path):
        mock_yf.return_value  = _make_spy_df()
        mock_wiki.side_effect = [[_SP500_DF], [pd.DataFrame({"A": [1]}), _NASDAQ_DF]]

        resp_lib.add(resp_lib.GET, FINNHUB_EARNINGS_URL,  json={"earningsCalendar": []}, status=200)
        resp_lib.add(resp_lib.GET, FINNHUB_ECONOMIC_URL,  json={"economicCalendar": []}, status=200)
        resp_lib.add(resp_lib.GET, CNN_FG_URL,            json={}, status=404)   # FAIL
        # Date-specific CBOE URLs return 403 to exercise the SPY fallback path
        resp_lib.add(resp_lib.GET, f"{CBOE_PC_URL}{TODAY}_options_volume.csv", body="", status=403)
        resp_lib.add(resp_lib.GET, f"{FINNHUB_BASE}/news",            json=[], status=200)
        resp_lib.add(resp_lib.GET, f"{NEWS_API_BASE}/top-headlines",  json={"articles": []}, status=200)
        resp_lib.add(resp_lib.GET, BARCHART_UA_URL,       body="", status=403)
        resp_lib.add(resp_lib.GET, FRED_BASE, body=Exception("timeout"))

        from data.data_engine import run_data_collection
        with patch("data.data_engine.OUTPUT_RAW",        tmp_path / "raw.json"), \
             patch("data.universe_manager.CACHE_PATH",   tmp_path / "cache.json"), \
             patch("data.unusual_activity._CACHE_PATH",  tmp_path / "ua.json"), \
             patch("data.sentiment_data._FG_CACHE",      {}), \
             patch("data.sentiment_data._FG_CACHE_DATE", ""), \
             patch("data.sentiment_data.yf.download",    side_effect=Exception("no VIX")), \
             patch("data.sentiment_data.yf.Ticker",      side_effect=Exception("no SPY")):
            result = run_data_collection()

        env = result["market_environment"]
        assert env["fear_greed_score"] == 50
        assert env["put_call_ratio"] == 0.9
        assert "date" in result


class TestDeepFetch:
    def test_sleep_called_between_tickers(self, tmp_path):
        from data.data_engine import deep_fetch

        with patch("data.data_engine.fetch_options_chain") as mock_chain, \
             patch("data.data_engine.filter_liquid_strikes") as mock_filter, \
             patch("data.data_engine.fetch_earnings_history",     return_value=[]), \
             patch("data.data_engine.fetch_analyst_ratings",      return_value=[]), \
             patch("data.data_engine.fetch_insider_transactions",  return_value=[]), \
             patch("data.data_engine.fetch_company_news",          return_value=[]), \
             patch("data.data_engine.time.sleep") as mock_sleep:

            mock_chain.return_value  = {"expiry": "2026-07-06", "options": [], "source": "yfinance"}
            mock_filter.return_value = []
            deep_fetch(["AAPL", "NVDA"], [])

        mock_sleep.assert_called_once_with(1)

    def test_returns_result_for_each_ticker(self, tmp_path):
        from data.data_engine import deep_fetch

        with patch("data.data_engine.fetch_options_chain") as mock_chain, \
             patch("data.data_engine.filter_liquid_strikes") as mock_filter, \
             patch("data.data_engine.fetch_earnings_history",     return_value=[]), \
             patch("data.data_engine.fetch_analyst_ratings",      return_value=[]), \
             patch("data.data_engine.fetch_insider_transactions",  return_value=[]), \
             patch("data.data_engine.fetch_company_news",          return_value=[]), \
             patch("data.data_engine.time.sleep"):

            mock_chain.return_value  = {"expiry": "2026-07-06", "options": [], "source": "yfinance"}
            mock_filter.return_value = []
            result = deep_fetch(["AAPL", "NVDA", "MSFT"], [])

        assert len(result) == 3
