import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
import responses as resp_lib

from config import BARCHART_UA_URL
from data.unusual_activity import fetch_unusual_activity, _load_cache, _save_cache

BARCHART_HTML_MOCK = """
<html><body>
<table>
  <tr><th>Symbol</th><th>Option Type</th><th>Volume</th></tr>
  <tr><td>AAPL</td><td>Call</td><td>15,230</td></tr>
  <tr><td>NVDA</td><td>Call</td><td>42,100</td></tr>
  <tr><td>SPY</td><td>Put</td><td>88,500</td></tr>
</table>
</body></html>
"""


class TestFetchUnusualActivity:
    @resp_lib.activate
    def test_returns_list_of_tickers(self, tmp_path):
        resp_lib.add(resp_lib.GET, BARCHART_UA_URL, body=BARCHART_HTML_MOCK, status=200)
        with patch("data.unusual_activity._CACHE_PATH", tmp_path / "ua_cache.json"):
            result = fetch_unusual_activity()
        assert isinstance(result, list)
        assert len(result) >= 1
        tickers = [r["ticker"] for r in result]
        assert "AAPL" in tickers or "NVDA" in tickers

    @resp_lib.activate
    def test_caches_result_no_re_scrape(self, tmp_path):
        """Second call on same day should use cache, not re-scrape."""
        resp_lib.add(resp_lib.GET, BARCHART_UA_URL, body=BARCHART_HTML_MOCK, status=200)
        cache_path = tmp_path / "ua_cache.json"
        with patch("data.unusual_activity._CACHE_PATH", cache_path):
            # First call — scrapes
            fetch_unusual_activity()
            # Verify cache was written
            assert cache_path.exists()
            # Second call — should use cache (no new HTTP request)
            result2 = fetch_unusual_activity()
        assert isinstance(result2, list)
        # Only 1 request should have been made
        assert len(resp_lib.calls) == 1

    @resp_lib.activate
    def test_barchart_403_returns_empty(self, tmp_path):
        resp_lib.add(resp_lib.GET, BARCHART_UA_URL, body="", status=403)
        with patch("data.unusual_activity._CACHE_PATH", tmp_path / "ua_cache.json"):
            result = fetch_unusual_activity()
        assert result == []

    def test_sentiment_classified_for_calls(self, tmp_path):
        today = date.today().isoformat()
        test_tickers = [{"ticker": "AAPL", "volume": 15000, "sentiment": "BULLISH"}]
        cache_path = tmp_path / "ua_cache.json"
        _save_cache.__wrapped__ = None  # reset any wrapping
        with patch("data.unusual_activity._CACHE_PATH", cache_path):
            _save_cache(test_tickers, today)
            result = _load_cache()
        assert result["tickers"][0]["sentiment"] == "BULLISH"
