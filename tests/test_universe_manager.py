import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from data.universe_manager import get_sp500, get_nasdaq100, build_universe, prefilter_universe
from config import UNIVERSE_MIN_PRICE, UNIVERSE_MIN_MARKET_CAP

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_sp500_df():
    return pd.DataFrame({
        "Symbol":      ["AAPL", "MSFT", "XOM"],
        "Security":    ["Apple", "Microsoft", "ExxonMobil"],
        "GICS Sector": ["Technology", "Technology", "Energy"],
    })


def _make_nasdaq_df():
    return pd.DataFrame({
        "Ticker":   ["NVDA", "AAPL"],   # AAPL overlaps with S&P 500
        "Company":  ["NVIDIA", "Apple"],
        "Sector":   ["Technology", "Technology"],
    })


class TestGetConstituents:
    @patch("data.universe_manager.pd.read_html")
    def test_get_sp500_returns_list(self, mock_read):
        mock_read.return_value = [_make_sp500_df()]
        result = get_sp500()
        assert len(result) == 3
        symbols = [r["symbol"] for r in result]
        assert "AAPL" in symbols

    @patch("data.universe_manager.pd.read_html")
    def test_get_nasdaq100_returns_list(self, mock_read):
        # Return multiple tables; first one without ticker column, second with it
        mock_read.return_value = [
            pd.DataFrame({"Irrelevant": [1, 2]}),
            _make_nasdaq_df(),
        ]
        result = get_nasdaq100()
        assert len(result) == 2

    @patch("data.universe_manager.pd.read_html")
    def test_wikipedia_failure_returns_empty(self, mock_read):
        mock_read.side_effect = Exception("Network error")
        result = get_sp500()
        assert result == []


class TestBuildUniverse:
    @patch("data.universe_manager.pd.read_html")
    def test_deduplicates_overlapping_symbols(self, mock_read, tmp_path):
        # AAPL appears in both S&P 500 and Nasdaq
        mock_read.side_effect = [
            [_make_sp500_df()],                      # S&P 500 call
            [pd.DataFrame({"Irrelevant": [1]}), _make_nasdaq_df()],  # Nasdaq call
        ]
        with patch("data.universe_manager.CACHE_PATH", tmp_path / "cache.json"):
            result = build_universe()
        symbols = [r["symbol"] for r in result]
        assert symbols.count("AAPL") == 1
        assert len(symbols) == len(set(symbols))

    @patch("data.universe_manager.pd.read_html")
    def test_fallback_to_cache_on_failure(self, mock_read, tmp_path):
        cache = [{"symbol": "CACHED", "sector": "Tech"}]
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps(cache))
        mock_read.side_effect = Exception("Wikipedia down")
        with patch("data.universe_manager.CACHE_PATH", cache_path):
            result = build_universe()
        assert any(r["symbol"] == "CACHED" for r in result)


class TestPrefilterUniverse:
    def _price_data(self, tickers_and_prices):
        return {t: {"price": p, "market_cap": m} for t, p, m in tickers_and_prices}

    def test_passes_valid_ticker(self):
        universe   = [{"symbol": "AAPL", "sector": "Tech"}]
        price_data = self._price_data([("AAPL", 150.0, 3_000_000_000_000)])
        result     = prefilter_universe(universe, price_data)
        assert len(result) == 1

    def test_excludes_low_price(self):
        universe   = [{"symbol": "XYZ", "sector": "Tech"}]
        price_data = self._price_data([("XYZ", 5.0, 5_000_000_000)])
        result     = prefilter_universe(universe, price_data)
        assert len(result) == 0

    def test_excludes_low_market_cap(self):
        universe   = [{"symbol": "TINY", "sector": "Tech"}]
        price_data = self._price_data([("TINY", 50.0, 500_000_000)])
        result     = prefilter_universe(universe, price_data)
        assert len(result) == 0

    def test_boundary_price_exactly_at_min(self):
        universe   = [{"symbol": "EDGE", "sector": "Tech"}]
        price_data = self._price_data([("EDGE", UNIVERSE_MIN_PRICE, UNIVERSE_MIN_MARKET_CAP + 1)])
        result     = prefilter_universe(universe, price_data)
        assert len(result) == 1

    def test_missing_price_data_excluded(self):
        universe = [{"symbol": "GHOST", "sector": "Tech"}]
        result   = prefilter_universe(universe, {})
        assert len(result) == 0
