import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import responses as resp_lib

from config import FINNHUB_BASE, FINNHUB_API_KEY
from data.earnings_data import (
    fetch_earnings_calendar, fetch_earnings_history,
    compute_implied_move, compute_hist_avg_move,
    fetch_analyst_ratings, classify_earnings_candidate,
)

FIXTURES = Path(__file__).parent / "fixtures"

TODAY    = date.today()
DATE_3D  = (TODAY + timedelta(days=3)).isoformat()
DATE_10D = (TODAY + timedelta(days=10)).isoformat()

FINNHUB_EARNINGS_URL = f"{FINNHUB_BASE}/calendar/earnings"


class TestFetchEarningsCalendar:
    @resp_lib.activate
    def test_returns_correct_structure(self):
        payload = {"earningsCalendar": [
            {"symbol": "NVDA", "date": DATE_3D,  "eps": None, "epsEstimated": 0.72},
            {"symbol": "AAPL", "date": DATE_10D, "eps": None, "epsEstimated": 1.55},
        ]}
        resp_lib.add(resp_lib.GET, FINNHUB_EARNINGS_URL, json=payload, status=200)
        result = fetch_earnings_calendar(days_ahead=14)
        assert len(result) == 2
        tickers = [r["ticker"] for r in result]
        assert "NVDA" in tickers and "AAPL" in tickers

    @resp_lib.activate
    def test_days_away_computed_correctly(self):
        payload = {"earningsCalendar": [{"symbol": "TEST", "date": DATE_3D}]}
        resp_lib.add(resp_lib.GET, FINNHUB_EARNINGS_URL, json=payload, status=200)
        result = fetch_earnings_calendar(days_ahead=14)
        assert len(result) == 1
        assert result[0]["days_away"] == 3

    @resp_lib.activate
    def test_failure_returns_empty_list(self):
        resp_lib.add(resp_lib.GET, FINNHUB_EARNINGS_URL, json={}, status=503)
        result = fetch_earnings_calendar()
        assert result == []

    @resp_lib.activate
    def test_empty_response_returns_empty_list(self):
        resp_lib.add(resp_lib.GET, FINNHUB_EARNINGS_URL,
                     json={"earningsCalendar": []}, status=200)
        result = fetch_earnings_calendar()
        assert result == []


class TestFetchEarningsHistory:
    @resp_lib.activate
    def test_returns_8_quarters(self):
        payload = [
            {"period": f"2025-Q{i}", "actual": 1.0 + i*0.1, "estimate": 1.0,
             "priceChangePercent": 5.0 + i}
            for i in range(8)
        ]
        resp_lib.add(
            resp_lib.GET,
            f"{FINNHUB_BASE}/stock/earnings",
            json=payload, status=200,
        )
        result = fetch_earnings_history("AAPL")
        assert len(result) == 8

    @resp_lib.activate
    def test_surprise_pct_computed(self):
        payload = [{"period": "2025-Q1", "actual": 1.10, "estimate": 1.0, "priceChangePercent": 5.0}]
        resp_lib.add(resp_lib.GET, f"{FINNHUB_BASE}/stock/earnings", json=payload, status=200)
        result = fetch_earnings_history("AAPL")
        assert abs(result[0]["surprise_pct"] - 10.0) < 0.1

    @resp_lib.activate
    def test_rate_limit_returns_empty(self):
        resp_lib.add(resp_lib.GET, f"{FINNHUB_BASE}/stock/earnings", json={}, status=429)
        result = fetch_earnings_history("AAPL")
        assert result == []


class TestComputeImpliedMove:
    def _make_chain(self, strike, call_bid, call_ask, put_bid, put_ask):
        return [
            {"option_type": "call", "strike": strike, "bid": call_bid, "ask": call_ask,
             "open_interest": 1000, "expiration_date": "2026-07-06", "greeks": {}},
            {"option_type": "put",  "strike": strike, "bid": put_bid,  "ask": put_ask,
             "open_interest": 1000, "expiration_date": "2026-07-06", "greeks": {}},
        ]

    def test_standard_calculation(self):
        chain  = self._make_chain(100, 2.95, 3.05, 2.75, 2.85)
        result = compute_implied_move(chain, 100.0)
        assert result is not None
        assert abs(result - 0.058) < 0.002

    def test_empty_chain_returns_none(self):
        assert compute_implied_move([], 100.0) is None

    def test_zero_price_returns_none(self):
        chain = self._make_chain(100, 3.0, 3.1, 2.9, 3.0)
        assert compute_implied_move(chain, 0.0) is None


class TestComputeHistAvgMove:
    def test_mean_absolute_move(self):
        history = [
            {"price_change_pct": 8.0},
            {"price_change_pct": -6.0},
            {"price_change_pct": 4.0},
            {"price_change_pct": -10.0},
        ]
        result = compute_hist_avg_move(history)
        assert result is not None
        assert abs(result - 0.07) < 0.001

    def test_none_values_ignored(self):
        history = [{"price_change_pct": 5.0}, {"price_change_pct": None}]
        result  = compute_hist_avg_move(history)
        assert result is not None
        assert abs(result - 0.05) < 0.001

    def test_empty_history_returns_none(self):
        assert compute_hist_avg_move([]) is None


class TestClassifyEarningsCandidate:
    def test_within_threshold_returns_true(self):
        earnings = [{"ticker": "NVDA", "date": DATE_3D, "days_away": 3}]
        result   = classify_earnings_candidate("NVDA", earnings, days_away_threshold=7)
        assert result["is_earnings_candidate"] is True
        assert result["days_away"] == 3

    def test_outside_threshold_returns_false(self):
        earnings = [{"ticker": "NVDA", "date": DATE_10D, "days_away": 10}]
        result   = classify_earnings_candidate("NVDA", earnings, days_away_threshold=7)
        assert result["is_earnings_candidate"] is False

    def test_different_ticker_returns_false(self):
        earnings = [{"ticker": "AAPL", "date": DATE_3D, "days_away": 3}]
        result   = classify_earnings_candidate("NVDA", earnings)
        assert result["is_earnings_candidate"] is False
