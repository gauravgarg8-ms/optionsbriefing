import json
from datetime import date, timedelta
from pathlib import Path

import pytest
import responses as resp_lib

from config import TRADIER_BASE, LIQUIDITY_MAX_BID_ASK_PCT, LIQUIDITY_MIN_OI
from data.options_data import compute_bid_ask_pct, filter_liquid_strikes, fetch_options_chain

FIXTURES = Path(__file__).parent / "fixtures"

# Dynamic expiry always 30 days out — stays within DEEP_FETCH_DTE_MIN/MAX (25–45)
_EXPIRY_30D = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")


class TestComputeBidAskPct:
    def test_standard_calculation(self):
        opt = {"bid": 2.0, "ask": 2.06}
        pct = compute_bid_ask_pct(opt)
        # mid = 2.03, spread = 0.06, pct = 0.06/2.03 ≈ 2.96%
        assert abs(pct - 0.06 / 2.03) < 0.0001

    def test_zero_mid_returns_100_pct(self):
        opt = {"bid": 0.0, "ask": 0.0}
        assert compute_bid_ask_pct(opt) == 1.0

    def test_wide_spread_22_pct(self):
        opt = {"bid": 1.0, "ask": 1.25}
        # mid = 1.125, spread = 0.25, pct = 0.25/1.125 ≈ 22.2%
        pct = compute_bid_ask_pct(opt)
        assert pct > 0.20

    def test_tight_spread_passes(self):
        opt = {"bid": 5.0, "ask": 5.10}
        pct = compute_bid_ask_pct(opt)
        assert pct < LIQUIDITY_MAX_BID_ASK_PCT


class TestFilterLiquidStrikes:
    def _make_opt(self, strike, bid, ask, oi, opt_type="put"):
        mid = (bid + ask) / 2 if (bid + ask) > 0 else 1.0
        return {
            "strike": strike, "bid": bid, "ask": ask, "open_interest": oi,
            "option_type": opt_type, "expiration_date": "2026-07-06",
            "greeks": {"mid_iv": 0.35}
        }

    def test_liquid_option_passes(self):
        opts = [self._make_opt(130, 2.0, 2.06, 2000)]
        result = filter_liquid_strikes(opts)
        assert len(result) == 1

    def test_low_oi_filtered_out(self):
        opts = [self._make_opt(130, 2.0, 2.06, 300)]  # OI < 500
        result = filter_liquid_strikes(opts)
        assert len(result) == 0

    def test_wide_spread_filtered_out(self):
        opts = [self._make_opt(130, 1.0, 1.25, 2000)]  # spread ~22%
        result = filter_liquid_strikes(opts)
        assert len(result) == 0

    def test_mixed_options_only_liquid_pass(self):
        opts = [
            self._make_opt(130, 2.0, 2.06, 2000),  # liquid
            self._make_opt(125, 1.0, 1.25, 300),   # wide spread + low OI
            self._make_opt(120, 1.5, 1.56, 800),   # liquid
        ]
        result = filter_liquid_strikes(opts)
        assert len(result) == 2
        strikes = [o["strike"] for o in result]
        assert 130 in strikes and 120 in strikes

    def test_bid_ask_pct_stored_on_passing_options(self):
        opts = [self._make_opt(130, 2.0, 2.06, 2000)]
        result = filter_liquid_strikes(opts)
        assert "_bid_ask_pct" in result[0]


class TestFetchOptionsChain:
    def _make_mock_yf_ticker(self, expiry=None):
        from unittest.mock import MagicMock
        import pandas as pd
        expiry = expiry or _EXPIRY_30D
        mock_t = MagicMock()
        mock_t.options = [expiry]
        mock_chain = MagicMock()
        mock_chain.puts  = pd.DataFrame([{
            "strike": 130.0, "bid": 2.0, "ask": 2.1, "lastPrice": 2.05,
            "volume": 500, "openInterest": 1000, "impliedVolatility": 0.42
        }])
        mock_chain.calls = pd.DataFrame([{
            "strike": 140.0, "bid": 3.0, "ask": 3.1, "lastPrice": 3.05,
            "volume": 800, "openInterest": 1500, "impliedVolatility": 0.40
        }])
        mock_t.option_chain.return_value = mock_chain
        return mock_t

    def test_yfinance_primary_when_no_token(self):
        """With no TRADIER_TOKEN, yfinance is used directly."""
        from unittest.mock import patch
        with patch("data.options_data.TRADIER_TOKEN", ""), \
             patch("data.options_data.yf.Ticker") as mock_ticker:
            mock_ticker.return_value = self._make_mock_yf_ticker()
            result = fetch_options_chain("NVDA")
        assert result["source"] == "yfinance"
        assert len(result["options"]) == 2   # 1 put + 1 call
        assert result["expiry"] == _EXPIRY_30D

    @resp_lib.activate
    def test_tradier_used_when_token_set(self):
        """With TRADIER_TOKEN set, Tradier is tried first and succeeds."""
        from unittest.mock import patch
        chain_data = json.loads((FIXTURES / "sample_chain_nvda.json").read_text())
        options    = chain_data["options"]["option"]

        resp_lib.add(resp_lib.GET, f"{TRADIER_BASE}/markets/options/expirations",
                     json={"expirations": {"date": [_EXPIRY_30D]}}, status=200)
        resp_lib.add(resp_lib.GET, f"{TRADIER_BASE}/markets/options/chains",
                     json={"options": {"option": options}}, status=200)

        with patch("data.options_data.TRADIER_TOKEN", "fake-token"):
            result = fetch_options_chain("NVDA")
        assert result["source"] == "tradier"
        assert len(result["options"]) == 3
        assert result["expiry"] == _EXPIRY_30D

    @resp_lib.activate
    def test_tradier_failure_falls_back_to_yfinance(self):
        """If Tradier is configured but returns 500, falls back to yfinance."""
        from unittest.mock import patch
        resp_lib.add(resp_lib.GET, f"{TRADIER_BASE}/markets/options/expirations",
                     json={"expirations": {"date": [_EXPIRY_30D]}}, status=200)
        resp_lib.add(resp_lib.GET, f"{TRADIER_BASE}/markets/options/chains",
                     json={}, status=500)

        with patch("data.options_data.TRADIER_TOKEN", "fake-token"), \
             patch("data.options_data.yf.Ticker") as mock_ticker:
            mock_ticker.return_value = self._make_mock_yf_ticker()
            result = fetch_options_chain("NVDA")
        assert result["source"] == "yfinance"
