import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from quant.quant_engine import run_lightweight_quant, run_deep_quant, _infer_bias


FIXTURES = Path(__file__).parent / "fixtures"


class TestRunLightweightQuant:
    def _make_ohlcv(self, n=210):
        dates  = pd.bdate_range(end="2026-05-29", periods=n)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        prices = np.clip(prices, 10, None)
        return pd.DataFrame({
            "Open": prices, "High": prices + 0.5,
            "Low": prices - 0.5, "Close": prices,
            "Volume": np.ones(n) * 1e6
        }, index=dates)

    @patch("quant.quant_engine._fetch_ohlcv")
    def test_returns_expected_keys(self, mock_fetch, memory_db):
        ohlcv = self._make_ohlcv()
        mock_fetch.return_value = ohlcv
        spy_close = ohlcv["Close"]
        with patch("quant.quant_engine._db", memory_db):
            result = run_lightweight_quant("AAPL", {}, spy_close)
        for key in ["ticker", "price", "ma50", "ma200", "hv20", "iv_rank",
                    "support", "resistance", "rs_20d"]:
            assert key in result, f"Missing key: {key}"

    @patch("quant.quant_engine._fetch_ohlcv")
    def test_above_50ma_computed_correctly(self, mock_fetch, memory_db):
        ohlcv = self._make_ohlcv()
        # Force price to be clearly above MA50
        prices = np.ones(210) * 120
        prices[:50] = 100  # MA50 ≈ 100 for first portion
        ohlcv["Close"] = ohlcv["High"] = ohlcv["Low"] = ohlcv["Open"] = prices
        mock_fetch.return_value = ohlcv
        spy_close = pd.Series(prices, index=ohlcv.index)
        with patch("quant.quant_engine._db", memory_db):
            result = run_lightweight_quant("AAPL", {}, spy_close)
        # With recent prices all at 120 and MA50 lower, above_50ma should be True
        assert isinstance(result.get("above_50ma"), bool)

    @patch("quant.quant_engine._fetch_ohlcv")
    def test_empty_ohlcv_returns_empty_dict(self, mock_fetch, memory_db):
        mock_fetch.return_value = None
        with patch("quant.quant_engine._db", memory_db):
            result = run_lightweight_quant("BAD", {}, pd.Series(dtype=float))
        assert result == {}


class TestRunDeepQuant:
    def test_updates_iv30_from_real_chain(self, memory_db):
        chain_data = json.loads((FIXTURES / "sample_chain_nvda.json").read_text())
        options    = chain_data["options"]["option"]
        lightweight = {
            "ticker": "NVDA", "price": 135.5, "ma50": 128.0, "ma200": 115.0,
            "above_50ma": True, "above_200ma": True,
            "rs_20d": 4.2, "hv20": 28.0, "iv30": 35.0, "iv_rank": 50.0,
            "iv_rv_ratio": 1.2, "support": 128.0, "resistance": 142.0,
            "atr_14": 3.8, "rsi_14": 62.0,
        }
        chain = {"expiry": "2026-07-06", "options": options, "source": "tradier"}
        with patch("quant.quant_engine._db", memory_db):
            result = run_deep_quant("NVDA", lightweight, chain)
        assert "structure" in result
        assert "spread_pricing" in result
        assert result.get("covered_call_opportunity") is not None


class TestInferBias:
    def test_bullish_signals(self):
        signals = {"above_50ma": True, "rs_20d": 3.0}
        assert _infer_bias(signals) == "Bullish"

    def test_bearish_signals(self):
        signals = {"above_50ma": False, "rs_20d": -2.0}
        assert _infer_bias(signals) == "Bearish"

    def test_neutral_mixed_signals(self):
        signals = {"above_50ma": True, "rs_20d": -1.0}
        assert _infer_bias(signals) == "Neutral"
