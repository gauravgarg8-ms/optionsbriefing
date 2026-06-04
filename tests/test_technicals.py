import numpy as np
import pandas as pd
import pytest

from quant.technicals import compute_technical_levels, compute_rs


class TestComputeTechnicalLevels:
    def test_returns_all_keys(self, synthetic_ohlcv):
        result = compute_technical_levels("AAPL", synthetic_ohlcv)
        for key in ["support", "resistance", "atr_14", "rsi_14", "support_52w", "resistance_52w"]:
            assert key in result

    def test_support_lte_current_price(self, synthetic_ohlcv):
        result = compute_technical_levels("AAPL", synthetic_ohlcv)
        current_price = float(synthetic_ohlcv["Close"].iloc[-1])
        assert result["support"] <= current_price + 0.01

    def test_resistance_gte_current_price(self, synthetic_ohlcv):
        result = compute_technical_levels("AAPL", synthetic_ohlcv)
        current_price = float(synthetic_ohlcv["Close"].iloc[-1])
        assert result["resistance"] >= current_price - 0.01

    def test_atr_positive(self, synthetic_ohlcv):
        result = compute_technical_levels("AAPL", synthetic_ohlcv)
        assert result["atr_14"] > 0

    def test_rsi_flat_prices_near_50(self, flat_ohlcv):
        result = compute_technical_levels("FLAT", flat_ohlcv)
        # Flat prices: equal up/down, RSI should be near 50
        assert 40 <= result["rsi_14"] <= 60

    def test_rsi_consecutive_up_days(self):
        n = 50
        dates = pd.bdate_range(end="2026-05-29", periods=n)
        prices = np.linspace(100, 120, n)  # pure uptrend
        df = pd.DataFrame(
            {"Open": prices, "High": prices + 0.5,
             "Low": prices - 0.5, "Close": prices,
             "Volume": np.ones(n) * 1e6},
            index=dates,
        )
        result = compute_technical_levels("UP", df)
        assert result["rsi_14"] >= 80

    def test_rsi_consecutive_down_days(self):
        n = 50
        dates = pd.bdate_range(end="2026-05-29", periods=n)
        prices = np.linspace(120, 100, n)  # pure downtrend
        df = pd.DataFrame(
            {"Open": prices, "High": prices + 0.5,
             "Low": prices - 0.5, "Close": prices,
             "Volume": np.ones(n) * 1e6},
            index=dates,
        )
        result = compute_technical_levels("DOWN", df)
        assert result["rsi_14"] <= 20

    def test_wilder_atr_value(self):
        """Verify ATR(14) against a known hand-calculated result."""
        n = 30
        dates = pd.bdate_range(end="2026-05-29", periods=n)
        high  = np.full(n, 105.0)
        low   = np.full(n, 95.0)
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {"Open": close, "High": high, "Low": low, "Close": close,
             "Volume": np.ones(n) * 1e6},
            index=dates,
        )
        result = compute_technical_levels("TEST", df)
        # True Range for every day = High - Low = 10 (no gap days)
        # ATR(14) should converge to ~10
        assert abs(result["atr_14"] - 10.0) < 1.0

    def test_insufficient_data_returns_defaults(self):
        dates = pd.bdate_range(end="2026-05-29", periods=5)
        df = pd.DataFrame(
            {"Open": [100]*5, "High": [101]*5, "Low": [99]*5,
             "Close": [100]*5, "Volume": [1e6]*5},
            index=dates,
        )
        result = compute_technical_levels("SHORT", df)
        assert result["rsi_14"] == 50.0


class TestComputeRS:
    def test_identical_series_returns_zero(self, synthetic_ohlcv):
        close = synthetic_ohlcv["Close"]
        rs = compute_rs(close, close)
        assert abs(rs) < 0.001

    def test_outperforming_stock_positive_rs(self):
        dates = pd.bdate_range(end="2026-05-29", periods=25)
        stock = pd.Series(np.linspace(100, 120, 25), index=dates)  # +20%
        spy   = pd.Series(np.linspace(100, 110, 25), index=dates)  # +10%
        rs = compute_rs(stock, spy)
        assert rs > 0

    def test_underperforming_stock_negative_rs(self):
        dates = pd.bdate_range(end="2026-05-29", periods=25)
        stock = pd.Series(np.linspace(100, 95, 25), index=dates)   # -5%
        spy   = pd.Series(np.linspace(100, 110, 25), index=dates)  # +10%
        rs = compute_rs(stock, spy)
        assert rs < 0

    def test_insufficient_data_returns_zero(self):
        series = pd.Series([100.0, 101.0, 102.0])
        rs = compute_rs(series, series, days=20)
        assert rs == 0.0
