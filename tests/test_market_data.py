import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from data.market_data import classify_vix, classify_spy_trend, fetch_vix_spy, fetch_sector_rotation


class TestClassifyVix:
    def test_calm_boundary(self):
        label, mult = classify_vix(14.9)
        assert label == "calm"
        assert mult == 1.0

    def test_normal_lower_boundary(self):
        label, mult = classify_vix(15.0)
        assert label == "normal"
        assert mult == 1.0

    def test_normal_mid(self):
        label, mult = classify_vix(20.0)
        assert label == "normal"

    def test_elevated_lower_boundary(self):
        label, mult = classify_vix(25.0)
        assert label == "elevated"
        assert mult == 0.75

    def test_elevated_upper(self):
        label, mult = classify_vix(34.9)
        assert label == "elevated"
        assert mult == 0.75

    def test_crisis_lower_boundary(self):
        label, mult = classify_vix(35.0)
        assert label == "crisis"
        assert mult == 0.50

    def test_crisis_high(self):
        label, mult = classify_vix(80.0)
        assert label == "crisis"
        assert mult == 0.50


class TestClassifySpyTrend:
    def test_strong_uptrend(self):
        assert classify_spy_trend(550, 530, 500) == "strong_uptrend"

    def test_moderate_uptrend(self):
        # price > ma200 but <= ma50
        assert classify_spy_trend(510, 530, 500) == "moderate_uptrend"

    def test_short_term_weak(self):
        # price < ma50 and price within 5% of ma200
        assert classify_spy_trend(495, 530, 500) == "short_term_weak"

    def test_bear_risk(self):
        # price well below ma200
        assert classify_spy_trend(450, 530, 500) == "bear_risk"


class TestFetchVixSpy:
    def _make_price_series(self, n=210, start=100.0, step=0.1):
        dates = pd.bdate_range(end="2026-05-29", periods=n)
        prices = [start + i * step for i in range(n)]
        df = pd.DataFrame({"Close": prices}, index=dates)
        df["High"] = df["Close"] * 1.005
        df["Low"]  = df["Close"] * 0.995
        return df

    @patch("data.market_data.yf.download")
    def test_returns_expected_keys(self, mock_download):
        spy_df = self._make_price_series(210)
        vix_df = pd.DataFrame({"Close": [18.4]}, index=pd.bdate_range(end="2026-05-29", periods=1))
        mock_download.side_effect = [vix_df, spy_df]

        result = fetch_vix_spy()
        for key in ["vix", "vix_regime", "spy_price", "spy_ma50", "spy_ma200", "spy_trend", "spy_hv20",
                    "spy_prior_close", "spy_5d_high", "spy_5d_low"]:
            assert key in result

    @patch("data.market_data.yf.download")
    def test_ma50_computed_correctly(self, mock_download):
        n = 210
        spy_df = self._make_price_series(n, start=100.0, step=1.0)
        vix_df = pd.DataFrame({"Close": [20.0]}, index=pd.bdate_range(end="2026-05-29", periods=1))
        mock_download.side_effect = [vix_df, spy_df]

        result = fetch_vix_spy()
        expected_ma50 = float(spy_df["Close"].rolling(50).mean().iloc[-1])
        assert abs(result["spy_ma50"] - expected_ma50) < 0.01

    @patch("data.market_data.yf.download")
    def test_failure_returns_defaults(self, mock_download):
        mock_download.side_effect = Exception("Network error")
        result = fetch_vix_spy()
        assert result["vix"] == 20.0
        assert result["vix_regime"] == "normal"
        assert result["spy_trend"] == "unknown"


class TestFetchSectorRotation:
    @patch("data.market_data.yf.download")
    def test_returns_leading_lagging(self, mock_download):
        etfs = ["XLK", "XLE", "XLF", "XLV", "XLU", "XLI", "XLB", "XLP"]
        n = 25
        dates = pd.bdate_range(end="2026-05-29", periods=n)
        # XLK goes up most, XLU goes down most
        data = {etf: [100.0 + (i * (j + 1) * 0.1) for i in range(n)]
                for j, etf in enumerate(etfs)}
        closes = pd.DataFrame(data, index=dates)
        mock_df = MagicMock()
        mock_df.__getitem__ = lambda self, key: closes if key == "Close" else closes
        mock_df.empty = False
        mock_download.return_value = mock_df

        result = fetch_sector_rotation()
        assert "leading_sectors" in result
        assert "lagging_sectors" in result
        assert len(result["leading_sectors"]) == 3
        assert len(result["lagging_sectors"]) == 3

    @patch("data.market_data.yf.download")
    def test_failure_returns_empty(self, mock_download):
        mock_download.side_effect = Exception("Network error")
        result = fetch_sector_rotation()
        assert result["leading_sectors"] == []
        assert result["lagging_sectors"] == []
