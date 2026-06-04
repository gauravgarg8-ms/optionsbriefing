"""
Integration tests for the 2A → 2B → 2C screening engine orchestrator.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from screening.screening_engine import run_screening

FIXTURES = Path(__file__).parent / "fixtures"

MARKET_ENV = {
    "leading_sectors": ["XLK", "XLI"], "lagging_sectors": ["XLU", "XLP"],
    "fear_greed_score": 61, "put_call_ratio": 0.82,
    "market_sentiment": "BULLISH", "structure_bias": "NEUTRAL_TO_DEBIT",
    "sector_news": {},
}


def _make_quant_signal(ticker, iv_rank=65, price=50, avg_vol=10000,
                       above_50=True, above_200=True, rs_20d=3.0,
                       sector="XLK", bid_ask_pct=0.02, oi=5000,
                       hv20=25.0, iv_rv=1.8, pop=0.72, ev=20.0):
    return {
        "ticker": ticker, "price": price, "iv_rank": iv_rank,
        "avg_options_vol": avg_vol, "above_50ma": above_50, "above_200ma": above_200,
        "rs_20d": rs_20d, "sector": sector, "iv_rv_ratio": iv_rv, "hv20": hv20,
        "bid_ask_pct": bid_ask_pct, "oi_target": oi,
        "structure": "bull_put_spread", "direction": "Bullish",
        "bs": {"pop": pop, "ev": ev, "delta": -0.25, "vega": -0.15},
        "sentiment_flags": [], "warning_flags": [],
        "support": price * 0.93, "resistance": price * 1.08,
        "ma50": price * 0.97, "ma200": price * 0.90,
    }


def _make_quant_data(tickers, **kwargs):
    signals = {t: _make_quant_signal(t, **kwargs) for t in tickers}
    return {
        "market_environment": MARKET_ENV,
        "quant_signals":      signals,
        "earnings_calendar":  [],
        "tbill_rate":         0.051,
    }


class TestRunScreeningOrchestration:
    def _mock_deep_fetch(self, tickers, earnings_calendar):
        """Returns empty deep data so screening_engine uses lightweight data."""
        return {t: {"chain": {"options": [], "expiry": None},
                    "earnings_history": [], "analyst_ratings": [],
                    "insider_transactions": [], "company_news": [],
                    "earnings_info": {"is_earnings_candidate": False,
                                      "earnings_date": None, "days_away": None,
                                      "has_earnings": False}}
                for t in tickers}

    @patch("screening.screening_engine.OUTPUT_SCREENED")
    def test_returns_candidates_for_valid_pool(self, mock_path, tmp_path):
        mock_path.__str__ = lambda s: str(tmp_path / "screened.json")
        quant_data = _make_quant_data(["AAPL", "NVDA", "MSFT"])

        mock_engine = MagicMock()
        mock_engine.deep_fetch.side_effect = self._mock_deep_fetch

        result = run_screening(quant_data, data_engine=mock_engine)
        assert isinstance(result.get("candidates"), list)
        assert "market_environment" in result

    @patch("screening.screening_engine.OUTPUT_SCREENED")
    def test_2c_called_only_for_top_15(self, mock_path, tmp_path):
        """deep_fetch should be called with at most 15 tickers."""
        mock_path.__str__ = lambda s: str(tmp_path / "screened.json")
        # Create 30 high-IV candidates
        quant_data = _make_quant_data([f"T{i}" for i in range(30)])

        mock_engine = MagicMock()
        mock_engine.deep_fetch.side_effect = self._mock_deep_fetch

        run_screening(quant_data, data_engine=mock_engine)

        called_tickers = mock_engine.deep_fetch.call_args[0][0]
        assert len(called_tickers) <= 15

    @patch("screening.screening_engine.OUTPUT_SCREENED")
    def test_no_candidates_emits_no_trade_day(self, mock_path, tmp_path):
        """If no tickers pass any screen, output should have no_trade_day=True."""
        mock_path.__str__ = lambda s: str(tmp_path / "screened.json")
        # All tickers have IV rank=30 (won't pass high_iv), below 50 MA (won't pass low_iv_trend),
        # and are in neutral sector (won't pass bearish either)
        quant_data = _make_quant_data(
            ["A", "B", "C"],
            iv_rank=30, above_50=False, rs_20d=-1.0, sector="XLF",
        )

        mock_engine = MagicMock()
        mock_engine.deep_fetch.side_effect = self._mock_deep_fetch

        result = run_screening(quant_data, data_engine=mock_engine)
        assert result["no_trade_day"] is True

    @patch("screening.screening_engine.OUTPUT_SCREENED")
    def test_screened_json_written(self, mock_path, tmp_path):
        output_file = tmp_path / "screened.json"
        mock_path.__str__   = lambda s: str(output_file)
        mock_path.parent    = tmp_path
        mock_path.write_text = output_file.write_text

        quant_data = _make_quant_data(["NVDA"])
        mock_engine = MagicMock()
        mock_engine.deep_fetch.side_effect = self._mock_deep_fetch

        with patch("screening.screening_engine.OUTPUT_SCREENED", output_file):
            run_screening(quant_data, data_engine=mock_engine)

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "candidates" in data

    @patch("screening.screening_engine.OUTPUT_SCREENED")
    def test_reduced_opportunity_flag_set_correctly(self, mock_path, tmp_path):
        """If fewer than 5 candidates score ≥65, reduced_opportunity_day should be True."""
        mock_path.__str__ = lambda s: str(tmp_path / "screened.json")
        # Only 2 candidates, both borderline scores
        quant_data = _make_quant_data(
            ["A", "B"],
            iv_rank=55, avg_vol=600, bid_ask_pct=0.07, oi=600,  # low-medium scores
        )
        mock_engine = MagicMock()
        mock_engine.deep_fetch.side_effect = self._mock_deep_fetch

        result = run_screening(quant_data, data_engine=mock_engine)
        # With only 2 candidates, can't have 5 scoring ≥65
        assert result["reduced_opportunity_day"] is True
