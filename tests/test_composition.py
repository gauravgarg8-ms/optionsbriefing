import pytest
from screening.composition import (
    select_final_10, check_reduced_opportunity, enforce_earnings_limit,
    compute_portfolio_exposure, compose_final_output,
)
from config import SCORE_HIGH_CONFIDENCE, MAX_EARNINGS_PLAYS

MARKET_ENV = {"leading_sectors": ["XLK"], "lagging_sectors": ["XLU"]}


def _make_candidate(ticker, score, sector="XLK", earnings_days=99,
                    delta=-0.28, vega=-0.18):
    return {
        "ticker": ticker, "score": score, "sector": sector,
        "earnings_days_away": earnings_days,
        "bs": {"delta": delta, "vega": vega, "pop": 0.70, "ev": 20.0},
    }


class TestSelectFinal10:
    def test_returns_at_most_10(self):
        candidates = [_make_candidate(f"T{i}", 80-i) for i in range(15)]
        result = select_final_10(candidates)
        assert len(result) == 10

    def test_returns_fewer_when_less_available(self):
        candidates = [_make_candidate("A", 70), _make_candidate("B", 65)]
        result = select_final_10(candidates)
        assert len(result) == 2


class TestCheckReducedOpportunity:
    def test_enough_high_scores_returns_false(self):
        candidates = [_make_candidate(f"T{i}", SCORE_HIGH_CONFIDENCE + 5) for i in range(5)]
        assert check_reduced_opportunity(candidates) is False

    def test_too_few_high_scores_returns_true(self):
        # Only 3 candidates score ≥ SCORE_HIGH_CONFIDENCE (need 5)
        candidates = [_make_candidate(f"T{i}", SCORE_HIGH_CONFIDENCE + 1) for i in range(3)]
        candidates += [_make_candidate(f"L{i}", 50) for i in range(5)]
        assert check_reduced_opportunity(candidates) is True

    def test_zero_candidates_returns_true(self):
        assert check_reduced_opportunity([]) is True


class TestEnforceEarningsLimit:
    def test_trims_excess_earnings_plays(self):
        # 6 earnings plays (limit = 4) — trim to top 4 by score
        candidates = [
            _make_candidate(f"E{i}", score=80-i, earnings_days=5) for i in range(6)
        ]
        result = enforce_earnings_limit(candidates)
        earnings_plays = [c for c in result if c["earnings_days_away"] <= 7]
        assert len(earnings_plays) == MAX_EARNINGS_PLAYS

    def test_keeps_highest_scoring_earnings_plays(self):
        candidates = [
            _make_candidate("HIGH", score=90, earnings_days=5),
            _make_candidate("MED",  score=70, earnings_days=5),
            _make_candidate("LOW",  score=50, earnings_days=5),
            _make_candidate("VLOW", score=45, earnings_days=5),
            _make_candidate("VVLOW", score=46, earnings_days=5),
        ]
        result = enforce_earnings_limit(candidates)
        tickers = [c["ticker"] for c in result if c["earnings_days_away"] <= 7]
        assert "HIGH" in tickers
        assert len(tickers) == MAX_EARNINGS_PLAYS

    def test_does_not_trim_when_under_limit(self):
        candidates = [_make_candidate(f"E{i}", score=80, earnings_days=5) for i in range(3)]
        result = enforce_earnings_limit(candidates)
        assert len([c for c in result if c["earnings_days_away"] <= 7]) == 3


class TestComputePortfolioExposure:
    def test_detects_sector_concentration(self):
        # 3 positions in XLK exceeds MAX_SECTOR_POSITIONS (2)
        candidates = [
            _make_candidate("A", 80, sector="XLK"),
            _make_candidate("B", 75, sector="XLK"),
            _make_candidate("C", 70, sector="XLK"),
        ]
        result = compute_portfolio_exposure(candidates)
        assert "XLK" in result["concentrated_sectors"]
        assert any("concentration" in w.lower() for w in result["portfolio_warnings"])

    def test_no_concentration_with_two_per_sector(self):
        candidates = [
            _make_candidate("A", 80, sector="XLK"),
            _make_candidate("B", 75, sector="XLK"),
            _make_candidate("C", 70, sector="XLE"),
        ]
        result = compute_portfolio_exposure(candidates)
        assert "XLK" not in result["concentrated_sectors"]

    def test_earnings_plays_counted_correctly(self):
        candidates = [
            _make_candidate("E1", 80, earnings_days=5),
            _make_candidate("E2", 80, earnings_days=6),
            _make_candidate("N1", 80, earnings_days=30),
        ]
        result = compute_portfolio_exposure(candidates)
        assert result["earnings_plays_count"] == 2

    def test_vega_direction_net_short(self):
        candidates = [
            _make_candidate("A", 80, vega=-0.18),
            _make_candidate("B", 75, vega=-0.22),
        ]
        result = compute_portfolio_exposure(candidates)
        assert result["vega_direction"] == "Net Short Vega"


class TestComposeFinalOutput:
    def test_no_candidates_triggers_no_trade_day(self):
        result = compose_final_output([], MARKET_ENV)
        assert result["no_trade_day"] is True
        assert result["candidates"] == []

    def test_enough_candidates_no_trade_day_false(self):
        candidates = [_make_candidate(f"T{i}", SCORE_HIGH_CONFIDENCE + 5) for i in range(7)]
        result = compose_final_output(candidates, MARKET_ENV)
        assert result["no_trade_day"] is False

    def test_portfolio_check_included(self):
        candidates = [_make_candidate("NVDA", 80)]
        result = compose_final_output(candidates, MARKET_ENV)
        assert "portfolio_check" in result
        assert "net_portfolio_delta" in result["portfolio_check"]
