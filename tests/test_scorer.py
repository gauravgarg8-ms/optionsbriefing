import pytest
from screening.scorer import compute_score, compute_confidence, filter_by_score
from config import SCORE_FLOOR

MARKET_ENV = {
    "leading_sectors": ["XLK", "XLI"],
    "lagging_sectors": ["XLU", "XLP"],
}


def _make_candidate(**kwargs):
    defaults = {
        "ticker": "NVDA", "sector": "XLK",
        "structure": "bull_put_spread", "direction": "Bullish",
        "iv_rank": 72, "iv_rv_ratio": 1.8, "hv20": 28.0,
        "above_50ma": True, "above_200ma": True, "rs_20d": 4.2,
        "avg_options_vol": 45000, "bid_ask_pct": 0.021, "oi_target": 8500,
        "earnings_days_away": 83,
        "bs": {"pop": 0.71, "ev": 22.5, "delta": -0.28, "vega": -0.18},
        "sentiment_flags": [], "warning_flags": [],
    }
    defaults.update(kwargs)
    return defaults


class TestComputeScore:
    def test_high_quality_credit_candidate_scores_high(self):
        c = _make_candidate()
        score = compute_score(c, MARKET_ENV)
        assert score >= 55  # IV 72 → 15pts, good trend → 20pts, liquid → 20pts, etc.

    def test_illiquid_high_bid_ask_scores_zero(self):
        c = _make_candidate(bid_ask_pct=0.15)
        assert compute_score(c, MARKET_ENV) == 0

    def test_illiquid_low_oi_scores_zero(self):
        c = _make_candidate(oi_target=400)
        assert compute_score(c, MARKET_ENV) == 0

    def test_low_pop_excluded(self):
        c = _make_candidate(bs={"pop": 0.55, "ev": 10.0, "delta": -0.2, "vega": -0.1})
        assert compute_score(c, MARKET_ENV) == 0

    def test_negative_ev_excluded(self):
        c = _make_candidate(bs={"pop": 0.72, "ev": -50.0, "delta": -0.2, "vega": -0.1})
        assert compute_score(c, MARKET_ENV) == 0

    def test_score_within_0_to_100(self):
        for iv in [20, 40, 60, 80]:
            c = _make_candidate(iv_rank=iv)
            s = compute_score(c, MARKET_ENV)
            assert 0 <= s <= 100, f"Score {s} out of range for iv_rank={iv}"

    def test_bearish_candidate_scoring(self):
        c = _make_candidate(
            direction="Bearish", structure="bear_call_spread",
            above_50ma=False, rs_20d=-3.0, sector="XLU",
            iv_rank=65, iv_rv_ratio=2.0,
        )
        score = compute_score(c, MARKET_ENV)
        assert score > 0  # Should score positively

    def test_debit_candidate_low_iv_scores_well(self):
        c = _make_candidate(
            structure="bull_call_spread", direction="Bullish",
            iv_rank=20, iv_rv_ratio=0.7, sector="XLK",
        )
        score = compute_score(c, MARKET_ENV)
        assert score > 0

    def test_two_bonus_flags_adds_points(self):
        c_no_flags  = _make_candidate(sentiment_flags=[])
        c_two_flags = _make_candidate(sentiment_flags=["analyst_upgrade_7d", "unusual_options_aligned"])
        s_no  = compute_score(c_no_flags,  MARKET_ENV)
        s_two = compute_score(c_two_flags, MARKET_ENV)
        assert s_two > s_no


class TestComputeConfidence:
    def _c(self, iv_rank, iv_rv, above_50, above_200, rs_20d,
           news, sector_leading, ev, direction="Bullish",
           structure="bull_put_spread"):
        return {
            "ticker": "TEST", "direction": direction, "structure": structure,
            "iv_rank": iv_rank, "iv_rv_ratio": iv_rv,
            "above_50ma": above_50, "above_200ma": above_200, "rs_20d": rs_20d,
            "news_signal": news, "_leading_sectors": ["XLK"] if sector_leading else [],
            "sector": "XLK",
            "bs": {"ev": ev, "pop": 0.72},
        }

    def test_all_7_aligned_returns_high(self):
        c = self._c(72, 2.0, True, True, 3.0, "BULLISH", True, 50.0)
        result = compute_confidence(c)
        assert result["label"] == "High"
        assert result["count"] == 7

    def test_4_aligned_returns_medium(self):
        # IV rank and IV/RV aligned, trend aligned, RS aligned, but news/sector/EV not
        c = self._c(72, 2.0, True, True, 3.0, "NEUTRAL", False, -10.0)
        result = compute_confidence(c)
        assert result["label"] == "Medium"

    def test_1_aligned_returns_low(self):
        # Only EV positive
        c = self._c(30, 0.8, False, False, -1.0, "BEARISH", False, 10.0)
        result = compute_confidence(c)
        assert result["label"] == "Low"

    def test_score_string_format(self):
        c = self._c(72, 2.0, True, True, 3.0, "BULLISH", True, 50.0)
        result = compute_confidence(c)
        assert "7" in result["score_string"]
        assert "High" in result["score_string"]


class TestFilterByScore:
    def test_filters_below_floor(self):
        c1 = _make_candidate(ticker="HIGH", iv_rank=75, bs={"pop": 0.75, "ev": 30.0, "delta": -0.2, "vega": -0.1})
        c2 = _make_candidate(ticker="LOW",  bid_ask_pct=0.20)  # illiquid → score=0
        result = filter_by_score([c1, c2], MARKET_ENV)
        tickers = [r["ticker"] for r in result]
        assert "HIGH" in tickers
        assert "LOW" not in tickers

    def test_sorted_descending(self):
        c1 = _make_candidate(ticker="A", iv_rank=55,
                              bs={"pop": 0.71, "ev": 10.0, "delta": -0.2, "vega": -0.1})
        c2 = _make_candidate(ticker="B", iv_rank=80, sentiment_flags=["analyst_upgrade_7d", "unusual_options_aligned"],
                              bs={"pop": 0.80, "ev": 40.0, "delta": -0.2, "vega": -0.1})
        result = filter_by_score([c1, c2], MARKET_ENV)
        if len(result) == 2:
            assert result[0]["score"] >= result[1]["score"]
