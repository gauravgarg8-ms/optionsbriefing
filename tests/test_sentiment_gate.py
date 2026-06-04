import pytest
from screening.sentiment_gate import apply_sentiment_gate, apply_gate_to_pool

MARKET_ENV = {
    "fear_greed_score":  61,
    "put_call_ratio":    0.82,
    "market_sentiment":  "BULLISH",
    "leading_sectors":   ["XLK", "XLI"],
    "lagging_sectors":   ["XLU", "XLP"],
    "sector_news":       {},
}


def _make_candidate(**kwargs):
    defaults = {
        "ticker": "TEST", "direction": "Bullish", "sector": "XLK",
        "structure": "bull_put_spread", "iv_rank": 60,
        "sentiment_flags": [], "warning_flags": [],
        "recent_downgrade_days": 99, "recent_upgrade_days": 99,
        "unusual_options_aligned": False, "positive_company_news": False,
        "insider_buying": False, "news_signal": "NEUTRAL",
        "major_negative_event": False,
    }
    defaults.update(kwargs)
    return defaults


class TestDisqualifiers:
    def test_recent_downgrade_bullish_disqualifies(self):
        c = _make_candidate(recent_downgrade_days=2, direction="Bullish")
        _, is_disq = apply_sentiment_gate(c, MARKET_ENV)
        assert is_disq is True

    def test_downgrade_on_day_3_disqualifies(self):
        c = _make_candidate(recent_downgrade_days=3, direction="Bullish")
        _, is_disq = apply_sentiment_gate(c, MARKET_ENV)
        assert is_disq is True

    def test_downgrade_on_day_4_does_not_disqualify(self):
        c = _make_candidate(recent_downgrade_days=4, direction="Bullish")
        _, is_disq = apply_sentiment_gate(c, MARKET_ENV)
        assert is_disq is False

    def test_adverse_headline_bullish_disqualifies(self):
        env = {**MARKET_ENV, "sector_news": {"XLK": {"adverse_headline": True}}}
        c   = _make_candidate(direction="Bullish", sector="XLK")
        _, is_disq = apply_sentiment_gate(c, env)
        assert is_disq is True

    def test_adverse_headline_bearish_does_not_disqualify(self):
        env = {**MARKET_ENV, "sector_news": {"XLU": {"adverse_headline": True}}}
        c   = _make_candidate(direction="Bearish", sector="XLU")
        _, is_disq = apply_sentiment_gate(c, env)
        assert is_disq is False

    def test_major_negative_event_bearish_news_disqualifies(self):
        c = _make_candidate(news_signal="BEARISH", major_negative_event=True)
        _, is_disq = apply_sentiment_gate(c, MARKET_ENV)
        assert is_disq is True


class TestAmplifiers:
    def test_analyst_upgrade_within_7d_adds_flag(self):
        c = _make_candidate(recent_upgrade_days=5)
        enriched, _ = apply_sentiment_gate(c, MARKET_ENV)
        assert "analyst_upgrade_7d" in enriched["sentiment_flags"]

    def test_unusual_options_aligned_adds_flag(self):
        c = _make_candidate(unusual_options_aligned=True)
        enriched, _ = apply_sentiment_gate(c, MARKET_ENV)
        assert "unusual_options_aligned" in enriched["sentiment_flags"]

    def test_bullish_sentiment_leading_sector_adds_flag(self):
        c = _make_candidate(direction="Bullish", sector="XLK")
        enriched, _ = apply_sentiment_gate(c, MARKET_ENV)
        assert "bullish_sentiment_leading_sector" in enriched["sentiment_flags"]

    def test_extreme_fear_high_iv_adds_flag(self):
        env = {**MARKET_ENV, "fear_greed_score": 20, "market_sentiment": "BEARISH"}
        c   = _make_candidate(direction="Bullish", iv_rank=65)
        enriched, _ = apply_sentiment_gate(c, env)
        assert "extreme_fear_panic_premium" in enriched["sentiment_flags"]

    def test_insider_buying_adds_flag(self):
        c = _make_candidate(insider_buying=True)
        enriched, _ = apply_sentiment_gate(c, MARKET_ENV)
        assert "insider_buying" in enriched["sentiment_flags"]


class TestWarningFlags:
    def test_extreme_greed_debit_adds_warning(self):
        env = {**MARKET_ENV, "fear_greed_score": 80}
        c   = _make_candidate(direction="Bullish", structure="bull_call_spread")
        enriched, _ = apply_sentiment_gate(c, env)
        assert "extreme_greed_debit_half_size" in enriched["warning_flags"]

    def test_credit_structure_no_greed_warning(self):
        env = {**MARKET_ENV, "fear_greed_score": 80}
        c   = _make_candidate(direction="Bullish", structure="bull_put_spread")
        enriched, _ = apply_sentiment_gate(c, env)
        assert "extreme_greed_debit_half_size" not in enriched["warning_flags"]

    def test_high_put_call_debit_adds_warning(self):
        env = {**MARKET_ENV, "put_call_ratio": 1.5}
        c   = _make_candidate(direction="Bullish", structure="bull_call_spread")
        enriched, _ = apply_sentiment_gate(c, env)
        assert "high_put_call_debit_half_size" in enriched["warning_flags"]


class TestCoveredCallFlag:
    def test_bullish_high_iv_sets_flag(self):
        c = _make_candidate(direction="Bullish", iv_rank=50)
        enriched, _ = apply_sentiment_gate(c, MARKET_ENV)
        assert enriched["covered_call_opportunity"] is True

    def test_low_iv_rank_no_flag(self):
        c = _make_candidate(direction="Bullish", iv_rank=30)
        enriched, _ = apply_sentiment_gate(c, MARKET_ENV)
        assert enriched["covered_call_opportunity"] is False


class TestApplyGateToPool:
    def test_disqualified_removed_from_pool(self):
        pool = [
            _make_candidate(ticker="GOOD"),
            _make_candidate(ticker="BAD", recent_downgrade_days=1, direction="Bullish"),
        ]
        result = apply_gate_to_pool(pool, MARKET_ENV)
        tickers = [r["ticker"] for r in result]
        assert "GOOD" in tickers
        assert "BAD" not in tickers
