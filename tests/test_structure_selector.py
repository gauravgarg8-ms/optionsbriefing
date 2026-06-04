import pytest
from quant.structure_selector import select_structure, apply_golden_rules


class TestSelectStructure:
    # ── High IV zone (> 50%) ────────────────────────────────────────────────
    def test_high_iv_bullish_returns_bull_put_spread(self):
        assert select_structure(70, 2.0, "Bullish") == "bull_put_spread"

    def test_high_iv_bearish_returns_bear_call_spread(self):
        assert select_structure(70, 2.0, "Bearish") == "bear_call_spread"

    def test_high_iv_neutral_returns_iron_condor(self):
        assert select_structure(70, 2.0, "Neutral") == "iron_condor"

    # ── Low IV zone (< 30%) ─────────────────────────────────────────────────
    def test_low_iv_bullish_returns_bull_call_spread(self):
        assert select_structure(25, 0.8, "Bullish") == "bull_call_spread"

    def test_low_iv_bearish_returns_bear_put_spread(self):
        assert select_structure(25, 0.8, "Bearish") == "bear_put_spread"

    def test_low_iv_neutral_returns_long_straddle(self):
        assert select_structure(25, 0.8, "Neutral") == "long_straddle"

    # ── Middle zone Decision #7 tie-breaking (IV/RV threshold = 1.2) ───────
    def test_middle_zone_high_iv_rv_bullish_returns_credit(self):
        # IV/RV = 1.5 ≥ 1.2 → credit → bull_put_spread
        assert select_structure(40, 1.5, "Bullish") == "bull_put_spread"

    def test_middle_zone_high_iv_rv_bearish_returns_credit(self):
        assert select_structure(40, 1.5, "Bearish") == "bear_call_spread"

    def test_middle_zone_low_iv_rv_bullish_returns_debit(self):
        # IV/RV = 1.0 < 1.2 → debit → bull_call_spread
        assert select_structure(40, 1.0, "Bullish") == "bull_call_spread"

    def test_middle_zone_low_iv_rv_bearish_returns_debit(self):
        assert select_structure(40, 1.0, "Bearish") == "bear_put_spread"

    def test_middle_zone_exactly_at_threshold_is_credit(self):
        # IV/RV = 1.2 (at boundary) → credit
        assert select_structure(40, 1.2, "Bullish") == "bull_put_spread"

    def test_middle_zone_just_below_threshold_is_debit(self):
        # IV/RV = 1.19 → debit
        assert select_structure(40, 1.19, "Bullish") == "bull_call_spread"

    # ── Special cases ────────────────────────────────────────────────────────
    def test_bullish_income_returns_csp(self):
        assert select_structure(50, 1.5, "Bullish income") == "cash_secured_put"

    def test_earnings_implied_greater_hist_returns_credit(self):
        result = select_structure(
            60, 2.0, "Bullish",
            earnings_flag=True, implied_move_pct=0.08, hist_avg_move_pct=0.05
        )
        assert result == "earnings_credit_spread"

    def test_earnings_implied_less_hist_returns_straddle(self):
        result = select_structure(
            25, 0.8, "Bullish",
            earnings_flag=True, implied_move_pct=0.03, hist_avg_move_pct=0.07
        )
        assert result == "long_straddle"


class TestApplyGoldenRules:
    def test_debit_with_high_iv_rank_overridden_to_credit(self):
        # bull_call_spread with IV Rank 60 → should become bull_put_spread
        result = apply_golden_rules("bull_call_spread", 60)
        assert result == "bull_put_spread"

    def test_bear_put_spread_with_high_iv_rank_overridden(self):
        result = apply_golden_rules("bear_put_spread", 65)
        assert result == "bear_call_spread"

    def test_credit_with_low_iv_rank_overridden_to_debit(self):
        result = apply_golden_rules("bull_put_spread", 15)
        assert result == "bull_call_spread"

    def test_bear_call_spread_with_low_iv_rank_overridden(self):
        result = apply_golden_rules("bear_call_spread", 15)
        assert result == "bear_put_spread"

    def test_valid_credit_structure_unchanged(self):
        # IV Rank = 60 → bull_put_spread is correct — no override
        result = apply_golden_rules("bull_put_spread", 60)
        assert result == "bull_put_spread"

    def test_valid_debit_structure_unchanged(self):
        # IV Rank = 25 → bull_call_spread is correct — no override
        result = apply_golden_rules("bull_call_spread", 25)
        assert result == "bull_call_spread"

    def test_middle_zone_no_override(self):
        # IV Rank = 40 — neither rule triggers
        result = apply_golden_rules("bull_put_spread", 40)
        assert result == "bull_put_spread"
