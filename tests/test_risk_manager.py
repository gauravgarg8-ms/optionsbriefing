import pytest
from datetime import date, timedelta
from risk_manager import compute_trade_management, compute_all_trade_management


def _make_candidate(structure, net_credit=None, net_debit=None, max_profit=None,
                    max_loss=None, pop=0.72, expiry_days_ahead=38, ticker="TEST"):
    today  = date.today()
    expiry = (today + timedelta(days=expiry_days_ahead)).isoformat()
    pricing = {}
    if net_credit is not None:
        pricing = {"net_credit": net_credit, "max_profit": max_profit or net_credit*100,
                   "max_loss": max_loss or (5.0 - net_credit)*100}
    elif net_debit is not None:
        pricing = {"net_debit": net_debit, "max_profit": (10.0 - net_debit)*100,
                   "max_loss": net_debit*100}
    return {
        "ticker": ticker, "structure": structure, "expiry": expiry,
        "spread_pricing": pricing,
        "bs": {"pop": pop, "ev": 20.0, "delta": -0.25, "vega": -0.15},
    }


class TestComputeTradeManagement:
    def test_credit_profit_target_is_50_pct_max_profit(self):
        c = _make_candidate("bull_put_spread", net_credit=1.50, max_profit=150.0)
        r = compute_trade_management(c)
        assert abs(r["profit_target_usd"] - 75.0) < 0.01

    def test_credit_stop_loss_is_2x_credit(self):
        c = _make_candidate("bull_put_spread", net_credit=1.50)
        r = compute_trade_management(c)
        assert abs(r["stop_loss_usd"] - 300.0) < 0.01   # 1.50 × 2 × 100

    def test_debit_profit_target_is_premium_x_100(self):
        c = _make_candidate("bull_call_spread", net_debit=2.00)
        r = compute_trade_management(c)
        assert abs(r["profit_target_usd"] - 200.0) < 0.01

    def test_debit_stop_loss_is_premium_x_100(self):
        c = _make_candidate("bear_put_spread", net_debit=1.50)
        r = compute_trade_management(c)
        assert abs(r["stop_loss_usd"] - 150.0) < 0.01

    def test_date_21_dte_computed_correctly(self):
        c = _make_candidate("bull_put_spread", net_credit=1.50, expiry_days_ahead=38)
        r = compute_trade_management(c)
        expiry = date.fromisoformat(c["expiry"])
        expected_21 = (expiry - __import__("datetime").timedelta(days=21)).isoformat()
        assert r["date_21_dte"] == expected_21

    def test_avoid_hold_past_is_7_days_before_expiry(self):
        c = _make_candidate("bull_put_spread", net_credit=1.50, expiry_days_ahead=38)
        r = compute_trade_management(c)
        expiry = date.fromisoformat(c["expiry"])
        expected = (expiry - __import__("datetime").timedelta(days=7)).isoformat()
        assert r["avoid_hold_past"] == expected

    def test_pop_quality_high(self):
        c = _make_candidate("bull_put_spread", net_credit=1.50, pop=0.82)
        r = compute_trade_management(c)
        assert r["pop_quality"] == "High"

    def test_pop_quality_good(self):
        c = _make_candidate("bull_put_spread", net_credit=1.50, pop=0.75)
        r = compute_trade_management(c)
        assert r["pop_quality"] == "Good"
        assert r["pop_half_size"] is False

    def test_pop_quality_acceptable_and_half_size(self):
        c = _make_candidate("bull_put_spread", net_credit=1.50, pop=0.63)
        r = compute_trade_management(c)
        assert r["pop_quality"] == "Acceptable"
        assert r["pop_half_size"] is True

    def test_pop_quality_exclude(self):
        c = _make_candidate("bull_put_spread", net_credit=1.50, pop=0.55)
        r = compute_trade_management(c)
        assert r["pop_quality"] == "EXCLUDE"

    def test_no_expiry_dates_are_none(self):
        c = _make_candidate("bull_put_spread", net_credit=1.50)
        c["expiry"] = None
        r = compute_trade_management(c)
        assert r["date_21_dte"] is None
        assert r["avoid_hold_past"] is None

    def test_original_fields_preserved(self):
        c = _make_candidate("bull_put_spread", net_credit=1.50)
        r = compute_trade_management(c)
        assert r["ticker"] == "TEST"
        assert r["structure"] == "bull_put_spread"


class TestComputeAllTradeManagement:
    def test_applies_to_all_candidates(self):
        screened = {
            "candidates": [
                _make_candidate("bull_put_spread", net_credit=1.50, ticker="A"),
                _make_candidate("bear_call_spread", net_credit=1.20, ticker="B"),
            ]
        }
        result = compute_all_trade_management(screened)
        for c in result["candidates"]:
            assert "profit_target_usd" in c
            assert "date_21_dte" in c

    def test_empty_candidates_no_error(self):
        result = compute_all_trade_management({"candidates": []})
        assert result["candidates"] == []
