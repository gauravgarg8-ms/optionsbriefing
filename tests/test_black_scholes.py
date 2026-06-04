import math
import pytest
from scipy.stats import norm

from quant.black_scholes import (
    black_scholes, compute_greeks, compute_pop, compute_ev,
    price_bull_put_spread, price_bear_call_spread,
    price_bull_call_spread, price_bear_put_spread,
    price_iron_condor, price_long_straddle, price_long_strangle,
    price_cash_secured_put,
)


class TestBlackScholes:
    def test_atm_call_known_value(self):
        # S=K=100, T=1yr, r=0, sigma=0.2 → call ≈ 7.97, d1=0.1, d2=-0.1
        result = black_scholes(100, 100, 1.0, 0.0, 0.20)
        assert abs(result["d1"] - 0.10) < 0.001
        assert abs(result["d2"] - (-0.10)) < 0.001
        assert abs(result["call_price"] - 7.97) < 0.05

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        result = black_scholes(S, K, T, r, sigma)
        # C - P = S - K * e^(-rT)
        lhs = result["call_price"] - result["put_price"]
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 0.001

    def test_deep_itm_call_approaches_intrinsic(self):
        # Deep ITM call (S >> K): C ≈ S - K*e^(-rT)
        result = black_scholes(200, 100, 1.0, 0.0, 0.20)
        intrinsic = 200 - 100
        assert result["call_price"] >= intrinsic * 0.95

    def test_invalid_T_raises(self):
        with pytest.raises(ValueError, match="expiry"):
            black_scholes(100, 100, 0.0, 0.05, 0.25)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError, match="Sigma"):
            black_scholes(100, 100, 1.0, 0.05, 0.0)

    def test_negative_T_raises(self):
        with pytest.raises(ValueError):
            black_scholes(100, 100, -0.5, 0.05, 0.25)


class TestComputeGreeks:
    def test_returns_all_keys(self):
        g = compute_greeks(100, 100, 1.0, 0.05, 0.25)
        for k in ["delta_call", "delta_put", "gamma", "theta_call", "theta_put", "vega"]:
            assert k in g

    def test_delta_call_between_0_and_1(self):
        g = compute_greeks(100, 100, 1.0, 0.05, 0.25)
        assert 0 < g["delta_call"] < 1

    def test_delta_put_between_neg1_and_0(self):
        g = compute_greeks(100, 100, 1.0, 0.05, 0.25)
        assert -1 < g["delta_put"] < 0

    def test_delta_put_call_relationship(self):
        g = compute_greeks(100, 100, 1.0, 0.05, 0.25)
        assert abs(g["delta_call"] + abs(g["delta_put"]) - 1.0) < 0.001


class TestComputePop:
    def test_credit_structure_uses_cdf_d2(self):
        d2  = 0.5
        pop = compute_pop(d2, "bull_put_spread")
        assert abs(pop - norm.cdf(d2)) < 0.0001

    def test_debit_structure_uses_neg_d2(self):
        d2  = 0.5
        pop = compute_pop(d2, "bull_call_spread")
        assert abs(pop - norm.cdf(-d2)) < 0.0001

    def test_pop_in_range_0_to_1(self):
        for d2 in [-2, -1, 0, 1, 2]:
            for structure in ["bull_put_spread", "bull_call_spread", "iron_condor"]:
                pop = compute_pop(d2, structure)
                assert 0.0 <= pop <= 1.0


class TestComputeEV:
    def test_positive_ev(self):
        # PoP=0.8, profit=100, loss=400 → EV = 80 - 80 = 0... wait:
        # EV = 0.8*100 - 0.2*400 = 80 - 80 = 0
        ev = compute_ev(0.8, 100, 400)
        assert abs(ev - 0.0) < 0.01

    def test_clearly_positive_ev(self):
        # PoP=0.75, profit=200, loss=300 → EV = 150 - 75 = 75
        ev = compute_ev(0.75, 200, 300)
        assert ev > 0

    def test_negative_ev(self):
        # PoP=0.4, profit=100, loss=400 → EV = 40 - 240 = -200
        ev = compute_ev(0.4, 100, 400)
        assert ev < 0


class TestSpreadPricers:
    S, K_short, K_long = 100.0, 95.0, 90.0
    T, r, sigma = 30/365, 0.05, 0.30

    def test_bull_put_spread_max_loss_formula(self):
        result = price_bull_put_spread(self.S, self.K_short, self.K_long, self.T, self.r, self.sigma)
        expected_max_loss = round((self.K_short - self.K_long - result["net_credit"]) * 100, 2)
        assert abs(result["max_loss"] - expected_max_loss) < 0.01

    def test_bull_put_spread_max_profit_nonnegative(self):
        result = price_bull_put_spread(self.S, self.K_short, self.K_long, self.T, self.r, self.sigma)
        assert result["max_profit"] >= 0

    def test_bear_call_spread_max_loss_formula(self):
        k_short, k_long = 105.0, 110.0
        result = price_bear_call_spread(self.S, k_short, k_long, self.T, self.r, self.sigma)
        expected_max_loss = round((k_long - k_short - result["net_credit"]) * 100, 2)
        assert abs(result["max_loss"] - expected_max_loss) < 0.01

    def test_bull_call_spread_breakeven(self):
        k_long, k_short = 100.0, 110.0
        result = price_bull_call_spread(self.S, k_long, k_short, self.T, self.r, self.sigma)
        assert abs(result["breakeven"] - (k_long + result["net_debit"])) < 0.01

    def test_iron_condor_max_profit_equals_total_credit_x_100(self):
        result = price_iron_condor(
            self.S, 92.0, 88.0, 108.0, 112.0, self.T, self.r, self.sigma
        )
        assert abs(result["max_profit"] - result["total_credit"] * 100) < 0.01

    def test_long_straddle_breakevens(self):
        result = price_long_straddle(self.S, 100.0, self.T, self.r, self.sigma)
        assert abs(result["be_low"]  - (100.0 - result["net_debit"])) < 0.01
        assert abs(result["be_high"] - (100.0 + result["net_debit"])) < 0.01

    def test_long_straddle_max_profit_unlimited(self):
        result = price_long_straddle(self.S, 100.0, self.T, self.r, self.sigma)
        assert result["max_profit"] == "unlimited"

    def test_long_strangle_has_two_breakevens(self):
        result = price_long_strangle(self.S, 95.0, 105.0, self.T, self.r, self.sigma)
        assert "be_low" in result and "be_high" in result
        assert result["be_low"] < result["be_high"]

    def test_csp_effective_cost_basis(self):
        result = price_cash_secured_put(self.S, 95.0, self.T, self.r, self.sigma)
        expected_cost = round(95.0 - result["net_credit"], 2)
        assert abs(result["effective_cost_basis"] - expected_cost) < 0.01

    def test_csp_max_profit_equals_credit_x_100(self):
        result = price_cash_secured_put(self.S, 95.0, self.T, self.r, self.sigma)
        assert abs(result["max_profit"] - result["net_credit"] * 100) < 0.01

    def test_all_pricers_return_theoretical_flag(self):
        for result in [
            price_bull_put_spread(self.S, 95, 90, self.T, self.r, self.sigma),
            price_bear_call_spread(self.S, 105, 110, self.T, self.r, self.sigma),
            price_bull_call_spread(self.S, 100, 110, self.T, self.r, self.sigma),
            price_bear_put_spread(self.S, 100, 90, self.T, self.r, self.sigma),
            price_long_straddle(self.S, 100, self.T, self.r, self.sigma),
            price_long_strangle(self.S, 95, 105, self.T, self.r, self.sigma),
            price_cash_secured_put(self.S, 95, self.T, self.r, self.sigma),
        ]:
            assert result.get("theoretical") is True
