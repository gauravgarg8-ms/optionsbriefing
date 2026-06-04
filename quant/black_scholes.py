"""
Black-Scholes pricing and Greeks for 8 options strategies.

⚠️  All outputs are THEORETICAL. They ignore volatility skew, smile,
     early assignment, and bid/ask friction.
     Always verify against live broker chain before trading.
"""

import math

import numpy as np
from scipy.stats import norm
from loguru import logger

from errors import ErrorCode


# ── Core B-S formula ────────────────────────────────────────────────────────

def black_scholes(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Standard Black-Scholes for European options.
    S: spot price, K: strike, T: time to expiry (years), r: risk-free rate,
    sigma: annualised implied vol.
    Returns: d1, d2, call_price, put_price.
    """
    if T <= 0:
        logger.error(f"[{ErrorCode.E2002}] BS_INVALID_INPUT: T={T} ≤ 0")
        raise ValueError(f"Time to expiry must be > 0, got {T}")
    if sigma <= 0:
        logger.error(f"[{ErrorCode.E2002}] BS_INVALID_INPUT: sigma={sigma} ≤ 0")
        raise ValueError(f"Sigma must be > 0, got {sigma}")

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put  = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return {
        "d1": round(d1, 4),
        "d2": round(d2, 4),
        "call_price": round(max(call, 0.0), 4),
        "put_price":  round(max(put,  0.0), 4),
    }


def compute_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Position-level Greeks (per contract = 100 shares, so theta is per calendar day).
    """
    bs  = black_scholes(S, K, T, r, sigma)
    d1  = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2  = d1 - sigma * math.sqrt(T)
    pdf = norm.pdf(d1)
    delta_call = norm.cdf(d1)
    delta_put  = delta_call - 1
    gamma      = pdf / (S * sigma * math.sqrt(T))
    # Theta per calendar day (not trading day)
    theta_call = (
        -(S * pdf * sigma) / (2 * math.sqrt(T))
        - r * K * math.exp(-r * T) * norm.cdf(d2)
    ) / 365
    theta_put = (
        -(S * pdf * sigma) / (2 * math.sqrt(T))
        + r * K * math.exp(-r * T) * norm.cdf(-d2)
    ) / 365
    vega = S * pdf * math.sqrt(T) / 100  # per 1% change in vol
    return {
        "delta_call": round(delta_call, 4),
        "delta_put":  round(delta_put, 4),
        "gamma":      round(gamma, 6),
        "theta_call": round(theta_call, 4),
        "theta_put":  round(theta_put, 4),
        "vega":       round(vega, 4),
    }


def compute_pop(d2: float, structure: str) -> float:
    """
    Probability of Profit at expiry.
    Credit → N(d2) at short strike; Debit → N(-d2); Condor → min of both short legs.
    """
    if "condor" in structure:
        return round(float(norm.cdf(d2)), 4)   # simplified: use put-side d2
    if structure in ("bull_put_spread", "bear_call_spread", "cash_secured_put"):
        return round(float(norm.cdf(d2)), 4)
    return round(float(norm.cdf(-d2)), 4)


def compute_ev(pop: float, max_profit: float, max_loss: float) -> float:
    """
    Expected Value = (PoP × max_profit) − ((1 − PoP) × max_loss).
    Both max_profit and max_loss should be in dollars (contract level).
    """
    return round(float(pop * max_profit - (1 - pop) * max_loss), 2)


# ── 8 Strategy Pricers ──────────────────────────────────────────────────────

def price_bull_put_spread(S: float, k_short: float, k_long: float,
                          T: float, r: float, sigma: float) -> dict:
    """Sell put at k_short, buy put at k_long (k_long < k_short)."""
    short_bs = black_scholes(S, k_short, T, r, sigma)
    long_bs  = black_scholes(S, k_long,  T, r, sigma)
    net_credit = round(short_bs["put_price"] - long_bs["put_price"], 4)
    net_credit = max(net_credit, 0.0)
    spread_width = k_short - k_long
    max_profit   = round(net_credit * 100, 2)
    max_loss     = round((spread_width - net_credit) * 100, 2)
    breakeven    = round(k_short - net_credit, 2)
    d2           = short_bs["d2"]
    pop          = compute_pop(d2, "bull_put_spread")
    ev           = compute_ev(pop, max_profit, max_loss)
    return {
        "strategy":    "bull_put_spread",
        "net_credit":  net_credit,
        "max_profit":  max_profit,
        "max_loss":    max_loss,
        "breakeven":   breakeven,
        "pop":         pop,
        "ev":          ev,
        "d2_short":    round(d2, 4),
        "theoretical": True,
    }


def price_bear_call_spread(S: float, k_short: float, k_long: float,
                           T: float, r: float, sigma: float) -> dict:
    """Sell call at k_short, buy call at k_long (k_long > k_short)."""
    short_bs = black_scholes(S, k_short, T, r, sigma)
    long_bs  = black_scholes(S, k_long,  T, r, sigma)
    net_credit = round(short_bs["call_price"] - long_bs["call_price"], 4)
    net_credit = max(net_credit, 0.0)
    spread_width = k_long - k_short
    max_profit   = round(net_credit * 100, 2)
    max_loss     = round((spread_width - net_credit) * 100, 2)
    breakeven    = round(k_short + net_credit, 2)
    d2           = short_bs["d2"]
    pop          = compute_pop(d2, "bear_call_spread")
    ev           = compute_ev(pop, max_profit, max_loss)
    return {
        "strategy":    "bear_call_spread",
        "net_credit":  net_credit,
        "max_profit":  max_profit,
        "max_loss":    max_loss,
        "breakeven":   breakeven,
        "pop":         pop,
        "ev":          ev,
        "d2_short":    round(d2, 4),
        "theoretical": True,
    }


def price_bull_call_spread(S: float, k_long: float, k_short: float,
                           T: float, r: float, sigma: float) -> dict:
    """Buy call at k_long (ATM), sell call at k_short (OTM). k_short > k_long."""
    long_bs  = black_scholes(S, k_long,  T, r, sigma)
    short_bs = black_scholes(S, k_short, T, r, sigma)
    net_debit  = round(long_bs["call_price"] - short_bs["call_price"], 4)
    net_debit  = max(net_debit, 0.0)
    spread_width = k_short - k_long
    max_profit   = round((spread_width - net_debit) * 100, 2)
    max_loss     = round(net_debit * 100, 2)
    breakeven    = round(k_long + net_debit, 2)
    d2           = long_bs["d2"]
    pop          = compute_pop(d2, "bull_call_spread")
    ev           = compute_ev(pop, max_profit, max_loss)
    return {
        "strategy":    "bull_call_spread",
        "net_debit":   net_debit,
        "max_profit":  max_profit,
        "max_loss":    max_loss,
        "breakeven":   breakeven,
        "pop":         pop,
        "ev":          ev,
        "d2_long":     round(d2, 4),
        "theoretical": True,
    }


def price_bear_put_spread(S: float, k_long: float, k_short: float,
                          T: float, r: float, sigma: float) -> dict:
    """Buy put at k_long (ATM), sell put at k_short (OTM below). k_short < k_long."""
    long_bs  = black_scholes(S, k_long,  T, r, sigma)
    short_bs = black_scholes(S, k_short, T, r, sigma)
    net_debit  = round(long_bs["put_price"] - short_bs["put_price"], 4)
    net_debit  = max(net_debit, 0.0)
    spread_width = k_long - k_short
    max_profit   = round((spread_width - net_debit) * 100, 2)
    max_loss     = round(net_debit * 100, 2)
    breakeven    = round(k_long - net_debit, 2)
    d2           = long_bs["d2"]
    pop          = compute_pop(d2, "bear_put_spread")
    ev           = compute_ev(pop, max_profit, max_loss)
    return {
        "strategy":    "bear_put_spread",
        "net_debit":   net_debit,
        "max_profit":  max_profit,
        "max_loss":    max_loss,
        "breakeven":   breakeven,
        "pop":         pop,
        "ev":          ev,
        "d2_long":     round(d2, 4),
        "theoretical": True,
    }


def price_iron_condor(S: float, k_put_short: float, k_put_long: float,
                      k_call_short: float, k_call_long: float,
                      T: float, r: float, sigma: float) -> dict:
    """
    Sell put spread (k_put_long/k_put_short) + sell call spread (k_call_short/k_call_long).
    k_put_long < k_put_short < S < k_call_short < k_call_long
    """
    ps = price_bull_put_spread(S, k_put_short, k_put_long, T, r, sigma)
    cs = price_bear_call_spread(S, k_call_short, k_call_long, T, r, sigma)
    total_credit = round(ps["net_credit"] + cs["net_credit"], 4)
    put_width    = k_put_short  - k_put_long
    call_width   = k_call_long  - k_call_short
    wing_width   = max(put_width, call_width)
    max_profit   = round(total_credit * 100, 2)
    max_loss     = round((wing_width - total_credit) * 100, 2)
    be_low       = round(k_put_short  - total_credit, 2)
    be_high      = round(k_call_short + total_credit, 2)
    pop          = round(float((norm.cdf(ps["d2_short"]) + (1 - norm.cdf(-cs["d2_short"]))) / 2), 4)
    ev           = compute_ev(pop, max_profit, max_loss)
    return {
        "strategy":      "iron_condor",
        "total_credit":  total_credit,
        "max_profit":    max_profit,
        "max_loss":      max_loss,
        "be_low":        be_low,
        "be_high":       be_high,
        "pop":           pop,
        "ev":            ev,
        "theoretical":   True,
    }


def price_long_straddle(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Buy ATM call + ATM put at same strike K."""
    bs         = black_scholes(S, K, T, r, sigma)
    net_debit  = round(bs["call_price"] + bs["put_price"], 4)
    max_loss   = round(net_debit * 100, 2)
    be_low     = round(K - net_debit, 2)
    be_high    = round(K + net_debit, 2)
    be_pct     = round(net_debit / S * 100, 2)
    d2         = bs["d2"]
    pop        = compute_pop(d2, "long_straddle")
    ev         = compute_ev(pop, S * 0.10 * 100, max_loss)  # EV approximated with 10% move target
    return {
        "strategy":      "long_straddle",
        "net_debit":     net_debit,
        "max_profit":    "unlimited",
        "max_loss":      max_loss,
        "be_low":        be_low,
        "be_high":       be_high,
        "breakeven_pct": be_pct,
        "pop":           pop,
        "ev":            ev,
        "theoretical":   True,
    }


def price_long_strangle(S: float, k_put: float, k_call: float,
                        T: float, r: float, sigma: float) -> dict:
    """Buy OTM put at k_put + OTM call at k_call. k_put < S < k_call."""
    put_bs     = black_scholes(S, k_put,  T, r, sigma)
    call_bs    = black_scholes(S, k_call, T, r, sigma)
    net_debit  = round(put_bs["put_price"] + call_bs["call_price"], 4)
    max_loss   = round(net_debit * 100, 2)
    be_low     = round(k_put  - net_debit, 2)
    be_high    = round(k_call + net_debit, 2)
    d2         = put_bs["d2"]
    pop        = compute_pop(d2, "long_strangle")
    ev         = compute_ev(pop, S * 0.10 * 100, max_loss)
    return {
        "strategy":    "long_strangle",
        "net_debit":   net_debit,
        "max_profit":  "unlimited",
        "max_loss":    max_loss,
        "be_low":      be_low,
        "be_high":     be_high,
        "pop":         pop,
        "ev":          ev,
        "theoretical": True,
    }


def price_cash_secured_put(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Sell an OTM or ATM put and secure it with cash equal to strike × 100.
    Max loss = (K − net_credit) × 100 (assignment at K, net of premium).
    """
    bs         = black_scholes(S, K, T, r, sigma)
    net_credit = round(bs["put_price"], 4)
    max_profit = round(net_credit * 100, 2)
    # Max loss: assigned at K, effective cost basis = K - net_credit
    effective_cost = round(K - net_credit, 2)
    max_loss       = round(effective_cost * 100, 2)
    breakeven      = effective_cost
    d2             = bs["d2"]
    pop            = compute_pop(d2, "cash_secured_put")
    ev             = compute_ev(pop, max_profit, max_loss)
    return {
        "strategy":          "cash_secured_put",
        "net_credit":        net_credit,
        "max_profit":        max_profit,
        "max_loss":          max_loss,
        "breakeven":         breakeven,
        "effective_cost_basis": effective_cost,
        "pop":               pop,
        "ev":                ev,
        "theoretical":       True,
    }


# ── Strategy dispatcher ──────────────────────────────────────────────────────

PRICERS = {
    "bull_put_spread":  price_bull_put_spread,
    "bear_call_spread": price_bear_call_spread,
    "bull_call_spread": price_bull_call_spread,
    "bear_put_spread":  price_bear_put_spread,
    "iron_condor":      price_iron_condor,
    "long_straddle":    price_long_straddle,
    "long_strangle":    price_long_strangle,
    "cash_secured_put": price_cash_secured_put,
}
