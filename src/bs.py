"""Black-Scholes 欧式期权解析价格与解析 Greeks。"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]

# 数值上视为零的阈值
_T_EPS = 1e-14
_SIGMA_EPS = 1e-14


def _validate_inputs(S: float, K: float, T: float, r: float, sigma: float) -> None:
    if S <= 0 or K <= 0:
        raise ValueError("S 与 K 必须为正")
    if T < 0:
        raise ValueError("T 不能为负")
    if sigma < 0:
        raise ValueError("sigma 不能为负")


def d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    """计算 d1, d2；在 T 或 sigma 极小时返回极限情形下的值。"""
    _validate_inputs(S, K, T, r, sigma)

    if T <= _T_EPS:
        # 到期瞬间：d1,d2 仅用于连续极限，价格走内在价值分支
        if sigma <= _SIGMA_EPS:
            sign = 1.0 if S >= K else -1.0
            return sign * float("inf"), sign * float("inf")
        return float("inf") if S > K else float("-inf"), float("inf") if S > K else float("-inf")

    if sigma <= _SIGMA_EPS:
        # 无波动：S_T = S * exp(rT)，与行权价比较
        forward = S * math.exp(r * T)
        if forward > K:
            return float("inf"), float("inf")
        if forward < K:
            return float("-inf"), float("-inf")
        return 0.0, 0.0

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """欧式看涨/看跌期权 BS 价格。"""
    _validate_inputs(S, K, T, r, sigma)

    if T <= _T_EPS:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    if sigma <= _SIGMA_EPS:
        forward = S * math.exp(r * T)
        if option_type == "call":
            return math.exp(-r * T) * max(forward - K, 0.0)
        return math.exp(-r * T) * max(K - forward, 0.0)

    d1, d2 = d1_d2(S, K, T, r, sigma)
    disc_k = K * math.exp(-r * T)
    if option_type == "call":
        return S * norm.cdf(d1) - disc_k * norm.cdf(d2)
    return disc_k * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Delta：对标的现价的一阶导。"""
    _validate_inputs(S, K, T, r, sigma)

    if T <= _T_EPS:
        if option_type == "call":
            return 1.0 if S > K else (0.5 if S == K else 0.0)
        return -1.0 if S < K else (-0.5 if S == K else 0.0)

    if sigma <= _SIGMA_EPS:
        forward = S * math.exp(r * T)
        if option_type == "call":
            return math.exp(-r * T) if forward > K else 0.0
        return -math.exp(-r * T) if forward < K else 0.0

    d1, _ = d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Gamma：对 S 的二阶导（看涨看跌相同）。"""
    _validate_inputs(S, K, T, r, sigma)

    if T <= _T_EPS or sigma <= _SIGMA_EPS:
        return 0.0

    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega：对 sigma 的导数（未除以 100）。"""
    _validate_inputs(S, K, T, r, sigma)

    if T <= _T_EPS or sigma <= _SIGMA_EPS:
        return 0.0

    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * math.sqrt(T)


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Theta：对 T 的导数（时间流逝一年，即 dV/dT）。"""
    _validate_inputs(S, K, T, r, sigma)

    if T <= _T_EPS:
        return 0.0

    if sigma <= _SIGMA_EPS:
        forward = S * math.exp(r * T)
        if option_type == "call":
            if forward > K:
                return -r * math.exp(-r * T) * (forward - K)
            return 0.0
        if forward < K:
            return r * math.exp(-r * T) * (K - forward)
        return 0.0

    d1, d2 = d1_d2(S, K, T, r, sigma)
    sqrt_t = math.sqrt(T)
    common = -(S * norm.pdf(d1) * sigma) / (2.0 * sqrt_t)
    if option_type == "call":
        return common - r * K * math.exp(-r * T) * norm.cdf(d2)
    return common + r * K * math.exp(-r * T) * norm.cdf(-d2)


def rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Rho：对 r 的导数。"""
    _validate_inputs(S, K, T, r, sigma)

    if T <= _T_EPS:
        return 0.0

    if sigma <= _SIGMA_EPS:
        forward = S * math.exp(r * T)
        disc = math.exp(-r * T)
        if option_type == "call":
            if forward > K:
                return T * disc * (K - forward) + disc * (forward - K) * (-T)
            return 0.0
        if forward < K:
            return T * disc * (K - forward) + disc * (K - forward) * (-T)
        return 0.0

    _, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * math.exp(-r * T) * norm.cdf(d2)
    return -K * T * math.exp(-r * T) * norm.cdf(-d2)


def put_call_parity_residual(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Put-Call Parity 残差：C - P - (S - K*exp(-rT))，应接近 0。"""
    c = price(S, K, T, r, sigma, "call")
    p = price(S, K, T, r, sigma, "put")
    return c - p - (S - K * math.exp(-r * T))


def greeks_finite_diff(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    h_s: float = 1e-4,
    h_sigma: float = 1e-4,
    h_r: float = 1e-4,
    h_t: float = 1e-4,
) -> dict[str, float]:
    """中心差分估计 Greeks，用于与解析解对照。"""
    v0 = price(S, K, T, r, sigma, option_type)
    delta_fd = (
        price(S + h_s, K, T, r, sigma, option_type)
        - price(S - h_s, K, T, r, sigma, option_type)
    ) / (2.0 * h_s)
    gamma_fd = (
        price(S + h_s, K, T, r, sigma, option_type)
        - 2.0 * v0
        + price(S - h_s, K, T, r, sigma, option_type)
    ) / (h_s * h_s)
    vega_fd = (
        price(S, K, T, r, sigma + h_sigma, option_type)
        - price(S, K, T, r, sigma - h_sigma, option_type)
    ) / (2.0 * h_sigma)
    rho_fd = (
        price(S, K, T, r + h_r, sigma, option_type)
        - price(S, K, T, r - h_r, sigma, option_type)
    ) / (2.0 * h_r)
    # Theta：对 T 减小 h_t（时间向前）
    t_up = max(T - h_t, 0.0)
    t_dn = T + h_t
    theta_fd = (
        price(S, K, t_up, r, sigma, option_type)
        - price(S, K, t_dn, r, sigma, option_type)
    ) / (t_dn - t_up)

    return {
        "delta": delta_fd,
        "gamma": gamma_fd,
        "vega": vega_fd,
        "rho": rho_fd,
        "theta": theta_fd,
    }
