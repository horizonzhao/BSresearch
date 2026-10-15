"""蒙特卡洛 Greeks：有限差分、路径微分、似然比。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bs import delta as bs_delta
from bs import d1_d2, vega as bs_vega
from mc_pricing import terminal_stock
from scipy.stats import norm
from utils import DEFAULT_SEED, make_rng, standard_error

PayoffType = Literal["call", "digital"]
GreekType = Literal["delta", "vega"]
GreekMethod = Literal["finite_diff", "pathwise", "likelihood_ratio"]


@dataclass(frozen=True)
class GreekEstimate:
    """单次 Greek MC 估计。"""

    mean: float
    se: float
    ci_low: float
    ci_high: float
    n: int
    greek: GreekType
    method: GreekMethod
    payoff: PayoffType


def _validate(S: float, K: float, T: float, r: float, sigma: float) -> None:
    if S <= 0 or K <= 0:
        raise ValueError("S 与 K 必须为正")
    if T <= 0:
        raise ValueError("T 必须为正")
    if sigma <= 0:
        raise ValueError("sigma 必须为正")


def discounted_payoff(
    s_t: np.ndarray, K: float, r: float, T: float, payoff: PayoffType
) -> np.ndarray:
    """贴现 payoff。"""
    disc = math.exp(-r * T)
    if payoff == "call":
        return disc * np.maximum(s_t - K, 0.0)
    return disc * (s_t > K).astype(float)


def bs_digital_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """现金或无数字看涨 Delta 解析值。"""
    _, d2 = d1_d2(S, K, T, r, sigma)
    return math.exp(-r * T) * norm.pdf(d2) / (S * sigma * math.sqrt(T))


def _estimate(samples: np.ndarray, greek: GreekType, method: GreekMethod, payoff: PayoffType) -> GreekEstimate:
    samples = np.asarray(samples, dtype=float)
    mu = float(np.mean(samples))
    se = standard_error(samples)
    half = 1.96 * se
    return GreekEstimate(
        mean=mu,
        se=se,
        ci_low=mu - half,
        ci_high=mu + half,
        n=samples.size,
        greek=greek,
        method=method,
        payoff=payoff,
    )


def mc_greek(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    greek: GreekType = "delta",
    method: GreekMethod = "pathwise",
    payoff: PayoffType = "call",
    seed: int | None = DEFAULT_SEED,
    z: np.ndarray | None = None,
    h: float = 1e-3,
) -> GreekEstimate:
    """MC 估计 Delta 或 Vega。"""
    _validate(S, K, T, r, sigma)
    if n_paths < 1:
        raise ValueError("n_paths 至少为 1")

    rng = make_rng(seed)
    if z is None:
        z = rng.standard_normal(n_paths)
    else:
        z = np.asarray(z, dtype=float)
        if z.size != n_paths:
            raise ValueError("z 的长度须等于 n_paths")

    if method == "finite_diff":
        samples = _finite_diff_samples(S, K, T, r, sigma, z, greek, payoff, h)
    elif method == "pathwise":
        samples = _pathwise_samples(S, K, T, r, sigma, z, greek, payoff)
    elif method == "likelihood_ratio":
        samples = _likelihood_ratio_samples(S, K, T, r, sigma, z, greek, payoff)
    else:
        raise ValueError(f"未知 method: {method}")

    return _estimate(samples, greek, method, payoff)


def _finite_diff_samples(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    z: np.ndarray,
    greek: GreekType,
    payoff: PayoffType,
    h: float,
) -> np.ndarray:
    """中心差分 bump-reprice，使用公共随机数 z。"""
    if greek == "delta":
        s_up = terminal_stock(S + h, r, sigma, T, z)
        s_dn = terminal_stock(S - h, r, sigma, T, z)
        v_up = discounted_payoff(s_up, K, r, T, payoff)
        v_dn = discounted_payoff(s_dn, K, r, T, payoff)
        return (v_up - v_dn) / (2.0 * h)

    sig_up = sigma + h
    sig_dn = sigma - h
    if sig_dn <= 0:
        raise ValueError("sigma - h 必须为正，请减小 h")
    s_up = terminal_stock(S, r, sig_up, T, z)
    s_dn = terminal_stock(S, r, sig_dn, T, z)
    v_up = discounted_payoff(s_up, K, r, T, payoff)
    v_dn = discounted_payoff(s_dn, K, r, T, payoff)
    return (v_up - v_dn) / (2.0 * h)


def _pathwise_samples(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    z: np.ndarray,
    greek: GreekType,
    payoff: PayoffType,
) -> np.ndarray:
    """路径微分法；数字期权 Delta 几乎处处为 0，不能用于不光滑 payoff。"""
    s_t = terminal_stock(S, r, sigma, T, z)
    disc = math.exp(-r * T)
    indicator = s_t > K

    if greek == "delta":
        if payoff == "call":
            return disc * indicator * (s_t / S)
        # 数字期权：indicator 对 S 的 pathwise 导数为 0
        return np.zeros_like(z)

    if payoff != "call":
        raise ValueError("pathwise Vega 仅实现欧式看涨")

    sqrt_t = math.sqrt(T)
    return disc * indicator * s_t * (sqrt_t * z - sigma * T)


def _likelihood_ratio_samples(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    z: np.ndarray,
    greek: GreekType,
    payoff: PayoffType,
) -> np.ndarray:
    """似然比（score function）法。"""
    s_t = terminal_stock(S, r, sigma, T, z)
    y = discounted_payoff(s_t, K, r, T, payoff)
    sqrt_t = math.sqrt(T)

    if greek == "delta":
        score = z / (S * sigma * sqrt_t)
        return y * score

    # Vega 的 LR 权重
    score = (z * z - 1.0 - z * sigma * sqrt_t) / sigma
    return y * score


def compare_greek_methods(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    greek: GreekType = "delta",
    payoff: PayoffType = "call",
    seed: int | None = DEFAULT_SEED,
    h: float = 1e-3,
) -> dict[str, GreekEstimate]:
    """三种方法一次对比。"""
    methods: tuple[GreekMethod, ...] = ("finite_diff", "pathwise", "likelihood_ratio")
    out: dict[str, GreekEstimate] = {}
    for m in methods:
        out[m] = mc_greek(
            S, K, T, r, sigma, n_paths, greek, m, payoff, seed=seed, h=h
        )
    return out


def analytical_greek(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    greek: GreekType,
    payoff: PayoffType = "call",
) -> float:
    """解析对照（数字期权仅 Delta）。"""
    if payoff == "call":
        if greek == "delta":
            return bs_delta(S, K, T, r, sigma, "call")
        return bs_vega(S, K, T, r, sigma)
    if greek == "delta":
        return bs_digital_delta(S, K, T, r, sigma)
    raise ValueError("数字期权未定义解析 Vega")


def fd_bias_variance_scan(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    hs: np.ndarray | tuple[float, ...],
    n_paths: int,
    n_batches: int,
    greek: GreekType = "delta",
    payoff: PayoffType = "call",
    base_seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    有限差分步长扫描：返回 (h 数组, |偏差| 数组, 方差数组)。
    每个 batch 独立种子，同一 h 下估计 batch 均值的方差。
    """
    true = analytical_greek(S, K, T, r, sigma, greek, payoff)
    hs = np.asarray(hs, dtype=float)
    biases = np.zeros(hs.size)
    variances = np.zeros(hs.size)

    for i, h in enumerate(hs):
        means = np.zeros(n_batches)
        for b in range(n_batches):
            est = mc_greek(
                S,
                K,
                T,
                r,
                sigma,
                n_paths,
                greek=greek,
                method="finite_diff",
                payoff=payoff,
                seed=base_seed + i * 1000 + b,
                h=float(h),
            )
            means[b] = est.mean
        biases[i] = abs(float(np.mean(means)) - true)
        variances[i] = float(np.var(means, ddof=1)) if n_batches > 1 else 0.0

    return hs, biases, variances


def plot_greeks_methods(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    n_paths: int = 80_000,
    seed: int = DEFAULT_SEED,
    out_path: str = "figures/fig_greeks_methods.png",
) -> None:
    """欧式看涨 Delta/Vega：三种方法的 |偏差| 与 SE 对比图。"""
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for ax, greek in zip(axes, ("delta", "vega")):
        true = analytical_greek(S, K, T, r, sigma, greek, "call")
        res = compare_greek_methods(
            S, K, T, r, sigma, n_paths, greek=greek, payoff="call", seed=seed
        )
        labels = []
        bias_vals = []
        se_vals = []
        for m in ("finite_diff", "pathwise", "likelihood_ratio"):
            labels.append(m)
            bias_vals.append(abs(res[m].mean - true))
            se_vals.append(res[m].se)
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, bias_vals, width, label="|bias|")
        ax.bar(x + width / 2, se_vals, width, label="SE")
        ax.set_xticks(x, labels, rotation=15)
        ax.set_title(f"European call {greek} (true={true:.4f})")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fd_bias_variance(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    hs: tuple[float, ...] = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4),
    n_paths: int = 20_000,
    n_batches: int = 40,
    seed: int = DEFAULT_SEED,
    out_path: str = "figures/fig_fd_bias_variance.png",
) -> None:
    """有限差分 Delta 的偏差-方差权衡图。"""
    h_arr, bias, var = fd_bias_variance_scan(
        S, K, T, r, sigma, hs, n_paths, n_batches, base_seed=seed
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(h_arr, bias, "o-", label="|bias|")
    ax.loglog(h_arr, np.sqrt(var), "s-", label="std across batches")
    ax.set_xlabel("bump size h")
    ax.set_ylabel("error scale")
    ax.set_title("Finite-diff Delta: bias-variance tradeoff")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
