"""蒙特卡洛欧式期权定价与方差缩减。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bs import price as bs_price
from utils import DEFAULT_SEED, make_rng, standard_error

MCMethod = Literal["plain", "antithetic", "antithetic_cv"]


@dataclass(frozen=True)
class MCEstimate:
    """单次 MC 估计结果。"""

    mean: float
    se: float
    ci_low: float
    ci_high: float
    n: int
    method: MCMethod
    variance_reduction_ratio: float | None = None


def _validate(S: float, K: float, T: float, r: float, sigma: float) -> None:
    if S <= 0 or K <= 0:
        raise ValueError("S 与 K 必须为正")
    if T < 0:
        raise ValueError("T 不能为负")
    if sigma < 0:
        raise ValueError("sigma 不能为负")


def terminal_stock(
    S0: float, r: float, sigma: float, T: float, z: np.ndarray
) -> np.ndarray:
    """风险中性下从标准正态 z 生成到期标的价 S_T。"""
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T) * z
    return S0 * np.exp(drift + diffusion)


def discounted_call_payoff(S_T: np.ndarray, K: float, r: float, T: float) -> np.ndarray:
    """贴现后的欧式看涨 payoff。"""
    return np.exp(-r * T) * np.maximum(S_T - K, 0.0)


def _estimate_from_samples(samples: np.ndarray, method: MCMethod) -> MCEstimate:
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    mu = float(np.mean(samples))
    se = standard_error(samples)
    half = 1.96 * se
    return MCEstimate(
        mean=mu,
        se=se,
        ci_low=mu - half,
        ci_high=mu + half,
        n=n,
        method=method,
    )


def _apply_control_variate(
    y: np.ndarray, x: np.ndarray, x_mean: float
) -> np.ndarray:
    """控制变量修正：Y - b*(X - E[X])，b 为样本最优系数。"""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2:
        return y
    cov = np.cov(y, x, ddof=1)[0, 1]
    var_x = np.var(x, ddof=1)
    if var_x <= 0.0:
        return y
    b_star = cov / var_x
    return y - b_star * (x - x_mean)


def mc_european_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    method: MCMethod = "plain",
    seed: int | None = DEFAULT_SEED,
    z: np.ndarray | None = None,
) -> MCEstimate:
    """
    欧式看涨期权 MC 定价。

    method:
      - plain: 基础 MC
      - antithetic: 对偶变量 (Z 与 -Z)
      - antithetic_cv: 对偶 + 以 S_T 为控制变量
    """
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

    x_mean = S * math.exp(r * T)

    if method == "plain":
        s_t = terminal_stock(S, r, sigma, T, z)
        samples = discounted_call_payoff(s_t, K, r, T)
        return _estimate_from_samples(samples, method)

    s_plus = terminal_stock(S, r, sigma, T, z)
    s_minus = terminal_stock(S, r, sigma, T, -z)
    y_plus = discounted_call_payoff(s_plus, K, r, T)
    y_minus = discounted_call_payoff(s_minus, K, r, T)
    y = 0.5 * (y_plus + y_minus)
    x = 0.5 * (s_plus + s_minus)

    if method == "antithetic":
        return _estimate_from_samples(y, method)

    y_cv = _apply_control_variate(y, x, x_mean)
    return _estimate_from_samples(y_cv, method)


def compare_variance_reduction(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None = DEFAULT_SEED,
) -> dict[str, MCEstimate]:
    """三种估计器一次跑齐，并给出相对 plain 的方差缩减比例 (SE_plain/SE_method)^2。"""
    plain = mc_european_call(S, K, T, r, sigma, n_paths, "plain", seed=seed)
    anti = mc_european_call(S, K, T, r, sigma, n_paths, "antithetic", seed=seed)
    anti_cv = mc_european_call(S, K, T, r, sigma, n_paths, "antithetic_cv", seed=seed)

    def vr_ratio(base_se: float, se: float) -> float:
        if se <= 0.0:
            return float("nan")
        return (base_se / se) ** 2

    anti_vr = vr_ratio(plain.se, anti.se)
    cv_vr = vr_ratio(plain.se, anti_cv.se)

    anti = MCEstimate(
        mean=anti.mean,
        se=anti.se,
        ci_low=anti.ci_low,
        ci_high=anti.ci_high,
        n=anti.n,
        method=anti.method,
        variance_reduction_ratio=anti_vr,
    )
    anti_cv = MCEstimate(
        mean=anti_cv.mean,
        se=anti_cv.se,
        ci_low=anti_cv.ci_low,
        ci_high=anti_cv.ci_high,
        n=anti_cv.n,
        method=anti_cv.method,
        variance_reduction_ratio=cv_vr,
    )
    plain = MCEstimate(
        mean=plain.mean,
        se=plain.se,
        ci_low=plain.ci_low,
        ci_high=plain.ci_high,
        n=plain.n,
        method=plain.method,
        variance_reduction_ratio=1.0,
    )
    return {"plain": plain, "antithetic": anti, "antithetic_cv": anti_cv}


def se_vs_n(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    ns: tuple[int, ...],
    method: MCMethod = "plain",
    seed: int | None = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """给定路径数网格，返回 (N 数组, SE 数组)。"""
    se_list: list[float] = []
    for n in ns:
        est = mc_european_call(S, K, T, r, sigma, n, method=method, seed=seed)
        se_list.append(est.se)
    return np.array(ns, dtype=float), np.array(se_list, dtype=float)


def fit_loglog_slope(ns: np.ndarray, se: np.ndarray) -> float:
    """对 log(SE) ~ log(N) 做最小二乘拟合，返回斜率。"""
    ns = np.asarray(ns, dtype=float)
    se = np.asarray(se, dtype=float)
    mask = (ns > 0) & (se > 0)
    x = np.log(ns[mask])
    y = np.log(se[mask])
    if x.size < 2:
        return float("nan")
    coef = np.polyfit(x, y, 1)
    return float(coef[0])


def confidence_interval_coverage(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    n_trials: int,
    method: MCMethod = "plain",
    base_seed: int = DEFAULT_SEED,
) -> float:
    """重复试验，统计 BS 真值落在 95% CI 内的比例。"""
    true_val = bs_price(S, K, T, r, sigma, "call")
    hits = 0
    for t in range(n_trials):
        est = mc_european_call(
            S, K, T, r, sigma, n_paths, method=method, seed=base_seed + t
        )
        if est.ci_low <= true_val <= est.ci_high:
            hits += 1
    return hits / n_trials


def plot_mc_convergence(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    ns: tuple[int, ...] = (10**2, 10**3, 10**4, 10**5, 10**6),
    seed: int = DEFAULT_SEED,
    out_path: str = "figures/fig_mc_convergence.png",
) -> float:
    """
    绘制三种方法的 SE 随 N 变化的 log-log 图，返回 plain 方法的拟合斜率。
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    slopes: dict[str, float] = {}
    for method, label in (
        ("plain", "plain"),
        ("antithetic", "antithetic"),
        ("antithetic_cv", "antithetic + CV"),
    ):
        n_arr, se_arr = se_vs_n(S, K, T, r, sigma, ns, method=method, seed=seed)
        slope = fit_loglog_slope(n_arr, se_arr)
        slopes[method] = slope
        ax.loglog(n_arr, se_arr, "o-", label=f"{label} (slope={slope:.3f})")

    ax.set_xlabel("N (paths)")
    ax.set_ylabel("Standard error")
    ax.set_title("MC standard error vs number of paths")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return slopes["plain"]
