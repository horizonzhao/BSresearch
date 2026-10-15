"""GBM 下离散 Delta 对冲模拟与 P&L 分析。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from bs import price as bs_price
from mc_pricing import fit_loglog_slope
from processes import (
    HestonParams,
    brownian_increments,
    default_misspecified_heston_params,
    heston_call_fourier,
    simulate_heston_paths,
)
from scipy.optimize import brentq
from scipy.stats import kurtosis, skew
from utils import DEFAULT_SEED, make_rng, standard_error

DriftMode = Literal["risk_neutral", "physical"]
VolSpec = Literal["sqrt_v0", "sqrt_theta", "implied"]

DEFAULT_REBALANCE_GRID = (4, 8, 16, 32, 64, 128, 256, 512)


@dataclass(frozen=True)
class HedgeStats:
    """单次对冲实验的 P&L 汇总统计。"""

    n_rebalances: int
    n_paths: int
    mean: float
    std: float
    se_mean: float
    q01: float
    q05: float
    q50: float
    q95: float
    q99: float


@dataclass(frozen=True)
class ExtendedHedgeStats(HedgeStats):
    """含偏度、峰度的 P&L 统计。"""

    skewness: float
    excess_kurtosis: float
    label: str = ""


def _validate_hedge_inputs(
    S0: float, K: float, T: float, r: float, sigma: float, n_rebalances: int
) -> None:
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 与 K 必须为正")
    if T <= 0:
        raise ValueError("T 必须为正")
    if sigma <= 0:
        raise ValueError("sigma 必须为正")
    if n_rebalances < 1:
        raise ValueError("n_rebalances 至少为 1")


def _d1_call(S: np.ndarray, K: float, T: float, r: float, sigma: float) -> np.ndarray:
    sqrt_t = math.sqrt(T)
    return (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)


def delta_call_vec(
    S: np.ndarray, K: float, T: float, r: float, sigma: float
) -> np.ndarray:
    """向量化欧式看涨 Delta。"""
    S = np.asarray(S, dtype=float)
    if T <= 1e-14:
        return np.where(S > K, 1.0, np.where(S == K, 0.5, 0.0))
    return norm.cdf(_d1_call(S, K, T, r, sigma))


def gamma_call_vec(
    S: np.ndarray, K: float, T: float, r: float, sigma: float
) -> np.ndarray:
    """向量化欧式看涨 Gamma。"""
    S = np.asarray(S, dtype=float)
    if T <= 1e-14:
        return np.zeros_like(S)
    d1 = _d1_call(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def gbm_paths_exact(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = DEFAULT_SEED,
    dW: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    精确 GBM 路径，返回 (S, dW)。

    S 形状 (n_paths, n_steps + 1)，含 t=0 与 t=T。
    """
    if n_steps < 1:
        raise ValueError("n_steps 至少为 1")
    dt = T / n_steps
    rng = make_rng(seed)
    inc = brownian_increments(n_paths, n_steps, dt, rng, dW=dW)
    log_inc = (mu - 0.5 * sigma * sigma) * dt + sigma * inc
    s = np.empty((n_paths, n_steps + 1), dtype=float)
    s[:, 0] = S0
    s[:, 1:] = S0 * np.exp(np.cumsum(log_inc, axis=1))
    return s, inc


def bs_implied_vol(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    vol_low: float = 1e-4,
    vol_high: float = 3.0,
) -> float:
    """由欧式看涨市场价反推 BS 隐含波动率。"""
    if market_price <= 0.0:
        raise ValueError("market_price 必须为正")
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    if market_price < intrinsic - 1e-10:
        raise ValueError("market_price 低于内在价值")

    def objective(sig: float) -> float:
        return bs_price(S, K, T, r, sig, "call") - market_price

    lo, hi = vol_low, vol_high
    if objective(lo) > 0.0:
        return lo
    if objective(hi) < 0.0:
        return hi
    return float(brentq(objective, lo, hi))


def hedge_sigma_for_spec(
    spec: VolSpec,
    params: HestonParams,
    S0: float,
    K: float,
    T: float,
    r: float,
    seed: int = DEFAULT_SEED,
) -> float:
    """误设实验：对冲者使用的 BS 波动率。"""
    if spec == "sqrt_v0":
        return math.sqrt(params.v0)
    if spec == "sqrt_theta":
        return math.sqrt(params.theta)
    if spec == "implied":
        # Heston 半解析价格反推 BS 隐含波动率（seed 保留以兼容旧调用）
        heston_p = heston_call_fourier(S0, K, T, params, r=r)
        return bs_implied_vol(S0, K, T, r, heston_p)
    raise ValueError(f"未知 VolSpec: {spec}")


def hedge_pnl_on_paths(
    paths: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma_hedge: float,
    n_rebalances: int,
    cost_k: float = 0.0,
) -> np.ndarray:
    """
    在给定股价路径上做 BS Delta 对冲。

    cost_k：比例交易成本，每次调仓扣 k * |Δ股数| * S。
    """
    paths = np.asarray(paths, dtype=float)
    n_paths, n_cols = paths.shape
    if n_cols != n_rebalances + 1:
        raise ValueError("paths 列数须为 n_rebalances + 1")

    s0 = paths[:, 0]
    dt = T / n_rebalances
    v0 = bs_price(float(s0[0]), K, T, r, sigma_hedge, "call")
    delta = delta_call_vec(s0, K, T, r, sigma_hedge)
    cash = np.full(n_paths, v0, dtype=float) - delta * s0
    if cost_k > 0.0:
        cash -= cost_k * np.abs(delta) * s0

    for i in range(1, n_rebalances):
        cash *= math.exp(r * dt)
        t_rem = T - i * dt
        delta_new = delta_call_vec(paths[:, i], K, t_rem, r, sigma_hedge)
        trade = delta_new - delta
        if cost_k > 0.0:
            cash -= cost_k * np.abs(trade) * paths[:, i]
        cash -= trade * paths[:, i]
        delta = delta_new

    cash *= math.exp(r * dt)
    portfolio = delta * paths[:, -1] + cash
    payoff = np.maximum(paths[:, -1] - K, 0.0)
    return portfolio - payoff


def simulate_hedge_pnl(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_rebalances: int,
    n_paths: int,
    drift_mode: DriftMode = "risk_neutral",
    mu: float | None = None,
    seed: int | None = DEFAULT_SEED,
    paths: np.ndarray | None = None,
    dW: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    离散 Delta 对冲 P&L（每条路径一个值）。

    n_rebalances：再平衡次数 N，时点 t_i = i*Δt，i=0..N-1，到期在 T=N*Δt。
    不使用未来信息：t_i 的 Delta 仅依赖 S_{t_i} 与剩余期限。
    """
    _validate_hedge_inputs(S0, K, T, r, sigma, n_rebalances)
    if n_paths < 1:
        raise ValueError("n_paths 至少为 1")

    if drift_mode == "risk_neutral":
        stock_mu = r
    else:
        if mu is None:
            raise ValueError("physical 模式须提供 mu")
        stock_mu = mu

    if paths is None:
        paths, dW = gbm_paths_exact(
            S0, stock_mu, sigma, T, n_rebalances, n_paths, seed=seed, dW=dW
        )
    else:
        paths = np.asarray(paths, dtype=float)
        if paths.shape != (n_paths, n_rebalances + 1):
            raise ValueError("paths 形状须为 (n_paths, n_rebalances + 1)")
        dW = None

    pnl = hedge_pnl_on_paths(paths, K, T, r, sigma, n_rebalances, cost_k=0.0)
    return pnl, dW


def summarize_pnl(pnl: np.ndarray, n_rebalances: int) -> HedgeStats:
    """P&L 分布的均值、标准差与分位数。"""
    pnl = np.asarray(pnl, dtype=float)
    n = pnl.size
    return HedgeStats(
        n_rebalances=n_rebalances,
        n_paths=n,
        mean=float(np.mean(pnl)),
        std=float(np.std(pnl, ddof=1)) if n > 1 else 0.0,
        se_mean=standard_error(pnl),
        q01=float(np.quantile(pnl, 0.01)),
        q05=float(np.quantile(pnl, 0.05)),
        q50=float(np.quantile(pnl, 0.50)),
        q95=float(np.quantile(pnl, 0.95)),
        q99=float(np.quantile(pnl, 0.99)),
    )


def summarize_pnl_extended(
    pnl: np.ndarray, n_rebalances: int, label: str = ""
) -> ExtendedHedgeStats:
    """P&L 扩展统计（偏度、峰度）。"""
    base = summarize_pnl(pnl, n_rebalances)
    pnl = np.asarray(pnl, dtype=float)
    return ExtendedHedgeStats(
        n_rebalances=base.n_rebalances,
        n_paths=base.n_paths,
        mean=base.mean,
        std=base.std,
        se_mean=base.se_mean,
        q01=base.q01,
        q05=base.q05,
        q50=base.q50,
        q95=base.q95,
        q99=base.q99,
        skewness=float(skew(pnl, bias=False)) if pnl.size > 2 else float("nan"),
        excess_kurtosis=float(kurtosis(pnl, fisher=True, bias=False))
        if pnl.size > 3
        else float("nan"),
        label=label,
    )


def simulate_misspec_hedge_pnl(
    S0: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    vol_spec: VolSpec,
    n_rebalances: int,
    n_paths: int,
    seed: int | None = DEFAULT_SEED,
    paths: np.ndarray | None = None,
) -> np.ndarray:
    """真实路径 Heston，对冲用 BS Delta（波动率可能误设）。"""
    sigma_hedge = hedge_sigma_for_spec(vol_spec, params, S0, K, T, r, seed=seed or DEFAULT_SEED)
    if paths is None:
        paths = simulate_heston_paths(
            S0, T, n_rebalances, params, "full_truncation", n_paths, seed=seed, r=r
        )
    return hedge_pnl_on_paths(paths, K, T, r, sigma_hedge, n_rebalances, cost_k=0.0)


def misspec_study(
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    params: HestonParams | None = None,
    rhos: tuple[float, ...] = (-0.7, 0.0, 0.7),
    vol_specs: tuple[VolSpec, ...] = ("sqrt_v0", "sqrt_theta", "implied"),
    n_rebalances: int = 64,
    n_paths: int = 20_000,
    seed: int = DEFAULT_SEED,
) -> list[ExtendedHedgeStats]:
    """模型误设：扫描 rho 与波动率选择。"""
    if params is None:
        params = default_misspecified_heston_params()

    results: list[ExtendedHedgeStats] = []
    for ri, rho in enumerate(rhos):
        p = HestonParams(
            kappa=params.kappa,
            theta=params.theta,
            xi=params.xi,
            rho=rho,
            v0=params.v0,
            mu=r,
        )
        paths = simulate_heston_paths(
            S0, T, n_rebalances, p, "full_truncation", n_paths, seed=seed + ri, r=r
        )
        for spec in vol_specs:
            sigma_h = hedge_sigma_for_spec(spec, p, S0, K, T, r, seed=seed + 100 + ri)
            pnl = hedge_pnl_on_paths(paths, K, T, r, sigma_h, n_rebalances, cost_k=0.0)
            label = f"rho={rho:+.1f}, {spec}, sig={sigma_h:.3f}"
            results.append(summarize_pnl_extended(pnl, n_rebalances, label=label))
    return results


def cost_tradeoff_curve(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    ns: tuple[int, ...] = DEFAULT_REBALANCE_GRID,
    cost_k: float = 0.001,
    n_paths: int = 15_000,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    扫描 N，返回 (N 数组, P&L 标准差, P&L 5% 分位数)。

    含比例交易成本 k（假设值，如 0.001=0.1%）。
    """
    stds: list[float] = []
    q05s: list[float] = []
    for j, n in enumerate(ns):
        paths, _ = gbm_paths_exact(S0, r, sigma, T, n, n_paths, seed=seed + j)
        pnl = hedge_pnl_on_paths(paths, K, T, r, sigma, n, cost_k=cost_k)
        st = summarize_pnl(pnl, n)
        stds.append(st.std)
        q05s.append(st.q05)
    return (
        np.array(ns, dtype=float),
        np.array(stds, dtype=float),
        np.array(q05s, dtype=float),
    )


def gamma_ito_step_error(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_rebalances: int,
    n_paths: int,
    seed: int | None = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    逐步伊藤修正项贡献：0.5 * Gamma * S^2 * σ^2 * (ΔW^2 - Δt)，ΔW 为标准布朗增量，
    即流程文档中 0.5 * Gamma * S^2 * ((σΔW)^2 - σ^2 Δt) 的写法。

    返回 (每步绝对贡献均值, 每步均值的 Gamma 权重)，用于解释平值/临近到期误差更大。
    """
    _validate_hedge_inputs(S0, K, T, r, sigma, n_rebalances)
    dt = T / n_rebalances
    paths, dW = gbm_paths_exact(S0, r, sigma, T, n_rebalances, n_paths, seed=seed)

    step_mean_abs = np.zeros(n_rebalances, dtype=float)
    step_gamma_mean = np.zeros(n_rebalances, dtype=float)
    for i in range(n_rebalances):
        t_rem = T - i * dt
        s_i = paths[:, i]
        g = gamma_call_vec(s_i, K, t_rem, r, sigma)
        inc = dW[:, i]
        contrib = 0.5 * g * s_i * s_i * sigma * sigma * (inc * inc - dt)
        step_mean_abs[i] = float(np.mean(np.abs(contrib)))
        step_gamma_mean[i] = float(np.mean(g))
    return step_mean_abs, step_gamma_mean


def hedge_stats_vs_n(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    ns: tuple[int, ...] = DEFAULT_REBALANCE_GRID,
    n_paths: int = 20_000,
    drift_mode: DriftMode = "risk_neutral",
    mu: float | None = None,
    seed: int = DEFAULT_SEED,
) -> list[HedgeStats]:
    """扫描再平衡次数 N，返回各 N 的 P&L 统计。"""
    stats: list[HedgeStats] = []
    for j, n in enumerate(ns):
        pnl, _ = simulate_hedge_pnl(
            S0,
            K,
            T,
            r,
            sigma,
            n,
            n_paths,
            drift_mode=drift_mode,
            mu=mu,
            seed=seed + j,
        )
        stats.append(summarize_pnl(pnl, n))
    return stats


def fit_std_vs_n_slope(ns: np.ndarray, stds: np.ndarray) -> float:
    """拟合 log(std) ~ log(N) 的斜率，理论约为 -0.5。"""
    return fit_loglog_slope(ns, stds)


def plot_hedge_pnl_dist(
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    ns_plot: tuple[int, ...] = (4, 64, 512),
    n_paths: int = 20_000,
    seed: int = DEFAULT_SEED,
    out_path: str = "figures/fig_hedge_pnl_dist.png",
) -> None:
    """不同 N 下 P&L 直方图对比。"""
    fig, axes = plt.subplots(1, len(ns_plot), figsize=(4 * len(ns_plot), 4), sharey=True)
    if len(ns_plot) == 1:
        axes = [axes]

    for ax, n in zip(axes, ns_plot):
        pnl, _ = simulate_hedge_pnl(
            S0, K, T, r, sigma, n, n_paths, seed=seed + n
        )
        st = summarize_pnl(pnl, n)
        ax.hist(pnl, bins=60, density=True, alpha=0.75, color="steelblue")
        ax.axvline(0.0, color="black", lw=1)
        ax.set_title(f"N={n}\nmean={st.mean:.3f}, std={st.std:.3f}")
        ax.set_xlabel("Hedge P&L")

    axes[0].set_ylabel("Density")
    fig.suptitle("Discrete delta-hedging P&L (short call, BS delta)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hedge_std_vs_n(
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    ns: tuple[int, ...] = DEFAULT_REBALANCE_GRID,
    n_paths: int = 20_000,
    seed: int = DEFAULT_SEED,
    out_path: str = "figures/fig_hedge_std_vs_N.png",
) -> float:
    """P&L 标准差随 N 的 log-log 图，返回拟合斜率。"""
    stats = hedge_stats_vs_n(S0, K, T, r, sigma, ns, n_paths, seed=seed)
    n_arr = np.array([s.n_rebalances for s in stats], dtype=float)
    std_arr = np.array([s.std for s in stats], dtype=float)
    mean_arr = np.array([s.mean for s in stats], dtype=float)
    slope = fit_std_vs_n_slope(n_arr, std_arr)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.loglog(n_arr, std_arr, "o-", label=f"std (slope={slope:.3f})")
    ax1.set_xlabel("N (rebalances)")
    ax1.set_ylabel("P&L std")
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.semilogx(n_arr, mean_arr, "s--", color="C1", alpha=0.8, label="mean")
    ax2.set_ylabel("P&L mean")
    ax2.axhline(0.0, color="gray", lw=0.8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Hedging error vs rebalance frequency (theory: std ~ N^{-1/2})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return slope


def plot_misspec_pnl(
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    params: HestonParams | None = None,
    rho: float = -0.7,
    n_rebalances: int = 64,
    n_paths: int = 18_000,
    seed: int = DEFAULT_SEED,
    out_path: str = "figures/fig_misspec_pnl.png",
) -> list[ExtendedHedgeStats]:
    """误设 P&L 分布与 GBM 对照图。"""
    if params is None:
        params = default_misspecified_heston_params()
    p = HestonParams(
        kappa=params.kappa,
        theta=params.theta,
        xi=params.xi,
        rho=rho,
        v0=params.v0,
        mu=r,
    )

    paths_h = simulate_heston_paths(
        S0, T, n_rebalances, p, "full_truncation", n_paths, seed=seed, r=r
    )
    specs: tuple[tuple[VolSpec, str], ...] = (
        ("sqrt_v0", r"$\sqrt{v_0}$"),
        ("sqrt_theta", r"$\sqrt{\theta}$"),
        ("implied", "IV (Heston)"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    stats_out: list[ExtendedHedgeStats] = []

    # GBM 对照（Step 5 世界）
    pnl_gbm, _ = simulate_hedge_pnl(
        S0, K, T, r, math.sqrt(p.v0), n_rebalances, n_paths, seed=seed
    )
    st_gbm = summarize_pnl_extended(pnl_gbm, n_rebalances, label="GBM (Step 5)")
    stats_out.append(st_gbm)
    axes[0].hist(pnl_gbm, bins=50, density=True, alpha=0.75, color="gray")
    axes[0].axvline(0.0, color="black", lw=1)
    axes[0].set_title(
        f"GBM baseline\nmean={st_gbm.mean:.2f}, skew={st_gbm.skewness:.2f}"
    )
    axes[0].set_xlabel("P&L")

    for ax, (spec, title) in zip(axes[1:], specs):
        sig = hedge_sigma_for_spec(spec, p, S0, K, T, r, seed=seed)
        pnl = hedge_pnl_on_paths(paths_h, K, T, r, sig, n_rebalances, cost_k=0.0)
        st = summarize_pnl_extended(
            pnl, n_rebalances, label=f"Heston {spec}, rho={rho}"
        )
        stats_out.append(st)
        ax.hist(pnl, bins=50, density=True, alpha=0.75)
        ax.axvline(0.0, color="black", lw=1)
        ax.set_title(
            f"{title}\nmean={st.mean:.2f}, skew={st.skewness:.2f}\n"
            f"q01={st.q01:.2f}, q05={st.q05:.2f}"
        )
        ax.set_xlabel("P&L")

    axes[0].set_ylabel("Density")
    fig.suptitle(f"Model misspec hedging (Heston paths, rho={rho}, N={n_rebalances})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return stats_out


def plot_misspec_rho_skew(
    stats: list[ExtendedHedgeStats],
    out_path: str | None = None,
) -> None:
    """辅助：按 rho 绘制偏度（可选，主图已含 skew 标题）。"""
    if out_path is None:
        return
    rhos = []
    skews = []
    for st in stats:
        if "rho=" in st.label and "sqrt_v0" in st.label:
            rhos.append(float(st.label.split("rho=")[1].split(",")[0]))
            skews.append(st.skewness)
    if not rhos:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(rhos)), skews, tick_label=[str(x) for x in rhos])
    ax.set_xlabel("rho")
    ax.set_ylabel("P&L skewness")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cost_tradeoff(
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    ns: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256, 512),
    cost_ks: tuple[float, ...] = (0.0, 0.0005, 0.001),
    n_paths: int = 12_000,
    seed: int = DEFAULT_SEED,
    out_path: str = "figures/fig_cost_tradeoff.png",
) -> dict[str, int]:
    """
    交易成本 vs 再平衡频率权衡图。

    cost_ks 为假设比例费率（0.0005=0.05%, 0.001=0.1%）。
    返回各 k 下使 |q05| 最大的“左尾风险”或综合指标最小的 N（此处用 -q05 作左尾）。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    optimal: dict[str, int] = {}

    for k in cost_ks:
        n_arr, std_arr, q05_arr = cost_tradeoff_curve(
            S0, K, T, r, sigma, ns, cost_k=k, n_paths=n_paths, seed=seed
        )
        lbl = f"k={k:.4f}" if k > 0 else "k=0"
        axes[0].loglog(n_arr, std_arr, "o-", label=lbl)
        tail_risk = -q05_arr
        axes[1].semilogx(n_arr, tail_risk, "o-", label=lbl)
        if k > 0:
            best_idx = int(np.argmin(tail_risk + std_arr * 0.2))
            optimal[lbl] = int(n_arr[best_idx])

    axes[0].set_xlabel("N (rebalances)")
    axes[0].set_ylabel("P&L std")
    axes[0].set_title("Total P&L volatility vs N")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_xlabel("N (rebalances)")
    axes[1].set_ylabel("- q05 (left-tail loss)")
    axes[1].set_title("Left-tail risk vs N (U-shape when k>0)")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle("Transaction cost vs rebalance frequency (assumed proportional fee k)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return optimal
