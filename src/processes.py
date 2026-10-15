"""GBM / Heston 路径生成，Euler / Milstein / full truncation 离散格式。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from mc_pricing import fit_loglog_slope
from utils import DEFAULT_SEED, make_rng, standard_error

GBMScheme = Literal["exact", "euler", "milstein"]
HestonScheme = Literal["euler", "reflection", "full_truncation"]


@dataclass(frozen=True)
class ConvergencePoint:
    """单个时间步长下的收敛误差估计。"""

    dt: float
    n_steps: int
    mean_error: float
    se: float


def _validate_gbm(S0: float, mu: float, sigma: float, T: float) -> None:
    if S0 <= 0:
        raise ValueError("S0 必须为正")
    if T <= 0:
        raise ValueError("T 必须为正")
    if sigma < 0:
        raise ValueError("sigma 不能为负")


def brownian_increments(
    n_paths: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
    dW: np.ndarray | None = None,
) -> np.ndarray:
    """生成形状 (n_paths, n_steps) 的布朗增量，满足 Var(ΔW)=dt。"""
    if dW is None:
        return rng.standard_normal((n_paths, n_steps)) * math.sqrt(dt)
    dW = np.asarray(dW, dtype=float)
    if dW.shape != (n_paths, n_steps):
        raise ValueError("dW 形状须为 (n_paths, n_steps)")
    return dW


def gbm_terminal_from_increments(
    S0: float,
    mu: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
    scheme: GBMScheme = "exact",
) -> np.ndarray:
    """
    在同一组布朗增量 dW 下，生成 GBM 到期价格（每条路径一个值）。

    强收敛比较时，exact 与各离散格式必须使用相同的 dW。
    """
    _validate_gbm(S0, mu, sigma, dt * dW.shape[1])
    dW = np.asarray(dW, dtype=float)
    n_paths, n_steps = dW.shape

    if scheme == "exact":
        w_t = np.sum(dW, axis=1)
        t = n_steps * dt
        return S0 * np.exp((mu - 0.5 * sigma * sigma) * t + sigma * w_t)

    s = np.full(n_paths, S0, dtype=float)
    if scheme == "euler":
        for j in range(n_steps):
            inc = dW[:, j]
            s = s + mu * s * dt + sigma * s * inc
        return s

    if scheme == "milstein":
        sig2 = sigma * sigma
        for j in range(n_steps):
            inc = dW[:, j]
            s = s + mu * s * dt + sigma * s * inc + 0.5 * sig2 * s * (inc * inc - dt)
        return s

    raise ValueError(f"未知 GBM scheme: {scheme}")


def simulate_gbm_terminal(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    scheme: GBMScheme,
    n_paths: int,
    seed: int | None = DEFAULT_SEED,
    dW: np.ndarray | None = None,
) -> np.ndarray:
    """批量模拟 GBM 到期价。"""
    if n_steps < 1:
        raise ValueError("n_steps 至少为 1")
    dt = T / n_steps
    rng = make_rng(seed)
    inc = brownian_increments(n_paths, n_steps, dt, rng, dW=dW)
    return gbm_terminal_from_increments(S0, mu, sigma, dt, inc, scheme=scheme)


def gbm_strong_errors(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    ks: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    scheme: Literal["euler", "milstein"] = "euler",
    n_paths: int = 20_000,
    seed: int = DEFAULT_SEED,
) -> list[ConvergencePoint]:
    """强误差 e_strong = E|S_T - S_T^{Δt}|，同一布朗路径。"""
    rng = make_rng(seed)
    points: list[ConvergencePoint] = []
    for k in ks:
        n_steps = 2**k
        dt = T / n_steps
        dW = brownian_increments(n_paths, n_steps, dt, rng)
        exact = gbm_terminal_from_increments(S0, mu, sigma, dt, dW, "exact")
        disc = gbm_terminal_from_increments(S0, mu, sigma, dt, dW, scheme)
        abs_err = np.abs(exact - disc)
        mu_e = float(np.mean(abs_err))
        se = standard_error(abs_err)
        points.append(ConvergencePoint(dt=dt, n_steps=n_steps, mean_error=mu_e, se=se))
    return points


def gbm_weak_errors(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    K: float,
    ks: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    scheme: Literal["euler", "milstein"] = "euler",
    n_paths: int = 40_000,
    n_batches: int = 1,
    seed: int = DEFAULT_SEED,
) -> list[ConvergencePoint]:
    """
    弱误差 |E[(S_T-K)^+] - E[(S_T^{Δt}-K)^+]|。

    在细网格布朗增量上合并得到粗网格增量，使不同 Δt 共享同一底层路径，降低 MC 噪声。
    """
    if n_batches < 1:
        raise ValueError("n_batches 至少为 1")

    def payoff(s: np.ndarray) -> np.ndarray:
        return np.maximum(s - K, 0.0)

    k_max = max(ks)
    n_fine = 2**k_max
    dt_fine = T / n_fine

    points: list[ConvergencePoint] = []
    for k in ks:
        n_steps = 2**k
        if n_fine % n_steps != 0:
            raise ValueError("ks 须满足 2^k_max 能被 2^k 整除")
        ratio = n_fine // n_steps
        dt = T / n_steps

        batch_errs: list[float] = []
        for b in range(n_batches):
            rng = make_rng(seed + 1009 * k + b)
            dW_fine = brownian_increments(n_paths, n_fine, dt_fine, rng)
            exact = gbm_terminal_from_increments(
                S0, mu, sigma, dt_fine, dW_fine, "exact"
            )
            dW_coarse = dW_fine.reshape(n_paths, n_steps, ratio).sum(axis=2)
            disc = gbm_terminal_from_increments(
                S0, mu, sigma, dt, dW_coarse, scheme
            )
            err_path = abs(float(np.mean(payoff(exact))) - float(np.mean(payoff(disc))))
            batch_errs.append(err_path)

        batch_arr = np.asarray(batch_errs, dtype=float)
        points.append(
            ConvergencePoint(
                dt=dt,
                n_steps=n_steps,
                mean_error=float(np.mean(batch_arr)),
                se=standard_error(batch_arr) if batch_arr.size > 1 else float("nan"),
            )
        )
    return points


def convergence_slopes(
    points: list[ConvergencePoint],
) -> float:
    """对 log-log 上的 (dt, mean_error) 拟合斜率。"""
    dt_arr = np.array([p.dt for p in points], dtype=float)
    err_arr = np.array([p.mean_error for p in points], dtype=float)
    return fit_loglog_slope(dt_arr, err_arr)


@dataclass(frozen=True)
class HestonParams:
    """Heston 模型参数。"""

    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    mu: float = 0.05

    def feller_ratio(self) -> float:
        """2 kappa theta / xi^2，小于 1 时 Feller 条件不满足。"""
        return 2.0 * self.kappa * self.theta / (self.xi * self.xi)


def _heston_correlated_increments(
    n_paths: int,
    n_steps: int,
    dt: float,
    rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """生成相关的 (dW_S, dW_v)，增量方差均为 dt。"""
    sqrt_dt = math.sqrt(dt)
    z1 = rng.standard_normal((n_paths, n_steps))
    z2 = rng.standard_normal((n_paths, n_steps))
    dW_v = sqrt_dt * z1
    dW_s = sqrt_dt * (rho * z1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z2)
    return dW_s, dW_v


def simulate_heston_terminal(
    S0: float,
    T: float,
    n_steps: int,
    params: HestonParams,
    scheme: HestonScheme,
    n_paths: int,
    seed: int | None = DEFAULT_SEED,
    r: float = 0.05,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    模拟 Heston 到期标的价，并返回诊断信息。

    风险中性下 mu 通常取 r；此处 S 的漂移用 params.mu（默认可设为 r）。
    """
    if n_steps < 1:
        raise ValueError("n_steps 至少为 1")
    dt = T / n_steps
    rng = make_rng(seed)
    dW_s, dW_v = _heston_correlated_increments(n_paths, n_steps, dt, params.rho, rng)

    s = np.full(n_paths, S0, dtype=float)
    v = np.full(n_paths, params.v0, dtype=float)

    neg_steps = 0
    total_steps = n_paths * n_steps

    for j in range(n_steps):
        v_before = v.copy()
        if scheme == "euler":
            # 朴素 Euler：不做截断；sqrt 仅在 v>=0 时有定义
            sqrt_v = np.sqrt(np.maximum(v, 0.0))
            v = v + params.kappa * (params.theta - v) * dt + params.xi * sqrt_v * dW_v[:, j]
            neg_steps += int(np.sum(v < 0))
            sqrt_v_s = np.sqrt(np.maximum(v_before, 0.0))
        elif scheme == "reflection":
            v = v + params.kappa * (params.theta - v) * dt + params.xi * np.sqrt(np.abs(v)) * dW_v[:, j]
            v = np.abs(v)
            neg_steps += int(np.sum(v_before < 0))
            sqrt_v_s = np.sqrt(np.maximum(v_before, 0.0))
        elif scheme == "full_truncation":
            v_pos = np.maximum(v, 0.0)
            v = v + params.kappa * (params.theta - v_pos) * dt + params.xi * np.sqrt(v_pos) * dW_v[:, j]
            neg_steps += int(np.sum(v < 0))
            sqrt_v_s = np.sqrt(v_pos)
        else:
            raise ValueError(f"未知 Heston scheme: {scheme}")

        s = s + params.mu * s * dt + sqrt_v_s * s * dW_s[:, j]

    diag = {
        "negative_variance_fraction": neg_steps / total_steps if total_steps else 0.0,
        "negative_terminal_v_fraction": float(np.mean(v < 0)),
    }
    return s, diag


def simulate_heston_paths(
    S0: float,
    T: float,
    n_steps: int,
    params: HestonParams,
    scheme: HestonScheme = "full_truncation",
    n_paths: int = 10_000,
    seed: int | None = DEFAULT_SEED,
    r: float = 0.05,
) -> np.ndarray:
    """Heston 标的价路径，形状 (n_paths, n_steps + 1)。"""
    if n_steps < 1:
        raise ValueError("n_steps 至少为 1")
    dt = T / n_steps
    rng = make_rng(seed)
    dW_s, dW_v = _heston_correlated_increments(n_paths, n_steps, dt, params.rho, rng)

    p = HestonParams(
        kappa=params.kappa,
        theta=params.theta,
        xi=params.xi,
        rho=params.rho,
        v0=params.v0,
        mu=r,
    )

    s = np.full(n_paths, S0, dtype=float)
    v = np.full(n_paths, params.v0, dtype=float)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0

    for j in range(n_steps):
        if scheme == "euler":
            sqrt_v = np.sqrt(np.maximum(v, 0.0))
            v = v + p.kappa * (p.theta - v) * dt + p.xi * sqrt_v * dW_v[:, j]
            sqrt_v_s = np.sqrt(np.maximum(v, 0.0))
        elif scheme == "reflection":
            v = v + p.kappa * (p.theta - v) * dt + p.xi * np.sqrt(np.abs(v)) * dW_v[:, j]
            v = np.abs(v)
            sqrt_v_s = np.sqrt(np.maximum(v, 0.0))
        elif scheme == "full_truncation":
            v_pos = np.maximum(v, 0.0)
            v = v + p.kappa * (p.theta - v_pos) * dt + p.xi * np.sqrt(v_pos) * dW_v[:, j]
            sqrt_v_s = np.sqrt(v_pos)
        else:
            raise ValueError(f"未知 Heston scheme: {scheme}")

        s = s + p.mu * s * dt + sqrt_v_s * s * dW_s[:, j]
        paths[:, j + 1] = s

    return paths


def heston_call_mc(
    S0: float,
    K: float,
    T: float,
    params: HestonParams,
    scheme: HestonScheme,
    n_steps: int,
    n_paths: int,
    seed: int | None = DEFAULT_SEED,
    r: float = 0.05,
) -> tuple[float, float]:
    """Heston 下欧式看涨 MC 价格 (mean, se)。"""
    # 定价在风险中性测度下，漂移取 r
    p = HestonParams(
        kappa=params.kappa,
        theta=params.theta,
        xi=params.xi,
        rho=params.rho,
        v0=params.v0,
        mu=r,
    )
    s_t, _ = simulate_heston_terminal(S0, T, n_steps, p, scheme, n_paths, seed=seed, r=r)
    payoffs = np.exp(-r * T) * np.maximum(s_t - K, 0.0)
    return float(np.mean(payoffs)), standard_error(payoffs)


def heston_char_func(
    u: complex,
    S0: float,
    T: float,
    params: HestonParams,
    r: float = 0.05,
) -> complex:
    """
    风险中性测度下 ln S_T 的特征函数 E[exp(i u ln S_T)]。

    采用 Albrecher 等 (2007) 的 "little trap" 形式：
    g 取 (b - d)/(b + d) 且指数项为 exp(-dT)，避免复对数在长期限下跨越分支切割。
    """
    kappa, theta, xi, rho, v0 = (
        params.kappa,
        params.theta,
        params.xi,
        params.rho,
        params.v0,
    )
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    exp_dt = np.exp(-d * T)
    C = (kappa * theta / (xi * xi)) * (
        (b - d) * T - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = ((b - d) / (xi * xi)) * (1.0 - exp_dt) / (1.0 - g * exp_dt)
    return np.exp(iu * (math.log(S0) + r * T) + C + D * v0)


def heston_call_fourier(
    S0: float,
    K: float,
    T: float,
    params: HestonParams,
    r: float = 0.05,
) -> float:
    """
    Heston (1993) 半解析欧式看涨价：C = S0 P1 - K e^{-rT} P2。

    P2 = 1/2 + (1/pi) int_0^inf Re[e^{-iu lnK} phi(u) / (iu)] du
    P1 = 1/2 + (1/pi) int_0^inf Re[e^{-iu lnK} phi(u - i) / (iu phi(-i))] du，
    其中 phi(-i) = E[S_T] = S0 e^{rT}。
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 与 K 必须为正")
    if T <= 0:
        raise ValueError("T 必须为正")
    if params.xi <= 0:
        raise ValueError("xi 必须为正（xi=0 退化为 BS，请直接用 BS 公式）")

    log_k = math.log(K)
    forward = S0 * math.exp(r * T)

    def integrand_p1(u: float) -> float:
        phi = heston_char_func(u - 1j, S0, T, params, r)
        val = np.exp(-1j * u * log_k) * phi / (1j * u * forward)
        return float(val.real)

    def integrand_p2(u: float) -> float:
        phi = heston_char_func(u, S0, T, params, r)
        val = np.exp(-1j * u * log_k) * phi / (1j * u)
        return float(val.real)

    i1, _ = quad(integrand_p1, 0.0, np.inf, limit=500, epsabs=1e-12, epsrel=1e-10)
    i2, _ = quad(integrand_p2, 0.0, np.inf, limit=500, epsabs=1e-12, epsrel=1e-10)
    p1 = 0.5 + i1 / math.pi
    p2 = 0.5 + i2 / math.pi
    return S0 * p1 - K * math.exp(-r * T) * p2


def heston_negative_variance_by_dt(
    S0: float,
    T: float,
    params: HestonParams,
    ks: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
    scheme: HestonScheme = "euler",
    n_paths: int = 5_000,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (dt 数组, 负方差步占比数组)。"""
    dts: list[float] = []
    fracs: list[float] = []
    for k in ks:
        n_steps = 2**k
        _, diag = simulate_heston_terminal(
            S0, T, n_steps, params, scheme, n_paths, seed=seed + k
        )
        dts.append(T / n_steps)
        fracs.append(diag["negative_variance_fraction"])
    return np.array(dts, dtype=float), np.array(fracs, dtype=float)


def default_misspecified_heston_params() -> HestonParams:
    """文档示例：Feller 条件不满足的一组参数。"""
    return HestonParams(kappa=1.0, theta=0.04, xi=0.8, rho=-0.7, v0=0.04, mu=0.05)


def plot_strong_weak_convergence(
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    K: float = 100.0,
    ks: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    n_paths_strong: int = 25_000,
    n_paths_weak: int = 60_000,
    seed: int = DEFAULT_SEED,
    out_path: str = "figures/fig_strong_weak_convergence.png",
) -> dict[str, float]:
    """绘制 GBM 强/弱收敛 log-log 图，返回各曲线拟合斜率。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    slopes: dict[str, float] = {}

    for ax, err_fn, title in (
        (axes[0], gbm_strong_errors, "Strong convergence"),
        (axes[1], gbm_weak_errors, "Weak convergence (call payoff)"),
    ):
        for scheme, color, marker in (
            ("euler", "C0", "o"),
            ("milstein", "C1", "s"),
        ):
            if err_fn is gbm_strong_errors:
                pts = gbm_strong_errors(
                    S0, mu, sigma, T, ks, scheme, n_paths_strong, seed=seed
                )
            else:
                pts = gbm_weak_errors(
                    S0, mu, sigma, T, K, ks, scheme, n_paths_weak, seed=seed
                )
            dt_arr = np.array([p.dt for p in pts])
            err_arr = np.array([p.mean_error for p in pts])
            slope = convergence_slopes(pts)
            slopes[f"{title}_{scheme}"] = slope
            ax.loglog(
                dt_arr,
                err_arr,
                f"{marker}-",
                color=color,
                label=f"{scheme} (slope={slope:.2f})",
            )
        ax.set_xlabel("Δt")
        ax.set_ylabel("Error")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("GBM Euler / Milstein discretization convergence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return slopes


def plot_heston_negative_variance(
    S0: float = 100.0,
    T: float = 1.0,
    params: HestonParams | None = None,
    ks: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
    n_paths: int = 8_000,
    seed: int = DEFAULT_SEED,
    out_path: str = "figures/fig_heston_negative_variance.png",
) -> None:
    """负方差比例随 Δt 变化，并对比各修正格式的定价偏差。"""
    if params is None:
        params = default_misspecified_heston_params()

    K = 100.0
    r = 0.05
    # 参考值：Heston 半解析傅里叶价格（与离散格式无关的独立基准）
    ref_price = heston_call_fourier(S0, K, T, params, r=r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dt_arr, frac = heston_negative_variance_by_dt(
        S0, T, params, ks, "euler", n_paths, seed=seed
    )
    axes[0].semilogx(dt_arr, frac, "o-", color="C3")
    axes[0].set_xlabel("Δt")
    axes[0].set_ylabel("Fraction of steps with v < 0")
    axes[0].set_title("Naive Euler: negative variance (Feller violated)")
    axes[0].grid(True, which="both", alpha=0.3)

    schemes: tuple[HestonScheme, ...] = ("euler", "reflection", "full_truncation")
    labels = ("euler", "reflection", "full trunc.")
    for scheme, label in zip(schemes, labels):
        biases: list[float] = []
        for k in ks:
            n_steps = 2**k
            price, _ = heston_call_mc(
                S0, K, T, params, scheme, n_steps, n_paths, seed=seed + k, r=r
            )
            biases.append(abs(price - ref_price))
        axes[1].loglog(dt_arr, biases, "o-", label=label)

    axes[1].set_xlabel("Δt")
    axes[1].set_ylabel("|MC price - Fourier reference|")
    axes[1].set_title(f"European call pricing bias vs Δt (Fourier ref={ref_price:.4f})")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    feller = params.feller_ratio()
    fig.suptitle(f"Heston (2κθ/ξ²={feller:.2f}<1): negative variance & pricing")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

