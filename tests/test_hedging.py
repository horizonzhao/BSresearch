"""离散 Delta 对冲验收测试。"""

from pathlib import Path

import numpy as np
import pytest

from hedging import (
    DEFAULT_REBALANCE_GRID,
    delta_call_vec,
    fit_std_vs_n_slope,
    gamma_ito_step_error,
    hedge_stats_vs_n,
    plot_hedge_pnl_dist,
    plot_hedge_std_vs_n,
    simulate_hedge_pnl,
    summarize_pnl,
)
from bs import delta as bs_delta


@pytest.fixture
def base_params() -> dict:
    return {"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}


def test_delta_vec_matches_bs(base_params):
    p = base_params
    s = np.array([80.0, 100.0, 120.0])
    for t in (1.0, 0.25):
        v = delta_call_vec(s, p["K"], t, p["r"], p["sigma"])
        for si, vi in zip(s, v):
            assert vi == pytest.approx(bs_delta(si, p["K"], t, p["r"], p["sigma"], "call"))


def test_pnl_mean_near_zero_risk_neutral(base_params):
    p = base_params
    pnl, _ = simulate_hedge_pnl(**p, n_rebalances=256, n_paths=15_000, seed=1)
    st = summarize_pnl(pnl, 256)
    assert abs(st.mean) < 3 * st.se_mean + 0.05


def test_pnl_std_decreases_with_n(base_params):
    p = base_params
    st4 = summarize_pnl(
        simulate_hedge_pnl(**p, n_rebalances=4, n_paths=12_000, seed=2)[0], 4
    )
    st256 = summarize_pnl(
        simulate_hedge_pnl(**p, n_rebalances=256, n_paths=12_000, seed=2)[0], 256
    )
    assert st256.std < st4.std


def test_std_vs_n_loglog_slope(base_params):
    p = base_params
    ns = (8, 16, 32, 64, 128, 256)
    stats = hedge_stats_vs_n(**p, ns=ns, n_paths=8_000, seed=10)
    n_arr = np.array([s.n_rebalances for s in stats], dtype=float)
    std_arr = np.array([s.std for s in stats], dtype=float)
    slope = fit_std_vs_n_slope(n_arr, std_arr)
    assert slope == pytest.approx(-0.5, abs=0.15)


def test_physical_drift_mean_can_differ_from_zero(base_params):
    p = base_params
    pnl_rn, _ = simulate_hedge_pnl(
        **p, n_rebalances=64, n_paths=20_000, drift_mode="risk_neutral", seed=3
    )
    pnl_ph, _ = simulate_hedge_pnl(
        **p,
        n_rebalances=64,
        n_paths=20_000,
        drift_mode="physical",
        mu=0.12,
        seed=3,
    )
    st_rn = summarize_pnl(pnl_rn, 64)
    st_ph = summarize_pnl(pnl_ph, 64)
    assert abs(st_rn.mean) < abs(st_ph.mean) + 0.02


def test_gamma_step_larger_near_expiry(base_params):
    """临近到期的逐步 Gamma / 伊藤项贡献应更大（ATM）。"""
    p = base_params
    step_abs, gamma_mean = gamma_ito_step_error(
        **p, n_rebalances=32, n_paths=10_000, seed=4
    )
    assert gamma_mean[-1] > gamma_mean[0]
    assert step_abs[-4:].mean() > step_abs[:4].mean() * 0.8


def test_invalid_n_rebalances(base_params):
    with pytest.raises(ValueError):
        simulate_hedge_pnl(**base_params, n_rebalances=0, n_paths=100)


def test_generate_hedge_figures(base_params):
    root = Path(__file__).resolve().parents[1]
    fig1 = root / "figures" / "fig_hedge_pnl_dist.png"
    fig2 = root / "figures" / "fig_hedge_std_vs_N.png"
    plot_hedge_pnl_dist(**base_params, n_paths=8_000, seed=5, out_path=str(fig1))
    slope = plot_hedge_std_vs_n(
        **base_params,
        ns=(4, 8, 16, 32, 64, 128),
        n_paths=8_000,
        seed=5,
        out_path=str(fig2),
    )
    assert fig1.is_file()
    assert fig2.is_file()
    assert slope == pytest.approx(-0.5, abs=0.2)
