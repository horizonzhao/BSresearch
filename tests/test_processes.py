"""离散化格式（GBM / Heston）验收测试。"""

from pathlib import Path

import numpy as np
import pytest

from processes import (
    brownian_increments,
    convergence_slopes,
    default_misspecified_heston_params,
    gbm_strong_errors,
    gbm_terminal_from_increments,
    gbm_weak_errors,
    heston_call_mc,
    heston_negative_variance_by_dt,
    plot_heston_negative_variance,
    plot_strong_weak_convergence,
    simulate_gbm_terminal,
)


@pytest.fixture
def gbm_params() -> dict:
    return {"S0": 100.0, "mu": 0.05, "sigma": 0.2, "T": 1.0}


def test_same_brownian_path_for_strong_convergence(gbm_params):
    """强收敛：exact 与离散格式在同一 dW 下应可复现。"""
    p = gbm_params
    n_paths, n_steps = 500, 64
    dt = p["T"] / n_steps
    rng = np.random.default_rng(0)
    dW = brownian_increments(n_paths, n_steps, dt, rng)
    exact = gbm_terminal_from_increments(p["S0"], p["mu"], p["sigma"], dt, dW, "exact")
    euler = gbm_terminal_from_increments(p["S0"], p["mu"], p["sigma"], dt, dW, "euler")
    assert exact.shape == euler.shape
    assert np.mean(np.abs(exact - euler)) > 0


def test_milstein_closer_than_euler_for_strong_error(gbm_params):
    p = gbm_params
    n_paths, n_steps = 8_000, 32
    dt = p["T"] / n_steps
    rng = np.random.default_rng(1)
    dW = brownian_increments(n_paths, n_steps, dt, rng)
    exact = gbm_terminal_from_increments(p["S0"], p["mu"], p["sigma"], dt, dW, "exact")
    eul = gbm_terminal_from_increments(p["S0"], p["mu"], p["sigma"], dt, dW, "euler")
    mil = gbm_terminal_from_increments(p["S0"], p["mu"], p["sigma"], dt, dW, "milstein")
    err_e = float(np.mean(np.abs(exact - eul)))
    err_m = float(np.mean(np.abs(exact - mil)))
    assert err_m < err_e


def test_gbm_strong_convergence_slopes(gbm_params):
    p = gbm_params
    ks = (3, 4, 5, 6, 7)
    e_pts = gbm_strong_errors(**p, ks=ks, scheme="euler", n_paths=12_000, seed=10)
    m_pts = gbm_strong_errors(**p, ks=ks, scheme="milstein", n_paths=12_000, seed=10)
    s_e = convergence_slopes(e_pts)
    s_m = convergence_slopes(m_pts)
    assert s_e == pytest.approx(0.5, abs=0.25)
    assert s_m == pytest.approx(1.0, abs=0.35)


def test_gbm_weak_convergence_slopes(gbm_params):
    p = gbm_params
    ks = (2, 3, 4, 5, 6, 7, 8)
    K = 100.0
    e_pts = gbm_weak_errors(**p, K=K, ks=ks, scheme="euler", n_paths=30_000, seed=11)
    m_pts = gbm_weak_errors(**p, K=K, ks=ks, scheme="milstein", n_paths=30_000, seed=11)
    s_e = convergence_slopes(e_pts)
    s_m = convergence_slopes(m_pts)
    assert s_e == pytest.approx(1.0, abs=0.45)
    assert s_m == pytest.approx(1.0, abs=0.45)


def test_simulate_gbm_reproducible(gbm_params):
    p = gbm_params
    a = simulate_gbm_terminal(p["S0"], p["mu"], p["sigma"], p["T"], 16, "euler", 1000, seed=3)
    b = simulate_gbm_terminal(p["S0"], p["mu"], p["sigma"], p["T"], 16, "euler", 1000, seed=3)
    np.testing.assert_allclose(a, b)


def test_heston_feller_violated():
    params = default_misspecified_heston_params()
    assert params.feller_ratio() < 1.0


def test_heston_negative_variance_decreases_with_dt():
    params = default_misspecified_heston_params()
    dt_arr, frac = heston_negative_variance_by_dt(
        100.0, 1.0, params, ks=(4, 5, 6, 7, 8), n_paths=3_000, seed=5
    )
    assert np.all(frac >= 0)
    assert frac[0] >= frac[-1] * 0.5


def test_heston_full_truncation_pricing_stable():
    params = default_misspecified_heston_params()
    p_fine, _ = heston_call_mc(100.0, 100.0, 1.0, params, "full_truncation", 512, 6_000, seed=1)
    p_coarse, _ = heston_call_mc(100.0, 100.0, 1.0, params, "full_truncation", 16, 6_000, seed=1)
    assert p_fine > 0
    assert abs(p_fine - p_coarse) < 5.0


def test_generate_step4_figures(gbm_params):
    root = Path(__file__).resolve().parents[1]
    fig1 = root / "figures" / "fig_strong_weak_convergence.png"
    fig2 = root / "figures" / "fig_heston_negative_variance.png"
    ks = (3, 4, 5, 6, 7)
    slopes = plot_strong_weak_convergence(
        **gbm_params,
        K=100.0,
        ks=ks,
        n_paths_strong=8_000,
        n_paths_weak=25_000,
        seed=42,
        out_path=str(fig1),
    )
    plot_heston_negative_variance(
        ks=(4, 5, 6, 7),
        n_paths=4_000,
        seed=42,
        out_path=str(fig2),
    )
    assert fig1.is_file()
    assert fig2.is_file()
    assert slopes["Strong convergence_euler"] == pytest.approx(0.5, abs=0.3)
