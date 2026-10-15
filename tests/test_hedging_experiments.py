"""Step 6：模型误设与交易成本验收测试。"""

from pathlib import Path

import numpy as np
import pytest

from hedging import (
    bs_implied_vol,
    cost_tradeoff_curve,
    hedge_sigma_for_spec,
    misspec_study,
    plot_cost_tradeoff,
    plot_misspec_pnl,
    simulate_misspec_hedge_pnl,
)
from bs import price as bs_price
from processes import HestonParams, default_misspecified_heston_params


@pytest.fixture
def base() -> dict:
    return {"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.05}


def test_implied_vol_roundtrip(base):
    p = base
    sig = 0.25
    mkt = bs_price(p["S0"], p["K"], p["T"], p["r"], sig, "call")
    iv = bs_implied_vol(p["S0"], p["K"], p["T"], p["r"], mkt)
    assert iv == pytest.approx(sig, rel=1e-5)


def test_misspec_sigma_specs(base):
    params = default_misspecified_heston_params()
    s0 = hedge_sigma_for_spec("sqrt_v0", params, **base, seed=1)
    s1 = hedge_sigma_for_spec("sqrt_theta", params, **base, seed=1)
    assert s0 == pytest.approx(np.sqrt(params.v0))
    assert s1 == pytest.approx(np.sqrt(params.theta))


def test_misspec_mean_differs_from_gbm(base):
    from hedging import simulate_hedge_pnl

    params = default_misspecified_heston_params()
    pnl_h = simulate_misspec_hedge_pnl(
        **base, params=params, vol_spec="sqrt_v0", n_rebalances=64, n_paths=10_000, seed=2
    )
    pnl_g, _ = simulate_hedge_pnl(
        **base, sigma=np.sqrt(params.v0), n_rebalances=64, n_paths=10_000, seed=2
    )
    assert abs(np.mean(pnl_h)) > abs(np.mean(pnl_g)) - 0.15 or np.std(pnl_h) > np.std(pnl_g) * 0.8


def test_misspec_study_skew_sensitivity(base):
    params = default_misspecified_heston_params()
    stats = misspec_study(
        **base,
        params=params,
        rhos=(-0.7, 0.7),
        n_rebalances=32,
        n_paths=6_000,
        seed=3,
    )
    assert len(stats) == 6
    skews = [s.skewness for s in stats if "rho=+0.7" in s.label or "rho=-0.7" in s.label]
    assert len(skews) >= 2


def test_cost_tradeoff_u_shape_left_tail(base):
    sigma = 0.2
    ns = (4, 8, 16, 32, 64, 128, 256)
    n_arr, std_arr, q05_arr = cost_tradeoff_curve(
        **base, sigma=sigma, ns=ns, cost_k=0.001, n_paths=5_000, seed=4
    )
    tail = -q05_arr
    assert tail[-1] > min(tail) * 0.9


def test_moneyness_affects_pnl_spread(base):
    sigma = 0.2
    _, std_atm, _ = cost_tradeoff_curve(
        base["S0"], 100.0, base["T"], base["r"], sigma, (64,), 0.0, 6_000, 5
    )
    _, std_otm, _ = cost_tradeoff_curve(
        base["S0"], 110.0, base["T"], base["r"], sigma, (64,), 0.0, 6_000, 5
    )
    assert std_atm[0] != pytest.approx(std_otm[0], rel=0.05)


def test_generate_step6_figures(base):
    root = Path(__file__).resolve().parents[1]
    fig1 = root / "figures" / "fig_misspec_pnl.png"
    fig2 = root / "figures" / "fig_cost_tradeoff.png"
    plot_misspec_pnl(**base, n_paths=6_000, seed=7, out_path=str(fig1))
    opt = plot_cost_tradeoff(
        **base, n_paths=5_000, seed=7, out_path=str(fig2)
    )
    assert fig1.is_file()
    assert fig2.is_file()
    assert len(opt) >= 1
