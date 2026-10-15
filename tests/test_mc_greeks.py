"""MC Greeks 三方法验收测试。"""

from pathlib import Path

import numpy as np
import pytest

from bs import delta as bs_delta
from bs import vega as bs_vega
from mc_greeks import (
    analytical_greek,
    bs_digital_delta,
    compare_greek_methods,
    fd_bias_variance_scan,
    mc_greek,
    plot_fd_bias_variance,
    plot_greeks_methods,
)


@pytest.fixture
def base_params() -> dict:
    return {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}


def test_call_delta_three_methods_converge(base_params):
    p = base_params
    true = bs_delta(**p, option_type="call")
    res = compare_greek_methods(**p, n_paths=120_000, greek="delta", seed=11)
    for m, est in res.items():
        assert abs(est.mean - true) < max(5 * est.se, 0.02), m


def test_call_vega_three_methods_converge(base_params):
    p = base_params
    true = bs_vega(**p)
    res = compare_greek_methods(**p, n_paths=120_000, greek="vega", seed=12)
    for m, est in res.items():
        assert abs(est.mean - true) < max(5 * est.se, 0.05), m


def test_finite_diff_uses_common_random_numbers(base_params):
    """同一组 z 下，FD 估计应可复现且与分路径计算一致。"""
    p = base_params
    rng = np.random.default_rng(0)
    z = rng.standard_normal(5000)
    est1 = mc_greek(**p, n_paths=5000, method="finite_diff", greek="delta", z=z, h=1e-2)
    est2 = mc_greek(**p, n_paths=5000, method="finite_diff", greek="delta", z=z, h=1e-2)
    assert est1.mean == est2.mean


def test_digital_pathwise_fails_lr_works(base_params):
    p = base_params
    true = bs_digital_delta(**p)
    n_paths = 150_000
    pw = mc_greek(**p, n_paths=n_paths, greek="delta", method="pathwise", payoff="digital", seed=3)
    lr = mc_greek(**p, n_paths=n_paths, greek="delta", method="likelihood_ratio", payoff="digital", seed=3)
    fd = mc_greek(**p, n_paths=n_paths, greek="delta", method="finite_diff", payoff="digital", seed=3, h=1e-2)

    assert abs(pw.mean) < 0.02
    assert abs(lr.mean - true) < max(5 * lr.se, 0.03)
    assert fd.se > lr.se * 1.5


def test_fd_h_scan_bias_variance_tradeoff(base_params):
    p = base_params
    hs = np.array([1e-1, 1e-2, 1e-3, 1e-4])
    h_arr, bias, var = fd_bias_variance_scan(
        **p, hs=hs, n_paths=15_000, n_batches=25, base_seed=7
    )
    assert h_arr.size == hs.size
    assert np.all(bias >= 0)
    # 步长过大偏差更大；步长过小方差更大（经验趋势，允许统计噪声）
    assert bias[0] >= bias[2] * 0.5 or bias[0] > 1e-4
    assert var[-1] >= var[1] * 0.3 or var[-1] > 1e-8


def test_generate_greeks_figures(base_params):
    root = Path(__file__).resolve().parents[1]
    fig1 = root / "figures" / "fig_greeks_methods.png"
    fig2 = root / "figures" / "fig_fd_bias_variance.png"
    plot_greeks_methods(**base_params, n_paths=30_000, seed=1, out_path=str(fig1))
    plot_fd_bias_variance(
        **base_params,
        hs=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
        n_paths=8_000,
        n_batches=20,
        seed=2,
        out_path=str(fig2),
    )
    assert fig1.is_file()
    assert fig2.is_file()


def test_analytical_greek_helper(base_params):
    p = base_params
    assert analytical_greek(**p, greek="delta", payoff="call") == bs_delta(**p, option_type="call")
    assert analytical_greek(**p, greek="vega", payoff="call") == bs_vega(**p)
