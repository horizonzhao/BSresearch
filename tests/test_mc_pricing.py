"""蒙特卡洛定价与方差缩减验收测试。"""

from pathlib import Path

import numpy as np
import pytest

from bs import price as bs_price
from mc_pricing import (
    compare_variance_reduction,
    confidence_interval_coverage,
    fit_loglog_slope,
    mc_european_call,
    plot_mc_convergence,
    se_vs_n,
)


@pytest.fixture
def base_params() -> dict:
    return {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}


def test_mc_near_bs_analytical(base_params):
    p = base_params
    true = bs_price(**p, option_type="call")
    est = mc_european_call(**p, n_paths=200_000, method="plain", seed=1)
    assert abs(est.mean - true) < 3 * est.se


def test_variance_reduction_se_order(base_params):
    p = base_params
    res = compare_variance_reduction(**p, n_paths=50_000, seed=42)
    assert res["antithetic"].se < res["plain"].se
    assert res["antithetic_cv"].se < res["antithetic"].se
    assert res["antithetic"].variance_reduction_ratio > 1.0
    assert res["antithetic_cv"].variance_reduction_ratio > res["antithetic"].variance_reduction_ratio


def test_loglog_slope_near_minus_half(base_params):
    p = base_params
    ns = (500, 2_000, 8_000, 32_000)
    n_arr, se_arr = se_vs_n(**p, ns=ns, method="plain", seed=0)
    slope = fit_loglog_slope(n_arr, se_arr)
    assert slope == pytest.approx(-0.5, abs=0.15)


def test_confidence_interval_coverage(base_params):
    p = base_params
    cov = confidence_interval_coverage(
        **p, n_paths=5_000, n_trials=200, method="plain", base_seed=100
    )
    assert 0.88 <= cov <= 1.0


def test_ci_contains_mean_properties(base_params):
    p = base_params
    est = mc_european_call(**p, n_paths=10_000, seed=7)
    assert est.ci_low < est.mean < est.ci_high
    assert est.se > 0


def test_invalid_n_paths(base_params):
    with pytest.raises(ValueError):
        mc_european_call(**base_params, n_paths=0)


def test_generate_convergence_figure(base_params, tmp_path):
    """生成流程文档要求的收敛图（写入项目 figures 目录）。"""
    root = Path(__file__).resolve().parents[1]
    out = root / "figures" / "fig_mc_convergence.png"
    ns = (10**2, 10**3, 10**4, 10**5)
    slope = plot_mc_convergence(**base_params, ns=ns, seed=42, out_path=str(out))
    assert out.is_file()
    assert slope == pytest.approx(-0.5, abs=0.2)
