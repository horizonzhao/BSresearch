"""Black-Scholes 解析解与 Greeks 验收测试。"""

import sys
from pathlib import Path

# 支持在 IDE 中直接运行本文件（不经过 pytest 时 conftest 不会加载）
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import math

import numpy as np
import pytest

from bs import (
    delta,
    gamma,
    greeks_finite_diff,
    price,
    put_call_parity_residual,
    rho,
    theta,
    vega,
)


@pytest.fixture
def base_params() -> dict:
    return {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}


def test_put_call_parity(base_params):
    res = put_call_parity_residual(**base_params)
    assert abs(res) < 1e-12


def test_put_call_parity_many_cases():
    rng = np.random.default_rng(0)
    for _ in range(50):
        S = float(rng.uniform(50, 150))
        K = float(rng.uniform(50, 150))
        T = float(rng.uniform(0.01, 3.0))
        r = float(rng.uniform(-0.01, 0.1))
        sigma = float(rng.uniform(0.05, 0.8))
        res = put_call_parity_residual(S, K, T, r, sigma)
        assert abs(res) < 1e-10


def test_greeks_vs_finite_diff(base_params):
    p = base_params
    ana = {
        "delta": delta(**p, option_type="call"),
        "gamma": gamma(**p),
        "vega": vega(**p),
        "rho": rho(**p, option_type="call"),
        "theta": theta(**p, option_type="call"),
    }
    fd = greeks_finite_diff(**p, option_type="call", h_s=1e-3, h_sigma=1e-3)
    for key in ("delta", "gamma", "vega", "rho"):
        rel = abs(ana[key] - fd[key]) / max(abs(ana[key]), 1e-8)
        assert rel < 1e-6, f"{key} rel err {rel}"


def test_call_put_delta_relationship(base_params):
    p = base_params
    dc = delta(**p, option_type="call")
    dp = delta(**p, option_type="put")
    assert dc - dp == pytest.approx(1.0, rel=0, abs=1e-12)


def test_boundary_small_T(base_params):
    p = {**base_params, "T": 1e-15}
    c = price(**p, option_type="call")
    assert c == pytest.approx(max(p["S"] - p["K"], 0.0), abs=1e-12)
    assert gamma(**p) == 0.0


def test_boundary_small_sigma(base_params):
    p = {**base_params, "sigma": 0.0}
    forward = p["S"] * math.exp(p["r"] * p["T"])
    expected = math.exp(-p["r"] * p["T"]) * max(forward - p["K"], 0.0)
    c = price(**p, option_type="call")
    assert c == pytest.approx(expected, rel=0, abs=1e-12)
    assert delta(**p, option_type="call") in (0.0, math.exp(-p["r"] * p["T"]))


def test_invalid_inputs():
    with pytest.raises(ValueError):
        price(0, 100, 1, 0.05, 0.2)
    with pytest.raises(ValueError):
        price(100, 100, -1, 0.05, 0.2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
