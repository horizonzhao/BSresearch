"""随机数、统计量与画图辅助（后续步骤扩展）。"""

from __future__ import annotations

import numpy as np

# 全局默认随机种子，保证实验可复现
DEFAULT_SEED = 42


def make_rng(seed: int | None = None) -> np.random.Generator:
    """创建 numpy 随机数生成器。"""
    if seed is None:
        seed = DEFAULT_SEED
    return np.random.default_rng(seed)


def standard_error(samples: np.ndarray) -> float:
    """样本均值的标准误差 SE = s / sqrt(n)。"""
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if n < 2:
        return float("nan")
    return float(np.std(samples, ddof=1) / np.sqrt(n))


def ci95_mean(samples: np.ndarray) -> tuple[float, float, float]:
    """返回 (均值, 95% 置信区间下界, 上界)，正态近似。"""
    samples = np.asarray(samples, dtype=float)
    mu = float(np.mean(samples))
    se = standard_error(samples)
    half = 1.96 * se
    return mu, mu - half, mu + half
