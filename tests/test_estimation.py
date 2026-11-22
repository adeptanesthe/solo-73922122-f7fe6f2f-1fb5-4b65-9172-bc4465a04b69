"""约束最小二乘求解器测试：与 scipy 式枚举/投影参考实现交叉验证。"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.estimation import CATEGORIES, _solve_simplex, estimate


def _reference_simplex_solve(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """暴力参考解：在单纯形网格 + 边界组合上搜索，用于校验求解器。

    同时枚举所有活动集顶点及内部细网格点，取 SSE 最小者。
    """
    n = A.shape[1]
    best_x, best_sse = None, np.inf

    # 顶点（n 个）与边内点
    grid = np.linspace(0.0, 1.0, 401)
    if n == 3:
        for a in grid:
            for b_ in grid:
                if a + b_ > 1.0 + 1e-12:
                    break
                x = np.array([a, b_, 1.0 - a - b_])
                sse = float(np.sum((A @ x - b) ** 2))
                if sse < best_sse:
                    best_sse, best_x = sse, x
    else:
        for support in range(1, n + 1):
            for idxs in itertools.combinations(range(n), support):
                for v in grid:
                    x = np.zeros(n)
                    if support == 1:
                        x[idxs[0]] = 1.0
                        sse = float(np.sum((A @ x - b) ** 2))
                        if sse < best_sse:
                            best_sse, best_x = sse, x
                        break
                    x[idxs[0]] = v
                    x[idxs[1]] = 1.0 - v
                    sse = float(np.sum((A @ x - b) ** 2))
                    if sse < best_sse:
                        best_sse, best_x = sse, x
    return best_x, best_sse


@pytest.mark.parametrize("seed", range(12))
def test_solver_matches_grid_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    A = rng.random((3, 3))
    b = rng.random(3)
    x, sse = _solve_simplex(A, b)
    # 可行性
    assert x.shape == (3,)
    assert np.all(x >= -1e-9)
    assert abs(x.sum() - 1.0) < 1e-8
    ref_x, ref_sse = _reference_simplex_solve(A, b)
    assert sse <= ref_sse + 1e-6
    np.testing.assert_allclose(A @ x - b, A @ ref_x - b, atol=2e-3)


def test_solver_perfect_fit_recovers_pi() -> None:
    # 满秩 P，q = P^T pi*，求解器应精确还原 pi*（含边界解）。
    P = np.array(
        [[0.8, 0.13, 0.07], [0.12, 0.78, 0.1], [0.05, 0.15, 0.8]]
    )
    for pi_star in [
        np.array([0.45, 0.35, 0.2]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.6, 0.4]),
        np.array([0.0, 0.0, 1.0]),
    ]:
        q = P.T @ pi_star
        x, sse = _solve_simplex(P.T, q)
        np.testing.assert_allclose(x, pi_star, atol=1e-8)
        assert sse < 1e-18


def test_solver_unconstrained_optimum_outside_simplex_gets_clipped() -> None:
    # 构造无约束最小二乘解出现负分量的场景，结果必须仍在单纯形内。
    A = np.eye(3)
    b = np.array([0.7, 0.9, -0.4])
    x, sse = _solve_simplex(A, b)
    assert np.all(x >= -1e-12)
    assert abs(x.sum() - 1.0) < 1e-10
    # 负目标分量对应坐标应被压到 0
    assert x[2] == pytest.approx(0.0, abs=1e-10)
    # 与解析边界解比较
    expected = np.array([0.4, 0.6, 0.0])
    np.testing.assert_allclose(x[:2], expected[:2], atol=1e-8)


def _machines(initial, inspected):
    return [
        {"id": f"M{i}", "initial": p, "inspected": y}
        for i, (p, y) in enumerate(zip(initial, inspected))
    ]


def test_estimate_gates_and_recovery() -> None:
    rng = np.random.default_rng(7)
    P = np.array(
        [[0.8, 0.13, 0.07], [0.12, 0.78, 0.1], [0.05, 0.15, 0.8]]
    )
    pi_star = np.array([0.45, 0.35, 0.2])

    # 1) 一台未检、零实检：缺样本，撤下估数
    res = estimate(_machines(["power"], [None]))
    assert not res.available
    assert "缺少实检类别样本" in res.reason

    # 2) 三类都有实检但初判完全一致 -> 混淆矩阵秩 1，撤下
    rows = []
    for cat in CATEGORIES:
        rows += _machines(["power", "power"], [cat, cat])
    rows += _machines(["drive"], [None])
    res = estimate(rows)
    assert not res.available
    assert "不满秩" in res.reason

    # 3) 满秩样本 + 大量未检件按真实比例与 P 生成，π 应接近 pi_star
    initial, inspected = [], []
    for true_idx in range(3):
        for _ in range(200):
            inspected.append(CATEGORIES[true_idx])
            initial.append(
                CATEGORIES[rng.choice(3, p=P[true_idx])]
            )
    for true_idx in range(3):
        for _ in range(int(pi_star[true_idx] * 300)):
            inspected.append(None)
            initial.append(
                CATEGORIES[rng.choice(3, p=P[true_idx])]
            )
    res = estimate(_machines(initial, inspected))
    assert res.available
    np.testing.assert_allclose(res.pi, pi_star, atol=0.06)
    # 折算台数 + 实检数 = 全批估计；和等于总台数
    assert sum(res.estimated_total.values()) == pytest.approx(
        res.n_total, abs=1e-8
    )
    for c in CATEGORIES:
        assert res.estimated_total[c] == pytest.approx(
            res.inspected_counts[c] + res.estimated_uninspected[c], abs=1e-10
        )


def test_estimate_all_inspected_returns_actuals() -> None:
    rows = _machines(
        ["power", "drive", "control", "power"],
        ["power", "power", "control", "control"],
    )
    res = estimate(rows)
    assert res.available and res.n_uninspected == 0
    assert res.pi is None and res.P is None
    assert res.actual_total == {"power": 2, "drive": 0, "control": 2}
    assert res.estimated_total == {"power": 2.0, "drive": 0.0, "control": 2.0}
