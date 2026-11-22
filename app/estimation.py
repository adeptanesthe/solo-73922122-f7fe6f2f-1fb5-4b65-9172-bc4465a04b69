"""故障占比估算核心逻辑（仅依赖 NumPy）。

业务口径
========
* 初判列 ``initial`` 与实检列 ``inspected`` 均取三类:
  ``power``（电源）、``drive``（驱动）、``control``（控制）。
* 对已拆检机器，按实检类别分组，统计每类实检样本中各初判类别的经验
  概率，组成 3x3 混淆概率矩阵 ``P``，行=实检类别，列=初判类别，每行和为 1。
* 未检件的初判经验分布为向量 ``q``（和为 1）。
* 在单纯形 ``pi >= 0 且 sum(pi) = 1`` 上最小化平方误差
  ``||P^T pi - q||_2^2``，得到未检件的故障占比估计 ``pi``。
* 折算台数 = ``pi * 未检台数``，再加上各类实检数得到全批估计。
* 门控：仍有未检件时，任一类实检样本缺失或混淆矩阵不满秩，则撤下估数，
  提示补检；全部拆完时直接给实数，不再做拟合。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CATEGORIES = ("power", "drive", "control")
CATEGORY_LABELS = {
    "power": "电源",
    "drive": "驱动",
    "control": "控制",
}
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

TOL = 1e-10


@dataclass
class EstimateResult:
    """一次估算的完整结果。

    ``available`` 为 False 时（需要补检），``pi`` 等拟合字段为 None。
    """

    available: bool
    reason: str | None
    n_total: int
    n_inspected: int
    n_uninspected: int
    inspected_counts: dict[str, int]
    uninspected_initial_counts: dict[str, int]
    P: list[list[float]] | None
    q: list[float] | None
    pi: list[float] | None
    sse: float | None
    estimated_uninspected: dict[str, float] | None
    estimated_total: dict[str, float] | None
    actual_total: dict[str, int] | None

    def to_payload(self) -> dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "categories": list(CATEGORIES),
            "labels": dict(CATEGORY_LABELS),
            "n_total": self.n_total,
            "n_inspected": self.n_inspected,
            "n_uninspected": self.n_uninspected,
            "inspected_counts": self.inspected_counts,
            "uninspected_initial_counts": self.uninspected_initial_counts,
            "confusion_matrix": self.P,
            "q": self.q,
            "pi": self.pi,
            "sse": self.sse,
            "estimated_uninspected": self.estimated_uninspected,
            "estimated_total": self.estimated_total,
            "actual_total": self.actual_total,
        }


def _solve_simplex(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """在单纯形 x>=0、sum(x)=1 上最小化 ||A x - b||_2^2。

    三维单纯形的最优解只可能落在：内部（无不等式活动）、某条边
    （一个分量为 0）或顶点（两个分量为 0）。因此枚举：

    1. 内部点：解等式约束 KKT，可行则采纳；
    2. 三条边：退化为标量二次函数，解析求最小点并截断到 [0,1]；
    3. 三个顶点：恒可行，保证算法在矩阵亏秩时仍有候选。

    纯 NumPy 实现，不依赖优化库。返回 ``(x, sse)``。
    """
    n = A.shape[1]
    candidates: list[np.ndarray] = []

    # 1) 内部点：KKT
    #    [2A^T A  1] [x]   [2 A^T b]
    #    [1^T     0] [lam] = [1      ]
    H = 2.0 * A.T @ A
    g = 2.0 * A.T @ b
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = H
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    rhs = np.concatenate([g, [1.0]])
    try:
        sol = np.linalg.solve(K, rhs)
        x = sol[:n]
        if np.all(x >= -TOL) and abs(float(np.sum(x)) - 1.0) <= 1e-8:
            candidates.append(np.clip(x, 0.0, None))
    except np.linalg.LinAlgError:
        pass

    # 2) 边：x_i = t, x_j = 1-t，其余为 0；解析最小化二次函数
    for i in range(n):
        for j in range(i + 1, n):
            u = A[:, i] - A[:, j]
            v = A[:, j] - b
            denom = float(u @ u)
            t_star = -float(v @ u) / denom if denom > TOL else 0.0
            t_star = min(1.0, max(0.0, t_star))
            x = np.zeros(n)
            x[i] = t_star
            x[j] = 1.0 - t_star
            candidates.append(x)

    # 3) 顶点
    for k in range(n):
        x = np.zeros(n)
        x[k] = 1.0
        candidates.append(x)

    best_x, best_sse = None, np.inf
    for x in candidates:
        sse = float(np.sum((A @ x - b) ** 2))
        if sse < best_sse:
            best_sse, best_x = sse, x
    return best_x, best_sse


def estimate(machines: list[dict]) -> EstimateResult:
    """根据当前拆检记录计算估算结果。

    ``machines`` 为字典列表，每项含 ``id``、``initial``、
    ``inspected``（None 表示未检）。
    """
    n_total = len(machines)
    inspected = [m for m in machines if m["inspected"] is not None]
    uninspected = [m for m in machines if m["inspected"] is None]
    n_insp, n_un = len(inspected), len(uninspected)

    inspected_counts = {c: 0 for c in CATEGORIES}
    for m in inspected:
        inspected_counts[m["inspected"]] += 1

    un_initial_counts = {c: 0 for c in CATEGORIES}
    for m in uninspected:
        un_initial_counts[m["initial"]] += 1

    actual_total = {
        c: inspected_counts[c] if n_un == 0 else None for c in CATEGORIES
    }

    base = dict(
        n_total=n_total,
        n_inspected=n_insp,
        n_uninspected=n_un,
        inspected_counts=inspected_counts,
        uninspected_initial_counts=un_initial_counts,
    )

    # 全部拆完：直接给实数，不做拟合。
    if n_un == 0:
        return EstimateResult(
            available=True,
            reason=None,
            P=None,
            q=None,
            pi=None,
            sse=None,
            estimated_uninspected=None,
            estimated_total={c: float(inspected_counts[c]) for c in CATEGORIES},
            actual_total=inspected_counts,
            **base,
        )

    # 门控 1：任一类实检样本缺失。
    missing = [
        CATEGORY_LABELS[c] for c in CATEGORIES if inspected_counts[c] == 0
    ]
    if missing:
        return EstimateResult(
            available=False,
            reason=(
                "仍有未检件，且缺少实检类别样本："
                + "、".join(missing)
                + "。请补检后再估算。"
            ),
            P=None,
            q=None,
            pi=None,
            sse=None,
            estimated_uninspected=None,
            estimated_total=None,
            actual_total=None,
            **base,
        )

    # 经验混淆概率矩阵 P：行=实检类别，列=初判类别。
    count = np.zeros((3, 3), dtype=float)
    for m in inspected:
        count[CAT_INDEX[m["inspected"]], CAT_INDEX[m["initial"]]] += 1.0
    row_sum = count.sum(axis=1, keepdims=True)
    P = count / row_sum

    # 门控 2：混淆矩阵不满秩。
    if np.linalg.matrix_rank(P, tol=1e-10) < 3:
        return EstimateResult(
            available=False,
            reason=(
                "仍有未检件，但实检样本得到的混淆概率矩阵不满秩"
                "（各类实检样本的初判分布线性相关），无法唯一拟合，"
                "请补充更多拆检样本。"
            ),
            P=P.tolist(),
            q=None,
            pi=None,
            sse=None,
            estimated_uninspected=None,
            estimated_total=None,
            actual_total=None,
            **base,
        )

    # 未检件初判经验分布 q。
    q_arr = np.array(
        [un_initial_counts[c] for c in CATEGORIES], dtype=float
    ) / n_un

    # 最小化 ||P^T pi - q||^2，约束 pi>=0、sum(pi)=1。
    pi, sse = _solve_simplex(P.T, q_arr)
    pi = np.clip(pi, 0.0, 1.0)
    pi = pi / pi.sum()

    estimated_uninspected = {
        c: float(pi[i] * n_un) for i, c in enumerate(CATEGORIES)
    }
    estimated_total = {
        c: inspected_counts[c] + estimated_uninspected[c]
        for c in CATEGORIES
    }

    return EstimateResult(
        available=True,
        reason=None,
        P=P.tolist(),
        q=q_arr.tolist(),
        pi=pi.tolist(),
        sse=sse,
        estimated_uninspected=estimated_uninspected,
        estimated_total=estimated_total,
        actual_total=None,
        **base,
    )
