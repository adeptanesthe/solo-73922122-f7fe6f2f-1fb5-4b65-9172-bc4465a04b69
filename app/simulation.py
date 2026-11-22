"""全批复检后的抽检回放试算。

业务口径
========
* 仅在**全批全部拆完**后允许试算：此时每台机器都有实检类别，可当作
  “真值”做回放。
* 按给定随机种子生成 ``ROUNDS``（固定 100）条**不放回**的随机拆检顺序；
  每条顺序对每个档位 ``k`` 取前 ``k`` 台作为已检（沿用其真实实检类别），
  其余机器只保留分类器初判（``inspected=None``），再调用现有
  ``estimate`` 估算。
* 单轮达标：估数可用，且电源/驱动/控制三类**估计台数**与全检实数
  （``actual_total``）的绝对差都不超过允许误差（台）。
* 各档达标次数 = 100 轮中达标的轮数；推荐量为达标次数 ≥
  ``PASS_THRESHOLD``（90）的**最小**档位；没有任何档位满足则为 ``None``。

并发与失效
==========
* 试算任务在发起时对全检记录做一次快照，并记录存储版本号
  ``base_version``；计算在独立线程中进行，不阻塞事件循环，因此试算
  期间其他窗口仍可登记/更正/撤回或换批。
* 存储每次写入（含更正后改回、重新上传换批）都会递增版本号，并使所有
  基准版本不一致的任务立即失效；计算完成后还会在版本校验通过时才落盘
  结果，迟到结果无法把已撤下的建议恢复。
* 失效任务的 ``result`` 会被清空，只保留失效原因。
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass

from .estimation import CATEGORIES, estimate

ROUNDS = 100
PASS_THRESHOLD = 90
MAX_SIZES = 6


class SimulationError(ValueError):
    """试算参数或状态错误，映射为 HTTP 400。"""


def normalize_sizes(n_total: int, sizes: list[int]) -> list[int]:
    """校验并规范化档位（去重检查 + 升序），非法时抛 ``SimulationError``。"""
    if not sizes:
        raise SimulationError("请至少填写 1 档拆检台数")
    if len(sizes) > MAX_SIZES:
        raise SimulationError(f"最多填写 {MAX_SIZES} 档拆检台数")
    if len(set(sizes)) != len(sizes):
        raise SimulationError("各档拆检台数不能重复")
    for size in sizes:
        if not isinstance(size, int) or isinstance(size, bool):
            raise SimulationError("拆检台数必须为整数")
        if size < 1 or size >= n_total:
            raise SimulationError(
                f"每档拆检台数须不少于 1 台且少于全批 {n_total} 台，"
                f"当前填写 {size} 台不合法"
            )
    return sorted(sizes)


def run_simulation(
    machines: list[dict],
    sizes: list[int],
    seed: int,
    tolerance: int,
    rounds: int = ROUNDS,
) -> dict:
    """对一份**全检快照**执行 100 轮回放试算，返回各档达标统计。

    ``machines`` 中每台都须有非空 ``inspected``；函数不修改传入列表。
    ``sizes`` 会先升序排列，以便各档共享同一条随机顺序的前缀。
    """
    sizes = sorted(sizes)
    n_total = len(machines)
    if n_total == 0:
        raise SimulationError("全检记录为空，无法试算")
    pending = sum(1 for m in machines if not m.get("inspected"))
    if pending:
        raise SimulationError(f"仍有 {pending} 台未拆检，无法试算")
    if tolerance < 0:
        raise SimulationError("允许误差须为非负整数（台）")
    actual_total = {c: 0 for c in CATEGORIES}
    for m in machines:
        actual_total[m["inspected"]] += 1

    # 每轮重新生成一条完整随机顺序，所有档位共用同一条顺序。
    # rng 只用于打乱：第 r 轮各档看到的前缀逐级加长，语义为“同一顺序开头”。
    rng = random.Random(seed)
    pass_counts = {k: 0 for k in sizes}
    unavailable_counts = {k: 0 for k in sizes}

    insp0 = [m["inspected"] for m in machines]
    base = [
        {"id": m["id"], "initial": m["initial"], "inspected": None}
        for m in machines
    ]

    for _ in range(rounds):
        order = list(range(n_total))
        rng.shuffle(order)
        trial = [dict(m) for m in base]
        checked = 0
        for size in sizes:
            # 前缀只增不减：本档在前一档基础上继续取顺序中的后续机器。
            while checked < size:
                trial[order[checked]]["inspected"] = insp0[order[checked]]
                checked += 1
            est = estimate(trial)
            if not est.available:
                unavailable_counts[size] += 1
                continue
            if all(
                abs(est.estimated_total[c] - actual_total[c])
                <= tolerance + 1e-9
                for c in CATEGORIES
            ):
                pass_counts[size] += 1

    size_results = []
    for size in sizes:
        passed = pass_counts[size]
        size_results.append(
            {
                "size": size,
                "pass_count": passed,
                "fail_count": rounds
                - passed
                - unavailable_counts[size],
                "unavailable_count": unavailable_counts[size],
                "pass_rate": passed / rounds,
                "meets_threshold": passed >= PASS_THRESHOLD,
            }
        )

    recommended = min(
        (
            item["size"]
            for item in size_results
            if item["pass_count"] >= PASS_THRESHOLD
        ),
        default=None,
    )

    return {
        "rounds": rounds,
        "threshold": PASS_THRESHOLD,
        "tolerance": tolerance,
        "seed": seed,
        "n_total": n_total,
        "actual_total": actual_total,
        "sizes": size_results,
        "recommended_size": recommended,
    }


@dataclass
class SimulationJob:
    """一次试算任务；状态为 running / done / invalidated / failed。"""

    job_id: str
    base_version: int
    tolerance: int
    seed: int
    sizes: list[int]
    n_total: int
    state: str = "running"
    reason: str | None = None
    result: dict | None = None

    def to_payload(self) -> dict:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "base_version": self.base_version,
            "tolerance": self.tolerance,
            "seed": self.seed,
            "sizes": list(self.sizes),
            "n_total": self.n_total,
            "rounds": ROUNDS,
            "threshold": PASS_THRESHOLD,
            "reason": self.reason,
            "result": self.result,
        }


class SimulationManager:
    """进程内试算任务表。

    所有方法都由事件循环中的持锁代码同步调用（存储锁），因此内部不再
    单独加锁；``run_simulation`` 在独立线程中运行，不接触本对象。
    """

    def __init__(self, max_jobs: int = 50) -> None:
        self._jobs: dict[str, SimulationJob] = {}
        self._max_jobs = max_jobs

    def create(
        self,
        base_version: int,
        tolerance: int,
        seed: int,
        sizes: list[int],
        n_total: int,
    ) -> SimulationJob:
        job = SimulationJob(
            job_id=uuid.uuid4().hex,
            base_version=base_version,
            tolerance=tolerance,
            seed=seed,
            sizes=list(sizes),
            n_total=n_total,
        )
        self._jobs[job.job_id] = job
        self._prune()
        return job

    def get(self, job_id: str) -> SimulationJob | None:
        return self._jobs.get(job_id)

    def invalidate_stale_sync(self, current_version: int) -> None:
        """试算基准版本与当前存储版本不一致的任务全部作废。"""
        for job in self._jobs.values():
            if job.base_version == current_version:
                continue
            if job.state == "done" or job.state == "running":
                job.state = "invalidated"
                job.reason = (
                    "试算期间拆检记录已变更"
                    f"（发起时版本 v{job.base_version}，当前版本 "
                    f"v{current_version}），本次结果作废，请重新试算。"
                )
                job.result = None

    def finish_sync(
        self, job_id: str, result: dict, current_version: int
    ) -> bool:
        """计算完成后落盘；版本已变或任务已失效时不写入结果。"""
        job = self._jobs.get(job_id)
        if job is None or job.state != "running":
            return False
        if job.base_version != current_version:
            self.invalidate_stale_sync(current_version)
            return False
        job.result = result
        job.state = "done"
        return True

    def fail_sync(self, job_id: str, reason: str) -> None:
        job = self._jobs.get(job_id)
        if job is not None and job.state == "running":
            job.state = "failed"
            job.reason = reason
            job.result = None

    def _prune(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return
        for job_id in list(self._jobs):
            if len(self._jobs) <= self._max_jobs:
                break
            if self._jobs[job_id].state != "running":
                del self._jobs[job_id]


simulations = SimulationManager()
