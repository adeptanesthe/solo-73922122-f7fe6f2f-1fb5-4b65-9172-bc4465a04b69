"""批次与拆检记录的内存存储。

演示环境使用单进程内存存储，所有读写经过同一把异步锁，保证两个浏览器
窗口同时登记/更正/撤回（即便操作同一台机器）时两笔修改都基于同一份
拆检记录顺序生效，实检数与估数始终来自同一快照。

注意：该实现与 uvicorn 单 worker 绑定（docker-compose 已固定
``--workers 1``）；多进程部署需替换为数据库等共享存储。
"""
from __future__ import annotations

import asyncio

from .estimation import CATEGORIES, CAT_INDEX, estimate


class StoreError(Exception):
    """业务校验错误，映射为 HTTP 400。"""


class BatchStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._machines: list[dict] = []
        self._id_index: dict[str, int] = {}

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def reset_locked(self, machines: list[dict]) -> None:
        """载入新批次（调用方须持锁）。"""
        self._machines = machines
        self._id_index = {m["id"]: i for i, m in enumerate(machines)}

    def snapshot_locked(self) -> dict:
        """基于同一份拆检记录生成机器清单与估算结果（持锁调用）。"""
        result = estimate(self._machines)
        return {
            "machines": [dict(m) for m in self._machines],
            "estimate": result.to_payload(),
        }

    def update_inspection_locked(
        self, machine_id: str, category: str | None
    ) -> dict:
        """登记/更正（category 为三类之一）或撤回（None）一台机器。"""
        if machine_id not in self._id_index:
            raise StoreError(f"机器编号 {machine_id} 不存在于当前批次")
        if category is not None and category not in CAT_INDEX:
            raise StoreError(
                f"非法实检类别 {category!r}，允许：{', '.join(CATEGORIES)}"
            )
        machine = self._machines[self._id_index[machine_id]]
        machine["inspected"] = category
        return self.snapshot_locked()

    def is_empty_locked(self) -> bool:
        return not self._machines


store = BatchStore()
