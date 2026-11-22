"""CSV 解析：机器编号 + 分类器初判，最多 500 台。"""
from __future__ import annotations

import csv
import io

from .estimation import CATEGORIES


class CsvError(ValueError):
    """CSV 内容或结构错误。"""


MAX_MACHINES = 500

_HEADER_ALIASES = {
    "id": "id",
    "机器编号": "id",
    "机器id": "id",
    "编号": "id",
    "machine_id": "id",
    "machine id": "id",
    "initial": "initial",
    "初判": "initial",
    "分类器初判": "initial",
    "初判类别": "initial",
    "initial_label": "initial",
}

_CATEGORY_ALIASES = {
    "power": "power",
    "电源": "power",
    "电源类": "power",
    "drive": "drive",
    "驱动": "drive",
    "驱动类": "drive",
    "control": "control",
    "控制": "control",
    "控制类": "control",
}


def _norm_header(name: str) -> str:
    return name.strip().lstrip("﻿").lower().replace(" ", "_")


def _map_header(name: str) -> str | None:
    key = _norm_header(name)
    if key in _HEADER_ALIASES:
        return _HEADER_ALIASES[key]
    return _HEADER_ALIASES.get(name.strip())


def _map_category(raw: str) -> str:
    key = raw.strip().lower()
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    raise CsvError(
        f"非法故障类别 {raw!r}，允许：电源/power、驱动/drive、控制/control"
    )


def parse_machines_csv(content: bytes) -> list[dict]:
    """解析上传的 CSV，返回 ``[{id, initial, inspected: None}, ...]``。"""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CsvError("CSV 必须为 UTF-8 编码") from exc

    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise CsvError("CSV 为空，至少需要表头和一行数据")

    columns = [_map_header(h) for h in header]
    if columns.count("id") != 1 or columns.count("initial") != 1:
        raise CsvError(
            "表头必须包含机器编号列与分类器初判列"
            "（可写作：机器编号/初判，或 id/initial）"
        )
    id_col = columns.index("id")
    initial_col = columns.index("initial")

    machines: list[dict] = []
    seen: set[str] = set()
    for line_no, row in enumerate(reader, start=2):
        if not row or all(not cell.strip() for cell in row):
            continue  # 容忍空行
        if len(row) < len(header):
            raise CsvError(f"第 {line_no} 行列数不足，应有 {len(header)} 列")
        machine_id = row[id_col].strip()
        if not machine_id:
            raise CsvError(f"第 {line_no} 行机器编号为空")
        if machine_id in seen:
            raise CsvError(f"机器编号重复：{machine_id}（第 {line_no} 行）")
        raw_label = row[initial_col].strip()
        if not raw_label:
            raise CsvError(f"第 {line_no} 行分类器初判为空")
        machines.append(
            {
                "id": machine_id,
                "initial": _map_category(raw_label),
                "inspected": None,
            }
        )
        seen.add(machine_id)
        if len(machines) > MAX_MACHINES:
            raise CsvError(f"一批最多 {MAX_MACHINES} 台，当前超出限制")

    if not machines:
        raise CsvError("CSV 没有任何机器记录")
    return machines
