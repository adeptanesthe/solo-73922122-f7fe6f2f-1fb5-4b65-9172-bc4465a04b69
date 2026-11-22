"""生成演示用 CSV（可信的项目脚本，非外部数据）。

真实故障占比按 电源 45% / 驱动 35% / 控制 20% 生成，分类器初判在真实
类别基础上按固定混淆概率加噪声，保证混淆矩阵满秩、估计可拟合。

用法：python scripts/generate_sample.py [台数] [输出路径]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

CATS = ("power", "drive", "control")
# 行=真实类别，列=初判类别
CONFUSION = np.array(
    [
        [0.80, 0.13, 0.07],
        [0.12, 0.78, 0.10],
        [0.05, 0.15, 0.80],
    ]
)
TRUE_RATIO = np.array([0.45, 0.35, 0.20])


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/sample_machines.csv")
    if n > 500:
        raise SystemExit("一批最多 500 台")
    rng = np.random.default_rng(20261122)
    labels = rng.choice(3, size=n, p=TRUE_RATIO)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["机器编号", "初判"])
        for i, true_idx in enumerate(labels, start=1):
            pred_idx = rng.choice(3, p=CONFUSION[true_idx])
            writer.writerow([f"M{i:04d}", CATS[pred_idx]])
    print(f"wrote {n} machines to {out}")


if __name__ == "__main__":
    main()
