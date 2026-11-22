"""API 端到端测试：上传、登记/更正/撤回、并发两窗口、门控、全检实数。"""
from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.csv_utils import MAX_MACHINES, parse_machines_csv
from app.main import app
from app.storage import store


@pytest.fixture(autouse=True)
async def _reset_store():
    async with store.lock:
        store.reset_locked([])
    yield
    async with store.lock:
        store.reset_locked([])


def _csv(text: str) -> bytes:
    return text.encode("utf-8")


def test_parse_csv_aliases_and_limits() -> None:
    machines = parse_machines_csv(
        _csv("机器编号,初判\nM1,电源\nM2,drive\nM3, 控制 \n")
    )
    assert [m["id"] for m in machines] == ["M1", "M2", "M3"]
    assert [m["initial"] for m in machines] == [
        "power",
        "drive",
        "control",
    ]
    assert all(m["inspected"] is None for m in machines)

    machines = parse_machines_csv(_csv("id,initial\nA,power\nB,control\n"))
    assert machines[0]["id"] == "A"

    for bad in [
        "编号,初判\n,电源\n",
        "机器编号,初判\nM1,电源\nM1,驱动\n",
        "机器编号,初判\nM1,爆炸\n",
        "机器编号\nM1\n",
        "\n",
    ]:
        with pytest.raises(Exception):
            parse_machines_csv(_csv(bad))

    over = "机器编号,初判\n" + "".join(
        f"M{i},电源\n" for i in range(MAX_MACHINES + 1)
    )
    with pytest.raises(Exception):
        parse_machines_csv(_csv(over))


async def test_full_workflow_inspect_correct_withdraw_and_gates() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        # 未上传时取批次 -> 404
        r = await client.get("/api/batch")
        assert r.status_code == 404

        # 9 台，初判多样，便于凑齐满秩样本
        content = _csv(
            "机器编号,初判\n"
            "M1,电源\nM2,电源\nM3,驱动\nM4,驱动\nM5,控制\nM6,控制\n"
            "M7,电源\nM8,驱动\nM9,控制\n"
        )
        r = await client.post(
            "/api/batch", files={"file": ("b.csv", content, "text/csv")}
        )
        assert r.status_code == 200
        snap = r.json()
        assert snap["estimate"]["n_total"] == 9
        assert snap["estimate"]["n_uninspected"] == 9
        assert not snap["estimate"]["available"]  # 零实检 -> 撤下

        # 非法类别 400
        r = await client.post(
            "/api/machines/M1/inspect", json={"category": "nuclear"}
        )
        assert r.status_code == 400
        # 不存在的机器 400
        r = await client.post(
            "/api/machines/NOPE/inspect", json={"category": "power"}
        )
        assert r.status_code == 400

        # 登记：让每类实检样本包含不同初判，保证撤回后仍可凑满秩
        plan = {
            "M1": "power", "M2": "power", "M3": "power",
            "M4": "drive", "M5": "drive", "M6": "drive",
            "M7": "control", "M8": "control", "M9": "control",
        }
        for mid, cat in plan.items():
            r = await client.post(
                f"/api/machines/{mid}/inspect", json={"category": cat}
            )
            assert r.status_code == 200

        snap = (await client.get("/api/batch")).json()
        est = snap["estimate"]
        assert est["n_inspected"] == 9 and est["n_uninspected"] == 0
        assert est["available"] and est["actual_total"] == {
            "power": 3, "drive": 3, "control": 3
        }

        # 更正 M1：power -> control，实检数随之变化
        r = await client.post(
            "/api/machines/M1/inspect", json={"category": "control"}
        )
        est = r.json()["estimate"]
        assert est["actual_total"] == {
            "power": 2, "drive": 3, "control": 4
        }

        # 撤回 M1：重新算作未检；此时有未检件且三类实检样本仍齐、矩阵满秩
        r = await client.post("/api/machines/M1/withdraw")
        est = r.json()["estimate"]
        assert est["n_inspected"] == 8 and est["n_uninspected"] == 1
        assert est["available"]
        m1 = next(m for m in r.json()["machines"] if m["id"] == "M1")
        assert m1["inspected"] is None
        # 估数 = 实检数 + 未检折算，总和守恒
        assert sum(est["estimated_total"].values()) == pytest.approx(9.0)
        for c in ("power", "drive", "control"):
            assert est["estimated_total"][c] == pytest.approx(
                est["inspected_counts"][c]
                + est["estimated_uninspected"][c]
            )

        # 继续撤回至某类实检样本归零 -> 估数撤下并提示补检
        for mid in ("M7", "M8", "M9"):
            r = await client.post(f"/api/machines/{mid}/withdraw")
        est = r.json()["estimate"]
        assert not est["available"]
        assert "控制" in est["reason"]


async def test_concurrent_two_windows_both_take_effect() -> None:
    """模拟两个窗口同时改不同机器（以及同一台），两笔都须生效。"""
    transport = ASGITransport(app=app)
    content = _csv(
        "机器编号,初判\n"
        "M1,电源\nM2,驱动\nM3,控制\nM4,电源\nM5,驱动\nM6,控制\n"
    )
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        await client.post(
            "/api/batch", files={"file": ("b.csv", content, "text/csv")}
        )

        # 不同机器并发登记：两笔都生效
        await asyncio.gather(
            client.post("/api/machines/M1/inspect", json={"category": "power"}),
            client.post("/api/machines/M2/inspect", json={"category": "drive"}),
            client.post("/api/machines/M3/inspect", json={"category": "control"}),
        )
        snap = (await client.get("/api/batch")).json()
        by_id = {m["id"]: m["inspected"] for m in snap["machines"]}
        assert by_id["M1"] == "power"
        assert by_id["M2"] == "drive"
        assert by_id["M3"] == "control"

        # 同一台机器两个窗口同时提交不同类别：两次写入都不丢，
        # 最终状态是其中之一（按锁顺序），且快照自洽
        r1, r2 = await asyncio.gather(
            client.post("/api/machines/M1/inspect", json={"category": "drive"}),
            client.post("/api/machines/M1/inspect", json={"category": "control"}),
        )
        assert r1.status_code == r2.status_code == 200
        final = (await client.get("/api/batch")).json()
        m1 = next(m for m in final["machines"] if m["id"] == "M1")
        assert m1["inspected"] in {"drive", "control"}
        # 实检数与机器明细始终一致（同一份拆检记录）
        counts = final["estimate"]["inspected_counts"]
        actual = {"power": 0, "drive": 0, "control": 0}
        for m in final["machines"]:
            if m["inspected"]:
                actual[m["inspected"]] += 1
        assert counts == actual

        # 并发撤回 + 登记不同机器，同样都生效
        await asyncio.gather(
            client.post("/api/machines/M2/withdraw"),
            client.post("/api/machines/M4/inspect", json={"category": "power"}),
        )
        snap = (await client.get("/api/batch")).json()
        by_id = {m["id"]: m["inspected"] for m in snap["machines"]}
        assert by_id["M2"] is None and by_id["M4"] == "power"
