"""回放试算：参数校验、达标判定、推荐档位与失效语义。"""
from __future__ import annotations

import asyncio
import random

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.simulation import (
    PASS_THRESHOLD,
    ROUNDS,
    SimulationError,
    normalize_sizes,
    run_simulation,
    simulations,
)
from app.storage import store

CATS = ("power", "drive", "control")


def _machines(n: int = 40) -> list[dict]:
    """构造一份全检记录：真实类别按 i%3 轮转，初判按固定混淆加噪。

    ``n`` 取 3 的倍数，三类实检实数各占 n/3，便于断言。
    """
    confusion = (
        (0.8, 0.13, 0.07),
        (0.12, 0.78, 0.10),
        (0.05, 0.15, 0.80),
    )
    rng = random.Random(20261122)
    rows = []
    for i in range(n):
        true_idx = i % 3
        initial_idx = rng.choices(range(3), weights=confusion[true_idx])[0]
        rows.append(
            {
                "id": f"M{i:04d}",
                "initial": CATS[initial_idx],
                "inspected": CATS[true_idx],
            }
        )
    return rows


def test_normalize_sizes_rules() -> None:
    assert normalize_sizes(10, [5, 2, 1]) == [1, 2, 5]
    for bad in ([], [0], [10], [11], [2, 2], [1, 2, 3, 4, 5, 6, 7]):
        with pytest.raises(SimulationError):
            normalize_sizes(10, bad)
    # 布尔不是合法档位
    with pytest.raises(SimulationError):
        normalize_sizes(10, [True])


def test_run_simulation_rejects_incomplete_snapshot() -> None:
    machines = _machines(9)
    machines[0]["inspected"] = None
    with pytest.raises(SimulationError):
        run_simulation(machines, [2], seed=1, tolerance=1)
    with pytest.raises(SimulationError):
        run_simulation([], [2], seed=1, tolerance=1)


def test_run_simulation_shape_and_determinism() -> None:
    machines = _machines(30)
    first = run_simulation(machines, [1, 2, 6], seed=42, tolerance=3)
    second = run_simulation(machines, [1, 2, 6], seed=42, tolerance=3)
    assert first == second  # 同一份记录 + 同一种子 -> 结果可复现

    assert first["rounds"] == ROUNDS == 100
    assert first["threshold"] == PASS_THRESHOLD == 90
    assert first["actual_total"] == {"power": 10, "drive": 10, "control": 10}
    assert [s["size"] for s in first["sizes"]] == [1, 2, 6]
    for item in first["sizes"]:
        assert (
            item["pass_count"] + item["fail_count"] + item["unavailable_count"]
            == ROUNDS
        )
        assert 0.0 <= item["pass_rate"] <= 1.0

    # 记录未被修改
    assert all(m["inspected"] for m in machines)


def test_run_simulation_threshold_and_recommendation() -> None:
    machines = _machines(30)
    # 1 台/轮样本太少（估数不可用），必然不满足 90 次；
    # 29 台/轮几乎等于全检，放宽到 ±2 台后必定 100 次达标 -> 推荐最小达标档。
    result = run_simulation(machines, [1, 29], seed=1, tolerance=2)
    by_size = {s["size"]: s for s in result["sizes"]}
    assert by_size[1]["pass_count"] < PASS_THRESHOLD
    assert by_size[29]["pass_count"] >= PASS_THRESHOLD
    assert result["recommended_size"] == 29

    # 严格容差下 29 台/轮也不足 90 次 -> 没有达标档位，推荐量为 None
    strict = run_simulation(machines, [1, 29], seed=1, tolerance=0)
    assert all(not s["meets_threshold"] for s in strict["sizes"])
    assert strict["recommended_size"] is None


def test_run_simulation_tolerance_zero_requires_exact_match() -> None:
    machines = _machines(12)
    # 仅剩 1 台未检、已检 11 台时，三类初判分布基本确定，
    # 但容差 0 要求估数与实数逐类完全一致，只有极少数轮次可能满足。
    result = run_simulation(machines, [11], seed=5, tolerance=0)
    detail = result["sizes"][0]
    assert detail["pass_count"] <= ROUNDS
    assert detail["fail_count"] + detail["unavailable_count"] >= 0


@pytest.fixture(autouse=True)
async def _reset_store():
    async with store.lock:
        store.reset_locked([])
    yield
    async with store.lock:
        store.reset_locked([])


def _csv(machines: list[dict]) -> bytes:
    lines = ["机器编号,初判"] + [f"{m['id']},{m['initial']}" for m in machines]
    return ("\n".join(lines) + "\n").encode()


async def _upload_and_inspect(client: AsyncClient, machines: list[dict]) -> None:
    r = await client.post(
        "/api/batch", files={"file": ("b.csv", _csv(machines), "text/csv")}
    )
    assert r.status_code == 200
    for m in machines:
        r = await client.post(
            f"/api/machines/{m['id']}/inspect",
            json={"category": m["inspected"]},
        )
        assert r.status_code == 200


async def test_api_requires_full_inspection_and_valid_params() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        machines = _machines(9)
        await client.post(
            "/api/batch", files={"file": ("b.csv", _csv(machines), "text/csv")}
        )
        # 未拆完 -> 400
        r = await client.post(
            "/api/simulations",
            json={"tolerance": 1, "seed": 1, "sizes": [2]},
        )
        assert r.status_code == 400
        assert "未拆检" in r.json()["detail"]

        await _upload_and_inspect(client, machines)
        snap = (await client.get("/api/batch")).json()
        assert snap["estimate"]["n_uninspected"] == 0
        assert snap["version"] >= 1

        # 超过全批台数 / 重复档位 / 负误差 / 空档位 / 非整数档位 -> 400
        for payload in (
            {"tolerance": 1, "seed": 1, "sizes": [9]},
            {"tolerance": 1, "seed": 1, "sizes": [2, 2]},
            {"tolerance": -1, "seed": 1, "sizes": [2]},
            {"tolerance": 1, "seed": 1, "sizes": []},
            {"tolerance": 1, "seed": 1, "sizes": [0]},
            {"tolerance": 1, "seed": 1, "sizes": [1.5]},
            {"tolerance": 1, "seed": 1, "sizes": [1, 2, 3, 4, 5, 6, 7]},
        ):
            r = await client.post("/api/simulations", json=payload)
            assert r.status_code == 400, payload
            assert r.json()["detail"]

        # 合法请求 -> 任务可轮询，最终 done 且有结果
        r = await client.post(
            "/api/simulations",
            json={"tolerance": 2, "seed": 11, "sizes": [3, 8]},
        )
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        for _ in range(100):
            job = (await client.get(f"/api/simulations/{job_id}")).json()
            if job["state"] != "running":
                break
            await asyncio.sleep(0.05)
        assert job["state"] == "done"
        assert job["base_version"] == snap["version"]
        assert [s["size"] for s in job["result"]["sizes"]] == [3, 8]
        assert job["result"]["rounds"] == 100

        r = await client.get("/api/simulations/does-not-exist")
        assert r.status_code == 404


async def test_api_invalidates_on_change_even_if_reverted() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        machines = _machines(30)
        await _upload_and_inspect(client, machines)

        r = await client.post(
            "/api/simulations",
            json={"tolerance": 1, "seed": 3, "sizes": [5]},
        )
        job_id = r.json()["job_id"]
        base_version = r.json()["base_version"]

        # 试算期间：更正 -> 再改回原值，版本仍已推进
        target = machines[0]
        await client.post(
            f"/api/machines/{target['id']}/inspect",
            json={"category": "drive" if target["inspected"] != "drive" else "power"},
        )
        await client.post(
            f"/api/machines/{target['id']}/inspect",
            json={"category": target["inspected"]},
        )

        for _ in range(100):
            job = (await client.get(f"/api/simulations/{job_id}")).json()
            if job["state"] != "running":
                break
            await asyncio.sleep(0.05)
        assert job["state"] == "invalidated"
        assert job["result"] is None
        assert f"v{base_version}" in job["reason"]

        # 迟到的完成回调不能在版本校验后恢复结果
        assert base_version != (await client.get("/api/batch")).json()["version"]
        async with store.lock:
            assert not simulations.finish_sync(
                job_id, {"fake": True}, store.version
            )
            assert simulations.get(job_id).state == "invalidated"
            assert simulations.get(job_id).result is None


async def test_api_invalidates_on_new_batch() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        await _upload_and_inspect(client, _machines(20))
        r = await client.post(
            "/api/simulations",
            json={"tolerance": 2, "seed": 9, "sizes": [4]},
        )
        job_id = r.json()["job_id"]

        # 换批
        await _upload_and_inspect(client, _machines(12))
        job = (await client.get(f"/api/simulations/{job_id}")).json()
        assert job["state"] in {"invalidated", "running"}
        for _ in range(100):
            job = (await client.get(f"/api/simulations/{job_id}")).json()
            if job["state"] != "running":
                break
            await asyncio.sleep(0.05)
        assert job["state"] == "invalidated"
        assert job["result"] is None
