"""FastAPI 入口：CSV 上传、拆检登记/更正/撤回、估算快照、回放试算。"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import simulation as simulation_module
from .csv_utils import CsvError, parse_machines_csv
from .estimation import CATEGORIES
from .simulation import SimulationError, normalize_sizes, simulations
from .storage import StoreError, store

app = FastAPI(title="返修机故障占比估算", version="1.1.0")

STATIC_DIR = Path(__file__).parent / "static"


class InspectPayload(BaseModel):
    category: str = Field(..., description="实检类别：power/drive/control")


class SimulationPayload(BaseModel):
    tolerance: int = Field(..., description="允许误差（台），非负整数")
    seed: int = Field(..., description="随机种子")
    sizes: list[int] = Field(
        ...,
        description="1～6 档拆检台数，每档至少 1 台且少于全批",
    )


@app.exception_handler(CsvError)
@app.exception_handler(StoreError)
@app.exception_handler(SimulationError)
async def _business_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """把请求体类型错误也统一成 400 + 中文原因，便于页面直接展示。"""
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    message = first.get("msg", "参数格式不正确")
    detail = f"参数不合法：{field} {message}" if field else f"参数不合法：{message}"
    return JSONResponse(status_code=400, content={"detail": detail})


@app.post("/api/batch")
async def upload_batch(request: Request) -> dict:
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "filename"):
        raise CsvError("请通过 file 字段上传 CSV 文件")
    content = await upload.read()
    if not content:
        raise CsvError("上传的文件为空")
    machines = parse_machines_csv(content)
    async with store.lock:
        store.reset_locked(machines)
        return store.snapshot_locked()


@app.get("/api/batch")
async def get_batch() -> dict:
    async with store.lock:
        if store.is_empty_locked():
            return JSONResponse(
                status_code=404,
                content={"detail": "尚未上传批次，请先上传 CSV"},
            )
        return store.snapshot_locked()


@app.post("/api/machines/{machine_id}/inspect")
async def inspect_machine(machine_id: str, payload: InspectPayload) -> dict:
    if payload.category not in CATEGORIES:
        raise StoreError(
            f"非法实检类别 {payload.category!r}，"
            f"允许：{', '.join(CATEGORIES)}"
        )
    async with store.lock:
        if store.is_empty_locked():
            raise StoreError("尚未上传批次，请先上传 CSV")
        return store.update_inspection_locked(machine_id, payload.category)


@app.post("/api/machines/{machine_id}/withdraw")
async def withdraw_machine(machine_id: str) -> dict:
    async with store.lock:
        if store.is_empty_locked():
            raise StoreError("尚未上传批次，请先上传 CSV")
        return store.update_inspection_locked(machine_id, None)


@app.post("/api/simulations")
async def start_simulation(payload: SimulationPayload) -> dict:
    """对发起时的全检记录快照启动 100 轮回放试算（不阻塞事件循环）。"""
    async with store.lock:
        machines, version = store.full_inspection_snapshot_locked()
        if payload.tolerance < 0:
            raise SimulationError("允许误差须为非负整数（台）")
        sizes = normalize_sizes(len(machines), payload.sizes)
        job = simulations.create(
            base_version=version,
            tolerance=payload.tolerance,
            seed=payload.seed,
            sizes=sizes,
            n_total=len(machines),
        )
    task = asyncio.create_task(
        _run_simulation_job(
            job.job_id,
            machines,
            sizes,
            payload.seed,
            payload.tolerance,
        )
    )
    # 防止任务被回收；任务自身持有完整上下文。
    app.state.simulation_tasks = getattr(app.state, "simulation_tasks", set())
    app.state.simulation_tasks.add(task)
    task.add_done_callback(app.state.simulation_tasks.discard)
    return job.to_payload()


async def _run_simulation_job(
    job_id: str,
    machines: list[dict],
    sizes: list[int],
    seed: int,
    tolerance: int,
) -> None:
    try:
        result = await asyncio.to_thread(
            simulation_module.run_simulation,
            machines,
            sizes,
            seed,
            tolerance,
        )
    except Exception as exc:  # noqa: BLE001 - 任务内错误只回填状态
        async with store.lock:
            simulations.fail_sync(job_id, f"试算失败：{exc}")
        return
    async with store.lock:
        # 版本校验通过才落盘；若期间记录已变更，此处会作废并丢弃结果。
        simulations.finish_sync(job_id, result, store.version)


@app.get("/api/simulations/{job_id}")
async def get_simulation(job_id: str) -> JSONResponse:
    async with store.lock:
        simulations.invalidate_stale_sync(store.version)
        job = simulations.get(job_id)
        if job is None:
            return JSONResponse(
                status_code=404,
                content={"detail": f"试算任务 {job_id} 不存在或已过期"},
            )
        return JSONResponse(content=job.to_payload())


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)
