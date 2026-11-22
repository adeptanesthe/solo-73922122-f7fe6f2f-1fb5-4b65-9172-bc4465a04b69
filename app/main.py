"""FastAPI 入口：CSV 上传、拆检登记/更正/撤回、估算快照。"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .csv_utils import CsvError, parse_machines_csv
from .estimation import CATEGORIES
from .storage import StoreError, store

app = FastAPI(title="返修机故障占比估算", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"


class InspectPayload(BaseModel):
    category: str = Field(..., description="实检类别：power/drive/control")


@app.exception_handler(CsvError)
@app.exception_handler(StoreError)
async def _business_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


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


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)
