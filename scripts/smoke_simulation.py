"""对运行中的服务验证抽检回放试算（纯标准库，可在容器外执行）。

覆盖：
1. 全批拆完 → 合法试算 → 轮询完成 → 推荐量；
2. 试算期间其他窗口更正（再改回原值）→ 结果作废，迟到结果不恢复建议。

用法：python scripts/smoke_simulation.py [base_url] [csv_path]
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
CSV_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/sample_machines.csv"
CATS = ("power", "drive", "control")


def call(method: str, path: str, body=None, raw: bytes | None = None,
         ctype: str | None = None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif raw is not None:
        data = raw
        headers["Content-Type"] = ctype or "application/octet-stream"
    req = urllib.request.Request(
        BASE + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" +
          (f" — {detail}" if detail else ""))
    if not cond:
        raise SystemExit(1)


def upload(path: str):
    with open(path, "rb") as f:
        boundary = "----smokesimboundary"
        payload = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"sample.csv\"\r\nContent-Type: text/csv\r\n\r\n"
        ).encode() + f.read() + f"\r\n--{boundary}--\r\n".encode()
        return call("POST", "/api/batch", raw=payload,
                    ctype=f"multipart/form-data; boundary={boundary}")


def wait_job(job_id: str, timeout: float = 60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, job = call("GET", f"/api/simulations/{job_id}")
        if status != 200 or job["state"] != "running":
            return job
        time.sleep(0.15)
    raise SystemExit("试算轮询超时")


# ── 场景 A：有效试算（全批拆完后） ────────────────────────────
status, snap = upload(CSV_PATH)
check("上传 CSV", status == 200)
n_total = snap["estimate"]["n_total"]
machines = snap["machines"]

# 未拆完时试算应被拒绝
status, err = call("POST", "/api/simulations",
                   {"tolerance": 5, "seed": 1, "sizes": [10]})
check("未全检时试算被拒", status == 400 and "未拆检" in err.get("detail", ""),
      err.get("detail"))

# 用带噪声的“真值”把全批拆完：约 75% 与初判一致，其余随机落到另两类，
# 从而形成满秩混淆矩阵，估数与实检实数存在真实偏差，可用于回放。
rng = random.Random(20261122)
for m in machines:
    if rng.random() < 0.75:
        inspected = m["initial"]
    else:
        inspected = rng.choice([c for c in CATS if c != m["initial"]])
    status, _ = call("POST", f"/api/machines/{m['id']}/inspect",
                     {"category": inspected})
check("全部登记实检", status == 200)
status, snap = call("GET", "/api/batch")
check("全批已拆完", snap["estimate"]["n_uninspected"] == 0)
base_version = snap["version"]
print(f"       全检实数：{snap['estimate']['actual_total']}")

sizes = [10, 20, 30, 40, 80, 119]
status, job = call("POST", "/api/simulations",
                   {"tolerance": 10, "seed": 20261122, "sizes": sizes})
check("发起试算返回 200", status == 200, f"job={job.get('job_id')}")
check("任务基准版本=发起时版本", job["base_version"] == base_version)
t0 = time.time()
job = wait_job(job["job_id"])
check("试算完成", job["state"] == "done", f"耗时 {time.time() - t0:.2f}s")
r = job["result"]
check("轮数为 100、门槛 90",
      r["rounds"] == 100 and r["threshold"] == 90)
check("各档达标次数在 0～100 且三类计数自洽", all(
    s["pass_count"] + s["fail_count"] + s["unavailable_count"] == 100
    for s in r["sizes"]))
check("档位按输入一一返回", [s["size"] for s in r["sizes"]] == sizes)
if r["recommended_size"] is None:
    check("无档位达标时推荐量为『未找到』",
          all(not s["meets_threshold"] for s in r["sizes"]))
else:
    rec = next(s for s in r["sizes"] if s["size"] == r["recommended_size"])
    smaller = [s for s in r["sizes"]
               if s["size"] < r["recommended_size"] and s["meets_threshold"]]
    check(f"推荐量 {r['recommended_size']} 台达标 {rec['pass_count']} 次且为最小达标档",
          rec["pass_count"] >= 90 and not smaller)
print("       各档达标次数：" +
      "，".join(f"{s['size']}台→{s['pass_count']}次" for s in r["sizes"]))

# ── 场景 B：试算中途记录变更（改回原值也算失效） ──────────────
status, job = call("POST", "/api/simulations",
                   {"tolerance": 5, "seed": 77,
                    "sizes": [10, 20, 30, 40, 60, 80]})
check("再次发起试算", status == 200)
job_id = job["job_id"]
target = machines[0]
other = "drive" if target["initial"] != "drive" else "power"
call("POST", f"/api/machines/{target['id']}/inspect", {"category": other})
call("POST", f"/api/machines/{target['id']}/inspect",
     {"category": target["initial"]})
job = wait_job(job_id)
check("期间更正（又改回）后任务失效",
      job["state"] == "invalidated" and job["result"] is None,
      job.get("reason", ""))
status, snap_after = call("GET", "/api/batch")
check("记录内容已改回、版本仍推进（改回也算失效）",
      snap_after["version"] > base_version)
# 再轮询两次，确认不会迟到恢复建议
time.sleep(0.3)
status, again = call("GET", f"/api/simulations/{job_id}")
check("迟到轮询不恢复旧建议",
      again["state"] == "invalidated" and again["result"] is None)

# ── 场景 C：换批后旧试算失效 ─────────────────────────────────
status, job = call("POST", "/api/simulations",
                   {"tolerance": 5, "seed": 88, "sizes": [10, 20]})
old_id = job["job_id"]
status, _ = upload(CSV_PATH)
job = wait_job(old_id)
check("换批后旧试算失效", job["state"] == "invalidated" and job["result"] is None)

print("\n试算冒烟全部通过 ✔")
