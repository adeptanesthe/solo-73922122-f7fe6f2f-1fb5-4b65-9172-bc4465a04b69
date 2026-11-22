"""对运行中的网页服务做端到端冒烟（纯标准库，可在容器外执行）。

覆盖：CSV 上传、登记/更正/撤回、估数门控（缺类/不满秩撤下、补检恢复）、
全检实数、台数守恒，以及两个“窗口”并发修改不同机器两笔都生效。

用法：python scripts/smoke_check.py [base_url] [csv_path]
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import urllib.request
import urllib.error

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
    req = urllib.request.Request(BASE + path, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        raise SystemExit(1)


# 1. 上传
with open(CSV_PATH, "rb") as f:
    boundary = "----smokeboundary"
    payload = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"sample.csv\"\r\nContent-Type: text/csv\r\n\r\n"
    ).encode() + f.read() + f"\r\n--{boundary}--\r\n".encode()
    status, snap = call("POST", "/api/batch", raw=payload,
                        ctype=f"multipart/form-data; boundary={boundary}")
check("上传 CSV", status == 200, f"status={status}")
est = snap["estimate"]
n_total = est["n_total"]
check("初始全部未检且估数撤下（缺实检样本）",
      est["n_uninspected"] == n_total and not est["available"])

# 2. 构造满秩拆检：三个实检类别的初判分布互不相同（行向量线性无关）
#    power: (3,2,1)  drive: (1,3,2)  control: (2,1,3)
machines = snap["machines"]
by_initial = {c: [m["id"] for m in machines if m["initial"] == c] for c in CATS}
layout = {
    "power": {"power": 3, "drive": 2, "control": 1},
    "drive": {"power": 1, "drive": 3, "control": 2},
    "control": {"power": 2, "drive": 1, "control": 3},
}
plan: dict[str, str] = {}
for true_cat, cells in layout.items():
    for init_cat, k in cells.items():
        for _ in range(k):
            mid = by_initial[init_cat].pop()
            plan[mid] = true_cat

for mid, cat in plan.items():
    status, s = call("POST", f"/api/machines/{mid}/inspect", {"category": cat})
    check(f"登记 {mid} -> {cat}", status == 200)
est = s["estimate"]
check("补检后估数恢复（满秩、三类齐）", est["available"],
      f"reason={est.get('reason')}")
check("台数守恒（估数总和=总台数）",
      abs(sum(est["estimated_total"].values()) - n_total) < 1e-8)
check("折算台数+实检数=全批估计", all(
    abs(est["estimated_total"][c] -
        (est["inspected_counts"][c] + est["estimated_uninspected"][c])) < 1e-9
    for c in CATS))
pi = est["pi"]
check("π 在单纯形内", all(x >= 0 for x in pi) and abs(sum(pi) - 1) < 1e-9)
print(f"       拟合 π = {dict(zip(CATS, (round(x, 4) for x in pi)))}，"
      f"SSE = {est['sse']:.6f}")

# 3. 更正一台：取一台已检机，改成另一类别，实检数应同步变化
mid = next(iter(plan))
old_counts = dict(est["inspected_counts"])
status, s = call("POST", f"/api/machines/{mid}/inspect",
                 {"category": "drive" if plan[mid] != "drive" else "power"})
new_cat = "drive" if plan[mid] != "drive" else "power"
check("更正返回 200", status == 200)
row = next(m for m in s["machines"] if m["id"] == mid)
check("机器明细已更正", row["inspected"] == new_cat)
check("实检数随同一份记录更新",
      s["estimate"]["inspected_counts"][new_cat] == old_counts[new_cat] + 1)
plan[mid] = new_cat

# 4. 撤回一台：重新算作未检
status, s = call("POST", f"/api/machines/{mid}/withdraw")
row = next(m for m in s["machines"] if m["id"] == mid)
check("撤回后该台重算未检", status == 200 and row["inspected"] is None)
check("撤回后未检数 +1", s["estimate"]["n_uninspected"] == est["n_uninspected"] + 1)

# 5. 门控：撤回某实检类别的全部样本 -> 估数撤下并提示补检
status, snap_all = call("GET", "/api/batch")
target = "control"
target_label = {"power": "电源", "drive": "驱动", "control": "控制"}[target]
victims = [m["id"] for m in snap_all["machines"]
           if m["inspected"] == target and m["id"] != mid]
for v in victims:
    status, s = call("POST", f"/api/machines/{v}/withdraw")
check("缺类门控触发", not s["estimate"]["available"]
      and target_label in s["estimate"]["reason"],
      s["estimate"].get("reason"))

# 6. 补检一台恢复该类样本（若矩阵仍满秩，估数恢复）
status, s = call("POST", f"/api/machines/{victims[0]}/inspect",
                 {"category": target})
check("补检后接口正常", status == 200)
print(f"       补检后估数状态: available={s['estimate']['available']}"
      f" reason={s['estimate'].get('reason')}")

# 7. 并发：两个窗口同时改不同机器，两笔都生效
m_a, m_b = [m["id"] for m in s["machines"] if m["inspected"] is None][:2]
with cf.ThreadPoolExecutor(max_workers=2) as ex:
    f1 = ex.submit(call, "POST", f"/api/machines/{m_a}/inspect",
                   {"category": "power"})
    f2 = ex.submit(call, "POST", f"/api/machines/{m_b}/inspect",
                   {"category": "control"})
    r1, r2 = f1.result(), f2.result()
check("并发两请求均 200", r1[0] == 200 and r2[0] == 200)
status, final = call("GET", "/api/batch")
by_id = {m["id"]: m["inspected"] for m in final["machines"]}
check("两窗口修改不同机器均生效",
      by_id[m_a] == "power" and by_id[m_b] == "control")
counts = final["estimate"]["inspected_counts"]
actual = {c: 0 for c in CATS}
for m in final["machines"]:
    if m["inspected"]:
        actual[m["inspected"]] += 1
check("实检数与明细一致（同一份拆检记录）", counts == actual)

# 8. 全部拆完 -> 实数
status, snap_all = call("GET", "/api/batch")
for m in snap_all["machines"]:
    if m["inspected"] is None:
        call("POST", f"/api/machines/{m['id']}/inspect",
             {"category": m["initial"]})
status, done = call("GET", "/api/batch")
e = done["estimate"]
check("全部拆完直接给实数",
      e["n_uninspected"] == 0 and e["available"] and e["pi"] is None)
check("实数合计=总台数", sum(e["actual_total"].values()) == n_total)
print(f"       实检实数: {e['actual_total']}")
print("\n全部冒烟检查通过 ✔")
