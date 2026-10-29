r"""训练监控看板：读 ``logs3/`` 的实时日志，起一个本地网页随时看进度。

为什么用它
----------
训练要跑十几个小时，没人愿意一直盯着 PowerShell 刷屏。这个脚本只做三件事：

1. 读 ``logs3/fold{N}/live.jsonl``（``train_3class.py`` 每 ``log_every`` 步追加一行）；
2. 顺带读 ``checkpoints3/*/history.json`` 与 ``summary.json`` 做兜底与汇总；
3. 起一个标准库 HTTP 服务，浏览器打开就能看曲线、ETA、GPU 占用。

零第三方依赖（只用 stdlib），不用装 flask / tensorboard；页面不引 CDN，
断网也能打开。

用法
----
.. code-block:: bash

    # 与训练同一个项目目录下，另开一个窗口
    python monitor.py                     # 默认 http://0.0.0.0:8787

    # 换端口 / 只允许本机访问 / 改日志目录
    python monitor.py --port 9000 --host 127.0.0.1
    python monitor.py --log-dir logs3 --out-dir checkpoints3

浏览器访问 ``http://<实验室机器IP>:8787``。远程看不了内网时，在自己电脑上转发：

.. code-block:: bash

    ssh -L 8787:127.0.0.1:8787 用户名@实验室机器IP
    # 然后本机浏览器打开 http://127.0.0.1:8787

注意
----
这是内网工具，没有鉴权。用 ``--host 127.0.0.1`` 可以限制成只有本机能看
（配合上面的 ssh 转发最安全）；绑 ``0.0.0.0`` 时至少要确认所在网段可信，
且 Windows 防火墙需要放行该端口：

.. code-block:: powershell

    netsh advfirewall firewall add rule name="ToxPath Monitor" `
      dir=in action=allow protocol=TCP localport=8787
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
BASELINE_DICE: float = 0.835
"""全部预测为空的基线分：本地 83.5% 的 patch 真值全空，画面里拿它当参照线。"""

STALL_SECONDS: float = 300.0
"""超过这个时间没有新日志，就认为训练停了（或卡住）。"""

SCALAR_METRICS: tuple[str, ...] = ("loss", "dice_c1", "dice_c2", "dice_fg",
                                   "iou_fg", "pixel_acc")


def _json_safe(value: Any) -> Any:
    """把 NaN / ±inf 换成 ``None``（递归）。

    老日志里可能留着 ``NaN`` 字面量（Python 的 json 默认允许），
    原样转给浏览器会让 ``JSON.parse`` 整页报错，这里统一洗成 null。
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# 读日志
# --------------------------------------------------------------------------- #
def read_live(path: Path) -> dict[str, Any] | None:
    """解析一个 ``live.jsonl``，返回 {start, steps, epochs, done}。

    文件可能正在被追加，末行写到一半属于正常情况：解析失败的行直接跳过。
    """
    if not path.is_file():
        return None

    start: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    done: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            event: str = str(rec.get("event", ""))
            if event == "start":
                start = rec
            elif event == "step":
                steps.append(rec)
            elif event == "epoch":
                epochs.append(rec)
            elif event == "done":
                done = rec

    if start is None and not steps and not epochs:
        return None
    return {"start": start, "steps": steps, "epochs": epochs, "done": done}


def eta_of(start: dict[str, Any] | None,
           steps: list[dict[str, Any]],
           epochs: list[dict[str, Any]],
           now: float) -> float | None:
    """估算该折剩余秒数：本 epoch 剩余步数 + 后续 epoch。

    用最近一个 step 的 ``it_s`` 推本 epoch 余量；用已完成 epoch 的平均耗时
    推后续 epoch（没有已完成 epoch 时退化成纯步数推算）。
    """
    if not start or not steps:
        return None
    total_steps: int = int(steps[-1].get("total", 0) or 0)
    done_steps: int = int(steps[-1].get("step", 0) or 0)
    it_s: float = float(steps[-1].get("it_s", 0.0) or 0.0)
    if total_steps <= 0 or it_s <= 0:
        return None

    remain_in_epoch: float = max(total_steps - done_steps, 0) * it_s

    epochs_cfg: int = int(start.get("epochs", 0) or 0)
    cur_epoch: int = int(steps[-1].get("epoch", 1) or 1)
    remain_epochs: int = max(epochs_cfg - cur_epoch, 0)

    completed: list[float] = [float(e.get("minutes", 0.0) or 0.0) * 60.0
                              for e in epochs]
    completed = [c for c in completed if c > 0]
    per_epoch: float = (sum(completed) / len(completed) if completed
                        else total_steps * it_s)
    return remain_in_epoch + remain_epochs * per_epoch


def read_folds(log_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    """汇总所有折：优先看 live.jsonl，没有就退回到 history.json。"""
    now: float = time.time()
    folds: list[dict[str, Any]] = []

    fold_dirs: list[Path] = sorted(
        log_dir.glob("fold*"),
        key=lambda p: int("".join(c for c in p.name if c.isdigit()) or 0),
    )
    for fd in fold_dirs:
        fold: int = int("".join(c for c in fd.name if c.isdigit()) or 0)
        live_path: Path = fd / "live.jsonl"
        parsed: dict[str, Any] | None = read_live(live_path)

        if parsed is None:
            # 老格式兜底：只有每 epoch 的 history.json
            hist_path: Path = out_dir / f"fold{fold}" / "history.json"
            if not hist_path.is_file():
                continue
            try:
                rows: list[dict[str, Any]] = json.loads(
                    hist_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            parsed = {"start": None, "steps": [], "done": None,
                      "epochs": [{"event": "epoch", "epoch": r.get("epoch"),
                                  "train_loss": r.get("loss"),
                                  **{k: r.get(k) for k in SCALAR_METRICS}}
                                 for r in rows]}

        mtime: float = live_path.stat().st_mtime if live_path.is_file() else 0.0
        age: float = now - mtime if mtime else 0.0

        if parsed["done"] is not None:
            status: str = "done"
        elif parsed["steps"] or parsed["epochs"]:
            status = "running" if age <= STALL_SECONDS else "stalled"
        else:
            status = "waiting"

        folds.append({
            "fold": fold,
            "status": status,
            "age_s": round(age, 1),
            "start": parsed["start"],
            "steps": parsed["steps"],
            "epochs": parsed["epochs"],
            "done": parsed["done"],
            "eta_s": None if status != "running" else
                     eta_of(parsed["start"], parsed["steps"], parsed["epochs"], now),
        })
    return folds


def gpu_stats() -> dict[str, Any] | None:
    """查一次 nvidia-smi；没有显卡或命令不存在时返回 None。"""
    cmd: list[str] = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    text: str = (proc.stdout or "").strip()
    if proc.returncode != 0 or not text:
        return None
    parts: list[str] = [p.strip() for p in text.splitlines()[0].split(",")]
    if len(parts) < 5:
        return None
    try:
        return {
            "name": parts[0],
            "util": float(parts[1]),
            "mem_used": int(float(parts[2])),
            "mem_total": int(float(parts[3])),
            "temp": int(float(parts[4])),
        }
    except ValueError:
        return None


def build_state(log_dir: Path, out_dir: Path) -> dict[str, Any]:
    """拼出 ``/api/state`` 的完整响应。"""
    folds: list[dict[str, Any]] = read_folds(log_dir, out_dir)

    summary: list[dict[str, Any]] = []
    summary_path: Path = out_dir / "summary.json"
    if summary_path.is_file():
        try:
            loaded: Any = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                summary = loaded
        except (json.JSONDecodeError, OSError):
            summary = []

    best: dict[str, Any] | None = None
    for f in folds:
        for e in f["epochs"]:
            value: Any = e.get("dice_fg")
            if isinstance(value, (int, float)) and value == value:  # 排除 nan
                if best is None or value > float(best["dice_fg"]):
                    best = {"fold": f["fold"], "epoch": e.get("epoch"),
                            "dice_fg": float(value)}

    return {
        "now": time.time(),
        "baseline": BASELINE_DICE,
        "stall_seconds": STALL_SECONDS,
        "folds": folds,
        "summary": summary,
        "best": best,
        "gpu": gpu_stats(),
    }


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
HTML_TEMPLATE: str = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ToxPath 训练监控</title>
<style>
  :root {
    --bg: #f6f7f9; --card: #ffffff; --line: #e5e7eb; --ink: #111827;
    --muted: #6b7280; --ok: #059669; --warn: #b45309; --bad: #b91c1c;
    --accent: #2563eb;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 20px 24px 40px; background: var(--bg); color: var(--ink);
    font: 14px/1.55 "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
  }
  h1 { font-size: 17px; margin: 0; letter-spacing: .2px; }
  .top { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px;
    border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid var(--line);
    background: #fff;
  }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
  .running .dot { background: var(--ok); animation: pulse 1.6s infinite; }
  .running { color: var(--ok); border-color: #a7f3d0; background: #ecfdf5; }
  .stalled .dot { background: var(--warn); }
  .stalled { color: var(--warn); border-color: #fde68a; background: #fffbeb; }
  .done { color: var(--accent); border-color: #bfdbfe; background: #eff6ff; }
  .done .dot { background: var(--accent); }
  @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: .35 } }
  .muted { color: var(--muted); font-size: 12px; }
  .cards { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin-bottom: 16px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
  .card .k { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
  .card .v { font-size: 20px; font-weight: 650; font-variant-numeric: tabular-nums; }
  .card .s { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); }
  .panel { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px 6px; }
  .panel h2 { font-size: 13px; margin: 0 0 6px; font-weight: 600; color: #374151; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 2px 0 4px; font-size: 12px; color: var(--muted); }
  .legend i { display: inline-block; width: 10px; height: 3px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }
  svg { display: block; width: 100%; height: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; }
  th, td { padding: 6px 8px; text-align: right; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 600; }
  .empty { color: var(--muted); padding: 28px 4px; text-align: center; font-size: 13px; }
  .delta-pos { color: #b91c1c; }  /* 涨：红 */
  .delta-neg { color: #047857; }  /* 跌：绿 */
</style>
</head>
<body>
  <div class="top">
    <h1>ToxPath 训练监控</h1>
    <span id="status" class="pill"><span class="dot"></span><span id="statusText">连接中</span></span>
    <span class="muted" id="stamp"></span>
  </div>

  <div class="cards" id="cards"></div>

  <div class="grid" style="margin-bottom:14px">
    <div class="panel">
      <h2>训练 loss（按 epoch 进度）</h2>
      <div class="legend" id="legendLoss"></div>
      <div id="chartLoss"></div>
    </div>
    <div class="panel">
      <h2>验证 dice_fg（虚线 = 全空基线 0.835）</h2>
      <div class="legend" id="legendDice"></div>
      <div id="chartDice"></div>
    </div>
  </div>

  <div class="panel" style="margin-bottom:14px">
    <h2>逐 epoch 明细</h2>
    <div id="tableHost"></div>
  </div>

  <div class="panel">
    <h2>多折汇总 summary.json</h2>
    <div id="summaryHost"></div>
  </div>

<script>
const REFRESH_MS = __REFRESH__;
const COLORS = ["#2563eb","#f59e0b","#059669","#dc2626","#7c3aed","#0891b2","#db2777"];
const state = { folds: [], baseline: 0.835 };

function fmtNum(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}
function fmtDur(sec) {
  if (sec === null || sec === undefined || !isFinite(sec) || sec < 0) return "—";
  const h = Math.floor(sec / 3600), m = Math.round((sec % 3600) / 60);
  if (h > 0) return h + " h " + m + " min";
  return Math.max(m, 1) + " min";
}
function color(i) { return COLORS[i % COLORS.length]; }

/* ---- 折的属性 ---- */
function foldMeta(f) {
  const st = f.start || {};
  const lastStep = f.steps.length ? f.steps[f.steps.length - 1] : null;
  const lastEpoch = f.epochs.length ? f.epochs[f.epochs.length - 1] : null;
  const total = lastStep ? lastStep.total : null;
  const doneStep = lastStep ? lastStep.step : 0;
  const frac = total ? doneStep / total : 0;
  const epochNo = lastStep ? lastStep.epoch : (lastEpoch ? lastEpoch.epoch : null);
  const epochsCfg = st.epochs || null;
  let overall = null;
  if (epochsCfg && epochNo) {
    overall = ((epochNo - 1) + frac) / epochsCfg;
  }
  let best = null;
  for (const e of f.epochs) {
    if (typeof e.dice_fg === "number" && !Number.isNaN(e.dice_fg)) {
      if (best === null || e.dice_fg > best) best = e.dice_fg;
    }
  }
  return { st, lastStep, lastEpoch, epochNo, epochsCfg, frac, overall, best };
}

function statusOf(f) {
  if (f.status === "done") return ["done", "已完成"];
  if (f.status === "running") return ["running", "训练中"];
  if (f.status === "stalled") return ["stalled", "无新日志 " + Math.round(f.age_s / 60) + " min"];
  return ["", "等待日志"];
}

/* ---- 卡片 ---- */
function renderCards() {
  const host = document.getElementById("cards");
  const folds = state.folds;
  if (!folds.length) {
    host.innerHTML = '<div class="card"><div class="k">状态</div><div class="v">等待日志</div>'
      + '<div class="s">训练启动后这里会自动出现</div></div>';
    return;
  }
  const active = folds.filter(f => f.status === "running").pop() || folds[folds.length - 1];
  const m = foldMeta(active);
  const st = m.st;
  const gpu = state.gpu;
  const best = state.best;

  const items = [
    ["当前折", "fold " + active.fold,
      (m.epochNo ? "epoch " + m.epochNo + (m.epochsCfg ? " / " + m.epochsCfg : "") : "—")
      + (m.lastStep ? " · step " + m.lastStep.step + " / " + m.lastStep.total : "")],
    ["本 epoch 进度", m.lastStep ? (m.frac * 100).toFixed(1) + "%" : "—",
      m.lastStep ? "总进度 " + (m.overall * 100).toFixed(1) + "%" : ""],
    ["速度", m.lastStep ? (1 / m.lastStep.it_s).toFixed(2) + " it/s" : "—",
      m.lastStep ? m.lastStep.it_s.toFixed(2) + " s/样本" : ""],
    ["预计剩余", fmtDur(active.eta_s),
      st ? st.image_size + " px · batch " + st.batch_size + "×" + st.grad_accum : ""],
    ["训练 loss", m.lastStep ? fmtNum(m.lastStep.loss) : "—",
      m.lastEpoch ? "epoch 均值 " + fmtNum(m.lastEpoch.train_loss) : ""],
    ["验证 dice_fg", m.lastEpoch ? fmtNum(m.lastEpoch.dice_fg) : "—",
      best ? "全局最佳 " + fmtNum(best.dice_fg) + "（fold " + best.fold + " / ep " + best.epoch + "）" : ""],
  ];
  if (gpu) {
    items.push(["GPU", gpu.util.toFixed(0) + "%",
      gpu.mem_used + " / " + gpu.mem_total + " MiB · " + gpu.temp + "°C"]);
  }
  host.innerHTML = items.map(([k, v, s]) =>
    '<div class="card"><div class="k">' + k + '</div><div class="v">' + v + '</div>'
    + (s ? '<div class="s">' + s + '</div>' : '') + '</div>').join("");
}

/* ---- 折线图（纯 SVG，无外部依赖） ---- */
function drawChart(hostId, series, opt) {
  const host = document.getElementById(hostId);
  const W = Math.max(host.clientWidth || 520, 320), H = opt.height || 230;
  const pad = { l: 52, r: 16, t: 12, b: 28 };
  const pts = series.flatMap(s => s.points);
  if (!pts.length) {
    host.innerHTML = '<div class="empty">等待日志…</div>';
    return;
  }
  let xmax = Math.max(...pts.map(p => p[0]), 0.001);
  let ymin = Math.min(...pts.map(p => p[1]));
  let ymax = Math.max(...pts.map(p => p[1]));
  if (opt.includeZero) ymin = Math.min(ymin, 0);
  if (opt.baseline !== undefined) ymin = Math.min(ymin, opt.baseline);
  if (ymax - ymin < 1e-6) { ymax += 0.1; ymin -= 0.1; }
  const padY = (ymax - ymin) * 0.08;
  ymin -= padY; ymax += padY;

  const X = v => pad.l + (v / xmax) * (W - pad.l - pad.r);
  const Y = v => H - pad.b - ((v - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);

  let svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" width="' + W + '" height="' + H + '">';
  /* 网格 + y 轴刻度 */
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const v = ymin + (ymax - ymin) * i / yTicks;
    const y = Y(v);
    svg += '<line x1="' + pad.l + '" y1="' + y + '" x2="' + (W - pad.r) + '" y2="' + y
      + '" stroke="#f1f3f5" stroke-width="1"/>';
    svg += '<text x="' + (pad.l - 8) + '" y="' + (y + 4) + '" text-anchor="end" '
      + 'font-size="11" fill="#6b7280">' + v.toFixed(opt.yDigits === undefined ? 2 : opt.yDigits) + '</text>';
  }
  /* x 轴刻度 */
  const xTicks = Math.min(6, Math.max(2, Math.round(xmax)));
  for (let i = 0; i <= xTicks; i++) {
    const v = xmax * i / xTicks;
    const x = X(v);
    svg += '<text x="' + x + '" y="' + (H - 8) + '" text-anchor="middle" font-size="11" fill="#6b7280">'
      + (opt.xLabel ? opt.xLabel(v) : v.toFixed(1)) + '</text>';
  }
  /* 基线 */
  if (opt.baseline !== undefined) {
    const y = Y(opt.baseline);
    svg += '<line x1="' + pad.l + '" y1="' + y + '" x2="' + (W - pad.r) + '" y2="' + y
      + '" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="5 4"/>';
    svg += '<text x="' + (W - pad.r) + '" y="' + (y - 6) + '" text-anchor="end" font-size="11" '
      + 'fill="#6b7280">基线 ' + opt.baseline.toFixed(3) + '</text>';
  }
  /* 折线 */
  series.forEach(s => {
    if (!s.points.length) return;
    const d = s.points.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join(" ");
    svg += '<path d="' + d + '" fill="none" stroke="' + s.color + '" stroke-width="1.8" '
      + 'stroke-linejoin="round" stroke-linecap="round"/>';
    if (s.points.length <= 40) {
      s.points.forEach(p => {
        svg += '<circle cx="' + X(p[0]).toFixed(1) + '" cy="' + Y(p[1]).toFixed(1)
          + '" r="2.6" fill="' + s.color + '"/>';
      });
    }
  });
  svg += '</svg>';
  host.innerHTML = svg;
}

function legend(hostId, series) {
  document.getElementById(hostId).innerHTML = series
    .map(s => '<span><i style="background:' + s.color + '"></i>' + s.name + '</span>').join("");
}

function renderCharts() {
  const lossSeries = state.folds.map((f, i) => ({
    name: "fold " + f.fold, color: color(i),
    points: f.steps.map(s => [s.epoch - 1 + s.step / (s.total || 1), s.loss])
  })).filter(s => s.points.length);
  const diceSeries = state.folds.map((f, i) => ({
    name: "fold " + f.fold, color: color(i),
    points: f.epochs.filter(e => typeof e.dice_fg === "number" && !Number.isNaN(e.dice_fg))
      .map(e => [e.epoch, e.dice_fg])
  })).filter(s => s.points.length);

  legend("legendLoss", lossSeries);
  legend("legendDice", diceSeries);
  drawChart("chartLoss", lossSeries, { yDigits: 3, xLabel: v => "ep " + (v + 1).toFixed(0) });
  drawChart("chartDice", diceSeries, { yDigits: 3, baseline: state.baseline, includeZero: true });
}

/* ---- 明细表 ---- */
function renderTable() {
  const rows = [];
  state.folds.forEach(f => f.epochs.forEach(e => rows.push({ fold: f.fold, e })));
  const host = document.getElementById("tableHost");
  if (!rows.length) { host.innerHTML = '<div class="empty">还没有完成任何 epoch</div>'; return; }
  rows.sort((a, b) => a.fold - b.fold || (a.e.epoch - b.e.epoch));
  let html = '<table><thead><tr><th>fold</th><th>epoch</th><th>train loss</th><th>val loss</th>'
    + '<th>dice_fg</th><th>dice_c1</th><th>dice_c2</th><th>pixel acc</th><th>耗时</th></tr></thead><tbody>';
  for (const r of rows) {
    const e = r.e;
    html += '<tr><td>' + r.fold + '</td><td>' + (e.epoch ?? "—") + '</td>'
      + '<td>' + fmtNum(e.train_loss) + '</td><td>' + fmtNum(e.loss) + '</td>'
      + '<td>' + fmtNum(e.dice_fg) + '</td><td>' + fmtNum(e.dice_c1) + '</td>'
      + '<td>' + fmtNum(e.dice_c2) + '</td><td>' + fmtNum(e.pixel_acc) + '</td>'
      + '<td>' + (e.minutes ? e.minutes.toFixed(1) + " min" : "—") + '</td></tr>';
  }
  host.innerHTML = html + '</tbody></table>';
}

function renderSummary() {
  const host = document.getElementById("summaryHost");
  const s = state.summary;
  if (!s || !s.length) { host.innerHTML = '<div class="empty">多折跑完才会写 summary.json</div>'; return; }
  let html = '<table><thead><tr><th>fold</th><th>验证切片</th><th>训练切片</th>'
    + '<th>best dice_fg</th><th>last dice_fg</th></tr></thead><tbody>';
  for (const r of s) {
    html += '<tr><td>' + r.fold + '</td><td>' + (r.val_slides || []).length + '</td>'
      + '<td>' + (r.n_train_slides ?? "—") + '</td>'
      + '<td>' + fmtNum(r.best_dice_fg) + '</td><td>' + fmtNum(r.last_dice_fg) + '</td></tr>';
  }
  host.innerHTML = html + '</tbody></table>';
}

function renderStatus() {
  const folds = state.folds;
  const running = folds.filter(f => f.status === "running");
  const el = document.getElementById("status");
  const txt = document.getElementById("statusText");
  if (running.length) {
    el.className = "pill running";
    txt.textContent = "训练中 · fold " + running.map(f => f.fold).join(",");
  } else if (folds.some(f => f.status === "done")) {
    el.className = "pill done";
    txt.textContent = "已完成";
  } else if (folds.length) {
    const [cls, label] = statusOf(folds[folds.length - 1]);
    el.className = "pill " + cls;
    txt.textContent = label;
  } else {
    el.className = "pill";
    txt.textContent = "没有日志";
  }
  document.getElementById("stamp").textContent =
    "更新于 " + new Date(state.now * 1000).toLocaleTimeString() + " · 每 " + (REFRESH_MS / 1000) + " s 自动刷新";
}

function render() {
  renderStatus();
  renderCards();
  renderCharts();
  renderTable();
  renderSummary();
}

async function tick() {
  try {
    const resp = await fetch("/api/state", { cache: "no-store" });
    const payload = await resp.json();
    Object.assign(state, payload);
    render();
  } catch (err) {
    document.getElementById("statusText").textContent = "连接断开";
  }
}

/* 首屏 + 定时刷新 */
(async function () {
  const resp = await fetch("/api/state", { cache: "no-store" });
  Object.assign(state, await resp.json());
  render();
  setInterval(tick, REFRESH_MS);
  window.addEventListener("resize", () => renderCharts());
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #
class MonitorHandler(BaseHTTPRequestHandler):
    """只处理两个路由：``/`` 返回页面，``/api/state`` 返回 JSON。"""

    log_dir: Path = Path("logs3")
    out_dir: Path = Path("checkpoints3")
    refresh_ms: int = 5000

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802（BaseHTTPRequestHandler 规定的名字）
        path: str = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            html: str = HTML_TEMPLATE.replace("__REFRESH__", str(self.refresh_ms))
            self._send(html.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/state":
            try:
                payload: dict[str, Any] = build_state(self.log_dir, self.out_dir)
            except Exception as exc:  # noqa: BLE001（看板不能因为解析失败就 500 挂掉）
                payload = {"now": time.time(), "folds": [], "error": f"{type(exc).__name__}: {exc}"}
            self._send(json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return

        self.send_error(404, "Not Found")

    def log_message(self, fmt: str, *args: Any) -> None:
        """静音访问日志：这个服务每分钟会被轮询十几次，刷屏没意义。"""
        return


def lan_addresses() -> list[str]:
    """列出本机可被局域网访问的 IP。"""
    ips: list[str] = []
    try:
        hostname: str = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip: str = str(info[4][0])
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except OSError:
        pass
    return ips


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ToxPath 训练监控看板（零依赖）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log-dir", type=Path, default=Path("logs3"),
                   help="train_3class.py 的 --log-dir")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints3"),
                   help="train_3class.py 的 --out-dir")
    p.add_argument("--host", type=str, default="0.0.0.0",
                   help="监听地址；只想本机看就填 127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--refresh", type=float, default=5.0,
                   help="页面自动刷新间隔（秒）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_dir: Path = args.log_dir.resolve()
    out_dir: Path = args.out_dir.resolve()

    MonitorHandler.log_dir = log_dir
    MonitorHandler.out_dir = out_dir
    MonitorHandler.refresh_ms = int(max(args.refresh, 1.0) * 1000)

    print(f"[monitor] 日志目录 {log_dir}")
    print(f"[monitor] 产物目录 {out_dir}")
    if not log_dir.is_dir():
        print(f"[warn   ] 日志目录还不存在，训练启动后会自动创建")

    server = ThreadingHTTPServer((args.host, args.port), MonitorHandler)
    print(f"[monitor] 本机   http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        for ip in lan_addresses():
            print(f"[monitor] 局域网 http://{ip}:{args.port}")
        print("[monitor] 别的网段访问不了时，用 ssh -L "
              f"{args.port}:127.0.0.1:{args.port} 用户名@机器IP 转发")
    print("[monitor] Ctrl+C 退出（不影响训练）")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[monitor] 已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
