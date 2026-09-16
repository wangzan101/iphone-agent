"""学习曲线评测（spec 2026-09-12 §8）：同一道题连做几遍，开孪生提示 vs 不给任何位置提示。

两组各用一个全新工作区（只拷主屏布局表），记忆关、技能库空、无人值守（confirm=None）——
两组只差 RunConfig.location_hints，差异才归得到孪生上（codex 评审第 2、3 条）。
顺序 ABBA：上一次把 App 停在哪一屏，对两组的影响对称（§8.3）。
⚠ 三道题 × 五遍是案例证据，不是统计证明（§1）。这里只负责跑和记，结论写进 docs/34。
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from iphone_agent.workspace import RunConfig, Workspace

ARMS = ("on", "off")


def order(rep: int) -> tuple[str, str]:
    return ("on", "off") if rep % 2 == 1 else ("off", "on")


def prepare_workspaces(src: Workspace, root: Path) -> dict[str, Workspace]:
    out: dict[str, Workspace] = {}
    for arm in ARMS:
        ws = Workspace(Path(root) / arm)
        ws.twin_device.mkdir(parents=True, exist_ok=True)
        layout = src.twin_device / "layout.json"
        if layout.exists():
            shutil.copy2(layout, ws.twin_device / "layout.json")
        ws.runs.mkdir(parents=True, exist_ok=True)
        out[arm] = ws
    return out


def _read_json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return d if isinstance(d, dict) else {}


def read_step_records(run_dir: Path) -> list[dict]:
    """steps.jsonl 读成一条条动作记录（跳过坏行；`kind == procedure_step` 的技能步不算模型动作）。
    curve/ab 两处评测共用这一个读法，别各写一遍（controller m8）。"""
    recs = []
    try:
        for line in (Path(run_dir) / "steps.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and "action" in r and r.get("kind") != "procedure_step":
                recs.append(r)
    except (OSError, UnicodeDecodeError):
        pass
    return recs


def run_metrics(run_dir: Path) -> dict:
    run = _read_json(Path(run_dir) / "run.json")
    usage = run.get("usage") or {}
    tokens = usage.get("total_tokens") or (int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0))
    recs = read_step_records(run_dir)
    started, ended = run.get("started_at"), run.get("ended_at")
    first = (recs[0].get("observation") or {}).get("screen") if recs else None
    return {"steps": run.get("steps"), "tokens": tokens,
            "wall_s": round(ended - started, 1) if isinstance(started, (int, float)) and isinstance(ended, (int, float)) else None,
            "location_steps": sum(1 for r in recs if (r.get("hints") or {}).get("location")),
            "route_steps": sum(1 for r in recs if (r.get("hints") or {}).get("route")),
            "start_screen": (first or {}).get("name") if isinstance(first, dict) else None,
            "twin": (run.get("twin") or {}).get("record")}


def run_curve(session, tasks, reps: int, root: Path, *, runner=None, verifier=None, log=print) -> dict:
    from iphone_agent.skills.store import SkillStore
    if runner is None:
        from iphone_agent.harness.loop import run_task as runner
    if verifier is None:
        from iphone_agent.eval.verify import verify_run as verifier
    wss = prepare_workspaces(session.workspace, root)
    out = {"ts": time.strftime("%Y%m%d-%H%M%S"), "reps": reps, "root": str(root),
           "tasks": [t.id for t in tasks], "runs": []}
    for rep in range(1, reps + 1):
        for t in tasks:
            for arm in order(rep):
                ws = wss[arm]
                log(f"# {t.id} · 第 {rep}/{reps} 遍 · {arm}")
                # ⚠ 2026-09-12 终审 FR#4：twin_hints 默认读 IPHONE_TWIN_HINTS，关掉时开组会退回旧 screenmap
                #   提示（spec §8.2），两组差的就不止孪生了。两组都钉死 True，只让 location_hints 不同。
                rc = RunConfig(location_hints=(arm == "on"), memory=False, twin_hints=True)
                store = SkillStore(personal=ws.dot / "skills", shared=ws.dot / "skills-shared")
                res = runner(t.prompt, session.dev, session.per, session.model, ws.runs,
                             workspace=ws, skill_store=store, confirm=None, run_config=rc)
                v = verifier(t, res.run_dir)
                out["runs"].append({"rep": rep, "task": t.id, "arm": arm, "run": str(res.run_dir),
                                    "status": v.status, "reasons": list(getattr(v, "reasons", []) or []),
                                    "metrics": run_metrics(res.run_dir)})
                log(f"  {v.status}  {Path(res.run_dir).name}")
    return out


def table(res: dict) -> list[str]:
    lines = ["| 题 | 遍 | 开：过/步/token/秒/起点 | 关：过/步/token/秒/起点 | 步数差（开−关） |",
             "|---|---|---|---|---|"]
    by = {(r["task"], r["rep"], r["arm"]): r for r in res.get("runs", [])}
    for tid in res.get("tasks", []):
        for rep in range(1, int(res.get("reps") or 0) + 1):
            cells, steps = [], {}
            for arm in ARMS:
                r = by.get((tid, rep, arm))
                if r is None:
                    cells.append("-")
                    continue
                m = r["metrics"]
                steps[arm] = m.get("steps")
                cells.append(f"{r['status']}/{m.get('steps')}/{m.get('tokens')}/{m.get('wall_s')}/{m.get('start_screen') or '-'}")
            diff = (steps["on"] - steps["off"]) if all(isinstance(steps.get(a), int) for a in ARMS) else "-"
            lines.append(f"| {tid} | {rep} | {cells[0]} | {cells[1]} | {diff} |")
    return lines


def save(res: dict, results: Path) -> Path:
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    p = results / f"curve-{res['ts']}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    p.with_suffix(".md").write_text("\n".join(table(res)) + "\n", encoding="utf-8")
    return p
