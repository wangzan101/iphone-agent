"""按需看图的真机 A/B（spec 2026-09-14 §10.3）：always（A）对 on_demand（B）。只负责跑和记；
归因、孪生闸门的人工核对、切不切默认，写进本地报告 docs/35-按需看图AB.md。

两组各用一个全新工作区（只拷主屏布局表）、记忆关、技能库空、无人值守（confirm=None），装配沿用 eval/curve.py。
顺序 A B B A A B = curve.order 展开 3 遍。硬闸题多跑一层「同一模型、allow_coord_tap=False」。
⚠ 每组 3 遍是冒烟对照，不宣称统计显著；p90 只作参考。
⚠ 阶段计时会重叠（settle 把自己的 capture 算在里面），summary 只能把各阶段并排列出来对照，
   不能把几个阶段加成一个总数——那个总数没有意义（controller 裁决）。
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from iphone_agent.eval.curve import _read_json, order, prepare_workspaces, read_step_records
from iphone_agent.eval.verify import load_tasks
from iphone_agent.workspace import RunConfig

ARMS = {"on": "always", "off": "on_demand"}      # 沿用 curve 的工作区名：on 目录 = A 组（always）
REGULAR = ("settings-ios-version", "settings-storage-free", "notes-recent-titles", "reminders-list-counts")
# ⚠ 2026-09-16：硬闸题原来用 write/yimu-add-expense-lunch，真机预跑里跑不通 —— 记账页会记住上次没填完的
# 金额（起点不干净），而无人值守下「删除/退格」是危险词被拒，模型无路可退，把小键盘的「×」当成清空键，
# 金额变成 341×13111。换成同一个 App 的只读版 yimu-account-list：不写数据、起点稳定，考点不变
# （页面元素多、入口是无字图标、账户名就是当年「工资账户」那类列表项）。
HARD = (("vision", "yimu-account-list"), ("vision", "weather-city-list"), ("vision", "settings-camera-grid"))
LAYERS = ("as_is", "no_coord")


def load_ab_tasks(root: Path) -> tuple[list, list, list[str]]:
    """常规 7 题（顶层 4 道 + curve/ 3 道）与硬闸 3 题。按路径显式加载，和 write/ 同一个手法。"""
    root = Path(root)
    top, errors = load_tasks(root / "tasks")
    curve, e2 = load_tasks(root / "tasks" / "curve")
    errors += e2
    regular = [t for t in top if t.id in REGULAR] + curve
    hard = []
    for sub, tid in HARD:
        ts, e = load_tasks(root / "tasks" / sub)
        errors += e
        hard += [t for t in ts if t.id == tid]
    return regular, hard, errors


def ready(task) -> str | None:
    """两道新题补上 answer_contains 之前不开 A/B（spec §10.3）。"""
    if any(c.get("type") == "answer_contains" for c in task.checks):
        return None
    return f"{task.id} 还没有 answer_contains（首跑前人工确认答案后补上）"


class NoCoordModel:
    """同一模型、allow_coord_tap=False（spec §10.3 第二层）。只换这一个对象上的 resolved，不改用户的 config.toml。

    ⚠ 不是「工具表里没有坐标」——工具表永远来自 `tool_defs(True, …)`（harness/loop.py 里固定传 True），
    模型看到的 tap 参数一直有 x/y。`allow_coord_tap=False` 真正改的是另外两处：
    1. system_prompt 不提坐标这件事（harness/prompt.py），模型理论上不该主动去找 x/y；
    2. `validate_action` 直接拒绝坐标 tap（ValidationError），并计入 `MAX_CONSECUTIVE_REJECTIONS`（harness/loop.py）。
    也就是说这一层测的是「模型手边还有坐标字段、但一用就被拒」这条路能不能把任务做成，
    不是「模型压根看不到坐标参数」。"""

    def __init__(self, model):
        self._m = model
        rm = model.resolved
        self.resolved = dataclasses.replace(rm, model=dataclasses.replace(rm.model, allow_coord_tap=False))

    def decide(self, *a, **k):
        return self._m.decide(*a, **k)

    def __getattr__(self, name):
        return getattr(self._m, name)


def _pct(xs: list[int], p: float) -> int | None:
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def run_metrics(run_dir: Path) -> dict:
    """spec §10.3「记录」一节要的数，全部来自留档。steps.jsonl 的读法沿用 curve.read_step_records
    （controller m8：不再另写一遍）。"""
    run = _read_json(Path(run_dir) / "run.json")
    recs = read_step_records(run_dir)
    step_ms = sorted(r["timing"]["step_ms"] for r in recs if isinstance(r.get("timing"), dict))
    # 阶段计时会重叠（settle 把自己的 capture 算在里面），这里只按阶段名分别累加、并排看，绝不相加成一个总数。
    phases: dict[str, int] = {}
    for r in recs:
        for k, v in ((r.get("timing") or {}).get("phases") or {}).items():
            phases[k] = phases.get(k, 0) + int(v.get("ms") or 0)
    by_model = [(r.get("action") or {}).get("name") for r in recs if r.get("by") != "program"]
    per = run.get("perception") or {}
    return {"steps": run.get("steps"), "step_ms_p50": _pct(step_ms, 0.5), "step_ms_p90": _pct(step_ms, 0.9),
            "startup_timing": run.get("startup_timing"), "phases_ms": phases,
            "observe": by_model.count("observe"), "zoom": by_model.count("zoom"),
            "fallback_by": per.get("fallback_by"), "parse_by": per.get("parse_by"), "label": per.get("label"),
            "taps": run.get("taps"),
            "stale_element_id": sum(1 for r in recs if r.get("validation") == "stale_element_id")}


def run_ab(session, regular, hard, reps: int, root: Path, *, runner=None, verifier=None, log=print) -> dict:
    from iphone_agent.skills.store import SkillStore
    if runner is None:
        from iphone_agent.harness.loop import run_task as runner
    if verifier is None:
        from iphone_agent.eval.verify import verify_run as verifier
    wss = prepare_workspaces(session.workspace, root)
    plan = [(t, "as_is") for t in regular] + [(t, layer) for t in hard for layer in LAYERS]
    out = {"ts": time.strftime("%Y%m%d-%H%M%S"), "reps": reps, "root": str(root),
           "tasks": [t.id for t in [*regular, *hard]], "runs": []}
    for rep in range(1, reps + 1):
        for t, layer in plan:
            for arm in order(rep):
                mode, ws = ARMS[arm], wss[arm]
                log(f"# {t.id} · {layer} · 第 {rep}/{reps} 遍 · {mode}")
                # 两组只差 screen_parse：twin_hints 钉死 True（curve.py 终审 FR#4 同一个坑），location_hints 相同。
                rc = RunConfig(screen_parse=mode, memory=False, twin_hints=True, location_hints=True)
                model = session.model if layer == "as_is" else NoCoordModel(session.model)
                store = SkillStore(personal=ws.dot / "skills", shared=ws.dot / "skills-shared")
                res = runner(t.prompt, session.dev, session.per, model, ws.runs,
                             workspace=ws, skill_store=store, confirm=None, run_config=rc)
                v = verifier(t, res.run_dir)
                out["runs"].append({"rep": rep, "task": t.id, "layer": layer, "arm": mode, "run": str(res.run_dir),
                                    "status": v.status, "reasons": list(getattr(v, "reasons", []) or []),
                                    "metrics": run_metrics(res.run_dir)})
                log(f"  {v.status}  {Path(res.run_dir).name}")
    return out


def table(res: dict) -> list[str]:
    lines = ["| 题 | 层 | 遍 | 组 | 判定 | 步 | step p50/p90 ms | 阶段 ms（并排，不相加） | observe | zoom | 兜底 | 坐标点击（框外） |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.get("runs", []):
        m = r["metrics"]
        fb = sum((m.get("fallback_by") or {}).values())
        taps = m.get("taps") or {}
        phases = " ".join(f"{k}={v}" for k, v in (m.get("phases_ms") or {}).items()) or "-"
        lines.append(f"| {r['task']} | {r['layer']} | {r['rep']} | {r['arm']} | {r['status']} | {m.get('steps')} | "
                     f"{m.get('step_ms_p50')}/{m.get('step_ms_p90')} | {phases} | {m.get('observe')} | {m.get('zoom')} | {fb} | "
                     f"{taps.get('by_coord', 0)}（{taps.get('coord_unclassified', 0)}） |")
    return lines


def save(res: dict, results: Path) -> Path:
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    p = results / f"ab-{res['ts']}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    p.with_suffix(".md").write_text("\n".join(table(res)) + "\n", encoding="utf-8")
    return p
