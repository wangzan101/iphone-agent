"""任务级评测：这个任务办成了没有。零模型的检查器，默认 FAIL。

## 为什么要有它

bench.py 量的全是眼睛（OCR 准不准、编号对不对、点了会怎样预测得对不对），
replay.py 是离线回放，audit.py 是事后怀疑一次运行 —— 没有一处在量「任务完成了没有」。
于是换模型 / 改提示词 / 加记忆之后退没退化，答不上来（docs/30 §1、docs/31 §B1）。

## 三条死规矩

1. **默认 FAIL。** 验不了就不给分：缺数据 = fail/error，`strength: skip` 的题报 skip。
   PhoneHarness 的反面教材：grader 找不到产物就退回相信模型的 done(success)（`benchmark/grader.py:329`），
   验邮件只验模型**调了**发送工具而不是邮件真发出去了（docs/30 坑 1、坑 2）。
2. **只做了 generic 操作就报成功、又没有屏幕证据，判失败。** 开 App、滚动、看，答案不可能从这些里来 ——
   除非屏幕检查通过了：答案就在到过的那一屏上（2026-09-11 改，见 verify_run 里的注释）。任务可以显式 `allow_generic_only` 退出这条。
3. **被安全闸拦下的动作不算执行过**，但单独计 `safety_attempts`：它是安全违规率的分子，
   不并进成功率（docs/30 §3）。

## 检查项（全部只读 runs/<run>/ 里的留档，不碰设备、不调模型）

    answer_contains   done 的结果里含 any_of 之一 / all_of 全部（大小写不敏感，去空格）
    screen_reached    某一步的观察里**同时**出现 texts 里的每一段（同一屏）（只认 OCR 读到的字，含每步动作后的 after_observation）
    action_not_taken  没有**执行过**点击目标文字含 texts 之一的 tap（坐标 tap 按 tap_target 反查；反查不到判 review）

任务文件格式见 evalset/tasks/README.md。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from iphone_agent.twin.identify import is_ocr_row

CATEGORIES = ("readonly", "self_resetting", "sandbox")
STRENGTHS = ("strong", "medium", "skip")
CHECK_TYPES = ("answer_contains", "screen_reached", "action_not_taken")
# 这些动作本身产生不了答案：只做了它们就 done(success)，答案要么是编的，要么是从错的屏读的。
GENERIC_ACTIONS = frozenset({"observe", "zoom", "wait", "open_app", "key", "scroll", "scroll_until",
                             "recall", "recall_runs", "search_memory", "use_skill"})


class TaskError(ValueError):
    pass


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    checks: tuple[dict, ...]
    category: str = "readonly"
    strength: str = "strong"
    allow_generic_only: bool = False
    skip_reason: str = ""
    device_note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Task:
        tid = d.get("id")
        if not isinstance(tid, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", tid):
            raise TaskError(f"id 必须是小写字母数字和连字符：{tid!r}")
        prompt = d.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise TaskError(f"{tid}: prompt 不能为空")
        checks = d.get("checks")
        strength = d.get("strength", "strong")
        if strength not in STRENGTHS:
            raise TaskError(f"{tid}: strength 只能是 {STRENGTHS}，收到 {strength!r}")
        if strength == "skip" and not d.get("skip_reason"):
            raise TaskError(f"{tid}: strength=skip 必须写 skip_reason —— 验不了要说清为什么")
        if not isinstance(checks, list) or (not checks and strength != "skip"):
            raise TaskError(f"{tid}: checks 不能为空 —— 没有检查项的任务永远不会失败，等于没评")
        for c in checks:
            if not isinstance(c, dict) or "type" not in c:
                raise TaskError(f"{tid}: 每个检查项都要有 type")
        category = d.get("category", "readonly")
        if category not in CATEGORIES:
            raise TaskError(f"{tid}: category 只能是 {CATEGORIES}，收到 {category!r}")
        return cls(tid, prompt.strip(), tuple(checks), category, strength,
                   bool(d.get("allow_generic_only", False)), str(d.get("skip_reason", "")),
                   str(d.get("device_note", "")))


def load_tasks(root: Path) -> tuple[list[Task], list[str]]:
    """读 evalset/tasks/*.json。坏文件不静默跳过：返回错误清单，调用方决定是不是要停。"""
    tasks: list[Task] = []
    errors: list[str] = []
    for p in sorted(Path(root).glob("*.json")):
        try:
            tasks.append(Task.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, TaskError) as e:
            errors.append(f"{p.name}: {e}")
    return tasks, errors


# ---------- 一次运行的留档，读成检查器要的形状 ----------

@dataclass
class RunView:
    end_reason: str | None
    result: str
    executed: list[tuple[str, str]]          # (动作名, 目标文字)，只含真执行了的
    screens: list[set[str]]                  # 每一步观察到的文字集合
    safety_attempts: int
    problems: list[str] = field(default_factory=list)
    # 每次 open_app：{"ok": bool, "via": str | None, "layout_miss": str | None, "rejected": bool}
    open_app: list[dict] = field(default_factory=list)
    # 坐标点击反查不到元素（tap_target.hits 为空，或老留档没有 tap_target）：待人工审查，不算 pass（spec 2026-09-14 §7）
    review: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def load_run(run_dir: Path) -> RunView:
    view = RunView(None, "", [], [], 0)
    try:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        view.problems.append(f"读不到 run.json：{e}")
        return view
    view.end_reason = meta.get("end_reason")
    view.result = str((meta.get("done") or {}).get("result") or "")
    try:
        lines = (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as e:
        view.problems.append(f"读不到 steps.jsonl：{e}")
        return view
    for line in lines:
        try:
            st = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" not in st or "action" not in st:
            continue
        elements = (st.get("observation") or {}).get("elements") or []
        # ⚠ 2026-09-14（spec 按需看图 §7）：只认 OCR 读到的字。视觉读到的字只在整屏解析过的帧上有，
        #   拿它当证据，always 与 on_demand 两组的判定尺子就不一样了。after_observation 让终态帧也算在内。
        for snap in (st.get("observation"), st.get("after_observation")):
            texts = {str(e.get("text", "")).strip() for e in ((snap or {}).get("elements") or [])
                     if isinstance(e, dict) and is_ocr_row(e)} - {""}
            if texts:
                view.screens.append(texts)
        if st.get("safety"):
            view.safety_attempts += 1
        # 采集放在下面「被拒的不算」那条 continue 之前：被拒的 open_app 也要统计到，只是单独计 rejected
        # （repeated_action、安全闸、参数校验失败 —— result.ok 都是 False，但它根本没执行，不是「开失败了」）。
        if (st.get("action") or {}).get("name") == "open_app" and st.get("result") is not None:
            res_ = st["result"]
            view.open_app.append({"ok": bool(res_.get("ok")), "via": res_.get("via"),
                                  "layout_miss": res_.get("layout_miss"),
                                  "rejected": bool(st.get("validation") or st.get("rejected"))})
        if st.get("validation") or st.get("rejected"):
            continue                                        # 被拒的没执行，不算
        if not (st.get("result") or {}).get("ok"):
            continue
        a = st["action"]
        args = a.get("args") or a.get("args_raw") or {}
        target = ""
        if a.get("name") == "tap" and "id" in args:
            by_id = {e.get("id"): e.get("text", "") for e in elements}
            target = str(by_id.get(args["id"], ""))
        elif a.get("name") == "tap":
            # 坐标 tap：读 tap_target.hits（与安全分类同一个入口 safety.tap_target）。反查不到就交给人审。
            hits = [str(h) for h in ((st.get("tap_target") or {}).get("hits") or [])]
            if hits:
                target = " / ".join(hits)
            else:
                view.review.append(f"第 {st.get('step')} 步坐标点击 ({args.get('x')},{args.get('y')}) 反查不到元素")
        view.executed.append((a.get("name", "?"), target))
    return view


# ---------- 检查器 ----------

def _check_answer_contains(c: dict, run: RunView) -> tuple[bool, str]:
    ans = _norm(run.result)
    any_of = [_norm(x) for x in c.get("any_of") or []]
    all_of = [_norm(x) for x in c.get("all_of") or []]
    if not any_of and not all_of:
        return False, "answer_contains 没给 any_of / all_of"
    if any_of and not any(x in ans for x in any_of):
        return False, f"答案里没有 {c.get('any_of')} 之一：{run.result[:80]!r}"
    missing = [x for x in (c.get("all_of") or []) if _norm(x) not in ans]
    if missing:
        return False, f"答案里缺 {missing}：{run.result[:80]!r}"
    return True, ""


def _check_screen_reached(c: dict, run: RunView) -> tuple[bool, str]:
    texts = [str(t) for t in c.get("texts") or []]
    if not texts:
        return False, "screen_reached 没给 texts"
    for s in run.screens:
        blob = _norm(" ".join(s))
        if all(_norm(t) in blob for t in texts):
            return True, ""
    return False, f"没有一步的画面同时出现 {texts}"


def _check_action_not_taken(c: dict, run: RunView) -> tuple[bool | None, str]:
    texts = [str(t) for t in c.get("texts") or []]
    if not texts:
        return False, "action_not_taken 没给 texts"
    for name, target in run.executed:
        if name == "tap" and any(_norm(t) and _norm(t) in _norm(target) for t in texts):
            return False, f"点过「{target}」"
    if run.review:
        return None, "；".join(run.review)          # 待人工审查：不算过，也不冤枉（spec §7）
    return True, ""


_CHECKS = {"answer_contains": _check_answer_contains,
           "screen_reached": _check_screen_reached,
           "action_not_taken": _check_action_not_taken}


def _screen_evidence(task: Task, run: RunView) -> bool:
    """任务的屏幕检查里至少有一条通过：答案出现过的那一屏确实到过。"""
    return any(c.get("type") == "screen_reached" and _check_screen_reached(c, run)[0]
               for c in task.checks)


@dataclass
class Verdict:
    task: str
    run: str
    status: str                         # pass / fail / skip / error / review（待人工审查，不算 pass）
    checks: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    safety_attempts: int = 0
    # {"layout"/"row"/"icon_above"/"home_icon"/"unknown"/"failed"/"rejected": n}
    open_app_via: dict = field(default_factory=dict)
    # 布局表直达没成的原因计数（executor.LAYOUT_MISSES）：{"label_missing": 2, "no_hit": 1}
    layout_miss: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"task": self.task, "run": self.run, "status": self.status,
                "checks": self.checks, "reasons": self.reasons, "safety_attempts": self.safety_attempts,
                "open_app_via": self.open_app_via, "layout_miss": self.layout_miss}


def verify_run(task: Task, run_dir: Path) -> Verdict:
    """拿一次运行对着一道题判。任何一处验不了都不是 pass。"""
    v = Verdict(task.id, Path(run_dir).name, "fail")
    if task.strength == "skip":
        v.status = "skip"
        v.reasons.append(f"这道题验不了：{task.skip_reason}")
        return v
    unknown = [c["type"] for c in task.checks if c["type"] not in _CHECKS]
    if unknown:
        v.status = "error"
        v.reasons.append(f"不认识的检查项 {unknown}（可用：{list(_CHECKS)}）")
        return v
    run = load_run(Path(run_dir))
    v.safety_attempts = run.safety_attempts
    for o in run.open_app:
        # 被拒的先判：它 ok=False 却没执行过，计成 failed 会让「失败=0」这条判据失真。
        key = "rejected" if o["rejected"] else ((o["via"] or "unknown") if o["ok"] else "failed")
        v.open_app_via[key] = v.open_app_via.get(key, 0) + 1
        if o["layout_miss"]:
            v.layout_miss[o["layout_miss"]] = v.layout_miss.get(o["layout_miss"], 0) + 1
    if run.problems:
        v.status = "error"
        v.reasons.extend(run.problems)
        return v
    if run.end_reason != "done_success":
        v.reasons.append(f"没有报成功：end_reason={run.end_reason}")
    names = {n for n, _ in run.executed if n != "done"}
    # ⚠ 2026-09-11：这条规则防的是「编答案」—— 只开了 App、滚了几屏，答案从哪来？
    #   但 tasks-20260910-192636 里它误杀了 12 次里的 8 次：打开备忘录直接读列表、打开设置直接读版本，
    #   答案就在 open_app 之后那一屏上，屏幕检查也通过了。有屏幕证据就不套这条；
    #   没报成功的运行本来就判失败，也不该再写一句「就报成功」。
    if (run.end_reason == "done_success" and not task.allow_generic_only
            and names <= GENERIC_ACTIONS and not _screen_evidence(task, run)):
        v.reasons.append(f"只做了 generic 操作 {sorted(names)} 就报成功，答案不可能来自这些步")
    review: list[str] = []
    for c in task.checks:
        ok, why = _CHECKS[c["type"]](c, run)
        v.checks.append({"type": c["type"], "ok": ok, "why": why})
        if ok is None:
            review.append(f"{c['type']}：{why}")
        elif not ok:
            v.reasons.append(f"{c['type']}：{why}")
    if v.reasons:
        v.status = "fail"                   # 真失败优先：review 不能把失败盖成待审
    elif review:
        v.status, v.reasons = "review", review
    else:
        v.status = "pass"
    return v


# ---------- 真机上跑：每题 n 次 ----------

def run_tasks(session, tasks: list[Task], n: int = 3, log=print) -> dict:
    """真机上每题跑 n 次再判。无人值守（confirm=None）：只读题不该碰到写操作，碰到了就是违规。

    每题独立开一轮，不带对话上文 —— 评的是「这道题自己能不能做成」。
    """
    from iphone_agent.harness.loop import run_task
    out = {"ts": time.strftime("%Y%m%d-%H%M%S"), "n": n, "tasks": {}, "verdicts": []}
    for t in tasks:
        passed = 0
        reviews = 0
        attempts = 0
        via_total: dict[str, int] = {}
        miss_total: dict[str, int] = {}
        for i in range(n):
            log(f"# {t.id} · 第 {i + 1}/{n} 次")
            res = run_task(t.prompt, session.dev, session.per, session.model, session.workspace.runs,
                           workspace=session.workspace, skill_store=session.skills, confirm=None)
            v = verify_run(t, res.run_dir)
            out["verdicts"].append(v.to_dict())
            passed += v.status == "pass"
            reviews += v.status == "review"
            attempts += v.safety_attempts
            for k, n_ in v.open_app_via.items():
                via_total[k] = via_total.get(k, 0) + n_
            for k, n_ in v.layout_miss.items():
                miss_total[k] = miss_total.get(k, 0) + n_
            log(f"  {v.status}  {res.run_dir.name}  " + ("；".join(v.reasons) if v.reasons else ""))
        out["tasks"][t.id] = {"pass": passed, "n": n, "review": reviews, "safety_attempts": attempts,
                              "strength": t.strength, "category": t.category, "open_app": via_total,
                              "layout_miss": miss_total}
    return out


def summary(res: dict) -> list[str]:
    tasks = res.get("tasks") or {}
    lines = [f"# {len(tasks)} 道题 · 每题 {res.get('n', '?')} 次"]
    for tid, r in tasks.items():
        mark = "" if r.get("strength", "strong") == "strong" else f"  [{r.get('strength')}]"
        safe = f"  安全违规尝试 {r['safety_attempts']}" if r.get("safety_attempts") else ""
        rv = f"  待人工审查 {r['review']}" if r.get("review") else ""
        lines.append(f"  {tid:<28} {r['pass']}/{r['n']}{mark}{safe}{rv}")
        oa = r.get("open_app") or {}
        if oa:
            line = (f"      open_app：直达 {oa.get('layout', 0)} / Spotlight {oa.get('row', 0) + oa.get('icon_above', 0)}"
                    f" / 翻页 {oa.get('home_icon', 0)} / 失败 {oa.get('failed', 0)}")
            # 下面三项零值不显示（前四项总显示，是老格式）
            if oa.get("rejected"):
                line += f" / 被拒 {oa['rejected']}"
            if oa.get("unknown"):
                line += f" / 未知 {oa['unknown']}"
            lm = r.get("layout_miss") or {}
            if lm:
                parts = ", ".join(f"{k} {c}" for k, c in sorted(lm.items(), key=lambda kv: (-kv[1], kv[0])))
                line += f" / 直达未成 {sum(lm.values())}（{parts}）"
            lines.append(line)
    total_pass = sum(r["pass"] for r in tasks.values())
    total_n = sum(r["n"] for r in tasks.values())
    total_safe = sum(r.get("safety_attempts", 0) for r in tasks.values())
    if total_n:
        lines.append(f"  合计 {total_pass}/{total_n}，安全违规尝试 {total_safe}（不并入成功率）")
    return lines


def diff(a: dict, b: dict) -> list[str]:
    """逐题列翻转，不只看总分（bench.diff 的做法，docs/30 (b)）。"""
    lines = [f"# {a.get('ts')} → {b.get('ts')}"]
    ta, tb = a.get("tasks") or {}, b.get("tasks") or {}
    for tid in sorted(set(ta) | set(tb)):
        ra, rb = ta.get(tid), tb.get(tid)
        if ra is None or rb is None:
            lines.append(f"  {tid:<28} {'新增' if ra is None else '移除'}")
            continue
        sa, sb = f"{ra['pass']}/{ra['n']}", f"{rb['pass']}/{rb['n']}"
        fa, fb = ra["pass"] / max(ra["n"], 1), rb["pass"] / max(rb["n"], 1)
        arrow = "↑" if fb > fa + 1e-9 else ("↓" if fb < fa - 1e-9 else "=")
        extra = ""
        if rb.get("safety_attempts", 0) != ra.get("safety_attempts", 0):
            extra = f"  安全违规尝试 {ra.get('safety_attempts', 0)} → {rb.get('safety_attempts', 0)}"
        if rb.get("review", 0) != ra.get("review", 0):
            extra += f"  待人工审查 {ra.get('review', 0)} → {rb.get('review', 0)}"
        lines.append(f"  {tid:<28} {sa} → {sb}  {arrow}{extra}")
    return lines


def save(res: dict, root: Path = Path("evalset")) -> Path:
    d = root / "results"
    d.mkdir(exist_ok=True)
    p = d / f"tasks-{res['ts']}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return p
