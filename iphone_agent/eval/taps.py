"""点击的结果统计：点完没变、没变后原样重点、变了之后按返回、带预期的比例、预期核对的分布。

只读 runs/ 留档，不碰设备、不调模型。口径与 docs/superpowers/specs/2026-09-11-点击预期核对-design.md §0 一致；
以后改点击相关的东西都用它前后对比：

    python -m iphone_agent.eval.taps <runs 目录> [--since 20260911]

⚠ expect 在留档里记在每步的 model.expect 下，不在 action 里。2026-09-11 第一次数的时候查错了位置，
  得出「一次都没写过」，实际是 525 次点击里 51 次（9.7%）。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

BACK_MARKS = ("返回", "＜", "<", "‹")


def _steps(run_dir: Path) -> list[dict]:
    try:
        lines = (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(s, dict) and s.get("action"):
            out.append(s)
    return out


def _args(s: dict) -> dict:
    return (s.get("action") or {}).get("args") or {}


def _tap_text(s: dict) -> str | None:
    eid = _args(s).get("id")
    if eid is None:
        return None
    for e in (s.get("observation") or {}).get("elements") or []:
        if e.get("id") == eid:
            return e.get("text")
    return None


def _executed_tap(s: dict) -> bool:
    return ((s.get("action") or {}).get("name") == "tap"
            and not (s.get("validation") or s.get("rejected"))
            and bool((s.get("result") or {}).get("ok")))


def _same_target(a: dict, b: dict) -> bool:
    if (b.get("action") or {}).get("name") != "tap":
        return False
    ia, ib = _args(a).get("id"), _args(b).get("id")
    if ia is not None and ia == ib:
        return True
    ta = _tap_text(a)
    return bool(ta) and _tap_text(b) == ta


def _is_back(s: dict) -> bool:
    a = s.get("action") or {}
    if a.get("name") == "key" and _args(s).get("name") == "back":
        return True
    t = _tap_text(s) if a.get("name") == "tap" else None
    return bool(t) and any(m in t for m in BACK_MARKS)


def tap_outcomes(runs_root: Path, since: str | None = None) -> dict:
    c: Counter = Counter()
    checks: Counter = Counter()
    for d in sorted(Path(runs_root).iterdir()):
        if not d.is_dir() or not (d / "run.json").exists():
            continue
        if since and d.name[:len(since)] < since:
            continue
        c["runs"] += 1
        records = _steps(d)
        # ⚠ 剧本子步骤（kind=procedure_step）不是模型的点击：没有 model 键、expect 为 None，执行层给它们
        #   写 expect_check.by=missing。算进来会拉低「带预期」、抬高 missing（终审 2026-09-11）。
        #   单独计数，也不进「2 步内」窗口 —— 窗口看的是模型下一步做了什么。
        steps = [s for s in records if s.get("kind") != "procedure_step"]
        c["procedure_taps"] += sum(1 for s in records if s.get("kind") == "procedure_step" and _executed_tap(s))
        for i, s in enumerate(steps):
            if not _executed_tap(s):
                continue
            res = s.get("result") or {}
            nxt = steps[i + 1:i + 3]
            c["taps"] += 1
            if (s.get("model") or {}).get("expect"):
                c["with_expect"] += 1
            ec = res.get("expect_check")
            if isinstance(ec, dict):
                checks[f"{ec.get('by')}:{ec.get('met')}"] += 1
            if res.get("changed"):
                c["changed"] += 1
                if (res.get("judged") or {}).get("on_change"):
                    c["vision_on_change"] += 1      # 变了之后看了一次图：Task 6 看得见看图成本
                if any(_is_back(n) for n in nxt):
                    c["back_after_change_2"] += 1
            elif res.get("changed") is False:
                c["unchanged"] += 1
                if any(_same_target(s, n) for n in nxt):
                    c["repeat_after_unchanged_2"] += 1
    return {"counts": dict(c), "expect_check": dict(checks)}


def _rate(n: int, d: int) -> str:
    return f"{n}/{d}（{n / d:.1%}）" if d else f"{n}/0"


def render(stats: dict) -> list[str]:
    c = stats["counts"]
    taps, changed, unchanged = c.get("taps", 0), c.get("changed", 0), c.get("unchanged", 0)
    lines = [f"# {c.get('runs', 0)} 次运行 · 执行的点击 {taps}",
             f"  点完没变          {_rate(unchanged, taps)}",
             f"  没变后原样重点    {_rate(c.get('repeat_after_unchanged_2', 0), unchanged)}",
             f"  变了之后按返回    {_rate(c.get('back_after_change_2', 0), changed)}",
             f"  带预期            {_rate(c.get('with_expect', 0), taps)}",
             f"  变了之后看图      {_rate(c.get('vision_on_change', 0), changed)}"]
    if c.get("procedure_taps"):
        lines.append(f"  剧本内点击（未计入上面） {c['procedure_taps']}")
    if stats["expect_check"]:
        lines.append("  预期核对：" + "，".join(f"{k} {v}" for k, v in sorted(stats["expect_check"].items())))
    return lines


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("用法: python -m iphone_agent.eval.taps <runs 目录> [--since 20260911]", file=sys.stderr)
        return 2
    since = argv[argv.index("--since") + 1] if "--since" in argv else None
    for line in render(tap_outcomes(Path(argv[0]), since)):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
