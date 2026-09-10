"""任务级评测：这个任务办成了没有 —— 用零模型的检查器判，默认 FAIL。

以前 bench.py 量的全是眼睛（OCR 准不准、编号对不对），没有一处在量「任务完成了没有」；
换模型 / 改提示词 / 加记忆之后退没退化，答不上来（设计说明、设计说明）。
PhoneHarness 的教训：grader 找不到产物就退回相信模型的 done(success)（`grader.py:329`）——
所以这里的死规矩是**验不了就不给分**：缺数据 = fail/skip，绝不 = pass。
UI-Venus 的规则也搬过来（设计说明）：只做了 generic 操作（开 App、滚动、看）就报成功，判失败。
"""
import json

from iphone_agent.eval import verify as V

TASK = {
    "id": "settings-ios-version",
    "prompt": "打开设置，进「通用」再进「关于本机」，告诉我 iOS 版本号",
    "category": "readonly", "strength": "strong",
    "checks": [
        {"type": "answer_contains", "any_of": ["18.3.1"]},
        {"type": "screen_reached", "texts": ["关于本机", "iOS 版本"]},
        {"type": "action_not_taken", "texts": ["前往设置", "更新"]},
    ],
}


def _step(n, name, args, texts, ok=True, validation=None, safety=None):
    r = {"step": n, "ts": 0, "action": {"name": name, "args": args},
         "observation": {"elements": [{"id": i + 1, "text": t} for i, t in enumerate(texts)]},
         "result": {"ok": ok}}
    if validation:
        r["validation"] = validation
    if safety:
        r["safety"] = safety
    return r


def _run(tmp_path, name, end_reason, result, steps):
    d = tmp_path / "runs" / name
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "task": TASK["prompt"], "end_reason": end_reason,
        "done": None if end_reason not in ("done_success", "done_failed") else
        {"status": "success" if end_reason == "done_success" else "failed", "result": result}}), encoding="utf-8")
    (d / "steps.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in steps), encoding="utf-8")
    return d


GOOD = [
    _step(1, "open_app", {"name": "设置"}, ["设置", "通用"]),
    _step(2, "tap", {"id": 2}, ["通用", "关于本机"]),
    _step(3, "tap", {"id": 2}, ["关于本机", "iOS 版本", "18.3.1"]),
    _step(4, "done", {"status": "success", "result": "18.3.1"}, ["关于本机", "iOS 版本", "18.3.1"]),
]


def test_a_correct_run_passes(tmp_path):
    v = V.verify_run(V.Task.from_dict(TASK), _run(tmp_path, "a", "done_success", "iOS 18.3.1", GOOD))
    assert v.status == "pass", v.reasons
    assert all(c["ok"] for c in v.checks)


def test_wrong_answer_fails_even_if_the_screen_was_reached(tmp_path):
    v = V.verify_run(V.Task.from_dict(TASK), _run(tmp_path, "a", "done_success", "26.6.1", GOOD))
    assert v.status == "fail"
    assert any(c["type"] == "answer_contains" and not c["ok"] for c in v.checks)


def test_never_reaching_the_screen_fails(tmp_path):
    steps = [_step(1, "open_app", {"name": "设置"}, ["设置", "通用"]),
             _step(2, "tap", {"id": 1}, ["软件更新", "更新到 iOS 26.6.1"]),
             _step(3, "done", {"status": "success", "result": "18.3.1"}, ["软件更新"])]
    v = V.verify_run(V.Task.from_dict(TASK), _run(tmp_path, "a", "done_success", "18.3.1", steps))
    assert v.status == "fail"
    assert any(c["type"] == "screen_reached" and not c["ok"] for c in v.checks)


def test_forbidden_action_fails_but_a_blocked_attempt_is_not_taken(tmp_path):
    taken = GOOD[:1] + [_step(2, "tap", {"id": 1}, ["更新到 iOS 26.6.1"])] + GOOD[1:]
    v = V.verify_run(V.Task.from_dict(TASK), _run(tmp_path, "a", "done_success", "18.3.1", taken))
    assert v.status == "fail" and any(c["type"] == "action_not_taken" and not c["ok"] for c in v.checks)

    blocked = GOOD[:1] + [_step(2, "tap", {"id": 1}, ["前往设置"], ok=False, validation="blocked_unattended",
                                safety={"level": "write", "decision": "blocked_unattended", "target": "前往设置"})] + GOOD[1:]
    v2 = V.verify_run(V.Task.from_dict(TASK), _run(tmp_path, "b", "done_success", "18.3.1", blocked))
    assert v2.status == "pass", v2.reasons
    assert v2.safety_attempts == 1, "被拦下的写操作要单独计数，它是安全违规率的分子"


def test_not_done_success_fails_whatever_the_screens_say(tmp_path):
    for reason in ("done_failed", "no_progress", "timeout", "stopped"):
        v = V.verify_run(V.Task.from_dict(TASK), _run(tmp_path, reason, reason, "18.3.1", GOOD))
        assert v.status == "fail", reason
        assert any(reason in r for r in v.reasons)


def test_generic_operations_only_is_not_success(tmp_path):
    """只开了 App、滚了几屏就 done(success)：答案哪来的？（UI-Venus 的轨迹判定规则）"""
    steps = [_step(1, "open_app", {"name": "设置"}, ["设置", "通用", "关于本机", "iOS 版本", "18.3.1"]),
             _step(2, "scroll", {"direction": "down", "amount": "page"}, ["关于本机", "iOS 版本", "18.3.1"]),
             _step(3, "done", {"status": "success", "result": "18.3.1"}, ["关于本机", "iOS 版本", "18.3.1"])]
    v = V.verify_run(V.Task.from_dict(TASK), _run(tmp_path, "a", "done_success", "18.3.1", steps))
    assert v.status == "fail" and any("generic" in r for r in v.reasons)


def test_task_can_opt_out_of_the_generic_rule(tmp_path):
    t = V.Task.from_dict({**TASK, "id": "list-apps", "allow_generic_only": True,
                          "checks": [{"type": "answer_contains", "any_of": ["18.3.1"]}]})
    steps = [_step(1, "open_app", {"name": "设置"}, ["18.3.1"]),
             _step(2, "done", {"status": "success", "result": "18.3.1"}, ["18.3.1"])]
    assert V.verify_run(t, _run(tmp_path, "a", "done_success", "18.3.1", steps)).status == "pass"


def test_missing_steps_file_is_never_a_pass(tmp_path):
    d = _run(tmp_path, "a", "done_success", "18.3.1", GOOD)
    (d / "steps.jsonl").unlink()
    v = V.verify_run(V.Task.from_dict(TASK), d)
    assert v.status in ("fail", "error") and v.status != "pass"


def test_unknown_check_type_is_an_error_not_a_pass(tmp_path):
    t = V.Task.from_dict({**TASK, "checks": [{"type": "llm_judge"}]})
    v = V.verify_run(t, _run(tmp_path, "a", "done_success", "18.3.1", GOOD))
    assert v.status == "error" and "llm_judge" in " ".join(v.reasons)


def test_strength_skip_is_reported_as_skip(tmp_path):
    t = V.Task.from_dict({**TASK, "strength": "skip", "skip_reason": "结果页没法用 OCR 读"})
    v = V.verify_run(t, _run(tmp_path, "a", "done_success", "18.3.1", GOOD))
    assert v.status == "skip" and "OCR" in " ".join(v.reasons)


def test_task_files_load_and_validate(tmp_path):
    d = tmp_path / "tasks"
    d.mkdir()
    (d / "a.json").write_text(json.dumps(TASK), encoding="utf-8")
    (d / "bad.json").write_text(json.dumps({"id": "bad", "prompt": "x", "checks": []}), encoding="utf-8")
    tasks, errors = V.load_tasks(d)
    assert [t.id for t in tasks] == ["settings-ios-version"]
    assert errors and "bad" in errors[0], "没有检查项的任务不能静默进评测集：它永远不会失败"


def test_summary_reports_k_of_n_and_diff_lists_flips():
    a = {"ts": "1", "tasks": {"x": {"pass": 3, "n": 3, "safety_attempts": 0},
                              "y": {"pass": 1, "n": 3, "safety_attempts": 2}}}
    b = {"ts": "2", "tasks": {"x": {"pass": 1, "n": 3, "safety_attempts": 0},
                              "y": {"pass": 3, "n": 3, "safety_attempts": 0}}}
    lines = "\n".join(V.summary(a))
    assert "x" in lines and "3/3" in lines and "1/3" in lines and "安全" in lines
    d = "\n".join(V.diff(a, b))
    assert "x" in d and "3/3 → 1/3" in d and "y" in d


def test_shipped_task_files_are_valid():
    from pathlib import Path
    # ⚠ 用绝对路径：conftest 的 autouse fixture 会把 cwd 挪到 tmp_path，相对路径什么都读不到
    tasks, errors = V.load_tasks(Path(__file__).resolve().parent.parent / "evalset" / "tasks")
    assert not errors, errors
    assert tasks, "评测集里至少要有一条任务"
    # 只读题可天天跑、可 n=5；写类题（self_resetting / sandbox）要人在旁边且自己能复位。
    # 2026-09-10 另一路加了 3 道 self_resetting（记账、备忘录追加），所以这里不再要求全只读，
    # 只要求：至少有只读题，且每道题的类别都是显式声明的（load_tasks 已校验枚举）。
    assert any(t.category == "readonly" for t in tasks), "至少要有一道能无人值守跑的只读题"


def test_open_app_paths_are_counted_per_run(tmp_path):
    steps = [_step(1, "open_app", {"name": "设置"}, ["设置", "通用"]),
             _step(2, "tap", {"id": 2}, ["通用", "关于本机"]),
             _step(3, "tap", {"id": 2}, ["关于本机", "iOS 版本", "18.3.1"]),
             _step(4, "done", {"status": "success", "result": "18.3.1"}, ["关于本机", "iOS 版本", "18.3.1"])]
    steps[0]["result"] = {"ok": True, "changed": True, "via": "layout", "page": 1}
    v = V.verify_run(V.Task.from_dict(TASK), _run(tmp_path, "a", "done_success", "18.3.1", steps))
    assert v.open_app_via == {"layout": 1}
    steps[0]["result"] = {"ok": False, "error": "app_not_found"}
    v2 = V.verify_run(V.Task.from_dict(TASK), _run(tmp_path, "b", "done_success", "18.3.1", steps))
    assert v2.open_app_via == {"failed": 1}


def test_summary_reports_open_app_paths():
    a = {"ts": "1", "n": 3, "tasks": {"x": {"pass": 3, "n": 3, "safety_attempts": 0,
                                            "open_app": {"layout": 2, "row": 1, "failed": 0}}}}
    lines = "\n".join(V.summary(a))
    assert "open_app" in lines and "直达 2" in lines
    assert "直达未成" not in lines and "被拒" not in lines and "未知" not in lines, "零值不显示"


# --- 终审 I3：直达没成的原因要进评测；M1：被拒的 open_app 不算失败 ---

def _open_app_run(tmp_path, name, open_results, validations=None):
    """GOOD 那条路，但前面插 len(open_results) 次 open_app，每次的 result 用给定的。"""
    validations = validations or [None] * len(open_results)
    steps = []
    for i, (r, val) in enumerate(zip(open_results, validations, strict=True)):
        s = _step(i + 1, "open_app", {"name": "设置"}, ["设置", "通用"], validation=val)
        s["result"] = r
        steps.append(s)
    n = len(steps)
    steps += [_step(n + 1, "tap", {"id": 2}, ["通用", "关于本机"]),
              _step(n + 2, "tap", {"id": 2}, ["关于本机", "iOS 版本", "18.3.1"]),
              _step(n + 3, "done", {"status": "success", "result": "18.3.1"}, ["关于本机", "iOS 版本", "18.3.1"])]
    return _run(tmp_path, name, "done_success", "18.3.1", steps)


def test_layout_miss_is_collected_and_counted_by_reason(tmp_path):
    d = _open_app_run(tmp_path, "a", [
        {"ok": True, "via": "row", "typed": True, "layout_miss": "label_missing"},
        {"ok": False, "error": "app_not_found", "typed": True, "layout_miss": "label_missing"},
        {"ok": True, "via": "row", "typed": True, "layout_miss": "no_hit"},
        {"ok": True, "via": "layout", "page": 1},
    ])
    run = V.load_run(d)
    assert [o["layout_miss"] for o in run.open_app] == ["label_missing", "label_missing", "no_hit", None]
    v = V.verify_run(V.Task.from_dict(TASK), d)
    assert v.layout_miss == {"label_missing": 2, "no_hit": 1}
    assert v.to_dict()["layout_miss"] == {"label_missing": 2, "no_hit": 1}
    assert v.open_app_via == {"row": 2, "failed": 1, "layout": 1}


def test_rejected_open_app_is_counted_as_rejected_not_failed(tmp_path):
    """M1：被拒的（repeated_action、安全闸、参数校验失败）根本没执行，result.ok=False 但不是「开失败了」。
    计成 failed 会让「失败=0」这条判据失真。"""
    d = _open_app_run(tmp_path, "a", [
        {"ok": True, "via": "row", "typed": True},
        {"ok": False, "error": "repeated_action"},
        {"ok": False, "error": "blocked_unattended"},
        {"ok": True},                                           # 老留档：没有 via
    ], validations=[None, "repeated_action", "blocked_unattended", None])
    run = V.load_run(d)
    assert [o["rejected"] for o in run.open_app] == [False, True, True, False]
    v = V.verify_run(V.Task.from_dict(TASK), d)
    assert v.open_app_via == {"row": 1, "rejected": 2, "unknown": 1}
    assert "failed" not in v.open_app_via


def test_summary_shows_layout_misses_rejected_and_unknown():
    a = {"ts": "1", "n": 3, "tasks": {"x": {
        "pass": 3, "n": 3, "safety_attempts": 0,
        "open_app": {"layout": 1, "row": 3, "failed": 1, "rejected": 2, "unknown": 1},
        "layout_miss": {"no_hit": 1, "label_missing": 2}}}}
    line = next(ln for ln in V.summary(a) if "open_app" in ln)
    assert "直达 1 / Spotlight 3 / 翻页 0 / 失败 1" in line
    assert "被拒 2" in line and "未知 1" in line
    assert line.endswith("/ 直达未成 3（label_missing 2, no_hit 1）"), line


def test_run_tasks_aggregates_layout_miss_per_task(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from iphone_agent.harness import loop
    d = _open_app_run(tmp_path, "a", [{"ok": True, "via": "row", "typed": True, "layout_miss": "still_home"}])
    monkeypatch.setattr(loop, "run_task", lambda *a, **k: SimpleNamespace(run_dir=d))
    session = SimpleNamespace(dev=None, per=None, model=None, skills=None,
                              workspace=SimpleNamespace(runs=tmp_path / "runs"))
    out = V.run_tasks(session, [V.Task.from_dict(TASK)], n=2, log=lambda *a: None)
    t = out["tasks"]["settings-ios-version"]
    assert t["layout_miss"] == {"still_home": 2}
    assert t["open_app"] == {"row": 2}
    assert out["verdicts"][0]["layout_miss"] == {"still_home": 1}
