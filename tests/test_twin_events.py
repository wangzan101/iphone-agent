"""孪生的输入：留档 → 事件、当前 App 状态机（spec §1、§2）。"""
import json

from iphone_agent.twin.events import AppTracker, Event, Snapshot, extract_events


def obs(frame, texts=("a",), w=624, h=1388):
    return {"frame_file": frame, "width_px": w, "height_px": h,
            "elements": [{"id": i + 1, "text": t, "confidence": 0.9, "box": [0, 0, 1, 1],
                          "center": [300, 300 + 60 * i], "source": "ocr"} for i, t in enumerate(texts)]}


def rec(step, frame, name, after=None, result=None, args=None, **extra):
    r = {"step": step, "ts": 1000.0 + step, "observation": obs(frame),
         "action": {"name": name, "args_raw": args or {}, "args": args or {}},
         "result": result or {"ok": True, "changed": True}}
    if after:
        r["after_frame_file"] = after
    r.update(extra)
    return json.dumps(r, ensure_ascii=False)


def test_event_id_uses_line_number_not_step():
    # 真实留档 runs/20260909-150415-5d25 第 7、8 行：被拒的 tap 与随后的 type 都是 step 6
    lines = [rec(6, "f1.png", "tap", validation="repeated_action"),
             rec(6, "f1.png", "type", after="f2.png"), rec(7, "f2.png", "done")]
    assert [e.event_id for e in extract_events("r1", lines)] == ["r1:2", "r1:3"]


def test_skips_phase_error_rejected_and_top_level_procedure_records():
    lines = [json.dumps({"step": 0, "phase": "screenmap", "error": "x"}),
             json.dumps({"step": 1, "error_phase": "loop", "error": "x"}),
             json.dumps({"step": 1, "rejected": "multiple_calls"}),
             rec(1, "f1.png", "proc_read_version", after="f3.png"),     # 顶层剧本记录：工具名不是设备动作
             rec(1, "f1.png", "tap", after="f2.png", kind="procedure_step", parent_call_id="c1"),
             rec(2, "f2.png", "tap", after="f3.png", kind="procedure_step", parent_call_id="c1"),
             rec(2, "f3.png", "done")]
    assert [e.event_id for e in extract_events("r", lines)] == ["r:5", "r:6", "r:7"]


def test_after_is_paired_by_frame_file_and_missing_after_is_none():
    lines = [rec(1, "f1.png", "tap", after="f3.png"),
             json.dumps({"step": 2, "error_phase": "loop", "error": "x"}),
             rec(2, "f3.png", "tap", after="f4.png")]            # f4 从没作为观察出现
    a, b = extract_events("r", lines)
    assert a.after is not None and a.after_event_id == "r:3"
    assert b.after is None and b.after_event_id is None


def test_args_fall_back_to_args_raw_and_bad_lines_are_skipped():
    r = json.loads(rec(1, "f1.png", "tap"))
    r["action"] = {"name": "tap", "args_raw": {"id": 3}}
    evs = extract_events("r", ["{bad json", "", json.dumps(r)])
    assert evs[0].args == {"id": 3} and evs[0].event_id == "r:3"


def ev(name, args=None, result=None, texts=("a",)):
    return Event("r:1", 0.0, Snapshot.from_observation(obs("f.png", texts)), {"name": name, "args": args or {}},
                 result if result is not None else {"ok": True, "changed": True}, None, None)


def test_open_app_counts_only_when_identity_verified():
    t = AppTracker()
    t.advance(ev("open_app", {"name": "设置"}, {"ok": True, "changed": True, "identity": {"verified": True}}))
    assert t.owner == "she-zhi" and t.display["she-zhi"] == "设置"


def test_open_app_without_verified_identity_is_unknown():
    # runs/20260910-193640-bfd7：open_app 设置 返回 ok，画面是备忘录 —— 没核过身份就不能当开对
    for result in ({"ok": True, "changed": True}, {"ok": True, "changed": True, "identity": {"verified": None}}):
        t = AppTracker()
        t.advance(ev("open_app", {"name": "设置"}, result))
        assert t.owner == "unknown"


def test_failed_open_app_from_inside_an_app_becomes_unknown():
    # 每条 open_app 路线失败前都先回第一张主屏（executor.return_to_first_home_page），
    # 错 App 的路线又在错 App 里返回 ok=False —— 失败之后手机不可能还停在原来那个 App。
    t = AppTracker()
    t.owner = "bei-wang-lu"
    t.advance(ev("open_app", {"name": "设置"}, {"ok": False, "error": "x"}))
    assert t.owner == "unknown"


def test_open_app_ok_but_unchanged_becomes_unknown():
    t = AppTracker()
    t.owner = "bei-wang-lu"
    t.advance(ev("open_app", {"name": "设置"}, {"ok": True, "changed": False}))
    assert t.owner == "unknown"


def test_home_spotlight_switcher_and_handover():
    t = AppTracker()
    t.owner = "she-zhi"
    t.advance(ev("key", {"name": "app_switcher"}))
    assert t.owner == "unknown"
    t.advance(ev("key", {"name": "home"}))
    assert t.owner == "system"
    t.owner = "she-zhi"
    t.advance(ev("key", {"name": "spotlight"}))
    assert t.owner == "system"
    t.owner = "she-zhi"
    t.advance(ev("handover", {"need": "登录"}))
    assert t.owner == "unknown"


def test_icon_tap_on_home_enters_app():
    t = AppTracker()
    t.advance(ev("tap", {"id": 1, "target": "icon_above"},
                 {"ok": True, "changed": True, "identity": {"verified": True}}, texts=("备忘录",)))
    assert t.owner == "bei-wang-lu" and t.display["bei-wang-lu"] == "备忘录"


def test_icon_tap_on_home_without_verified_identity_is_unknown():
    # 2026-09-11（fbc1 用户决定）：点图标进 App 与 open_app 同一条规则——
    # identity 缺失（老留档）/ verified None（没核成）/ verified False（核过说不是）都不能进 App。
    for result in ({"ok": True, "changed": True},
                   {"ok": True, "changed": True, "identity": {"verified": None}},
                   {"ok": True, "changed": True, "identity": {"verified": False}}):
        t = AppTracker()
        t.advance(ev("tap", {"id": 1, "target": "icon_above"}, result, texts=("备忘录",)))
        assert t.owner == "unknown", result


def _snap(els):
    return Snapshot.from_observation({"width_px": 624, "height_px": 1388, "elements": els})


def test_leaving_system_by_a_non_icon_tap_is_unknown_and_later_tab_icons_do_not_enter_apps():
    # ⚠ 真实留档 runs/20260908-022024-e257（闸门 B 错归）：Spotlight 里点结果文字「记账本」（target=text）
    #   进了 App，状态机仍当 system；之后在 App 里点底部 tab 图标（icon_above，id 45 的 OCR 读成「G」）
    #   被当成「主屏点图标」，整段被记到了 App「g」。元素照抄第 5、13 行。
    t = AppTracker()
    spotlight = _snap([{"id": 4, "text": "一木", "center": [120, 1283], "confidence": 1.0},
                       {"id": 5, "text": "记账本", "center": [104, 341], "confidence": 1.0}])
    t.advance(Event("r:5", 0.0, spotlight, {"name": "tap", "args": {"id": 5, "target": "text", "x": 104, "y": 341}},
                    {"ok": True, "changed": True}, None, None))
    assert t.owner == "unknown"
    ledger = _snap([{"id": 43, "text": "食品餐饮-午餐", "center": [181, 1230], "confidence": 1.0},
                    {"id": 44, "text": "@", "center": [71, 1285], "confidence": 0.3},
                    {"id": 45, "text": "G", "center": [311, 1284], "confidence": 0.5}])
    t.advance(Event("r:13", 0.0, ledger, {"name": "tap", "args": {"id": 45, "target": "icon_above", "x": 311, "y": 1170}},
                    {"ok": True, "changed": True}, None, None))
    assert t.owner == "unknown"


def test_system_actions_that_may_launch_an_app_become_unknown_others_stay_system():
    # 系统界面上画面变了、又不是认得出的「主屏点图标」：点了 Spotlight 结果 / 小组件 / 图标下的文字，
    # 或在 Spotlight 里按回车（直接开最佳结果）—— 都可能进了某个 App，不知道是哪个 → unknown。
    for name, args in (("tap", {"id": 1, "target": "text"}), ("tap", {"id": 1, "target": "row_right"}),
                       ("tap", {"x": 10, "y": 10}), ("tap", {"id": 99, "target": "icon_above"}),
                       ("key", {"name": "return"})):
        t = AppTracker()
        t.advance(ev(name, args))
        assert t.owner == "unknown", (name, args)
    # 画面没变的点击、翻页、在搜索框里打字、等待：仍在系统界面
    for name, args, result in (("tap", {"id": 1, "target": "text"}, {"ok": True, "changed": False}),
                               ("scroll", {"direction": "left"}, None), ("type", {"text": "一木"}, None),
                               ("wait", {}, None)):
        t = AppTracker()
        t.advance(ev(name, args, result))
        assert t.owner == "system", (name, args)
    # 已在 App 里：点文字不改归属（在 App 内导航）
    t = AppTracker()
    t.owner = "she-zhi"
    t.advance(ev("tap", {"id": 1, "target": "text"}))
    assert t.owner == "she-zhi"


def test_mark_unknown():
    t = AppTracker()
    t.owner = "she-zhi"
    t.mark_unknown()
    assert t.owner == "unknown"


def test_read_run_events_returns_empty_on_missing_directory(tmp_path):
    """读不了等于没有事件，不抛（不变式 5）。"""
    from iphone_agent.twin.events import read_run_events
    missing_dir = tmp_path / "nonexistent"
    assert read_run_events(missing_dir) == []


def test_read_run_events_returns_empty_on_invalid_utf8(tmp_path):
    """读 steps.jsonl 碰上坏编码也返回空，不抛（不变式 5）。"""
    from iphone_agent.twin.events import read_run_events
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    # 写入包含无效 UTF-8 的 steps.jsonl
    steps_file = run_dir / "steps.jsonl"
    steps_file.write_bytes(b"\xff\xfe{bad")
    assert read_run_events(run_dir) == []


def test_snapshot_reads_label_status_and_candidates():
    obs = {"elements": [], "width_px": 1, "height_px": 2, "frame_file": "frame_003.png",
           "screen": {"app": "设置", "name": "通用", "same_as": None, "anchors": []},
           "perception": {"ocr": 0, "vision": "ok", "screen": "ok"},
           "screen_candidates": [{"index": 1, "app_id": "she-zhi", "screen_id": "s_1", "name": "通用"}, "bad"]}
    s = Snapshot.from_observation(obs)
    assert s.frame_file == "frame_003.png" and s.label["name"] == "通用" and s.label_status == "ok"
    assert s.candidates == ({"index": 1, "app_id": "she-zhi", "screen_id": "s_1", "name": "通用"},)
    old = Snapshot.from_observation({"elements": [], "width_px": 1, "height_px": 2})
    assert old.frame_file == "" and old.label is None and old.label_status == "off" and old.candidates == ()


def test_tracker_remembers_a_verified_entry_once():
    t = AppTracker()
    t.advance(ev("open_app", {"name": "App Store"},
                 {"ok": True, "changed": True, "identity": {"verified": True, "label_app": "应用商店"}}))
    assert t.take_entry() == ("app-store", "应用商店") and t.take_entry() == (None, None)
    t.advance(ev("open_app", {"name": "设置"}, {"ok": True, "changed": True, "identity": {"verified": True}}))
    t.advance(ev("scroll", {"direction": "down"}, {"ok": True, "changed": True}))
    assert t.take_entry() == (None, None), "进入之后又做了别的动作：这次进入已经不是「上一个动作」"


def test_after_prefers_the_records_own_after_observation():
    """spec 2026-09-14 §8.1：最后一个动作之后没有下一条记录，after 也要在（带实时用过的标注）。"""
    before = {"frame_file": "frame_001.png", "width_px": 10, "height_px": 10, "elements": []}
    after = {"frame_file": "frame_002.png", "width_px": 10, "height_px": 10, "elements": [],
             "screen": {"app": "设置", "name": "通用", "same_as": None, "anchors": []}}
    line = json.dumps({"step": 1, "ts": 1.0, "action": {"name": "tap", "args": {"id": 1}},
                       "result": {"ok": True}, "observation": before,
                       "after_frame_file": "frame_002.png", "after_observation": after}, ensure_ascii=False)
    (ev,) = extract_events("r", [line])
    assert ev.after is not None and ev.after.frame_file == "frame_002.png" and ev.after.label["name"] == "通用"
    assert ev.after_event_id is None
