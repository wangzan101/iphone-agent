from iphone_agent.harness.history import HistoryRow, render_history, row_from_record, summarize_omitted


def _row(step, action, executed="yes", changed="yes", expected="unknown", note=""):
    return HistoryRow(str(step), action, executed, changed, expected, note)


def rec(step, name, args, result=None, **kw):
    r = {"step": step, "action": {"name": name, "args": args}, "model": {"reason": "r", "expect": None}}
    if result is not None:
        r["result"] = result
    r.update(kw)
    return r


def test_tap_by_id_shows_element_text():
    row = row_from_record(rec(1, "tap", {"id": 3, "target": "icon_above"},
                              {"ok": True, "changed": True}),
                          elements_by_id={3: "微信"})
    assert row.step == "1"
    assert row.action == 'tap "微信"(icon_above)'
    assert row.executed == "yes" and row.changed == "yes" and row.expected == "unknown"


def test_changed_column_distinguishes_local():
    row = row_from_record(rec(2, "tap", {"id": 1}, {"ok": True, "changed": True},
                              transition={"changed": False, "local_changed": True}))
    assert row.changed == "local"


def test_rejected_action_is_row_R():
    row = row_from_record(rec(3, "tap", {"id": 99}, {"ok": False, "error": "stale_element_id"},
                              validation="stale_element_id"))
    assert row.step == "R" and row.executed == "rejected" and row.changed == "n/a"
    assert "stale_element_id" in row.note


def test_recall_has_no_change_semantics():
    row = row_from_record(rec(4, "recall", {"name": "settings-entry"},
                              {"ok": True, "content": "正文第一行\n第二行"}))
    assert row.action == "recall settings-entry" and row.changed == "n/a"
    assert row.note == "正文第一行"


def test_eval_fills_expected_and_note():
    row = row_from_record(rec(5, "scroll", {"direction": "down"}, {"ok": True, "changed": True},
                              eval={"expected": "yes", "note": "看到了更多群"}))
    assert row.expected == "yes" and row.note == "看到了更多群"


def test_render_keeps_first_and_last_when_over_limit():
    rows = [HistoryRow(str(i), f"a{i}", "yes", "yes", "unknown", "") for i in range(1, 21)]
    s = render_history(rows, keep=5)
    lines = s.splitlines()
    assert lines[0] == "【到目前为止】"
    assert lines[2].startswith("1 |")
    assert "【更早】第 2–16 步（15 步）" in lines[3]
    assert lines[-1].startswith("20 |") and len(lines) == 2 + 1 + 2 + 4


def test_render_empty():
    assert render_history([], keep=12) == "【到目前为止】\n（还没有动作）"


def test_note_falls_back_to_reason_when_nothing_else_filled():
    """reason 里写的关键信息不该在 state 模式下消失（Important 1）。"""
    row = row_from_record(rec(6, "tap", {"id": 1}, {"ok": True, "changed": True},
                              model={"reason": "iOS 版本 18.6\n第二行", "expect": None}))
    assert row.note == "iOS 版本 18.6"


def test_note_prefers_eval_over_reason():
    row = row_from_record(rec(7, "tap", {"id": 1}, {"ok": True, "changed": True},
                              model={"reason": "iOS 版本 18.6", "expect": None},
                              eval={"expected": "yes", "note": "点开了设置"}))
    assert row.note == "点开了设置"


def test_rejected_row_appends_eval_note():
    row = row_from_record(rec(8, "tap", {"id": 99}, {"ok": False, "error": "stale_element_id"},
                              validation="stale_element_id",
                              eval={"expected": "no", "note": "元素编号已经变了"}))
    assert row.step == "R"
    assert row.note == "stale_element_id · 元素编号已经变了"


def test_rejected_row_without_eval_note_is_unchanged():
    row = row_from_record(rec(9, "tap", {"id": 99}, {"ok": False, "error": "stale_element_id"},
                              validation="stale_element_id"))
    assert row.note == "stale_element_id"


def test_summarize_omitted_groups_actions_and_keeps_failures():
    rows = [_row(2, 'tap "微信"(icon_above)'), _row(3, 'tap "家人群"'), _row(4, "scroll down"),
            _row(5, "scroll down", changed="no", expected="no", note="到底了"),
            _row(6, 'tap "发现"', expected="no", note="进了错页"),
            _row(7, "tap #99", executed="rejected", changed="n/a", expected="n/a", note="stale_element_id"),
            _row(8, "recall settings-entry", changed="n/a", note="通用在第二组"),
            _row(9, "collect down", changed="n/a", note="35 行，4 屏")]
    s = summarize_omitted(rows, list_max=8)
    lines = s.splitlines()
    assert lines[0] == "【更早】第 2–9 步（8 步）"
    assert "tap 4（目标 微信, 家人群, 发现, #99）" in s
    assert "scroll 2" in s and "recall 1" in s and "collect 1" in s
    assert "没达到预期的：5 scroll down（到底了）；6 tap \"发现\"（进了错页）" in s
    assert "被拒 1 次：stale_element_id ×1" in s
    assert "读到的：8 recall settings-entry（通用在第二组）；9 collect down（35 行，4 屏）" in s


def test_summarize_omitted_caps_lists_and_reports_totals():
    rows = [_row(i, f'tap "t{i}"', expected="no", note="x") for i in range(2, 22)]
    s = summarize_omitted(rows, list_max=3)
    assert "…共 20 个" in s                      # tap 目标列表被截
    assert "（共 20 条，列前 3）" in s              # 没达到预期的被截
    assert s.count("tap \"t") <= 3 + 3           # 目标列表 3 个 + 失败列表 3 条


def test_summarize_omitted_skips_empty_sections():
    rows = [_row(2, "scroll down"), _row(3, "scroll down")]
    s = summarize_omitted(rows, list_max=8)
    assert "没达到预期的" not in s and "被拒" not in s and "读到的" not in s
    assert "动作分布：scroll 2" in s


def test_summarize_omitted_rejected_section_is_bounded_by_reject_code():
    """90 条被拒、note 各不相同（拒绝码相同但 eval.note 不同）：被拒行不能按 90 个 key 展开。"""
    rows = [_row(i, "tap #1", executed="rejected", changed="n/a", expected="n/a",
                 note=f"stale_element_id · 第 {i} 次理由各不相同") for i in range(2, 92)]
    s = summarize_omitted(rows, list_max=8)
    reject_line = next(line for line in s.splitlines() if line.startswith("被拒"))
    assert len(reject_line) < 200
    assert reject_line == "被拒 90 次：stale_element_id ×90"


def test_summarize_omitted_rejected_section_caps_distinct_groups():
    """拒绝码本身各不相同时才需要截断并标注共 N 种。"""
    rows = [_row(i, "tap #1", executed="rejected", changed="n/a", expected="n/a",
                 note=f"reason_{i}") for i in range(2, 92)]
    s = summarize_omitted(rows, list_max=8)
    reject_line = next(line for line in s.splitlines() if line.startswith("被拒"))
    assert len(reject_line) < 300
    assert "共 90 种" in reject_line
    assert reject_line.count("×1") <= 8


def test_summarize_omitted_keeps_coordinate_tap_targets():
    rows = [_row(2, "tap (120,300)"), _row(3, "tap (10,20)"), _row(4, "tap (120,300)")]
    s = summarize_omitted(rows, list_max=8)
    assert "(120,300)" in s
    assert "(10,20)" in s


def test_summarize_omitted_caps_single_entry_action_length():
    """【更早】里每条不只限条数，单条本身（含 r.action）也不能不封顶（Important 2）：
    `type "` + 500 字长文本 + `"` 不能原样进【更早】。"""
    long_text = "字" * 500
    rows = [_row(2, f'type "{long_text}"', expected="no", note="没打进去")]
    s = summarize_omitted(rows, list_max=8)
    line = next(line for line in s.splitlines() if line.startswith("没达到预期的"))
    assert len(line) <= len("没达到预期的：") + 60 + 20


def test_summarize_omitted_caps_tap_target_length():
    """tap 目标本身也可能是长文本（元素标签），动作分布里的目标列表要截断。"""
    long_target = "标" * 200
    rows = [_row(2, f'tap "{long_target}"')]
    s = summarize_omitted(rows, list_max=8)
    line = next(line for line in s.splitlines() if line.startswith("动作分布"))
    assert long_target not in line


def test_render_history_uses_earlier_block_instead_of_ellipsis():
    rows = [_row(i, f"a{i}") for i in range(1, 21)]
    s = render_history(rows, keep=5)
    assert "[…" not in s
    assert "【更早】第 2–16 步（15 步）" in s
    assert s.splitlines()[-1].startswith("20 |")
