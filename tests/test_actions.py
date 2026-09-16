import math
from types import SimpleNamespace

import pytest

from iphone_agent import config
from iphone_agent.harness.actions import Action, ValidationError, validate_action


def obs(w=800, h=1700, elements=(), coord_mode="norm1000"):
    els = [SimpleNamespace(id=i, center=c) for i, c in elements]
    o = SimpleNamespace(observation_id=5, width_px=w, height_px=h, elements=els, coord_mode=coord_mode)
    o.element = lambda eid: next((e for e in els if e.id == eid), None)
    return o


def act(name, **args):
    return Action(name=name, args=args, reason="r", expect=None, call_id="c1")


def test_tap_by_id_resolves_center():
    a = validate_action(act("tap", id=2), obs(elements=[(2, (100, 200))]))
    assert a.args["x"] == 100 and a.args["y"] == 200 and a.args["id"] == 2


def test_tap_stale_id_rejected():
    with pytest.raises(ValidationError) as ei:
        validate_action(act("tap", id=9), obs(elements=[(2, (100, 200))]))
    assert ei.value.code == "stale_element_id"


def test_tap_array_coord_takes_center():
    a = validate_action(act("tap", x=[100, 200], y=[10, 20]), obs())
    assert (a.args["x"], a.args["y"]) == (150, 15)


def test_tap_small_overshoot_clamped_large_rejected():
    a = validate_action(act("tap", x=805, y=10), obs(w=800))   # 0.6% 越界 → 夹回
    assert a.args["x"] == 800
    with pytest.raises(ValidationError) as ei:
        validate_action(act("tap", x=900, y=10), obs(w=800))    # 12.5% → 拒绝
    assert ei.value.code == "out_of_image"


def test_tap_requires_exactly_one_target():
    for args in ({}, {"id": 1, "x": 1, "y": 1}, {"x": 1}):
        with pytest.raises(ValidationError) as ei:
            validate_action(act("tap", **args), obs(elements=[(1, (5, 5))]))
        assert ei.value.code == "invalid_args"


def test_nan_and_large_negative_rejected():
    for x in (math.nan, math.inf, -50):
        with pytest.raises(ValidationError):
            validate_action(act("tap", x=x, y=10), obs())


def test_type_length_and_wait_range():
    with pytest.raises(ValidationError):
        validate_action(act("type", text="x" * 501), obs())
    with pytest.raises(ValidationError):
        validate_action(act("wait", seconds=30), obs())
    assert validate_action(act("wait", seconds=2), obs()).args["seconds"] == 2


def test_enums_checked():
    with pytest.raises(ValidationError):
        validate_action(act("scroll", direction="sideways", amount="page"), obs())
    # scroll 现在支持四个方向（滚轮是镜像里唯一的滑动通道，没有 swipe 这个工具了）
    for d in ("up", "down", "left", "right"):
        assert validate_action(act("scroll", direction=d), obs()).args["direction"] == d
    with pytest.raises(ValidationError) as ei2:
        validate_action(act("swipe", kind="back"), obs())
    assert ei2.value.code == "unknown_tool"
    with pytest.raises(ValidationError):
        validate_action(act("done", status="maybe", result="x"), obs())
    with pytest.raises(ValidationError) as ei:
        validate_action(act("fly"), obs())
    assert ei.value.code == "unknown_tool"


def _act_with_name_arg(action_name, name_value):
    # act(name, **args) 的 name 形参和 key/open_app 自身的 "name" 参数同名会冲突，
    # 这里直接构造 Action 绕开这个命名碰撞。
    return Action(name=action_name, args={"name": name_value}, reason="r", expect=None, call_id="c1")


def test_key_enum_checked():
    with pytest.raises(ValidationError) as ei:
        validate_action(_act_with_name_arg("key", "volume_up"), obs())
    assert ei.value.code == "invalid_args"
    for value in ("home", "app_switcher", "spotlight"):
        validate_action(_act_with_name_arg("key", value), obs())


def test_open_app_name_validated():
    for bad in (123, "", "   ", "x" * 51):
        with pytest.raises(ValidationError) as ei:
            validate_action(_act_with_name_arg("open_app", bad), obs())
        assert ei.value.code == "invalid_args"
    validate_action(_act_with_name_arg("open_app", "设置"), obs())


def test_tap_bool_list_coord_rejected():
    with pytest.raises(ValidationError):
        validate_action(act("tap", x=[True, False], y=10), obs())


def test_clamp_tolerance_exact_boundary():
    a = validate_action(act("tap", x=816, y=10), obs(w=800))  # 2% 容差边界内 → 夹回
    assert a.args["x"] == 800
    with pytest.raises(ValidationError) as ei:
        validate_action(act("tap", x=817, y=10), obs(w=800))  # 刚超容差 → 拒绝
    assert ei.value.code == "out_of_image"


def test_scroll_amount_defaults_to_page():
    a = validate_action(act("scroll", direction="down"), obs())
    assert a.args["amount"] == "page"


# --- 高阶滚动的参数校验 ---

def test_scroll_until_and_collect_take_the_same_directions_as_scroll():
    """同一件事只允许有一个理解：方向的取值和含义必须与 scroll 逐字一致。"""
    from iphone_agent.harness.actions import ACTION_NAMES
    for name in ("scroll", "scroll_until", "collect"):
        assert name in ACTION_NAMES
    for d in ("up", "down", "left", "right"):
        assert validate_action(act("scroll_until", direction=d, text="x"), obs()).args["direction"] == d
        assert validate_action(act("collect", direction=d), obs()).args["direction"] == d
    with pytest.raises(ValidationError):
        validate_action(act("scroll_until", direction="sideways", text="x"), obs())
    with pytest.raises(ValidationError):
        validate_action(act("collect", direction="sideways"), obs())


def test_scroll_until_requires_a_non_empty_text():
    """谓词只能是数据 —— 调用方是个发 JSON 的模型，传不了回调函数。"""
    for bad in ({}, {"text": ""}, {"text": "   "}, {"text": 123},
                {"text": "长" * (config.SCROLL_UNTIL_TEXT_MAX + 1)}):
        with pytest.raises(ValidationError):
            validate_action(act("scroll_until", direction="down", **bad), obs())
    a = validate_action(act("scroll_until", direction="down", text="关于本机"), obs())
    assert a.args["text"] == "关于本机"


# --- tap 的两类目标（2026-09-07 真机实测） ---
# 主屏幕「设置」标签高 23px、中心 y=796：点 y=796 完全没反应；
# 点 y=727（上移 3 倍文字高）与 y=750（上移 2 倍）都成功打开 App。
# 规则确定且有容差，所以偏移由代码算 —— 坐标永远来自框，模型只负责选目标。

def obs_with_box(eid, center, box):
    from types import SimpleNamespace
    el = SimpleNamespace(id=eid, center=center, box=box)
    o = SimpleNamespace(observation_id=1, width_px=644, height_px=1436, elements=[el])
    o.element = lambda i: el if i == eid else None
    return o


def test_target_text_taps_the_label_itself():
    o = obs_with_box(16, (534, 796), (515, 785, 554, 808))
    a = validate_action(act("tap", id=16), o)
    assert (a.args["x"], a.args["y"]) == (534, 796)
    assert a.args["target"] == "text"


def test_target_icon_above_lifts_by_three_label_heights():
    """标签高 23 → 上移 69 → y=727，正是实测点得中图标的那个位置。"""
    o = obs_with_box(16, (534, 796), (515, 785, 554, 808))
    a = validate_action(act("tap", id=16, target="icon_above"), o)
    assert a.args["x"] == 534
    assert a.args["y"] == 796 - 3 * 23 == 727


def test_icon_above_never_goes_negative():
    o = obs_with_box(1, (100, 10), (90, 5, 110, 15))
    a = validate_action(act("tap", id=1, target="icon_above"), o)
    assert a.args["y"] == 0


def test_icon_above_requires_an_id_not_raw_coords():
    """偏移要靠标签的框算，光给坐标算不出来。"""
    o = obs_with_box(1, (100, 100), (90, 90, 110, 110))
    with pytest.raises(ValidationError) as ei:
        validate_action(act("tap", x=100, y=100, target="icon_above"), o)
    assert ei.value.code == "invalid_args"


def test_unknown_target_rejected():
    o = obs_with_box(1, (100, 100), (90, 90, 110, 110))
    with pytest.raises(ValidationError):
        validate_action(act("tap", id=1, target="icon_left"), o)


# --- 从行文字派生同一行上的无文字目标（2026-09-08 实测的版式）---
#
# 列表行：文字在左、chevron/开关/数值在右、图标在最左，三者同一个 y。
# 这么做而不是让模型估坐标，是因为标定出来模型指图标 p90 差 117px，
# 而一个图标才 120px 宽（docs/14）。

def test_row_right_keeps_the_row_and_moves_to_the_right_edge():
    o = obs_with_box(1, (100, 500), (80, 490, 160, 510))
    a = validate_action(act("tap", id=1, target="row_right"), o)
    assert a.args["y"] == 500, "换到了别的行上"
    assert a.args["x"] == round(o.width_px * config.ROW_RIGHT_RATIO)
    assert a.args["x"] > 100, "没有往右挪"


def test_row_left_keeps_the_row_and_moves_to_the_left_edge():
    o = obs_with_box(1, (300, 500), (280, 490, 360, 510))
    a = validate_action(act("tap", id=1, target="row_left"), o)
    assert a.args["y"] == 500
    assert a.args["x"] == round(o.width_px * config.ROW_LEFT_RATIO)
    assert a.args["x"] < 300, "没有往左挪"


def test_row_targets_need_an_id_not_raw_coordinates():
    """它们是从那个文字元素的框推出来的 —— 没有 id 就无从推起。"""
    o = obs_with_box(1, (100, 500), (80, 490, 160, 510))
    for t in ("row_right", "row_left"):
        with pytest.raises(ValidationError):
            validate_action(act("tap", x=10, y=10, target=t), o)


def test_coords_disabled_rejects_xy_with_its_own_code():
    """未标定的模型：工具面上没有 x/y，但模型仍可能硬塞。要用专门的码拒，提示它用 id。"""
    with pytest.raises(ValidationError) as ei:
        validate_action(act("tap", x=10, y=10), obs(), allow_coord_tap=False)
    assert ei.value.code == "coord_disabled" and "id" in ei.value.message


def test_coords_disabled_still_allows_id_paths():
    a = validate_action(act("tap", id=2, target="row_right"), obs(elements=[(2, (100, 200))]),
                        allow_coord_tap=False)
    assert a.args["x"] == round(800 * config.ROW_RIGHT_RATIO) and a.args["y"] == 200


def test_default_allow_coord_tap_keeps_old_behaviour():
    a = validate_action(act("tap", x=10, y=20), obs())
    assert a.args["x"] == 10 and a.args["y"] == 20


def test_action_dataclass_defaults():
    from iphone_agent.harness.actions import Action
    a = Action("tap", {"id": 1}, "r", None, "c1")
    assert a.eval is None and a.memory is None


def test_procedure_action_validates_declared_params():
    import pytest

    from iphone_agent.harness.actions import Action, ValidationError, validate_action
    from iphone_agent.skills import model as M
    p = M.Procedure(name="add", app="jizhangben", description="记一笔",
                    params={"金额": {"type": "string", "description": ""}}, returns=[], risk="read",
                    status="verified", expect_source="single_run", provenance=M.Provenance(),
                    steps=[M.Step(do="type", text="{金额}", expect=())])
    procs = {p.tool_name: p}
    ok = validate_action(Action("jizhangben__add", {"金额": "12"}, "r", None, "c"), None, procs)
    assert ok.args == {"金额": "12"}
    with pytest.raises(ValidationError) as e:
        validate_action(Action("jizhangben__add", {}, "r", None, "c"), None, procs)
    assert e.value.code == "invalid_args"
    with pytest.raises(ValidationError):
        validate_action(Action("jizhangben__add", {"金额": "12", "多余": "x"}, "r", None, "c"), None, procs)
    with pytest.raises(ValidationError) as e:
        validate_action(Action("nobody__x", {}, "r", None, "c"), None, procs)
    assert e.value.code == "unknown_tool"


def test_use_skill_validates_kind_and_name():
    import pytest

    from iphone_agent.harness.actions import NON_COUNTING, Action, ValidationError, validate_action
    assert "use_skill" in NON_COUNTING
    ok = validate_action(Action("use_skill", {"kind": "app", "name": "settings"}, "r", None, "c"), None)
    assert ok.args["kind"] == "app"
    with pytest.raises(ValidationError):
        validate_action(Action("use_skill", {"kind": "tool", "name": "x"}, "r", None, "c"), None)
    with pytest.raises(ValidationError):
        validate_action(Action("use_skill", {"kind": "app", "name": ""}, "r", None, "c"), None)


def test_done_remember_kind_is_validated():
    import pytest

    from iphone_agent.harness.actions import Action, ValidationError, validate_action
    base = {"status": "success", "result": "x"}
    ok = validate_action(Action("done", {**base, "remember": [
        {"name": "a", "description": "d", "content": "c"},
        {"name": "b", "description": "d", "content": "c", "kind": "app_note", "app": "settings"},
        {"name": "c-scene", "description": "d", "content": "1. …", "kind": "scenario", "apps": ["settings"]},
    ]}, "r", None, "c"), None)
    # 缺省的 kind 由 validate_action 显式补上；合并「面向规模」那条线之后，
    # 记忆那一档的名字从 "memory" 统一成 "knowledge"（还多了 playbook）。
    assert [it["kind"] for it in ok.args["remember"]] == ["knowledge", "app_note", "scenario"]
    with pytest.raises(ValidationError):
        validate_action(Action("done", {**base, "remember": [
            {"name": "a", "description": "d", "content": "c", "kind": "tool"}]}, "r", None, "c"), None)
    with pytest.raises(ValidationError):
        validate_action(Action("done", {**base, "remember": [
            {"name": "a", "description": "d", "content": "c", "kind": "app_note"}]}, "r", None, "c"), None)
