import pytest

from iphone_agent.harness.actions import ACTION_NAMES
from iphone_agent.harness.tools import tool_defs


def _tap(defs):
    return next(t for t in defs if t["function"]["name"] == "tap")


@pytest.mark.parametrize("allow", [True, False])
def test_every_tool_the_model_sees_is_a_tool_the_harness_validates(allow):
    """两份清单必须一一对应（从 test_qwen.py 迁来，对两种开关都成立）。

    schema 里有、ACTION_NAMES 里没有 → 模型调了就得到 unknown_tool；
    反过来 → 那个动作模型根本看不见。这类「同一件事在两处各写一份、只改一处」的
    分叉，这个项目已经踩过三次。
    """
    assert sorted(t["function"]["name"] for t in tool_defs(allow)) == sorted(ACTION_NAMES)


def test_coords_enabled_tap_has_xy():
    props = _tap(tool_defs(True))["function"]["parameters"]["properties"]
    assert "x" in props and "y" in props and "id" in props
    assert "坐标" in _tap(tool_defs(True))["function"]["description"]


def test_coords_disabled_tap_has_no_xy_and_no_mention():
    t = _tap(tool_defs(False))["function"]
    assert "x" not in t["parameters"]["properties"] and "y" not in t["parameters"]["properties"]
    assert "id" in t["parameters"]["properties"]
    assert "坐标" not in t["description"]


def test_observe_is_the_full_screen_look():
    """spec 2026-09-14 §4.1：observe 改义为看全屏，工具面上什么都不加。"""
    from iphone_agent.harness.tools import tool_defs
    d = {t["function"]["name"]: t["function"] for t in tool_defs(True)}["observe"]["description"]
    assert d.startswith("看全屏") and "zoom" in d and "慢" in d


def test_each_call_returns_fresh_list():
    a, b = tool_defs(True), tool_defs(True)
    assert a == b and a is not b


@pytest.mark.parametrize("allow", [True, False])
def test_tap_requires_an_expectation_with_quoted_key_text(allow):
    """2026-09-11：留档里只有 9.7% 的点击带 expect，写了的也都是「进入设置主界面」这种页面名，
    程序没法拿屏幕文字核对。tap 的 expect 必填，并要求把点完才出现的字放进「」。"""
    fn = _tap(tool_defs(allow))["function"]
    assert "expect" in fn["parameters"]["required"]
    desc = fn["parameters"]["properties"]["expect"]["description"]
    assert "「" in desc and "」" in desc and "必填" in desc


def test_tap_expectation_example_is_checkable_and_not_a_forbidden_setting():
    """终审 2026-09-11：示例「「飞行模式」变成打开」教的是核对不了的写法（那几个字点之前就在屏上），
    还拿项目明令不碰的设置（CLAUDE.md §8）当示范 —— 模型每一步都读工具说明。"""
    desc = _tap(tool_defs(True))["function"]["parameters"]["properties"]["expect"]["description"]
    assert "飞行模式" not in desc
    assert "别把你要点的" in desc


def test_other_tools_keep_expect_optional():
    for t in tool_defs(True):
        fn = t["function"]
        if fn["name"] != "tap" and "expect" in fn["parameters"]["properties"]:
            assert "expect" not in fn["parameters"]["required"], fn["name"]
