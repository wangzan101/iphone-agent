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


def test_each_call_returns_fresh_list():
    a, b = tool_defs(True), tool_defs(True)
    assert a == b and a is not b
