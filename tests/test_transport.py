import dataclasses
import json
from types import SimpleNamespace

import pytest

from iphone_agent.harness.tools import tool_defs
from iphone_agent.model.reply import ModelError
from iphone_agent.model.transports.chat_completions import ChatCompletionsTransport
from tests._resolved import fake_resolved


def resp(tool_calls, content=None):
    return {"model": "qwen3.7-plus-2026-08", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}]}


def tc(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class FakeCompletions:
    def __init__(self, reply=None, raise_=None):
        self.calls, self._reply, self._raise = [], reply, raise_

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise:
            raise self._raise
        return SimpleNamespace(model_dump=lambda: self._reply)


def _wired(transport, completions):
    transport._client_or_create = lambda: SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return completions


# ---- 解析（从 test_qwen.py 迁来，语义不变）----

def test_parse_single_tool_call_pixel_mode():
    t = ChatCompletionsTransport(fake_resolved(coord_mode="pixel", allow_coord_tap=True))
    r = t.parse_response(resp([tc("c1", "tap", {"x": 100, "y": 200, "reason": "点通用", "expect": "进入通用"})]),
                         (800, 1700))
    a = r.actions[0]
    assert a.name == "tap" and a.args == {"x": 100, "y": 200} and a.reason == "点通用" and a.expect == "进入通用"
    assert a.call_id == "c1" and r.model_version == "qwen3.7-plus-2026-08"


def test_norm1000_converts_to_pixels():
    t = ChatCompletionsTransport(fake_resolved())        # 内置 qwen 是 norm1000
    r = t.parse_response(resp([tc("c1", "tap", {"x": 500, "y": 500, "reason": "r"})]), (800, 1700))
    assert r.actions[0].args == {"x": 400, "y": 850}


def test_norm1000_tolerates_bbox_coordinates_takes_midpoint():
    t = ChatCompletionsTransport(fake_resolved())
    r = t.parse_response(resp([tc("c1", "tap", {"x": [400, 600], "y": [450, 550], "reason": "r"})]),
                         (800, 1700))
    assert r.actions[0].args == {"x": 400, "y": 850}


def test_norm1000_zoom_box_converts_to_pixels():
    """⚠ 2026-09-16 真机：桥只给 tap 换算，zoom 的框原样送进校验，被当像素比。
    624 宽的屏上 x>624 的框（右边 38%：搜索、⋮ 菜单、开关）永远点不到。"""
    t = ChatCompletionsTransport(fake_resolved())
    r = t.parse_response(resp([tc("c1", "zoom", {"x1": 750, "y1": 60, "x2": 980, "y2": 160, "reason": "r"})]),
                         (624, 1388))
    assert r.actions[0].args == {"x1": 468, "y1": 83, "x2": 612, "y2": 222}


def test_pixel_mode_zoom_box_untouched():
    t = ChatCompletionsTransport(fake_resolved(coord_mode="pixel", allow_coord_tap=True))
    r = t.parse_response(resp([tc("c1", "zoom", {"x1": 480, "y1": 890, "x2": 620, "y2": 940, "reason": "r"})]),
                         (624, 1388))
    assert r.actions[0].args == {"x1": 480, "y1": 890, "x2": 620, "y2": 940}


def test_norm1000_zoom_box_survives_validation_and_crops_the_asked_region():
    """桥 → 校验 一条路走完：那个被连拒五次的框（20260916-121854/off 那一跑）必须过，
    而且裁的就是模型框的那一块（右上角的搜索/菜单区）。"""
    from iphone_agent.harness.actions import validate_action
    t = ChatCompletionsTransport(fake_resolved())
    a = t.parse_response(resp([tc("c1", "zoom", {"x1": 750, "y1": 60, "x2": 980, "y2": 160, "reason": "r"})]),
                         (624, 1388)).actions[0]
    o = SimpleNamespace(observation_id=1, width_px=624, height_px=1388, elements=[],
                        coord_mode="norm1000", element=lambda _i: None)
    v = validate_action(a, o)
    assert (v.args["x1"], v.args["y1"], v.args["x2"], v.args["y2"]) == (468, 83, 612, 222)


def test_out_of_range_zoom_box_still_rejected_without_blaming_the_convention():
    """换算之后被拒的框是**真的**出界。提示语不能再教模型「要用归一化值」——
    它已经用了；那句话让它把正确的框换成像素再试，真机上连试五次熔断。"""
    import pytest as _pytest

    from iphone_agent.harness.actions import ValidationError, validate_action
    t = ChatCompletionsTransport(fake_resolved())
    a = t.parse_response(resp([tc("c1", "zoom", {"x1": 100, "y1": 100, "x2": 1200, "y2": 300, "reason": "r"})]),
                         (624, 1388)).actions[0]
    o = SimpleNamespace(observation_id=1, width_px=624, height_px=1388, elements=[],
                        coord_mode="norm1000", element=lambda _i: None)
    with _pytest.raises(ValidationError) as ei:
        validate_action(a, o)
    assert ei.value.code == "out_of_image"
    msg = ei.value.message
    assert "1200" in msg, f"要用模型自己写的那个数说话：{msg}"
    assert "不是像素" not in msg, f"提示语在教模型换约定，而它本来就是对的：{msg}"


def test_multiple_tool_calls_all_returned_for_rejection():
    t = ChatCompletionsTransport(fake_resolved())
    r = t.parse_response(resp([tc("c1", "tap", {"id": 1, "reason": "a"}),
                               tc("c2", "wait", {"seconds": 1, "reason": "b"})]), (800, 1700))
    assert [a.call_id for a in r.actions] == ["c1", "c2"]


def test_bad_json_arguments_becomes_action_with_error_marker():
    bad = {"id": "c1", "type": "function", "function": {"name": "tap", "arguments": "{not json"}}
    r = ChatCompletionsTransport(fake_resolved()).parse_response(resp([bad]), (800, 1700))
    assert r.actions[0].args == {"_parse_error": True}


def test_no_tool_calls_gives_empty_actions_and_text():
    r = ChatCompletionsTransport(fake_resolved()).parse_response(resp(None, content="我想想"), (800, 1700))
    assert r.actions == [] and r.text == "我想想"


def test_choices_empty_raises_model_error_not_index_error():
    t = ChatCompletionsTransport(fake_resolved())
    _wired(t, FakeCompletions(reply={"model": "m", "usage": {}, "choices": []}))
    with pytest.raises(ModelError):
        t.decide([{"role": "user", "content": "hi"}], (800, 1700))


# ---- 出站请求从两张表拼 ----

def test_outbound_request_for_builtin_qwen():
    """出站请求组装：tools / parallel_tool_calls / extra_body / model 全来自 profile。"""
    t = ChatCompletionsTransport(fake_resolved())
    c = _wired(t, FakeCompletions(reply=resp(None, content="ok")))
    t.decide([{"role": "user", "content": "hi"}], (800, 1700))
    kw = c.calls[0]
    assert kw["model"] == "qwen3.7-plus" and kw["tools"] == tool_defs(True)
    assert kw["parallel_tool_calls"] is False and kw["tool_choice"] == "auto" and kw["max_tokens"] == 1024
    assert kw["extra_body"] == {"enable_thinking": False}


def test_coords_disabled_model_gets_tools_without_xy():
    t = ChatCompletionsTransport(fake_resolved("alibaba:qwen-vl-max"))    # 未内置 → allow_coord_tap False
    c = _wired(t, FakeCompletions(reply=resp(None, content="ok")))
    t.decide([], (800, 1700))
    assert c.calls[0]["tools"] == tool_defs(False)
    assert "extra_body" not in c.calls[0]            # 没有专有参数就不发这个键


def test_omit_params_and_two_level_extra_body():
    r = fake_resolved()
    provider = dataclasses.replace(r.provider, omit_params={"parallel_tool_calls", "tool_choice"},
                                   extra_body={"top_k": 1, "enable_thinking": True})
    r = dataclasses.replace(r, provider=provider)
    t = ChatCompletionsTransport(r)
    c = _wired(t, FakeCompletions(reply=resp(None, content="ok")))
    t.decide([], (800, 1700))
    kw = c.calls[0]
    assert "parallel_tool_calls" not in kw and "tool_choice" not in kw
    assert kw["extra_body"] == {"top_k": 1, "enable_thinking": False}     # 模型级覆盖端点级


def test_model_error_carries_response_body_and_hint():
    """端点 400 的原文必须带出来 —— 用户要靠它判断是该填 omit_params 还是模型不吃图片。"""
    class FakeStatusError(Exception):
        status_code = 400
        body = {"error": {"message": "image input not supported"}}
    t = ChatCompletionsTransport(fake_resolved())
    _wired(t, FakeCompletions(raise_=FakeStatusError("400 Bad Request")))
    with pytest.raises(ModelError) as ei:
        t.decide([], (800, 1700))
    msg = str(ei.value)
    assert "image input not supported" in msg and "子项目 4" in msg


def test_client_uses_profile_base_url_key_and_headers(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            seen.update(kw)
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    r = fake_resolved()
    r = dataclasses.replace(r, provider=dataclasses.replace(r.provider, default_headers={"X-A": "1"}))
    ChatCompletionsTransport(r)._client_or_create()
    assert seen["api_key"] == "k" and seen["base_url"] == r.provider.base_url
    assert seen["default_headers"] == {"X-A": "1"} and seen["timeout"] > 0


# ---- eval / memory 字段透传（从已删除的 test_qwen.py 迁来，随 main 合并带入）----

def test_parse_pops_eval_and_memory():
    """eval / memory 与 reason、expect 同属「模型的话」，必须弹出 args，
    否则会混进动作参数被 validate_action 拒掉。"""
    t = ChatCompletionsTransport(fake_resolved(coord_mode="pixel", allow_coord_tap=True))
    r = t.parse_response(
        resp([tc("c1", "tap", {"id": 1, "reason": "r", "eval": "yes: 进了通用页", "memory": "已看 2 个群"})]),
        (800, 1700))
    a = r.actions[0]
    assert a.args == {"id": 1}
    assert a.eval == "yes: 进了通用页" and a.memory == "已看 2 个群"


def test_eval_and_memory_default_none():
    t = ChatCompletionsTransport(fake_resolved(coord_mode="pixel", allow_coord_tap=True))
    a = t.parse_response(resp([tc("c1", "tap", {"id": 1, "reason": "r"})]), (800, 1700)).actions[0]
    assert a.eval is None and a.memory is None


def test_decide_passes_tools_and_max_tokens_through():
    """调用方给的工具表和 token 预算要压过 profile；不给时仍走 profile 的默认。"""
    t = ChatCompletionsTransport(fake_resolved())
    c = _wired(t, FakeCompletions(reply=resp(None, content="ok")))
    my_tools = [{"type": "function",
                 "function": {"name": "route", "parameters": {"type": "object", "properties": {}}}}]
    t.decide([{"role": "user", "content": "hi"}], (800, 1700), tools=my_tools, max_tokens=200)
    assert c.calls[0]["tools"] is my_tools and c.calls[0]["max_tokens"] == 200
    t.decide([{"role": "user", "content": "hi"}], (800, 1700))
    assert c.calls[1]["tools"] == tool_defs(True) and c.calls[1]["max_tokens"] == 1024
