import json
from pathlib import Path

import pytest

from iphone_agent.harness import verdict as V

FIX = Path(__file__).parent / "fixtures" / "verdict"


def test_to_json_omits_empty_why_and_seen():
    assert V.Verdict("met", "text").to_json() == {"state": "met", "by": "text"}
    assert V.Verdict("not_met", "readback", seen="iOS8").to_json() == {
        "state": "not_met", "by": "readback", "seen": "iOS8"}


def test_new_results_carry_their_own_verdict():
    r = {"ok": True, "changed": True, "verdict": {"state": "not_met", "by": "readback", "seen": "iOS8"},
         "verified": "true"}                      # 旧字段说 true 也不看：新结果以 verdict 为准
    assert V.of("type", {"text": "18"}, r) == V.Verdict("not_met", "readback", seen="iOS8")


@pytest.mark.parametrize("name,args,result,state,by", [
    # 不改画面的动作、没执行成的动作：不该有结论
    ("observe", {}, {"ok": True, "changed": None}, "n/a", "neutral"),
    ("key", {"name": "left"}, {"ok": True, "changed": None}, "n/a", "neutral"),
    ("wait", {"seconds": 1}, {"ok": True, "changed": False}, "n/a", "neutral"),
    ("tap", {"x": 1, "y": 1}, {"ok": False, "error": "out_of_image"}, "n/a", "failed"),
    # tap
    ("tap", {}, {"ok": True, "changed": True, "expect_check": {"met": True, "by": "text", "matched": "关于本机"}}, "met", "text"),
    ("tap", {}, {"ok": True, "changed": True, "expect_check": {"met": True, "by": "vision"}}, "met", "vision"),
    ("tap", {}, {"ok": True, "changed": True, "expect_check": {"met": False, "by": "vision"},
                 "judged": {"worked": False, "why": "进了登录页", "on_change": True}}, "not_met", "vision"),
    # 变了 + 看图说没达到预期、没有 expect_check：2026-09-11 之前的旧留档形状
    ("tap", {}, {"ok": True, "changed": True, "judged": {"worked": False, "why": "进的是通知页"}}, "not_met", "vision"),
    ("tap", {}, {"ok": True, "changed": True, "expect_check": {"met": None, "by": "missing"}}, "unknown", "missing"),
    ("tap", {}, {"ok": True, "changed": True, "expect_check": {"met": None, "by": "unverified"}}, "unknown", "unverified"),
    # R7：像素说没变、没人复核 → not_met，不是 unknown
    ("tap", {}, {"ok": True, "changed": False, "expect_check": {"met": None, "by": "missing"}}, "not_met", "pixels"),
    ("tap", {}, {"ok": True, "changed": False, "judged": {"worked": False, "why": "没动"},
                 "expect_check": {"met": False, "by": "vision"}}, "not_met", "vision"),
    # 主屏点图标核出了别的 App
    ("tap", {"target": "icon_above"}, {"ok": True, "changed": True, "expect_check": {"met": None, "by": "missing"},
                                       "identity": {"verified": False, "seen": "Gemini", "by": "judge"}}, "not_met", "identity"),
    ("tap", {"target": "icon_above"}, {"ok": True, "changed": True, "expect_check": {"met": None, "by": "missing"},
                                       "identity": {"verified": True, "by": "screen_label", "seen": "设置"}}, "met", "identity"),
    # type
    ("type", {"text": "a"}, {"ok": True, "changed": True, "verified": "true"}, "met", "text"),
    ("type", {"text": "a"}, {"ok": True, "changed": False, "verified": "false"}, "not_met", "text"),
    ("type", {"text": "a"}, {"ok": True, "changed": True, "verified": "wrong"}, "not_met", "readback"),
    ("type", {"text": "a"}, {"ok": True, "changed": True, "verified": "unknown"}, "unknown", "unverified"),
    ("type", {"text": "你好"}, {"ok": True, "changed": True, "picked": "你好"}, "unknown", "unverified"),
    ("type", {"text": "你好"}, {"ok": True, "changed": False, "picked": "你好", "judged": {"worked": False}}, "not_met", "vision"),
    # picked 的结果现在也带 verified（Task 4 之后点候选走同一条核对）：有 verified 就按它走通用映射
    ("type", {"text": "你好"}, {"ok": True, "changed": True, "picked": "你好", "verified": "true"}, "met", "text"),
    ("type", {"text": "你好"}, {"ok": True, "changed": True, "picked": "你好", "verified": "wrong"}, "not_met", "readback"),
    # open_app
    ("open_app", {"name": "设置"}, {"ok": True, "changed": True, "identity": {"verified": True, "by": "screen_label"}}, "met", "identity"),
    ("open_app", {"name": "设置"}, {"ok": True, "changed": True, "identity": {"verified": None, "by": "judge"}}, "unknown", "unverified"),
    ("open_app", {"name": "设置"}, {"ok": True, "changed": True}, "unknown", "unverified"),
    # scroll / key / erase：预期可选
    ("scroll", {"direction": "down"}, {"ok": True, "changed": True}, "met", "pixels"),
    ("scroll", {"direction": "down"}, {"ok": True, "changed": False}, "not_met", "pixels"),
    ("key", {"name": "return"}, {"ok": True, "changed": True, "judged": {"worked": False, "why": "x", "on_change": True}}, "not_met", "vision"),
    ("erase", {"count": 2}, {"ok": True, "changed": True, "judged": {"worked": True, "on_change": True}}, "met", "vision"),
    # switch_ime：旧留档里它从来没有结论
    ("switch_ime", {}, {"ok": True, "changed": True}, "unknown", "unverified"),
    # scroll_until / collect
    ("scroll_until", {"text": "关于本机"}, {"ok": True, "found": True, "screens": 2}, "met", "text"),
    ("scroll_until", {"text": "关于本机"}, {"ok": True, "found": False, "screens": 3, "never_moved": True}, "unknown", "never_moved"),
    ("scroll_until", {"text": "关于本机"}, {"ok": True, "found": False, "screens": 3, "reached_end": True}, "not_met", "text"),
    ("collect", {}, {"ok": True, "count": 12, "screens": 3}, "met", "text"),
    ("collect", {}, {"ok": True, "count": 4, "screens": 3, "never_moved": True}, "unknown", "never_moved"),
    # 剧本工具
    ("she-zhi__kan-ban-ben", {}, {"ok": True}, "n/a", "procedure"),
])
def test_of_derives_from_legacy_fields(name, args, result, state, by):
    v = V.of(name, args, result)
    assert (v.state, v.by) == (state, by)


def _fixture(name):
    d = json.loads((FIX / name).read_text(encoding="utf-8"))
    return d["action"]["name"], d["action"].get("args") or {}, d["result"]


def test_scene_xhs_mistap_is_unknown_and_says_so():
    name, args, result = _fixture("tap_no_expect.json")
    v = V.of(name, args, result)
    assert (v.state, v.by) == ("unknown", "missing")
    text = V.say(v, name, result)
    assert "没有核实" in text
    assert "没有落在任何已知元素上" in text          # result.tapped.hits == []


def test_scene_safari_type_is_unknown_in_the_archive():
    name, args, result = _fixture("type_partial_landing.json")
    v = V.of(name, args, result)
    assert (v.state, v.by) == ("unknown", "unverified")
    assert "没有核实" in V.say(v, name, result)


def test_say_is_silent_for_met_and_neutral():
    assert V.say(V.Verdict("met", "text"), "tap", {}) is None
    assert V.say(V.Verdict("n/a", "neutral"), "observe", {}) is None


def test_say_reports_what_was_read_back():
    v = V.Verdict("not_met", "readback", seen="iOS8")
    assert V.say(v, "type", {"ok": True}, ) == "想打的字没有按原样落进去：输入框里现在是「iOS8」。"


def test_say_not_met_with_an_existing_hint_adds_nothing():
    v = V.Verdict("not_met", "vision", why="进了登录页")
    assert V.say(v, "tap", {"hint": "画面变了，但看图复核说没达到预期：进了登录页。先确认自己在哪。"}) is None


def test_quiet_silences_unverified_but_not_missing():
    assert V.say(V.Verdict("unknown", "unverified"), "tap", {}, quiet=True) is None
    assert "没有核实" in V.say(V.Verdict("unknown", "missing"), "tap", {}, quiet=True)
