"""看图解析多答「这一屏是什么」（spec 2026-09-12 §3）。"""
import hashlib

import pytest
from PIL import Image

from iphone_agent.driver.geometry import Frame, Rect
from iphone_agent.harness.runlog import frame_name, observation_snapshot
from iphone_agent.perceive.observe import Perceiver
from iphone_agent.perceive.ocr import RawBox
from iphone_agent.perceive.screen import (
    PROMPT,
    PROMPT_ELEMENTS,
    PROMPT_SCREEN,
    Candidate,
    ScreenLabel,
    parse_screen_full,
    parse_screen_label,
    screen_prompt,
)


class Asker:
    def __init__(self, data):
        self.data, self.prompts = data, []

    def ask_json(self, prompt, images, max_tokens=None):
        self.prompts.append(prompt)
        return self.data


class Ctx:
    def __init__(self, cands=(), boom=False):
        self.cands, self.boom, self.seen = list(cands), boom, []

    def candidates(self, texts):
        if self.boom:
            raise RuntimeError("twin broke")
        self.seen.append(list(texts))
        return self.cands


def img():
    return Image.new("RGB", (600, 1200), "white")


def frame(fid=1):
    return Frame(img(), 600, 1200, Rect(0, 0, 300, 600), 0.0, fid)


GOOD = {"app": "设置", "name": "通用", "same_as": None, "anchors": ["通用", "关于本机"]}
EL = [{"kind": "button", "label": "完成", "box": [0, 0, 100, 100]}]


def test_elements_prompt_is_byte_identical_to_before():
    assert PROMPT is PROMPT_ELEMENTS
    assert hashlib.sha256(PROMPT_ELEMENTS.encode("utf-8")).hexdigest() == \
        "f1470183d336b93d12d9b79a5d7680d360135e3ee82febabb04263657e3b5ac9"


def test_screen_prompt_shares_the_element_spec_and_lists_candidates():
    head = PROMPT_ELEMENTS.split("\n\n只输出一个 JSON 数组")[0]
    assert PROMPT_SCREEN.startswith(head)
    assert "same_as" in PROMPT_SCREEN and "「系统」" in PROMPT_SCREEN and "「不确定」" in PROMPT_SCREEN
    assert screen_prompt(()) == PROMPT_SCREEN
    p = screen_prompt([Candidate(1, "bei-wang-lu", "备忘录", "s_a", "iCloud全部备忘录"),
                       Candidate(2, "bei-wang-lu", "备忘录", "s_b", "文件夹")])
    assert "1. 备忘录 / iCloud全部备忘录\n2. 备忘录 / 文件夹" in p and p.startswith(PROMPT_SCREEN)


@pytest.mark.parametrize("data,status", [
    (None, "failed"), ("not json", "failed"), (3, "failed"),
    ([], "missing"), ({"elements": []}, "missing"),
    ({"screen": "x"}, "invalid"),
])
def test_label_status_values(data, status):
    assert parse_screen_label(data, 0) == (None, status)


@pytest.mark.parametrize("patch", [
    {"app": ""}, {"app": "  "}, {"app": 3}, {"app": "a" * 31},
    {"name": ""}, {"name": None},
    {"same_as": True}, {"same_as": 0}, {"same_as": 2}, {"same_as": "1"},
])
def test_invalid_screen_segments(patch):
    assert parse_screen_label({"screen": {**GOOD, **patch}}, 1) == (None, "invalid")


def test_valid_label_is_cleaned():
    raw = {**GOOD, "app": " 设置 ", "name": "名" * 50, "same_as": 1,
           "anchors": [" 通用 ", 3, "", "通用", "x" * 31, "a", "b", "c", "d", "e"]}
    lab, st = parse_screen_label({"screen": raw}, 1)
    assert st == "ok"
    assert lab == ScreenLabel("设置", "名" * 40, 1, ("通用", "a", "b", "c", "d"))
    assert lab.to_json() == {"app": "设置", "name": "名" * 40, "same_as": 1, "anchors": ["通用", "a", "b", "c", "d"]}
    assert parse_screen_label({"screen": {**GOOD, "anchors": "通用"}}, 0)[0].anchors == ()


def test_broken_screen_segment_keeps_elements():
    sp = parse_screen_full(img(), Asker({"screen": {"app": ""}, "elements": EL}))
    assert sp.vision == "ok" and [i.label for i in sp.items] == ["完成"]
    assert sp.label is None and sp.label_status == "invalid"
    old = parse_screen_full(img(), Asker(EL))
    assert old.vision == "ok" and old.label_status == "missing"
    assert parse_screen_full(img(), None).label_status == "off"
    assert parse_screen_full(img(), Asker(None)).label_status == "failed"


def test_observe_carries_label_candidates_and_status():
    asker = Asker({"screen": {**GOOD, "same_as": 1}, "elements": EL})
    per = Perceiver(ocr=lambda im: [RawBox("通用", 0.99, 0.1, 0.8, 0.3, 0.04)], asker=asker)
    ctx = Ctx([Candidate(1, "she-zhi", "设置", "s_1", "通用")])
    with per.task_scope(ctx, "always"):
        o = per.observe(frame())
    assert ctx.seen == [["通用"]]
    assert "1. 设置 / 通用" in asker.prompts[-1]
    assert o.screen.same_as == 1 and (o.perception["vision"], o.perception["screen"]) == ("ok", "ok")
    snap = observation_snapshot(o, frame_name(o.frame_id))
    assert snap["frame_file"] == "frame_001.png"
    assert snap["screen"]["name"] == "通用"
    assert snap["screen_candidates"] == [{"index": 1, "app_id": "she-zhi", "app_display": "设置",
                                          "screen_id": "s_1", "name": "通用"}]
    o2 = per.observe(frame(2))           # 解除绑定之后：没有候选段
    assert asker.prompts[-1] == PROMPT_SCREEN and o2.screen_candidates == []


def test_task_scope_is_released_on_exception():
    per = Perceiver(ocr=lambda im: [])
    with pytest.raises(RuntimeError), per.task_scope(Ctx(), "on_demand"):
        raise RuntimeError("x")
    assert per._identity is None
    assert per.mode == "always"


def test_broken_context_means_no_candidates_not_a_failed_observation():
    asker = Asker({"screen": GOOD, "elements": EL})
    per = Perceiver(ocr=lambda im: [RawBox("通用", 0.99, 0.1, 0.8, 0.3, 0.04)], asker=asker)
    with per.task_scope(Ctx(boom=True), "always"):
        o = per.observe(frame())
    assert asker.prompts[-1] == PROMPT_SCREEN and o.screen.name == "通用"


def test_screen_parse_off_is_off():
    per = Perceiver(ocr=lambda im: [], asker=Asker({"screen": GOOD, "elements": EL}))
    with per.task_scope(None, "off"):
        o = per.observe(frame())
    assert (o.perception["vision"], o.perception["screen"], o.perception["parse"]) == ("off", "off", None)
    assert o.screen is None and "screen" not in observation_snapshot(o, "f.png")


def test_zoom_keeps_the_elements_prompt():
    asker = Asker({"screen": GOOD, "elements": EL})
    per = Perceiver(ocr=lambda im: [], asker=asker)
    base = per.observe(frame())
    per.zoom(base, (0, 0, 300, 300))
    assert asker.prompts[-1] == PROMPT_ELEMENTS
