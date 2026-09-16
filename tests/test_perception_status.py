"""视觉解析失败要看得见（开发原则 §3、spec §3.1）：状态进 observation["perception"]。"""
from types import SimpleNamespace

from PIL import Image

from iphone_agent.driver.geometry import Frame, Rect
from iphone_agent.harness.runlog import observation_snapshot
from iphone_agent.perceive.observe import Perceiver
from iphone_agent.perceive.ocr import RawBox
from iphone_agent.perceive.screen import parse_screen, parse_screen_status


class Asker:
    def __init__(self, data):
        self.data = data

    def ask_json(self, prompt, images):
        return self.data


def img():
    return Image.new("RGB", (600, 1200), "white")


def test_status_values():
    assert parse_screen_status(img(), None) == ([], "off")
    assert parse_screen_status(img(), Asker(None)) == ([], "failed")
    assert parse_screen_status(img(), Asker("not json")) == ([], "failed")
    assert parse_screen_status(img(), Asker([])) == ([], "empty")
    assert parse_screen_status(img(), Asker([{"kind": "x"}]))[1] == "failed", "全是坏条目 ≠ 模型说没有"
    items, status = parse_screen_status(img(), Asker([{"kind": "button", "label": "完成", "box": [0, 0, 100, 100]}]))
    assert status == "ok" and [i.label for i in items] == ["完成"]
    assert [i.label for i in parse_screen(img(), Asker({"elements": [{"label": "a", "box": [0, 0, 9, 9]}]}))] == ["a"]


def test_observation_carries_perception_status_into_the_log():
    per = Perceiver(ocr=lambda im: [RawBox("设置", 0.99, 0.1, 0.8, 0.3, 0.04)], asker=Asker(None))
    o = per.observe(Frame(img(), 600, 1200, Rect(0, 0, 300, 600), 0.0, 1))
    p = o.perception
    assert (p["ocr"], p["vision"], p["screen"], p["parse"]) == (1, "failed", "failed", "always")
    assert observation_snapshot(o, "f.png")["perception"]["vision"] == "failed"


def test_observe_text_is_ocr_only_and_numbers_observations_like_observe():
    """翻主屏找 App 只要读标签（2026-09-14）：observe_text 不做整屏解析、不挑认屏候选，
    perception 如实写 off；observation_id 和 observe 共用同一个计数。"""

    class Counting:
        calls = 0

        def ask_json(self, prompt, images):
            Counting.calls += 1
            return None

    per = Perceiver(ocr=lambda im: [RawBox("设置", 0.99, 0.1, 0.8, 0.3, 0.04)], asker=Counting())
    f = Frame(img(), 600, 1200, Rect(0, 0, 300, 600), 0.0, 1)
    a = per.observe_text(f)
    assert Counting.calls == 0, "observe_text 不能问视觉"
    assert (a.perception["vision"], a.perception["screen"], a.perception["parse"]) == ("off", "off", None)
    assert a.screen is None and a.screen_candidates == []
    assert [e.text for e in a.elements] == ["设置"]
    b = per.observe(f)
    c = per.observe_text(f)
    assert Counting.calls == 1
    assert (a.observation_id, b.observation_id, c.observation_id) == (1, 2, 3)


def test_snapshot_without_perception_attribute_is_unchanged():
    o = SimpleNamespace(window_rect=Rect(0, 0, 1, 1), width_px=1, height_px=1, ahash=0, elements=[])
    assert "perception" not in observation_snapshot(o, "f.png")
