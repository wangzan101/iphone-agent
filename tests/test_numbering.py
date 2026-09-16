"""编号只由这一帧的 OCR 框决定（spec 2026-09-14 §6.1、§6.3）。"""
import pytest
from PIL import Image

from iphone_agent.driver.geometry import Frame, Rect
from iphone_agent.harness.actions import Action, ValidationError, validate_action
from iphone_agent.perceive.observe import Perceiver
from iphone_agent.perceive.ocr import RawBox

FULL = [RawBox("通用", 0.99, 0.1, 0.8, 0.3, 0.04), RawBox("关于本机", 0.99, 0.1, 0.6, 0.3, 0.04)]


class Asker:
    """整屏解析：一个套住「通用」的按钮（融合成 both），一个在最上面的纯视觉返回箭头。"""
    def ask_json(self, prompt, images, max_tokens=None):
        return {"screen": {"app": "设置", "name": "通用", "same_as": None, "anchors": []},
                "elements": [{"kind": "button", "label": "通用按钮", "box": [80, 150, 420, 210]},
                             {"kind": "icon", "label": "返回箭头", "box": [20, 50, 60, 90]}]}


def ocr(im):
    if im.size == (600, 1200):
        return list(FULL)
    return [RawBox("关于本机·放大", 0.99, 0.1, 0.4, 0.3, 0.2)]    # zoom 里重读出来的


def frame(fid=1):
    return Frame(Image.new("RGB", (600, 1200), "white"), 600, 1200, Rect(0, 0, 300, 600), 0.0, fid)


def ocr_ids(o):
    return [(e.id, e.text) for e in o.elements if e.source != "vision"]


def test_ocr_ids_do_not_depend_on_whether_the_full_screen_was_parsed():
    per = Perceiver(ocr=ocr, asker=Asker())
    with per.task_scope(None, "always"):
        full = per.observe(frame())
    with per.task_scope(None, "off"):
        plain = per.observe(frame(2))
    assert ocr_ids(full) == ocr_ids(plain) == [(1, "通用"), (2, "关于本机")]
    assert full.element(1).source == "both" and full.element(1).kind == "button"
    vis = [e for e in full.elements if e.source == "vision"]
    assert [(e.id, e.text) for e in vis] == [(3, "返回箭头")], "纯视觉项编在 OCR 后面，哪怕它在屏幕最上面"


def test_zoom_keeps_outside_ids_and_never_reuses_replaced_ones():
    per = Perceiver(ocr=ocr, asker=None)
    base = per.observe(frame())
    z = per.zoom(base, (0, 400, 600, 500))
    assert z.element(1).text == "通用", "放大区域外的编号不变"
    assert z.element(2) is None, "被替换掉的旧编号在这次 zoom 里不再使用"
    assert z.element(3).text == "关于本机·放大"
    with pytest.raises(ValidationError) as ei:
        validate_action(Action("tap", {"id": 2}, "r", None, "c"), z)
    assert ei.value.code == "stale_element_id"
