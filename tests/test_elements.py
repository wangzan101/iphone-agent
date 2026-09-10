from PIL import Image

from iphone_agent.driver.geometry import Frame, Rect
from iphone_agent.perceive.elements import build_observation
from iphone_agent.perceive.ocr import RawBox


def frame(w=200, h=400):
    return Frame(image=Image.new("RGB", (w, h), "white"), width_px=w, height_px=h,
                 window_rect=Rect(0, 0, w / 2, h / 2), ts=0.0, frame_id=7)


def test_ids_ordered_top_to_bottom_left_to_right():
    boxes = [
        RawBox("底部", 0.9, x=0.1, y=0.05, w=0.2, h=0.05),   # 下方
        RawBox("右上", 0.9, x=0.6, y=0.9, w=0.2, h=0.05),    # 上方偏右
        RawBox("左上", 0.9, x=0.1, y=0.9, w=0.2, h=0.05),    # 上方偏左
    ]
    obs = build_observation(frame(), boxes, observation_id=3)
    assert [e.text for e in obs.elements] == ["左上", "右上", "底部"]
    assert [e.id for e in obs.elements] == [1, 2, 3]
    assert obs.observation_id == 3 and obs.frame_id == 7


def test_elements_text_header_and_lines():
    obs = build_observation(frame(), [RawBox("通用", 0.87, 0.1, 0.5, 0.2, 0.05)], observation_id=2)
    lines = obs.elements_text.splitlines()
    # 头部必须带图像尺寸：模型要点 OCR 看不见的图标时，得知道坐标空间多大
    # （2026-09-07 实测：不给尺寸，模型按 850 宽估坐标，实际 644，越界被拒）
    assert lines[0].startswith("observation #2")
    assert "200x400" in lines[0]
    assert lines[1].startswith("[1] 通用 (")
    assert lines[1].endswith("0.87")


def test_marked_image_same_size_and_text_set():
    obs = build_observation(frame(), [RawBox("A", 0.9, 0.1, 0.5, 0.2, 0.05)], observation_id=1)
    assert obs.marked_image.size == obs.image.size
    assert obs.text_set == {"A"}
    assert isinstance(obs.ahash, int)


# --- 坐标约定：头部说的必须就是适配器做的 ---
#
# 2026-09-08 真机踩过：当时还是全局配置项，标定翻成了 norm1000，头部那行却还写着
# 「图像 624x1388 像素（坐标就用这个空间）」。模型照头部给了像素 y=1280，
# 适配器按 norm1000 换算成 1280/1000*1388 = 1777，越界被拒；模型换个像素值
# 再试，再拒 —— 连续五次被拒，任务终止。
#
# 一个规则、两个入口、只改了一个。下面两个测试把它们绑在一起。

def _head(mode):
    obs = build_observation(frame(), [RawBox("甲", 0.9, x=0.1, y=0.5, w=0.2, h=0.05)],
                            observation_id=1, coord_mode=mode)
    assert obs.coord_mode == mode            # 头部说的和字段记的必须是同一个
    return obs.elements_text.splitlines()[0]


def test_header_says_normalized_when_that_is_what_the_adapter_expects():
    head = _head("norm1000")
    assert "0-1000" in head, f"没告诉模型用归一化坐标：{head}"


def test_element_coordinates_use_the_same_convention_as_tap():
    """⚠ 同一段文字里不能有两套坐标约定。

    2026-09-08 真机踩到：头部写着「给坐标时用 0-1000 的归一化值」，
    而元素行印的是**像素**中心 (187,673)。模型从列表读到那种数、在那个尺度上
    做空间推理，然后自然按同一尺度给 tap 坐标 —— 连续 5 次 out_of_image 被熔断。
    它不是没看提示，是**列表本身在教它用像素**。
    """
    obs = build_observation(frame(200, 400),
                            [RawBox("甲", 0.9, x=0.4, y=0.5, w=0.2, h=0.05)],
                            observation_id=1, coord_mode="norm1000")
    line = obs.elements_text.splitlines()[1]
    xs = line[line.index("(") + 1:line.index(")")].split(",")
    x, y = int(xs[0]), int(xs[1])
    assert 0 <= x <= 1000 and 0 <= y <= 1000, f"元素坐标不在归一化范围里：{line}"
    assert x != 100, "看着像像素（0.5*200=100），没折算"


def test_pixel_mode_still_prints_pixels():
    obs = build_observation(frame(200, 400),
                            [RawBox("甲", 0.9, x=0.4, y=0.5, w=0.2, h=0.05)],
                            observation_id=1, coord_mode="pixel")
    line = obs.elements_text.splitlines()[1]
    x = int(line[line.index("(") + 1:line.index(",")])
    assert x == 100, f"像素模式下该印像素：{line}"


def test_header_says_pixels_when_the_adapter_takes_pixels():
    head = _head("pixel")
    assert "0-1000" not in head, f"适配器要像素，头部却让模型给归一化值：{head}"
    assert "200x400" in head


def test_out_of_image_error_explains_the_convention():
    """光说「越界」，模型只会换个像素值再试一遍。真机上它连试了五次。
    提示语读的是 obs.coord_mode —— 和头部同一个来源。"""
    from types import SimpleNamespace

    import pytest

    from iphone_agent.harness.actions import Action, ValidationError, validate_action
    o = SimpleNamespace(observation_id=1, width_px=624, height_px=1388, elements=[], coord_mode="norm1000")
    o.element = lambda eid: None
    with pytest.raises(ValidationError) as ei:
        validate_action(Action("tap", {"x": 222, "y": 1777}, "r", None, "c"), o)
    assert "0-1000" in ei.value.message, ei.value.message
    o.coord_mode = "pixel"
    with pytest.raises(ValidationError) as ei:
        validate_action(Action("tap", {"x": 222, "y": 1777}, "r", None, "c"), o)
    assert "0-1000" not in ei.value.message


def test_perceiver_passes_coord_mode_through():
    from iphone_agent.perceive.observe import Perceiver
    per = Perceiver(ocr=lambda img: [RawBox("甲", 0.9, x=0.1, y=0.5, w=0.2, h=0.05)], coord_mode="pixel")
    assert per.coord_mode == "pixel"
    o = per.observe(frame())
    assert o.coord_mode == "pixel" and "0-1000" not in o.elements_text.splitlines()[0]
