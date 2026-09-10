from iphone_agent.perceive.ocr import RawBox, vision_box_to_pixels


def test_vision_box_flips_y():
    # Vision：左下原点，(x, y, w, h) 归一化。图像 100x200。
    box = RawBox(text="通用", confidence=0.9, x=0.1, y=0.8, w=0.2, h=0.05)
    x1, y1, x2, y2 = vision_box_to_pixels(box, 100, 200)
    # x1 = 0.1*100 = 10; x2 = 0.3*100 = 30
    # y1 = (1-0.8-0.05)*200 = 30; y2 = (1-0.8)*200 = 40
    assert (x1, y1, x2, y2) == (10, 30, 30, 40)


def test_vision_box_bottom_left_origin_maps_to_bottom():
    box = RawBox(text="x", confidence=1.0, x=0.0, y=0.0, w=0.5, h=0.1)
    x1, y1, x2, y2 = vision_box_to_pixels(box, 100, 200)
    assert y2 == 200 and y1 == 180
