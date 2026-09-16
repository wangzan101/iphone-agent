from iphone_agent.model.coords import midpoint, to_pixels


def test_norm1000_scales_by_image_size():
    assert to_pixels({"x": 500, "y": 500, "id": 3}, "norm1000", (800, 1700)) == {"x": 400, "y": 850, "id": 3}


def test_pixel_mode_passes_through_untouched():
    args = {"x": 100, "y": 200}
    assert to_pixels(args, "pixel", (800, 1700)) is args


def test_bbox_takes_midpoint_before_scaling():
    """模型把坐标吐成 [x1, x2] 包围盒（B129，已知模型行为）：取中点再换算。"""
    assert to_pixels({"x": [400, 600], "y": [450, 550]}, "norm1000", (800, 1700)) == {"x": 400, "y": 850}
    assert midpoint([1, 3]) == 2 and midpoint(7) == 7 and midpoint([True, False]) == [True, False]


def test_missing_axis_left_alone():
    """只有 x 没有 y：不换算，validate_action 会拒（tap 需要 id 或 (x,y) 二选一）。"""
    assert to_pixels({"x": 500}, "norm1000", (800, 1700)) == {"x": 500}


def test_zoom_box_scales_all_four_corners():
    """⚠ 2026-09-16：zoom 的框以前**根本不换算**（只认 x/y 两个键），
    模型给的 0-1000 被当像素用 —— 624 宽的屏上，x>624 的框一律 out_of_image。"""
    assert to_pixels({"x1": 750, "y1": 60, "x2": 980, "y2": 160}, "norm1000", (624, 1388)) == {
        "x1": 468, "y1": 83, "x2": 612, "y2": 222}


def test_zoom_box_untouched_in_pixel_mode():
    args = {"x1": 480, "y1": 890, "x2": 620, "y2": 940}
    assert to_pixels(args, "pixel", (624, 1388)) is args


def test_zoom_box_with_a_bad_corner_left_for_validation():
    args = {"x1": "abc", "y1": 60, "x2": 980, "y2": 160}
    assert to_pixels(args, "norm1000", (624, 1388)) is args


def test_partial_zoom_box_left_alone():
    """少一个角就不是框：原样交给 validate_action 以 invalid_args 拒（x2 必须是有限数字）。"""
    args = {"x1": 750, "y1": 60, "x2": 980}
    assert to_pixels(args, "norm1000", (624, 1388)) == args


def test_unconvertible_value_left_for_validation():
    """原来 float("abc") 会在桥里炸成 ModelError，整个任务以 model_error 收场；
    现在原样交给 validate_action，它会以 invalid_args 拒绝并给模型提示，任务继续。"""
    args = {"x": "abc", "y": 5}
    assert to_pixels(args, "norm1000", (800, 1700)) is args
