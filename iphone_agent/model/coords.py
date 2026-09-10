"""桥解析完之后统一做的一步：把模型报的坐标按该模型的约定换成图像像素。

它不是协议的事 —— OpenAI 兼容桥和将来的 Anthropic 桥都用它。
"""
from __future__ import annotations


def midpoint(v):
    """容忍模型把坐标吐成 [x1, x2] 包围盒（B129，actions.py 的 _num 也防这个）：取中点。
    非法类型原样返回，留给 validate_action 校验拒绝。bool 是 int 子类，必须排除。"""
    if (isinstance(v, list) and len(v) == 2
            and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in v)):
        return (v[0] + v[1]) / 2
    return v


def to_pixels(args: dict, coord_mode: str, image_size: tuple[int, int]) -> dict:
    """norm1000 → 图像像素；pixel 原样。只在 x、y 都在时动。

    换算失败（值不是数字）**原样返回**而不是抛：让 validate_action 以 invalid_args 拒绝、
    给模型一句提示，任务继续 —— 而不是在桥里炸成 ModelError 让整个任务收场。
    """
    if coord_mode != "norm1000" or "x" not in args or "y" not in args:
        return args
    W, H = image_size
    try:
        x = round(float(midpoint(args["x"])) / 1000 * W)
        y = round(float(midpoint(args["y"])) / 1000 * H)
    except (TypeError, ValueError):
        return args
    return {**args, "x": x, "y": y}
