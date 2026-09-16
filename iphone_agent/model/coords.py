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


# 带坐标的参数形状，按「一整组都在才算数」判：点 (x, y) 和框 (x1, y1, x2, y2)。
# ⚠ 2026-09-16 真机事故，两种死法，根子都是「桥只给 tap 换算」：
#   1. 拒绝：模型给 zoom 框 {"x1":750,"y1":60,"x2":980,"y2":160}（规规矩矩的 0-1000），
#      框没换算就拿去和图像像素比，624 宽的屏上直接 out_of_image，而提示语还教它
#      「要用 0-1000 的归一化值」—— 它本来就是。屏幕右边约 38%（搜索、⋮ 菜单、开关）
#      整片不可达；那一跑连试 5 次，任务以 model_error 收场。
#   2. 静默错：x 小于 624 的框会被**收下**，然后按像素裁 —— 裁出来的不是模型框的那块，
#      它却以为自己看清了（例如 {"x1":480,"y1":890,"x2":620,"y2":940}）。
#   按需看图（on_demand）下最致命：元素表只有 OCR，zoom 是模型认图标和开关状态的主要手段。
#   所以这里认的是**坐标键**，不是动作名。加新动作时把它的坐标键加进这张表，
#   别去桥里按动作名分支 —— 一个规则一个入口（CLAUDE.md §7），
#   上一次同一条规矩写在两处，代价就是上面那两跑。
COORD_GROUPS = (("x", "y"), ("x1", "y1", "x2", "y2"))
_X_KEYS = frozenset({"x", "x1", "x2"})


def to_pixels(args: dict, coord_mode: str, image_size: tuple[int, int]) -> dict:
    """norm1000 → 图像像素；pixel 原样。只在某一组坐标键**整组都在**时动。

    换算失败（值不是数字）**原样返回**而不是抛：让 validate_action 以 invalid_args 拒绝、
    给模型一句提示，任务继续 —— 而不是在桥里炸成 ModelError 让整个任务收场。
    """
    if coord_mode != "norm1000":
        return args
    keys = next((g for g in COORD_GROUPS if all(k in args for k in g)), None)
    if keys is None:
        return args
    W, H = image_size
    try:
        conv = {k: round(float(midpoint(args[k])) / 1000 * (W if k in _X_KEYS else H))
                for k in keys}
    except (TypeError, ValueError):
        return args
    return {**args, **conv}
