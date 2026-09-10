"""changed 的定义是「检测到明显视觉差异」，不是「动作生效」（spec §5.2）。"""
from __future__ import annotations

from dataclasses import dataclass

from iphone_agent import config
from iphone_agent.perceive.hashing import hamming


@dataclass(frozen=True)
class ChangeResult:
    changed: bool
    hamming: int
    text_diff: int


def did_change(before, after) -> ChangeResult:
    hd = hamming(before.ahash, after.ahash)
    td = len(before.text_set ^ after.text_set)
    changed = hd > config.AHASH_CHANGED_THRESHOLD or td > config.TEXT_DIFF_THRESHOLD
    return ChangeResult(changed=changed, hamming=hd, text_diff=td)


def local_mad(before, after, cx: int, cy: int, radius: int | None = None) -> float:
    """两张图在 (cx, cy) 周围一小块里的平均绝对灰度差。

    ⚠ 为什么需要它：整屏判据（aHash 8x8 + 文本集合）**看不见小控件的变化**。
    2026-09-08 实测：开关翻转两次，整屏 aHash 汉明都是 0、文本差都是 0，
    而局部 MAD 是 17.0 / 16.98；点空白处三次局部 MAD 都是 0.0。

    不修的后果不是「少报一次变化」：模型点了开关，工具告诉它没变化，
    它以为没点中就再点一次 —— 又翻回去了，而且熔断还把这算成无进展。
    """
    from iphone_agent import config
    # 半径按图像宽度折算，不写死像素 —— 见 config.LOCAL_PATCH_RATIO 的注释。
    r = round(before.width * config.LOCAL_PATCH_RATIO) if radius is None else radius
    box = (max(0, cx - r), max(0, cy - r), cx + r, cy + r)
    a = before.crop(box).convert("L")
    b = after.crop(box).convert("L")
    if a.size != b.size or a.size[0] == 0 or a.size[1] == 0:
        return 0.0
    pa, pb = a.tobytes(), b.tobytes()
    if not pa:
        return 0.0
    return sum(abs(x - y) for x, y in zip(pa, pb, strict=True)) / len(pa)
