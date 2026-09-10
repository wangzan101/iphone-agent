"""坐标系：harness 只见图像像素（左上原点）；driver 只见全局屏幕坐标。本文件是两者之间唯一的换算。"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


class OutOfWindow(ValueError):
    """坐标落在窗口外。硬指标：零窗口外注入。"""


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def same_as(self, other: Rect, tol: float = 1.0) -> bool:
        return (abs(self.x - other.x) <= tol and abs(self.y - other.y) <= tol
                and abs(self.w - other.w) <= tol and abs(self.h - other.h) <= tol)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


@dataclass(frozen=True)
class Frame:
    image: Image.Image
    width_px: int
    height_px: int
    window_rect: Rect
    ts: float
    frame_id: int

    @property
    def scale(self) -> float:
        """倍率由实际图像宽度除以窗口宽度算出，不硬编码 Retina。"""
        return self.width_px / self.window_rect.w


def content_bounds(image, threshold: int = 20,
                   sample_fracs: tuple[float, ...] = (0.3, 0.5, 0.7)) -> Rect:
    """手机屏幕在抓帧图里的实际范围（图像像素）。

    ⚠ 窗口矩形不等于手机屏幕：镜像 App 在窗口内画了带圆角的机身，四周留黑边。
    按窗口几何算锚点的动作（滚轮的落点）必须用这个范围，否则锚点落在黑边里，
    hit-test 命不中手机屏幕 —— 2026-09-05 实测：边缘返回手势起点取「窗口左缘 +2 点」
    对应图像 x=4，灰度 0，完全没反应；x=16 才进入内容区（实测左右各约 12 图像像素黑边）。

    只在竖直方向中段取样：顶部有灵动岛（本来就是黑的）、四角是圆角，都会把边界量偏。
    """
    g = image.convert("L")
    W, H = g.size
    px = g.load()
    lefts, rights = [], []
    for fr in sample_fracs:
        y = min(H - 1, int(H * fr))
        xs = [x for x in range(W) if px[x, y] > threshold]
        if xs:
            lefts.append(min(xs))
            rights.append(max(xs))
    if not lefts:  # 整屏全黑（锁屏、DRM 遮挡）：退回整幅图，别让调用方拿到空矩形
        return Rect(0, 0, W, H)
    left, right = max(lefts), min(rights)  # 取最保守的一组，确保落在内容内
    tops, bots = [], []
    for fr in sample_fracs:  # 竖直方向同样多列取样再取保守值
        x = left + int((right - left) * fr)
        ys = [y for y in range(H) if px[x, y] > threshold]
        if ys:
            tops.append(min(ys))
            bots.append(max(ys))
    top, bot = (max(tops), min(bots)) if tops else (0, H - 1)
    return Rect(left, top, right - left + 1, bot - top + 1)


def image_to_screen(px: float, py: float, frame: Frame) -> tuple[int, int]:
    """图像像素 → 全局屏幕坐标。最外层整体取整一次（B012 教训）。"""
    if not (0 <= px <= frame.width_px and 0 <= py <= frame.height_px):
        raise OutOfWindow(f"image coord ({px},{py}) outside {frame.width_px}x{frame.height_px}")
    sx = round(frame.window_rect.x + px / frame.scale)
    sy = round(frame.window_rect.y + py / frame.scale)
    if not frame.window_rect.contains(sx, sy):
        raise OutOfWindow(f"screen coord ({sx},{sy}) outside window {frame.window_rect}")
    return sx, sy


# 左右留白：只需避开机身黑边。实测黑边占宽度 1.9%（图像 624 宽，手机屏幕 x∈[12,612]），
# 取 0.06 是它的三倍余量，同时把最容易露出来的左右两条边留给滚轮用。
SCROLL_INSET_X = 0.06


def scroll_anchor_candidates(rect: Rect, inset: float = 0.18,
                             steps: int = 3,
                             inset_x: float | None = None) -> list[tuple[float, float]]:
    """滚轮锚点的候选点，按「离中心由近及远」排序。

    滚轮是唯一没能后台化的动作：macOS 按**真实指针位置**路由滚轮事件，
    指针下方最上面的窗口才收得到。所以锚点被别的窗口盖住时，只能先把镜像抬起来
    —— 用户屏幕上就会闪一下。

    但「被盖住」通常只盖住一部分。镜像只要还露着一块，就该用那一块滚，
    根本不需要抬窗口。这个函数给出可选的落点。

    留白**左右和上下必须分开**。原来四周统一 18%，2026-09-08 真机撞到：
    浏览器盖住镜像、只在右边露出约 40px，而候选点 x 只覆盖窗口的 18%-82%，
    **露出来的那条正好落在从不去试的死区里** —— 于是判定「全被盖住」，
    退回抬窗口，再失败。

    两个方向的理由本来就不一样：

    · **上下**留 18% 是对的：太靠上会落到导航栏、太靠下会落到标签栏，那儿滚不动。
    · **左右**没有这个问题 —— 列表是通栏的，左右只需要避开机身黑边。
      实测黑边只占宽度的 1.9%（图像 624 宽、手机屏幕 x∈[12,612]），
      18% 是下限的 9 倍，白白把最容易露出来的那两条边废掉了。

    所以 `inset` 管上下，`inset_x` 管左右（默认 `SCROLL_INSET_X`）。

    中心排第一 —— 它最可能命中主列表，只有它被盖住时才考虑别处。
    `steps` 决定网格密度：目标是**镜像只要露出一小块就能用**，所以点取得密一些。
    调用方应当一次性取好窗口列表再逐点判断，别每个候选都去查一次系统。
    """
    ix = SCROLL_INSET_X if inset_x is None else inset_x
    cx, cy = rect.center
    x0, x1 = rect.x + rect.w * ix, rect.x + rect.w * (1 - ix)
    y0, y1 = rect.y + rect.h * inset, rect.y + rect.h * (1 - inset)
    pts = [(cx, cy)]
    for i in range(1, steps + 1):
        f = i / steps
        for x in (cx + (x1 - cx) * f, cx - (cx - x0) * f):
            for y in (cy, cy + (y1 - cy) * f, cy - (cy - y0) * f):
                pts.append((x, y))
        for y in (cy + (y1 - cy) * f, cy - (cy - y0) * f):
            pts.append((cx, y))
    seen, out = set(), []
    for p in pts:
        k = (round(p[0], 3), round(p[1], 3))
        if k not in seen:
            seen.add(k)
            out.append(p)
    return sorted(out, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
