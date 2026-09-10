import pytest
from PIL import Image

from iphone_agent.driver.geometry import Frame, OutOfWindow, Rect, image_to_screen, scroll_anchor_candidates


def make_frame(w_px, h_px, rect):
    return Frame(image=Image.new("RGB", (w_px, h_px)), width_px=w_px, height_px=h_px,
                 window_rect=rect, ts=0.0, frame_id=1)


def test_scale_from_image_not_hardcoded():
    f = make_frame(800, 1740, Rect(100, 50, 400, 870))
    assert f.scale == 2.0
    g = make_frame(720, 1566, Rect(100, 50, 400, 870))
    assert g.scale == pytest.approx(1.8)


def test_image_to_screen_integer_scale():
    f = make_frame(800, 1740, Rect(100, 50, 400, 870))
    assert image_to_screen(0, 0, f) == (100, 50)
    assert image_to_screen(800, 1740, f) == (500, 920)
    assert image_to_screen(401, 333, f) == (300, 216)  # 100+200.5 → 300(四舍五入到偶), 50+166.5 → 216


def test_image_to_screen_non_integer_scale_rounds_once():
    f = make_frame(720, 1566, Rect(100.5, 50, 400, 870))
    sx, sy = image_to_screen(361, 200, f)
    assert isinstance(sx, int) and isinstance(sy, int)
    # 100.5 + 361/1.8 = 301.05… → 301；50 + 200/1.8 = 161.1 → 161
    assert (sx, sy) == (301, 161)


def test_image_to_screen_rejects_outside_window():
    f = make_frame(800, 1740, Rect(100, 50, 400, 870))
    with pytest.raises(OutOfWindow):
        image_to_screen(-1, 10, f)
    with pytest.raises(OutOfWindow):
        image_to_screen(10, 1741, f)


def test_image_to_screen_rejects_outside_window_second_guard():
    # scale 只按宽度算（width_px / window_rect.w = 800/400 = 2）。
    # 图像 800x1800 的宽高比（800:1800）与窗口 400x870 的宽高比（400:870）不一致，
    # 所以 py=1800 虽然落在图像范围 [0,1800] 内、能通过第一道防线（入参坐标校验），
    # 但换算后 sy = 50 + 1800/2 = 950，超出窗口下边界 50+870=920，
    # 必须靠第二道防线（换算后坐标校验 window_rect.contains）拦截。
    f = make_frame(800, 1800, Rect(100, 50, 400, 870))
    with pytest.raises(OutOfWindow):
        image_to_screen(0, 1800, f)


def test_rect_same_as_tolerance():
    a = Rect(100, 50, 400, 870)
    assert a.same_as(Rect(100.4, 50, 400, 870))
    assert not a.same_as(Rect(103, 50, 400, 870))


# --- content_bounds：窗口矩形里手机屏幕的实际范围 ---
# 真实缺陷（2026-09-05）：镜像在窗口内画带圆角的机身，四周留黑边；按窗口几何算的
# 手势起点会落在黑边里，iOS 收不到触摸。

from PIL import ImageDraw  # noqa: E402

from iphone_agent.driver.geometry import content_bounds  # noqa: E402


def letterboxed(w, h, inset, fill="white"):
    img = Image.new("RGB", (w, h), "black")
    ImageDraw.Draw(img).rectangle([inset, inset, w - 1 - inset, h - 1 - inset], fill=fill)
    return img


def test_content_bounds_finds_the_inset_rect():
    b = content_bounds(letterboxed(200, 400, 12))
    assert (b.x, b.y, b.w, b.h) == (12, 12, 176, 376)  # [12,187] 与 [12,387]


def test_content_bounds_ignores_a_black_island_at_the_top_center():
    """顶部中间的灵动岛是黑的，不能被当成边界。"""
    img = letterboxed(200, 400, 10)
    ImageDraw.Draw(img).rectangle([80, 14, 120, 40], fill="black")  # 岛
    b = content_bounds(img)
    assert b.y == 10, "灵动岛把上边界量偏了"
    assert b.x == 10 and b.w == 180


def test_content_bounds_falls_back_to_whole_image_when_all_black():
    """锁屏或 DRM 遮挡时整屏全黑，不能返回空矩形让调用方算出 NaN。"""
    b = content_bounds(Image.new("RGB", (100, 200), "black"))
    assert (b.x, b.y, b.w, b.h) == (0, 0, 100, 200)


def test_content_bounds_on_a_frame_with_no_border_returns_full_image():
    b = content_bounds(Image.new("RGB", (100, 200), "white"))
    assert (b.x, b.y, b.w, b.h) == (0, 0, 100, 200)


class TestScrollAnchorCandidates:
    """滚轮锚点候选：镜像被盖住一部分时，用没被盖住的那块滚，不抬窗口。"""

    RECT = Rect(100, 200, 300, 600)

    def test_中心排第一(self):
        pts = scroll_anchor_candidates(self.RECT)
        assert pts[0] == self.RECT.center

    def test_上下留白之内(self):
        """上下留白是为了避开导航栏和标签栏 —— 那儿滚不动。"""
        inset, r = 0.18, self.RECT
        for _x, y in scroll_anchor_candidates(r, inset=inset):
            assert r.y + r.h * inset - 1e-6 <= y <= r.y + r.h * (1 - inset) + 1e-6

    def test_左右留白比上下窄得多(self):
        """⚠ 左右和上下的理由不一样，留白也必须分开。

        2026-09-08 真机撞到：浏览器盖住镜像、只在右边露出约 40px，
        而候选点 x 只覆盖窗口的 18%-82% —— **露出来的那条正好落在从不去试的死区里**，
        于是判定「全被盖住」，退回抬窗口，再失败。

        左右不存在导航栏/标签栏的问题（列表是通栏的），只需避开机身黑边，
        实测黑边占宽度 1.9%。
        """
        from iphone_agent.driver.geometry import SCROLL_INSET_X
        r = self.RECT
        assert SCROLL_INSET_X < 0.18 / 2, "左右留白没有真正放宽"
        assert SCROLL_INSET_X > 0.019 * 2, "留得太窄，会落到机身黑边上"
        xs = [x for x, _ in scroll_anchor_candidates(r)]
        assert min(xs) <= r.x + r.w * (SCROLL_INSET_X + 0.01)
        assert max(xs) >= r.x + r.w * (1 - SCROLL_INSET_X - 0.01)

    def test_按离中心距离升序(self):
        cx, cy = self.RECT.center
        d = [(x - cx) ** 2 + (y - cy) ** 2 for x, y in scroll_anchor_candidates(self.RECT)]
        assert d == sorted(d)

    def test_没有重复点(self):
        pts = scroll_anchor_candidates(self.RECT)
        assert len(pts) == len({(round(x, 3), round(y, 3)) for x, y in pts})

    def test_覆盖四个方向(self):
        """半边被盖住时必须还有另一半可用 —— 候选点要真的分散在四周。"""
        cx, cy = self.RECT.center
        pts = scroll_anchor_candidates(self.RECT)
        assert any(x < cx for x, _ in pts) and any(x > cx for x, _ in pts)
        assert any(y < cy for _, y in pts) and any(y > cy for _, y in pts)
