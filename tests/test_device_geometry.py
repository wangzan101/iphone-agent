"""content_rect 与滚轮：锚点必须按**内容矩形**算，不是窗口矩形。

真实缺陷（2026-09-05 真机）：镜像 App 在窗口内画带圆角的机身，四周留黑边
（实测左右各 6 点、上 38 点、下 6 点）—— 窗口矩形不等于手机屏幕。

同一轮还查明：鼠标拖拽在镜像里会被 iOS 当成长按+拖动物体（主屏幕进编辑模式、
图标被拖走），做不出滑动手势；滚轮才是唯一的滑动通道，竖直滚列表、横向翻页都可以。
所以驱动层没有 swipe，只有四方向的 scroll。
"""
from PIL import Image, ImageDraw

import iphone_agent.driver.device as device_mod
from iphone_agent.driver.device import Device
from iphone_agent.driver.geometry import Frame, Rect
from iphone_agent.driver.window import MirrorWindow

WINDOW = Rect(10, 20, 200, 400)
# 图像 2 倍分辨率；四周留黑边，模拟镜像的机身圆角
INSET_PX = 12


class FakeInjector:
    def __init__(self):
        self.calls = []

    def activate(self, timeout_s: float = 0.5):
        pass

    def mouse_move(self, x, y):
        self.calls.append(("move", x, y))

    def mouse_down(self, x, y):
        self.calls.append(("down", x, y))

    def mouse_drag(self, x, y):
        self.calls.append(("drag", x, y))

    def mouse_up(self, x, y):
        self.calls.append(("up", x, y))

    def warp(self, x, y):
        self.calls.append(("warp", x, y))

    def key_down(self, code, flags=0):
        self.calls.append(("key_down", code, flags))

    def key_up(self, code):
        self.calls.append(("key_up", code))

    def scroll_lines(self, v=0, h=0):
        self.calls.append(("scroll", v, h))

    def scroll_at(self, anchor, v=0, h=0, region=None):
        # 真实实现里 warp + MouseMoved + 滚轮是一整套，缺一不可（后台注入器的
        # mouse_move 是空操作，调用方拼装会静默丢掉 hit-test —— 踩过）
        self.region = region
        self.calls.append(("warp", anchor[0], anchor[1]))
        self.calls.append(("move", anchor[0], anchor[1]))
        self.calls.append(("scroll", v, h))

    def release_all(self):
        pass


def letterboxed_frame(rect: Rect, frame_id=1, scale=2) -> Frame:
    W, H = int(rect.w * scale), int(rect.h * scale)
    img = Image.new("RGB", (W, H), "black")
    ImageDraw.Draw(img).rectangle(
        [INSET_PX, INSET_PX, W - 1 - INSET_PX, H - 1 - INSET_PX], fill="white")
    return Frame(img, W, H, rect, 0.0, frame_id)


def make_device(monkeypatch, rect: Rect = WINDOW):
    win = MirrorWindow(window_id=1, pid=1, rect=rect)
    inj = FakeInjector()
    monkeypatch.setattr(device_mod, "current_rect", lambda window_id: rect)
    dev = Device(win=win, injector=inj)
    monkeypatch.setattr(dev, "capture", lambda: letterboxed_frame(dev.window_rect()))
    return dev, inj


def expected_content(rect: Rect, scale=2) -> Rect:
    inset = INSET_PX / scale
    return Rect(rect.x + inset, rect.y + inset, rect.w - 2 * inset, rect.h - 2 * inset)


def test_content_rect_is_inset_from_window_rect(monkeypatch):
    dev, _ = make_device(monkeypatch)
    c = dev.content_rect()
    exp = expected_content(WINDOW)
    assert (c.x, c.y, c.w, c.h) == (exp.x, exp.y, exp.w, exp.h)
    assert c.x > WINDOW.x and c.x + c.w < WINDOW.x + WINDOW.w  # 确实缩进了


def test_content_rect_cached_until_window_moves(monkeypatch):
    dev, _ = make_device(monkeypatch)
    captures = []
    monkeypatch.setattr(dev, "capture",
                        lambda: (captures.append(1), letterboxed_frame(dev.window_rect()))[1])
    dev.content_rect(); dev.content_rect(); dev.content_rect()
    assert len(captures) == 1  # 黑边只随窗口几何变，不该每次都抓帧

    moved = Rect(300, 40, 200, 400)
    monkeypatch.setattr(device_mod, "current_rect", lambda window_id: moved)
    c = dev.content_rect()
    assert len(captures) == 2  # 窗口动了必须重算
    assert c.x == expected_content(moved).x


def test_scroll_anchors_at_content_center(monkeypatch):
    dev, inj = make_device(monkeypatch)
    dev.scroll("down", "page")
    c = dev.content_rect()
    _, wx, wy = next(x for x in inj.calls if x[0] == "warp")
    assert (wx, wy) == tuple(map(int, c.center))
    assert c.contains(wx, wy)
    # 顺序必须是 warp → MouseMoved → 滚轮：macOS 把滚轮路由给指针下方的窗口，
    # 中间那个 MouseMoved 是用来触发 hit-test 的
    kinds = [x[0] for x in inj.calls]
    assert kinds.index("warp") < kinds.index("move") < kinds.index("scroll")
    # 内容矩形要传下去：后台注入器靠它在镜像**还露着的部分**里另挑落点，
    # 免得为了滚一下就把窗口抬到最前面。
    assert inj.region is not None and inj.region.same_as(c)


# --- 窗口句柄不缓存：镜像 App 重启后必须自愈 ---

def test_device_rediscovers_window_after_app_restart(monkeypatch):
    """真机踩到（2026-09-05）：镜像 App 重启后 window id 与 pid 全变，
    Device 握着旧句柄，后续所有操作以「pid 不存在」失败。"""
    import iphone_agent.driver.device as dm

    old = MirrorWindow(window_id=1, pid=100, rect=WINDOW)
    new = MirrorWindow(window_id=2, pid=200, rect=WINDOW)
    current = {"win": old}
    monkeypatch.setattr(dm, "find_mirror_window", lambda: current["win"])
    monkeypatch.setattr(dm, "current_rect", lambda wid: WINDOW)

    made = []
    monkeypatch.setattr(dm, "ForegroundInjector",
                        lambda pid: (made.append(pid), FakeInjector())[1])

    dev = Device()  # 不钉住 → 会自动发现
    assert dev.win.pid == 100 and made == [100]

    current["win"] = new  # App 重启
    dev.window_rect()
    assert dev.win.pid == 200, "没有重新发现窗口"
    assert made == [100, 200], "注入器没有跟着换到新 pid"


def test_pinned_window_is_not_rediscovered(monkeypatch):
    """显式传窗口的调用方（测试）不该被自动重新发现打断。"""
    import iphone_agent.driver.device as dm
    monkeypatch.setattr(dm, "find_mirror_window",
                        lambda: (_ for _ in ()).throw(AssertionError("钉住了还去发现")))
    monkeypatch.setattr(dm, "current_rect", lambda wid: WINDOW)
    dev = Device(win=MirrorWindow(window_id=9, pid=9, rect=WINDOW), injector=FakeInjector())
    dev.window_rect()
    assert dev.win.window_id == 9


def test_scroll_directions_map_to_the_right_wheel_axis(monkeypatch):
    """四个方向全走滚轮：竖直用 v 轴、横向用 h 轴，符号按「想看到的内容在哪边」。"""
    for direction, expect in [("down", ("v", -1)), ("up", ("v", 1)),
                              ("right", ("h", -1)), ("left", ("h", 1))]:
        dev, inj = make_device(monkeypatch)
        dev.scroll(direction, "page")
        _, v, h = next(x for x in inj.calls if x[0] == "scroll")
        axis, sign = expect
        if axis == "v":
            assert h == 0 and (v > 0) == (sign > 0), f"{direction} 的竖直分量不对"
        else:
            assert v == 0 and (h > 0) == (sign > 0), f"{direction} 的横向分量不对"


def test_scroll_rejects_unknown_direction(monkeypatch):
    import pytest
    dev, _ = make_device(monkeypatch)
    with pytest.raises(ValueError):
        dev.scroll("sideways")


# --- 打字：只发 down，重复字母先松开 ---

def _key_events(inj):
    return [c for c in inj.calls if c[0] in ("key_down", "key_up")]


def test_typing_sends_only_key_downs(monkeypatch):
    """真机实测：down+up 成对会让大小写按位置交替（abcdef→aBcDeF），
    只发 down 才正确。见 docs/14 与 runs/pinyin4-*。"""
    dev, inj = make_device(monkeypatch)
    dev.type("abcdef")
    ev = _key_events(inj)
    downs = [c for c in ev if c[0] == "key_down"]
    assert len(downs) == 6
    # 六个 down 必须连续发完，中间不能夹 up
    assert [c[0] for c in ev[:6]] == ["key_down"] * 6


def test_typing_releases_a_key_before_pressing_it_again(monkeypatch):
    """键按下不松，再按同一个键就不出字符 —— shezhi 会丢第二个 h 变成 shezi。
    所以判据是「这个键还按着吗」，不是「和前一个是否相同」（两个 h 并不相邻）。"""
    dev, inj = make_device(monkeypatch)
    dev.type("shezhi")
    ev = _key_events(inj)
    h = 4      # 'h' 的 keycode
    seq = [(c[0], c[1]) for c in ev]
    first_h = seq.index(("key_down", h))
    # 第二个 h 按下之前，必须先把它抬起来
    rest = seq[first_h + 1:]
    assert ("key_up", h) in rest, "重复的键没有先松开"
    assert rest.index(("key_up", h)) < rest.index(("key_down", h)), "松开必须在再次按下之前"


def test_typing_leaves_no_key_held(monkeypatch):
    """零残留按键是硬指标。"""
    dev, inj = make_device(monkeypatch)
    dev.type("hello")
    held = []
    for kind, code, *_ in _key_events(inj):
        if kind == "key_down":
            held.append(code)
        elif code in held:
            held.remove(code)
    assert held == [], f"打完还按着 {held}"


def test_shifted_text_falls_back_and_says_so(monkeypatch):
    """需要 Shift 的字符走不了只发 down 那条路（Shift 是修饰键，不能只按不放），
    返回值要如实标出来，让调用方知道结果可能不可靠。"""
    dev, inj = make_device(monkeypatch)
    assert dev.type("abc") == "keystrokes"
    dev2, _ = make_device(monkeypatch)
    assert dev2.type("Abc") == "shifted"
