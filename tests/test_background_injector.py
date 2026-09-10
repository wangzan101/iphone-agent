"""后台注入器：点击走 SkyLight 不抢焦点，其余委托前台。

真机实测（2026-09-07，macOS 15.6.1）：
  SkyLight 点击 + 不调 SetFrontProcess → 手机有反应、前台不变、用户能继续打字 ✅
  SkyLight 点击 + 调 SetFrontProcess   → 手机没反应、前台变成镜像、焦点被夺 ❌
  CGEventPostToPid 点击                 → 手机没反应 ❌
所以 SetFrontProcess 对我们不但没用还有害 —— 那是 yabai 为了改焦点做的事。
"""
import pytest

from iphone_agent.driver import injector as inj_mod
from iphone_agent.driver.geometry import Rect
from iphone_agent.driver.injector import BackgroundInjector


class FakeForeground:
    def __init__(self):
        self.calls = []
        self.raise_error = None      # 设上就让 activate 抛，用来测「抢不到焦点」

    def activate(self, timeout_s=0.5):
        self.calls.append(("activate",))
        if self.raise_error is not None:
            raise self.raise_error

    def warp(self, x, y):
        self.calls.append(("warp", x, y))

    def scroll_lines(self, v=0, h=0):
        self.calls.append(("scroll", v, h))

    def scroll_at(self, anchor, v=0, h=0, raise_window=True, region=None):
        self.calls.append(("scroll_at", anchor, v, h, raise_window))

    def key_down(self, code, flags=0):
        self.calls.append(("key_down", code, flags))

    def key_up(self, code):
        self.calls.append(("key_up", code))

    def release_all(self):
        self.calls.append(("release_all",))


@pytest.fixture
def rig(monkeypatch):
    posts = []
    monkeypatch.setattr(inj_mod.sl, "post",
                        lambda pid, wid, et, global_xy=None, window_xy=None, set_front=True:
                        posts.append({"pid": pid, "wid": wid, "type": et,
                                      "global": global_xy, "window": window_xy,
                                      "set_front": set_front}))
    fb = FakeForeground()
    bg = BackgroundInjector(pid=7, window_id=99, window_origin=lambda: (100, 50), fallback=fb)
    return bg, fb, posts


def test_click_never_sets_front_process(rig):
    """调 SetFrontProcess 会夺走键盘焦点，而且实测手机反而没反应。"""
    bg, _, posts = rig
    bg.mouse_down(150, 90)
    bg.mouse_up(150, 90)
    assert posts and all(p["set_front"] is False for p in posts)


def test_click_sends_both_global_and_window_coordinates(rig):
    """只填窗口内坐标时手机毫无反应（实测），所以全局坐标是必需的。"""
    bg, _, posts = rig
    bg.mouse_down(150, 90)
    p = posts[0]
    assert p["global"] == (150, 90)
    assert p["window"] == (50, 40), "窗口内坐标 = 全局坐标减窗口左上角"
    assert p["type"] == inj_mod.sl.EVENT_DOWN and p["wid"] == 99


def test_drag_uses_the_dragged_event_type(rig):
    bg, _, posts = rig
    bg.mouse_down(10, 10); bg.mouse_drag(20, 20); bg.mouse_up(20, 20)
    assert [p["type"] for p in posts] == [inj_mod.sl.EVENT_DOWN,
                                          inj_mod.sl.EVENT_DRAGGED,
                                          inj_mod.sl.EVENT_UP]


def test_mouse_move_does_not_touch_the_real_pointer(rig):
    """后台路径不该动用户的真实指针。"""
    bg, fb, posts = rig
    bg.mouse_move(1, 2)
    assert posts == [] and fb.calls == []


def test_activate_is_a_noop(rig):
    """不激活正是后台注入的意义。"""
    bg, fb, _ = rig
    bg.activate()
    assert fb.calls == []


def test_keyboard_goes_through_post_to_pid_without_stealing_focus(monkeypatch, rig):
    """实测：直接 CGEventPostToPid 发键就能用，什么前置都不需要，前台不变。
    （另试过用 SkyLight 记录发键盘事件，猜事件类型 10/11 —— 不成立。）"""
    bg, fb, posts = rig
    sent = []
    monkeypatch.setattr(inj_mod.Quartz, "CGEventCreateKeyboardEvent",
                        lambda src, code, down: {"code": code, "down": down, "flags": None})
    monkeypatch.setattr(inj_mod.Quartz, "CGEventSetFlags",
                        lambda ev, f: ev.__setitem__("flags", f))
    monkeypatch.setattr(inj_mod.Quartz, "CGEventPostToPid",
                        lambda pid, ev: sent.append((pid, ev)))

    bg.key_down(18, flags=0x100000)
    bg.key_up(18)

    assert [pid for pid, _ in sent] == [7, 7], "键盘事件要直投给镜像进程"
    assert sent[0][1]["down"] is True and sent[0][1]["flags"] == 0x100000
    assert sent[1][1]["down"] is False and sent[1][1]["flags"] == 0, \
        "抬起前必须清标志位，否则修饰键锁死"
    assert ("activate",) not in fb.calls, "键盘不该再激活前台"


def _stub_window_list(monkeypatch, topmost_id):
    """伪造窗口列表：让锚点处最上面的 layer-0 窗口是 topmost_id。"""
    monkeypatch.setattr(inj_mod.Quartz, "CGWindowListCopyWindowInfo", lambda *a: [
        {"kCGWindowLayer": 25, "kCGWindowNumber": 1,
         "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 9999, "Height": 9999}},  # 菜单栏之类，不算
        {"kCGWindowLayer": 0, "kCGWindowNumber": topmost_id,
         "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 9999, "Height": 9999}},
    ])


def _stub_restore(monkeypatch, restored):
    class FakeApp:
        def processIdentifier(self):
            return 4242

        def activateWithOptions_(self, opts):
            restored["app"] = True

    monkeypatch.setattr(inj_mod, "NSWorkspace", type("WS", (), {
        "sharedWorkspace": staticmethod(lambda: type("W", (), {
            "frontmostApplication": lambda self: FakeApp()})())}))
    monkeypatch.setattr(inj_mod.Quartz, "CGEventCreate", lambda src: object())
    monkeypatch.setattr(inj_mod.Quartz, "CGEventGetLocation", lambda ev: (11, 22))
    monkeypatch.setattr(inj_mod.Quartz, "CGWarpMouseCursorPosition",
                        lambda pos: restored.__setitem__("pointer", pos))


def test_scroll_does_not_raise_the_window_when_mirror_is_already_topmost(monkeypatch, rig):
    """关键优化：滚轮命中的是「指针下方最上面的窗口」，不要求它是前台。
    用户为了看手机本来就让镜像露着 —— 这时完全不该抬窗口，屏幕上什么都不该变。"""
    bg, fb, _ = rig
    restored = {}
    _stub_window_list(monkeypatch, topmost_id=99)   # 99 = 这个后台注入器的 window_id
    _stub_restore(monkeypatch, restored)

    bg.scroll_at((300, 400), v=-10)

    call = next(c for c in fb.calls if c[0] == "scroll_at")
    assert call[-1] is False, "镜像本来就在最上面，不该抬窗口"
    assert restored.get("pointer") == (11, 22), "指针仍要还回原位"
    assert "app" not in restored, "没抢焦点就不该去还原焦点"


def test_scroll_raises_only_when_something_covers_the_anchor(monkeypatch, rig):
    """被别的窗口盖住时才退回「抬升 → 滚 → 还原」
    （phone-harness 记的「Chrome 遮住镜像时滚动会滚到 Chrome 里」正是这种情形）。"""
    bg, fb, _ = rig
    restored = {}
    _stub_restore(monkeypatch, restored)
    seen = {"n": 0}
    # 一开始被别人盖着，抬完窗口就到最上面了 —— 这是真实世界里最常见的情形
    monkeypatch.setattr(type(bg), "_is_topmost_at",
                        lambda self, x, y, windows=None: seen.__setitem__("n", seen["n"] + 1)
                        or seen["n"] > 1)

    bg.scroll_at((300, 400), v=-10)

    assert ("activate",) in fb.calls, "被盖住时必须抬窗口，否则滚到别的 App 里去了"
    assert any(c[0] == "scroll_at" for c in fb.calls), "抬完在最上面了，就该滚"
    assert restored.get("pointer") == (11, 22)
    assert restored.get("app"), "抬了窗口就必须把焦点还回去"


def test_release_all_lifts_a_held_button(rig):
    """零残留按键是硬指标，后台路径也要守。"""
    bg, fb, posts = rig
    bg.mouse_down(10, 10)
    posts.clear()
    bg.release_all()
    assert ("release_all",) in fb.calls
    assert posts and posts[-1]["type"] == inj_mod.sl.EVENT_UP


# --- 锚点避让：镜像只要露出一小块，就在那一块上滚，不抬窗口 ---

def _stub_partial_cover(monkeypatch, mirror_id, cover_bounds):
    """镜像铺满，另一个窗口压在它上面、只盖住 cover_bounds 那一块。"""
    monkeypatch.setattr(inj_mod.Quartz, "CGWindowListCopyWindowInfo", lambda *a: [
        {"kCGWindowLayer": 0, "kCGWindowNumber": 777, "kCGWindowBounds": cover_bounds},
        {"kCGWindowLayer": 0, "kCGWindowNumber": mirror_id,
         "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1000, "Height": 1000}},
    ])


def test_scroll_moves_the_anchor_instead_of_raising_when_only_part_is_covered(
        monkeypatch, rig):
    """终端压住镜像中心时，换一个还露着的落点滚 —— 用户屏幕上连闪一下都不该有。"""
    bg, fb, _ = rig
    restored = {}
    region = Rect(0, 0, 1000, 1000)
    # 盖住中间那一块（含中心点 500,500），四周仍然露着
    _stub_partial_cover(monkeypatch, mirror_id=99,
                        cover_bounds={"X": 400, "Y": 400, "Width": 200, "Height": 200})
    _stub_restore(monkeypatch, restored)

    bg.scroll_at((500, 500), v=-10, region=region)

    call = next(c for c in fb.calls if c[0] == "scroll_at")
    assert call[-1] is False, "还露着一块就不该抬窗口"
    assert call[1] != (500, 500), "锚点必须换到没被盖住的地方"
    ax, ay = call[1]
    assert not (400 <= ax <= 600 and 400 <= ay <= 600), "换的落点仍在遮挡区里"
    assert region.contains(ax, ay), "落点必须还在手机屏幕内"
    assert "app" not in restored, "没抬窗口就不该去还原焦点"


def test_scroll_refuses_rather_than_scrolling_into_someone_elses_window(monkeypatch, rig):
    """⚠ 抬了窗口还是盖着，就**不能滚** —— 那一下会滚进盖住它的那个 App 里。

    这正是 phone-harness 记的「Chrome 遮住镜像时滚动会滚到 Chrome 里」。
    老实报错比闷头滚错地方强。
    """
    from iphone_agent.driver.injector import ActivateFailed
    bg, fb, _ = rig
    restored = {}
    _stub_partial_cover(monkeypatch, mirror_id=99,
                        cover_bounds={"X": 0, "Y": 0, "Width": 1000, "Height": 1000})
    _stub_restore(monkeypatch, restored)

    with pytest.raises(ActivateFailed):
        bg.scroll_at((500, 500), v=-10, region=Rect(0, 0, 1000, 1000))
    assert not any(c[0] == "scroll_at" for c in fb.calls), "盖着还滚了"
    assert restored.get("app"), "抬了窗口就必须把焦点还回去"


def test_a_failed_activate_does_not_block_scrolling_when_the_window_is_on_top(monkeypatch, rig):
    """⚠ 抬窗口失败**不等于**滚不了。

    2026-09-08 真机撞到：窗口确实被抬到了最上面（用户亲眼看到），但 activate()
    报 activate_failed（frontmost 仍是浏览器），于是滚轮**根本没发出去**就被拦了。

    「窗口在最上面」和「App 是活跃应用」是两件事，macOS 可以只做前者，
    而滚轮要的恰恰是前者。用错的尺子去量，就会把能用的情况判死。
    """
    from iphone_agent.driver.injector import ActivateFailed
    bg, fb, _ = rig
    restored = {}
    _stub_restore(monkeypatch, restored)
    seen = {"n": 0}

    def topmost(x, y, windows=None):
        seen["n"] += 1
        return seen["n"] > 1        # 第一次判「被盖」，抬完之后就在最上面了

    monkeypatch.setattr(type(bg), "_is_topmost_at", topmost)
    fb.raise_error = ActivateFailed("抢不到焦点")      # activate 失败

    bg.scroll_at((300, 400), v=-10)
    assert any(c[0] == "scroll_at" for c in fb.calls), "窗口明明在最上面，却没滚"


def test_anchor_search_queries_the_window_list_once(monkeypatch, rig):
    """十几个候选点不能各查一次系统：既慢，拿到的快照还可能互相不一致。"""
    bg, _, _ = rig
    calls = {"n": 0}

    def counting(*a):
        calls["n"] += 1
        return [{"kCGWindowLayer": 0, "kCGWindowNumber": 777,
                 "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1000, "Height": 1000}}]

    monkeypatch.setattr(inj_mod.Quartz, "CGWindowListCopyWindowInfo", counting)
    bg.pick_clear_anchor(Rect(0, 0, 1000, 1000), default=None)
    assert calls["n"] == 1
