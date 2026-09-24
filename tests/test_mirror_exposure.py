"""「镜像露没露出来」只有一条规则：滚轮挑落点、任务开跑前的检查、doctor 共用它。

⚠ 2026-09-18：点击（SkyLight 按窗口 id 投递）和键盘（CGEventPostToPid）在镜像被完全盖住时
照样能用；只有滚轮（含 open_app 里翻主屏页的横向滚轮）按真实指针下方最上面的窗口路由，
需要镜像露出一点。779d09a 把每次点击都挂到窗口层级检查上，误拦了后台点击，已在 58dad74 撤回 ——
这里的守卫测试钉住：点击和按键路径**不查窗口层级**。
"""
import pytest

from iphone_agent.driver import injector as inj_mod
from iphone_agent.driver.device import Device
from iphone_agent.driver.geometry import Rect
from iphone_agent.driver.injector import (
    ActivateFailed,
    BackgroundInjector,
    ForegroundInjector,
    exposed_anchor,
)
from iphone_agent.driver.window import MirrorWindow

MIRROR_ID = 99
CONTENT = Rect(100, 100, 300, 600)       # 手机屏幕（内容矩形），全局坐标
MIRROR = {"kCGWindowLayer": 0, "kCGWindowNumber": MIRROR_ID,
          "kCGWindowBounds": {"X": 94, "Y": 62, "Width": 312, "Height": 644}}


def _cover(x, y, w, h, wid=777):
    return {"kCGWindowLayer": 0, "kCGWindowNumber": wid,
            "kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h}}


# 几种真实会出现的窗口层级（列表按前后顺序，前面的压在上面）
STACKS = {
    "uncovered": [MIRROR],
    # 浏览器盖住镜像，只在右边露出约 20 点的一条（2026-09-08 真机的摆法）
    "right_strip": [_cover(0, 0, 380, 2000), MIRROR],
    # 终端压住中间一块，四周还露着
    "center_covered": [_cover(200, 300, 100, 200), MIRROR],
    # 完全盖住，一条缝都没有（2026-09-18 四次真机失败的现场）
    "fully_covered": [_cover(0, 0, 2000, 2000), MIRROR],
    # 两个窗口拼起来盖满
    "two_windows_cover": [_cover(0, 0, 250, 2000, 777), _cover(250, 0, 2000, 2000, 778), MIRROR],
}
EXPOSED = {"uncovered": True, "right_strip": True, "center_covered": True,
           "fully_covered": False, "two_windows_cover": False}


def _stub_windows(monkeypatch, stack, counter=None):
    def fake(*a):
        if counter is not None:
            counter["n"] += 1
        return stack
    monkeypatch.setattr(inj_mod.Quartz, "CGWindowListCopyWindowInfo", fake)


class FakeForeground:
    def __init__(self):
        self.calls = []

    def activate(self, timeout_s=0.5):
        self.calls.append(("activate",))
        raise ActivateFailed("抢不到焦点")

    def scroll_at(self, anchor, v=0, h=0, raise_window=True, region=None):
        self.calls.append(("scroll_at", anchor, raise_window))

    def release_all(self):
        pass


def _bg():
    return BackgroundInjector(pid=7, window_id=MIRROR_ID, window_origin=lambda: (94, 62),
                              fallback=FakeForeground())


def _device(monkeypatch, inj):
    dev = Device(win=MirrorWindow(MIRROR_ID, 7, Rect(94, 62, 312, 644)), injector=inj)
    monkeypatch.setattr(dev, "content_rect", lambda: CONTENT)
    monkeypatch.setattr(dev, "window_rect", lambda: Rect(94, 62, 312, 644))
    return dev


@pytest.mark.parametrize("name", sorted(STACKS))
def test_exposed_anchor_answers_whether_any_part_is_uncovered(monkeypatch, name):
    _stub_windows(monkeypatch, STACKS[name])
    p = exposed_anchor(MIRROR_ID, CONTENT)
    assert (p is not None) is EXPOSED[name]
    if p is not None:
        assert CONTENT.contains(*p), "露出的落点必须在手机屏幕内"


def test_exposed_anchor_accepts_a_prefetched_window_list(monkeypatch):
    counter = {"n": 0}
    _stub_windows(monkeypatch, [], counter)
    assert exposed_anchor(MIRROR_ID, CONTENT, windows=STACKS["right_strip"]) is not None
    assert counter["n"] == 0


@pytest.mark.parametrize("name", sorted(STACKS))
def test_device_exposure_and_the_scroll_path_agree(monkeypatch, name):
    """同一个窗口层级，「露没露出来」和「滚轮要不要抬窗口」必须给同一个答案 —— 一条规则一个入口。"""
    _stub_windows(monkeypatch, STACKS[name])
    monkeypatch.setattr(inj_mod.Quartz, "CGEventCreate", lambda src: object())
    monkeypatch.setattr(inj_mod.Quartz, "CGEventGetLocation", lambda ev: (1, 1))
    monkeypatch.setattr(inj_mod.Quartz, "CGWarpMouseCursorPosition", lambda pos: None)
    monkeypatch.setattr(inj_mod.time, "sleep", lambda s: None)

    class _App:
        def processIdentifier(self):
            return 7          # 等于镜像 pid：finally 里不去还原焦点

    monkeypatch.setattr(inj_mod, "NSWorkspace", type("WS", (), {
        "sharedWorkspace": staticmethod(lambda: type("W", (), {
            "frontmostApplication": lambda self: _App()})())}))
    bg = _bg()
    dev = _device(monkeypatch, bg)
    exposed = dev.mirror_exposed()
    assert exposed is EXPOSED[name]

    try:
        bg.scroll_at(tuple(map(int, CONTENT.center)), h=-10, region=CONTENT)
        scrolled_without_raising = ("activate",) not in bg._fb.calls
    except ActivateFailed:
        scrolled_without_raising = False
    assert scrolled_without_raising is exposed


def test_foreground_injector_does_not_need_exposure(monkeypatch):
    """前台注入每个动作前都抬窗口，盖没盖住都能滚 —— 不查窗口层级，直接算露着。"""
    counter = {"n": 0}
    _stub_windows(monkeypatch, STACKS["fully_covered"], counter)
    dev = _device(monkeypatch, ForegroundInjector(7))
    assert dev.mirror_exposed() is True
    assert counter["n"] == 0


def test_tap_and_keys_never_query_the_window_stack(monkeypatch):
    """⚠ 守卫：点击和按键在镜像被完全盖住时照样送得到（docs/14），绝不能挂到窗口层级上。
    779d09a 就是这么把后台点击误拦的（58dad74 撤回）。"""
    counter = {"n": 0}
    _stub_windows(monkeypatch, STACKS["fully_covered"], counter)
    posts = []
    monkeypatch.setattr(inj_mod.sl, "post", lambda *a, **kw: posts.append(a))
    monkeypatch.setattr(inj_mod.Quartz, "CGEventCreateKeyboardEvent", lambda *a: {})
    monkeypatch.setattr(inj_mod.Quartz, "CGEventSetFlags", lambda ev, f: None)
    keys = []
    monkeypatch.setattr(inj_mod.Quartz, "CGEventPostToPid", lambda pid, ev: keys.append(pid))
    import iphone_agent.driver.device as device_mod
    monkeypatch.setattr(device_mod.time, "sleep", lambda s: None)
    dev = _device(monkeypatch, _bg())

    dev.tap(200, 300)
    dev.key("home")
    dev._type_downs_only([0, 1])

    assert posts, "点击应当照常发出"
    assert keys, "按键应当照常发出"
    assert counter["n"] == 0, "点击/按键路径查了窗口层级"


def test_cli_and_web_render_the_mirror_hidden_end_readably(capsys, tmp_path):
    """CLI 和网页都走各自现有的结束渲染：一个短标签 + 那句完整的指引。"""
    from pathlib import Path

    from iphone_agent.cli.commands import END_WORD, ChatView
    from iphone_agent.driver.injector import MIRROR_HIDDEN_HINT
    from iphone_agent.harness.loop import RunResult
    from iphone_agent.web.server import _end_message

    res = RunResult("mirror_hidden", 0, None, None, tmp_path / "run-x", MIRROR_HIDDEN_HINT)
    ChatView().finish(res)
    out = capsys.readouterr().out
    assert END_WORD["mirror_hidden"] in out and MIRROR_HIDDEN_HINT in out

    msg = _end_message(res)
    assert msg["code"] == "end.mirrorHiddenDetail"
    i18n = (Path(__file__).parents[1] / "iphone_agent/web/static/i18n.js").read_text()
    app = (Path(__file__).parents[1] / "iphone_agent/web/static/app.js").read_text()
    assert '"end.mirrorHiddenDetail"' in i18n and '"end.mirrorHidden"' in i18n
    assert 'mirror_hidden: L("end.mirrorHidden")' in app
    for code in ("check.mirror.exposed.title", "check.exposed.ok", "check.exposed.covered",
                 "check.exposed.foreground", "check.exposed.why", "check.exposed.fix"):
        assert f'"{code}"' in i18n, code
    # 其余结束原因照旧走原始诊断
    other = RunResult("device_error", 1, None, None, tmp_path, "炸了")
    assert _end_message(other) == {"code": "event.error", "params": {"detail": "炸了"}}
