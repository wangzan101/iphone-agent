"""输入：优先逐键，粘贴只作非 ASCII 兜底。

真机实测（2026-09-07）：粘贴在四种情况下全部失败 —— Spotlight 里等 50ms / 1.5s /
剪贴板提前 3 秒写好、先点输入框聚焦、备忘录正文里光标就位。逐键输入两处全成。
所以键盘通道好的，坏的是 Mac→iPhone 剪贴板同步。
"""
import iphone_agent.driver.device as dm
from iphone_agent.driver.geometry import Rect
from iphone_agent.driver.injector import CHAR_KEYCODE, SHIFT_KEYCODE
from iphone_agent.driver.window import MirrorWindow

RECT = Rect(0, 0, 100, 200)


class RecInjector:
    def __init__(self):
        self.downs = []

    def activate(self, timeout_s=0.5):
        pass

    def key_down(self, code, flags=0):
        self.downs.append((code, flags))

    def key_up(self, code):
        pass

    def release_all(self):
        pass


def make(monkeypatch):
    monkeypatch.setattr(dm, "current_rect", lambda wid: RECT)
    inj = RecInjector()
    return dm.Device(win=MirrorWindow(1, 1, RECT), injector=inj), inj


def test_ascii_goes_through_keystrokes_not_clipboard(monkeypatch):
    dev, inj = make(monkeypatch)
    assert dev.type("hi") == "keystrokes"
    codes = [c for c, _ in inj.downs]
    assert CHAR_KEYCODE["h"] in codes and CHAR_KEYCODE["i"] in codes
    assert dm.KEY["v"] not in codes, "不该走粘贴"


def test_uppercase_holds_shift(monkeypatch):
    dev, inj = make(monkeypatch)
    dev.type("Hi")
    assert SHIFT_KEYCODE in [c for c, _ in inj.downs], "大写字母要按住 shift"


def test_digits_space_and_punctuation_are_typable(monkeypatch):
    dev, inj = make(monkeypatch)
    assert dev.type("a1 .-") == "keystrokes"


def test_non_ascii_falls_back_to_paste(monkeypatch):
    """中文打不出 keycode，只能退回粘贴 —— 而粘贴当前在真机上是坏的，
    调用方必须能接受它失败。"""
    dev, inj = make(monkeypatch)
    written = {}
    monkeypatch.setattr(dm, "NSPasteboard", type("PB", (), {
        "generalPasteboard": staticmethod(lambda: type("B", (), {
            "clearContents": lambda self: None,
            "setString_forType_": lambda self, s, t: written.update(text=s),
        })())}))
    assert dev.type("设置") == "paste"
    assert written["text"] == "设置"
    assert dm.KEY["v"] in [c for c, _ in inj.downs]


def test_mixed_ascii_and_chinese_also_falls_back(monkeypatch):
    dev, inj = make(monkeypatch)
    monkeypatch.setattr(dm, "NSPasteboard", type("PB", (), {
        "generalPasteboard": staticmethod(lambda: type("B", (), {
            "clearContents": lambda self: None,
            "setString_forType_": lambda self, s, t: None,
        })())}))
    assert dev.type("wifi设置") == "paste"
