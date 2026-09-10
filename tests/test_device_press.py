"""裁决五：press() 中间任何一步抛异常，已按下的键都要在 finally 里抬起（零残留按键是硬指标）。"""
import pytest

import iphone_agent.driver.device as device_mod
from iphone_agent.driver.device import Device
from iphone_agent.driver.geometry import Rect
from iphone_agent.driver.injector import KEY
from iphone_agent.driver.window import MirrorWindow


class FailingKeyInjector:
    """模仿 ForegroundInjector 的 _held_keys 记账；key_down 在指定 keycode 上抛异常。"""
    def __init__(self, fail_code):
        self.fail_code = fail_code
        self._held_keys: list[int] = []

    def activate(self, timeout_s: float = 0.5):
        pass

    def key_down(self, code, flags=0):
        if code == self.fail_code:
            raise RuntimeError("注入失败")
        self._held_keys.append(code)

    def key_up(self, code):
        if code in self._held_keys:
            self._held_keys.remove(code)

    def release_all(self):
        for code in list(reversed(self._held_keys)):
            self.key_up(code)


def make_device(monkeypatch, injector):
    rect = Rect(0, 0, 200, 400)
    win = MirrorWindow(window_id=1, pid=1, rect=rect)
    monkeypatch.setattr(device_mod, "current_rect", lambda window_id: rect)
    return Device(win=win, injector=injector)


def test_press_releases_modifier_when_main_key_fails(monkeypatch):
    inj = FailingKeyInjector(fail_code=KEY["v"])
    dev = make_device(monkeypatch, inj)

    with pytest.raises(RuntimeError):
        dev.press(KEY["v"], [KEY["cmd"]])

    assert inj._held_keys == []


def test_press_releases_earlier_modifier_when_second_modifier_fails(monkeypatch):
    inj = FailingKeyInjector(fail_code=KEY["a"])
    dev = make_device(monkeypatch, inj)

    with pytest.raises(RuntimeError):
        dev.press(KEY["v"], [KEY["cmd"], KEY["a"]])

    assert inj._held_keys == []


def test_press_succeeds_and_releases_everything(monkeypatch):
    inj = FailingKeyInjector(fail_code=-1)
    dev = make_device(monkeypatch, inj)
    dev.press(KEY["v"], [KEY["cmd"]])
    assert inj._held_keys == []


def _device_with(inj):
    """装一个只记录调用的假 injector，绕开真实窗口发现。"""
    import iphone_agent.driver.device as dm
    from iphone_agent.driver.geometry import Rect
    from iphone_agent.driver.window import MirrorWindow
    rect = Rect(0, 0, 100, 200)
    dm.current_rect = lambda wid: rect
    return dm.Device(win=MirrorWindow(window_id=1, pid=1, rect=rect), injector=inj)


def test_main_key_carries_the_modifier_flag_mask():
    """Cmd+1/2/3 是镜像 App 的 Mac 菜单快捷键，走 Mac 层而非转发给 iOS。
    只「真按住」Cmd 而事件上没有标志位的话，Mac 层收不到这个组合键
    （2026-09-05 实测 key home 完全没反应）。"""
    import Quartz

    from iphone_agent.driver.injector import KEY
    seen = []

    class Rec:
        def activate(self, timeout_s=0.5): pass
        def key_down(self, code, flags=0): seen.append((code, flags))
        def key_up(self, code): pass
        def release_all(self): pass

    dev = _device_with(Rec())
    dev.press(KEY["1"], [KEY["cmd"]])

    cmd_evt = [f for c, f in seen if c == KEY["cmd"]]
    main_evt = [f for c, f in seen if c == KEY["1"]]
    assert cmd_evt == [Quartz.kCGEventFlagMaskCommand], "修饰键自己的事件也要带标志位"
    assert main_evt == [Quartz.kCGEventFlagMaskCommand], "主键事件没带 Cmd 标志位"


def test_no_modifiers_means_no_flags():
    from iphone_agent.driver.injector import KEY
    seen = []

    class Rec:
        def activate(self, timeout_s=0.5): pass
        def key_down(self, code, flags=0): seen.append((code, flags))
        def key_up(self, code): pass
        def release_all(self): pass

    dev = _device_with(Rec())
    dev.press(KEY["return"])
    assert seen == [(KEY["return"], 0)]
