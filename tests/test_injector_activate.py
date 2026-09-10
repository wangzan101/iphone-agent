"""activate() 的控制流测试。

真实缺陷（2026-09-05 真机）：轮询用 time.sleep 时 NSWorkspace.frontmostApplication()
返回陈旧缓存，激活明明成功也读不到，导致 ActivateFailed。修复是改用泵 run loop 读取。
run loop 行为本身要真机才测得了，这里锁住 activate 的控制流。
"""
import pytest

from iphone_agent.driver.injector import ActivateFailed, ForegroundInjector


class FakeInjector(ForegroundInjector):
    """把「读前台 pid」「真正激活」「查进程」三处换成可编程的假实现。"""

    def __init__(self, pid, front_seq, app_exists=True):
        super().__init__(pid)
        self._front_seq = list(front_seq)
        self._app_exists = app_exists
        self.activate_calls = 0
        self.reads = 0

    def _running_app(self):
        return object() if self._app_exists else None

    def _do_activate(self, app):
        self.activate_calls += 1

    def _frontmost_pid(self, pump_s=0.02):
        # 序列耗尽后保持最后一个值：真实的前台 pid 总是有值的，
        # 「一直没切过去」应该模型成持续读到同一个别人的 pid，而不是读到 None。
        self.reads += 1
        if len(self._front_seq) > 1:
            return self._front_seq.pop(0)
        return self._front_seq[0] if self._front_seq else None


def test_already_frontmost_skips_activation():
    inj = FakeInjector(42, front_seq=[42])
    inj.activate()
    assert inj.activate_calls == 0  # 已在前台就不该再切一次窗口
    assert inj.reads == 1


def test_returns_once_target_becomes_frontmost():
    # 先读到别人，再读到目标 —— 模拟激活需要几十毫秒才生效
    inj = FakeInjector(42, front_seq=[719, 719, 42])
    inj.activate(timeout_s=2.0)
    assert inj.activate_calls == 1
    assert inj.reads == 3


def test_timeout_reports_both_target_and_actual_pid():
    inj = FakeInjector(42, front_seq=[719])
    with pytest.raises(ActivateFailed) as ei:
        inj.activate(timeout_s=0.05)
    msg = str(ei.value)
    assert "42" in msg and "719" in msg  # 目标和实际都要出现，否则没法归因


def test_missing_pid_raises_before_any_activation():
    inj = FakeInjector(42, front_seq=[42], app_exists=False)
    with pytest.raises(ActivateFailed, match="不存在"):
        inj.activate()
    assert inj.activate_calls == 0 and inj.reads == 0
