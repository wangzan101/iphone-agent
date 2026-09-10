"""结构化自检。

重点不是「检查得准不准」（那要真机），而是**这一层永远不抛、永远给得出下一步**：
界面拿着这份结果去引导一个不会用电脑的人，任何一项塌了都不能把整份结果带走。
"""
import pytest

from iphone_agent import health
from iphone_agent.health import FAIL, INFO, PASS, Check


class _Boom:
    """每一个属性都炸 —— 模拟镜像没连、权限没给的机器。"""
    def __getattr__(self, name):
        raise RuntimeError("镜像窗口不可用")


def test_guard_turns_exception_into_a_fail_check():
    def explode():
        raise RuntimeError("炸了")
    c = health._guard(explode, "x.y", "某项", "因为重要")
    assert c.status == FAIL
    assert c.id == "x.y" and c.title == "某项"
    assert "炸了" in c.detail
    assert c.fix, "失败的检查必须给出下一步，否则界面上就是个死胡同"


def test_run_checks_never_raises_on_a_broken_machine():
    """一台什么都没配好的机器上，run_checks 仍要返回完整的一份清单。"""
    checks = health.run_checks(_Boom())
    assert len(checks) == 7
    ids = [c.id for c in checks]
    assert ids == ["perm.accessibility", "perm.screen_recording", "mirror.window",
                   "mirror.capture", "perceive.ocr", "model", "injector"]


def test_every_failing_check_tells_you_what_to_do():
    for c in health.run_checks(_Boom()):
        if c.status == FAIL:
            assert c.fix, f"{c.id} 失败了却没说怎么修"
            assert c.why, f"{c.id} 没说为什么重要"


def test_permission_checks_carry_a_jump_url_when_failing(monkeypatch):
    """权限没给的时候必须给出能直接跳到那一页设置的 URL —— 让用户自己找是产品化里最掉人的一步。"""
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "ApplicationServices",
                        SimpleNamespace(AXIsProcessTrusted=lambda: False))
    monkeypatch.setitem(sys.modules, "Quartz",
                        SimpleNamespace(CGPreflightScreenCaptureAccess=lambda: False))
    a = health.check_accessibility()
    s = health.check_screen_recording()
    assert a.status == FAIL and a.fix_url == health.PREF_ACCESSIBILITY
    assert s.status == FAIL and s.fix_url == health.PREF_SCREEN_RECORDING


def test_granted_permissions_have_no_fix_url(monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "ApplicationServices",
                        SimpleNamespace(AXIsProcessTrusted=lambda: True))
    a = health.check_accessibility()
    assert a.status == PASS and a.fix_url == "" and a.fix == ""


def test_all_ok_ignores_non_blocking_entries():
    """injector 是说明性的一条：它不 pass 也不该把整体判成不可用。"""
    checks = [Check(id="a", title="A", status=PASS),
              Check(id="injector", title="注入路径", status=INFO, blocking=False)]
    assert health.all_ok(checks)
    assert not health.all_ok(checks + [Check(id="b", title="B", status=FAIL)])


def test_as_dicts_is_json_shaped():
    import json
    d = health.as_dicts(health.run_checks(_Boom()))
    json.dumps(d)                                     # 不能抛：这是要走 HTTP 的
    assert set(d[0]) == {"id", "title", "status", "detail", "why", "fix", "fix_url", "blocking"}


@pytest.mark.parametrize("status,ok", [(PASS, True), (INFO, True), (FAIL, False)])
def test_ok_property(status, ok):
    assert Check(id="x", title="x", status=status).ok is ok
