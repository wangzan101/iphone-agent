"""「连接暂停」自愈：点「继续」之后要给握手留够时间，握手期间不能再点。

iphone-use 真机记录（`iphone-use upstream: crates/server/src/main.rs:465-469`，2026-06-12）：
镜像重连握手要 10–30 秒，**握手中途落下的一次点击会取消它**，急着重试会把
「能连上」变成「连上就断」。所以点完「继续」只能看，不能点；看够一个握手窗口还没回来，
才允许第二次。
"""
from iphone_agent.harness.reconnect import ensure_connected

PAUSED = ["连接暂停", "继续"]
NORMAL = ["设置", "通用"]


def _taps(dev):
    return [c for c in dev.calls if c[0] == "tap"]


def test_recovery_inside_handshake_window_needs_only_one_tap(fake_env):
    """点一次 → 头两次回看仍是暂停页（握手中）→ 第三次回来了。全程只许点一次。"""
    dev, per, _ = fake_env([PAUSED, PAUSED, PAUSED, NORMAL, NORMAL])
    ok, why, obs = ensure_connected(dev, per, handshake_s=1.0, poll_s=0.01)
    assert ok, why
    assert len(_taps(dev)) == 1, f"握手期间不该再点：{dev.calls}"
    assert obs is not None and "设置" in [e.text for e in obs.elements]


def test_second_tap_only_after_handshake_window_expires(fake_env):
    """一直不回来：等满一个握手窗口才点第二次；第二次也等满才放弃。有界，不会死等。"""
    dev, per, _ = fake_env([PAUSED] * 40)
    ok, why, _ = ensure_connected(dev, per, handshake_s=0.05, poll_s=0.01)
    assert not ok
    assert len(_taps(dev)) == 2, f"两次机会用完就该放弃：{dev.calls}"
    assert "两次" in why
