"""恢复阶梯本身：顺序、每级之后重做、点击那种没法重做的、重启上限、留痕。"""
from types import SimpleNamespace

import pytest

from iphone_agent.harness.recovery import DeviceChannelDead, Recovery


def _dev(pid=1):
    win = SimpleNamespace(pid=pid, window_id=10)
    d = SimpleNamespace(win=win, released=0)
    d.release_all = lambda: setattr(d, "released", d.released + 1)
    d._resolve = lambda: d.win
    return d


def test_rungs_are_tried_cheapest_first_and_stop_at_the_first_redo_that_works():
    dev = _dev()
    calls = []
    rec = Recovery(dev, None, restart=lambda: calls.append("restart") or 0.1,
                   ensure=lambda d, p, obs=None: calls.append("ensure") or (True, "连接正常", None))
    attempts = []
    assert rec.climb("keyboard", redo=lambda: attempts.append(1) or len(attempts) == 2) is True
    assert [e["rung"] for e in rec.events] == ["resolve", "connect"]
    assert calls == ["ensure"] and dev.released == 1
    assert [e["ok"] for e in rec.events] == [False, True]


def test_exhausted_ladder_raises_with_what_was_tried():
    dev = _dev()
    rec = Recovery(dev, None, restart=lambda: 0.1, ensure=lambda d, p, obs=None: (True, "ok", None))
    with pytest.raises(DeviceChannelDead) as e:
        rec.climb("keyboard", redo=lambda: False)
    assert [t["rung"] for t in e.value.tried] == ["resolve", "connect", "restart"]
    assert rec.restarts == 1


def test_restart_budget_is_per_task():
    dev = _dev()
    n = []
    rec = Recovery(dev, None, restart=lambda: n.append(1) or 0.1,
                   ensure=lambda d, p, obs=None: (True, "ok", None), max_restarts=1)
    with pytest.raises(DeviceChannelDead):
        rec.climb("keyboard", redo=lambda: False)
    with pytest.raises(DeviceChannelDead) as e:
        rec.climb("keyboard", redo=lambda: False)
    assert len(n) == 1, "第二次爬梯不该再重启"
    assert e.value.tried[-1]["skipped"] is True and "不再重启" in e.value.tried[-1]["detail"]


def test_tap_channel_without_redo_counts_a_rung_that_actually_did_something():
    """点击没法重做：某一级真的做了事（pid 变了 / 连接恢复 / 重启了）就算救过，调用方清计数再看。"""
    dev = _dev(pid=1)
    rec = Recovery(dev, None, restart=lambda: 0.1, ensure=lambda d, p, obs=None: (True, "连接正常", None))
    # ① 窗口没变、② 本来就连着 → 都不算做了事；③ 重启算
    assert rec.climb("tap") is True
    assert [e["rung"] for e in rec.events] == ["resolve", "connect", "restart"]
    rec.drain()
    dev2 = _dev(pid=1)
    dev2._resolve = lambda: SimpleNamespace(pid=2, window_id=11)      # 镜像重启过，pid 变了
    rec2 = Recovery(dev2, None, restart=lambda: 0.1, ensure=lambda d, p, obs=None: (True, "ok", None))
    assert rec2.climb("tap") is True
    assert [e["rung"] for e in rec2.events] == ["resolve"]


def test_a_rung_that_blows_up_does_not_block_the_next_one():
    dev = _dev()
    def boom(d, p, obs=None):
        raise RuntimeError("窗口找不到")
    rec = Recovery(dev, None, restart=lambda: 0.1, ensure=boom)
    with pytest.raises(DeviceChannelDead) as e:
        rec.climb("keyboard", redo=lambda: False)
    connect = e.value.tried[1]
    assert connect["skipped"] and "RuntimeError" in connect["detail"]
    assert e.value.tried[2]["rung"] == "restart"


def test_drain_hands_events_to_the_run_log_once():
    dev = _dev()
    rec = Recovery(dev, None, restart=lambda: 0.1, ensure=lambda d, p, obs=None: (True, "ok", None))
    rec.climb("keyboard", redo=lambda: True)
    ev = rec.drain()
    assert len(ev) == 1 and ev[0]["rung"] == "resolve" and "ms" in ev[0]
    assert rec.drain() == []
