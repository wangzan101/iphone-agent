"""打开 App 后核身份：先看新帧标注，对不上再问（spec 2026-09-12 §5.3）。"""
from types import SimpleNamespace

import pytest

import iphone_agent.harness.judge as judge
from iphone_agent.harness.executor import Executor
from iphone_agent.perceive.screen import ScreenLabel
from iphone_agent.twin.context import ScreenIdentityContext
from iphone_agent.twin.live import LiveTwin


@pytest.fixture
def judge_calls(monkeypatch):
    calls = []

    def fake(image, name, asker):
        calls.append(name)
        return judge.AppCheck(is_app=True, confidence=0.9, seen=name, why="stub")
    monkeypatch.setattr(judge, "is_target_app", fake)
    return calls


def new(app=None):
    return SimpleNamespace(image=None, screen=ScreenLabel(app, "首页", None, ()) if app else None)


def ex(ctx=None):
    return Executor(None, None, asker=object(), identity=ctx)


def test_matching_label_skips_the_judge(judge_calls):
    got = ex(ScreenIdentityContext(LiveTwin(None, "r")))._identity("设置", new("设置"))
    assert got == {"verified": True, "by": "screen_label", "seen": "设置"} and judge_calls == []


def test_without_context_the_plain_name_still_counts(judge_calls):
    assert ex()._identity("设置", new("设置"))["by"] == "screen_label" and judge_calls == []


def test_mismatch_asks_the_judge_and_learns_the_alias(judge_calls):
    ctx = ScreenIdentityContext(LiveTwin(None, "r"))
    first = ex(ctx)._identity("App Store", new("应用商店"))
    assert first["by"] == "judge" and first["verified"] is True and first["label_app"] == "应用商店"
    assert judge_calls == ["App Store"]
    second = ex(ctx)._identity("App Store", new("应用商店"))
    assert second["by"] == "screen_label" and judge_calls == ["App Store"], "学到别名后第二次不再问"


def test_unsure_or_missing_label_asks_the_judge(judge_calls):
    assert ex()._identity("设置", new("不确定"))["by"] == "judge"
    assert "label_app" not in ex()._identity("设置", new())
    assert judge_calls == ["设置", "设置"]


def _od_per(reply):
    from iphone_agent.perceive.observe import Perceiver
    from iphone_agent.perceive.ocr import RawBox
    from tests.test_ensure_label import Asker
    return Perceiver(ocr=lambda im: [RawBox("通用", 0.99, 0.1, 0.8, 0.3, 0.04)], asker=Asker(reply))


def _frame():
    from PIL import Image

    from iphone_agent.driver.geometry import Frame, Rect
    return Frame(Image.new("RGB", (600, 1200), "white"), 600, 1200, Rect(0, 0, 300, 600), 0.0, 1)


def test_on_demand_identity_is_one_short_label_and_no_judge_when_it_matches(judge_calls):
    """spec 2026-09-14 §5.2：on_demand 下 new 没有整屏标注，_identity 先补一次短标注再比对。"""
    per = _od_per({"screen": {"app": "设置", "name": "通用", "same_as": None, "anchors": []}})
    with per.task_scope(None, "on_demand"):
        new = per.observe(_frame())
        ex_ = Executor(None, per, asker=per.asker)
        got = ex_._identity("设置", new)
        ex_._identity("设置", new)                         # 同一帧：幂等，不再标
        per.asker.replies.append({"screen": {"app": "设置", "name": "通用", "same_as": None, "anchors": []}})
        ex_._identity("设置", per.observe(_frame()))      # 开错一次换下一路 = 又一帧：又一次 identity 标注
    assert got["by"] == "screen_label" and judge_calls == []
    st = per.stats()
    assert st["label"]["identity"]["ok"] == 2 and st["label"]["adopt"]["ok"] == 0
    assert sum(st["parse_by"].values()) == 0


def test_on_demand_identity_mismatch_still_asks_the_judge(judge_calls):
    per = _od_per({"screen": {"app": "备忘录", "name": "首页", "same_as": None, "anchors": []}})
    with per.task_scope(None, "on_demand"):
        got = Executor(None, per, asker=per.asker)._identity("设置", per.observe(_frame()))
    assert got["by"] == "judge" and judge_calls == ["设置"] and got["label_app"] == "备忘录"
