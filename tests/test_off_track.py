"""动作结果的第三态：画面变了，但变到了不该到的地方。

以前只有「变 / 没变」。误点进一个**新的**错页面满足「变了且是新画面」，
于是被熔断器当成进展、把 no_progress 清零 —— 误点这类失败在它眼里是看不见的
（设计说明，Mobile-Agent-E 的 A/B/C；设计说明，mirroir 的 PostActionVerifier）。

信号从哪来：模型 103 次运行、1113 步里**一次都没填过**自评字段（2026-09-09 统计 runs/），
所以只能在「变了」这一侧也问一次判官 —— 但只在模型给了 expect 时问，
没有预期就没有可核对的东西，也省下这次调用。
"""
from iphone_agent.harness.actions import Action
from iphone_agent.harness.executor import Executor
from iphone_agent.harness.guard import ActionGuard, no_progress_hint
from iphone_agent.harness.history import row_from_record


def act(name, **args):
    return Action(name, args, "r", None, "c")


# ---------- guard ----------

def test_off_track_change_is_not_progress():
    g = ActionGuard()
    g.record_screen(1)
    g.record_outcome(act("tap", x=1, y=1), changed=True, new_screen_hash=2, off_track=True)
    assert g.no_progress == 1
    assert g.last_reason == "wrong_page"
    assert 2 in g._seen_screens, "错页面也是见过的画面，再回来算 revisited"
    says = no_progress_hint(g.last_reason)
    assert "预期" in says and "没点中" not in says, says


def test_off_track_does_not_wash_away_earlier_no_progress():
    g = ActionGuard()
    g.record_outcome(act("tap", x=1, y=1), changed=False, new_screen_hash=1)
    g.record_outcome(act("tap", x=2, y=2), changed=False, new_screen_hash=1)
    assert g.no_progress == 2
    v = g.record_outcome(act("tap", x=3, y=3), changed=True, new_screen_hash=5, off_track=True)
    assert g.no_progress == 3 and v == "warn"


def test_off_track_tap_did_land_so_dead_taps_reset():
    g = ActionGuard()
    g.record_outcome(act("tap", x=1, y=1), changed=False, new_screen_hash=1)
    assert g.dead_taps == 1
    g.record_outcome(act("tap", x=2, y=2), changed=True, new_screen_hash=5, off_track=True)
    assert g.dead_taps == 0, "点是点中了（画面变了），只是去错了地方，不算点击失灵"


# ---------- executor ----------

class _Judge:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def ask_json(self, prompt, images, max_tokens=None):
        self.calls += 1
        return self.replies.pop(0) if self.replies else None


def _tap_with_expect(fake_env, judge, expect):
    dev, per, _ = fake_env([["通用"], ["通知"], ["通知"], ["通知"]])
    obs = per.observe(dev.capture())
    ex = Executor(dev, per, asker=judge)
    e = obs.elements[0]
    res, _ = ex.run(Action("tap", {"id": 1, "x": e.center[0], "y": e.center[1]}, "r", expect, "c1"), obs)
    return res


def test_changed_but_judged_off_track_is_reported(fake_env):
    judge = _Judge({"worked": False, "confidence": 0.9, "why": "进的是通知页，不是关于本机"})
    res = _tap_with_expect(fake_env, judge, "进入关于本机页")
    assert res.ok and res.changed is True
    assert res.extra["judged"]["worked"] is False
    assert "通知页" in (res.hint or ""), "判官的原因要原文回给模型"
    assert "预期" in res.hint


def test_changed_and_judged_on_track_has_no_hint(fake_env):
    judge = _Judge({"worked": True, "confidence": 0.9, "why": "到了"})
    res = _tap_with_expect(fake_env, judge, "进入通知页")
    assert res.changed is True and res.extra["judged"]["worked"] is True
    assert not res.hint


def test_changed_without_expect_does_not_ask_the_judge(fake_env):
    judge = _Judge({"worked": False, "confidence": 0.9, "why": "x"})
    res = _tap_with_expect(fake_env, judge, None)
    assert res.changed is True and judge.calls == 0 and "judged" not in res.extra


def test_judge_failure_on_changed_side_falls_back_to_old_behaviour(fake_env):
    res = _tap_with_expect(fake_env, _Judge(), "进入关于本机页")     # 没回复 = 没问成
    assert res.changed is True and "judged" not in res.extra and not res.hint


# ---------- history ----------

def _rec(step, name, args, result, **kw):
    r = {"step": step, "action": {"name": name, "args": args},
         "model": {"reason": "r", "expect": None}, "result": result}
    r.update(kw)
    return r


def test_off_track_judgement_fills_expected_when_model_did_not():
    """expected 列本来靠模型自评，而它从来不填。判官看图判了「没达到预期」时，
    这一列就该有话说，且说清楚是看图判的。"""
    # ToolResult.to_json 把 extra 摊平到顶层，留档里 judged 就在 result 顶层
    row = row_from_record(_rec(3, "tap", {"id": 1},
                               {"ok": True, "changed": True,
                                "judged": {"worked": False, "confidence": 0.9,
                                           "why": "进的是通知页"}}))
    assert row.changed == "yes"
    assert row.expected == "no(看图)"
    assert "通知页" in row.note


def test_model_own_assessment_beats_the_judge_in_the_row():
    K = "ev" + "al"
    row = row_from_record(_rec(3, "tap", {"id": 1},
                               {"ok": True, "changed": True,
                                "judged": {"worked": False, "why": "x"}},
                               **{K: {"expected": "yes", "note": "其实到了"}}))
    assert row.expected == "yes" and row.note == "其实到了"
