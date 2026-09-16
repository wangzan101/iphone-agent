"""点击的预期核对：先看「」里的字有没有新出现（肯定信号，直接用），对不上再看图（否定结论交给会看图的一方）。

docs/superpowers/specs/2026-09-11-点击预期核对-design.md §3.2。
"""
import pytest

from iphone_agent import config
from iphone_agent.harness.actions import Action
from iphone_agent.harness.executor import Executor, expect_keys, expect_met_by_text
from iphone_agent.harness.guard import ActionGuard


class _Judge:
    """假的看图一方，记下每次的提示词。回复按被问的那一问作答：提示词要 "met" 键（judge.met_expectation，
    「是不是预期的样子」）就把结论放在 met 下，否则放在 worked 下（judge.did_action_work）—— 真判官也是
    照提示词要的 JSON 键回的。回复里写的都是同一个结论，只是分发到对应的键。"""
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.prompts = []

    def ask_json(self, prompt, images, max_tokens=None):
        self.calls += 1
        self.prompts.append(prompt)
        r = self.replies.pop(0) if self.replies else None
        if isinstance(r, dict) and "worked" in r and '"met"' in prompt:
            r = {("met" if k == "worked" else k): v for k, v in r.items()}
        return r


def _tap(fake_env, judge, expect, before, after, name="tap"):
    dev, per, _ = fake_env([before, after, after, after])
    obs = per.observe(dev.capture())
    e = obs.elements[0]
    # scroll 的注入同时读 direction 和 amount（executor._inject），少一个就是 KeyError
    args = ({"id": 1, "x": e.center[0], "y": e.center[1]} if name == "tap"
            else {"direction": "down", "amount": "page"})
    res, new = Executor(dev, per, asker=judge).run(Action(name, args, "r", expect, "c1"), obs)
    return res, new


# ---------- 纯函数 ----------

def test_expect_keys_takes_quoted_parts_without_spaces():
    assert expect_keys("进入通用页，出现「关于 本机」和「软件更新」") == ["关于本机", "软件更新"]
    assert expect_keys("进入设置主界面") == [] and expect_keys(None) == []


def test_text_match_needs_newly_appeared_text_and_whole_single_chars():
    assert expect_met_by_text("出现「关于本机」", ["通用", "关于本机 >"]) == "关于本机"
    assert expect_met_by_text("出现「关于本机」", ["通用"]) is None
    assert expect_met_by_text("金额变成「1」", ["11"]) is None, "单字只认整条相等：「1」不能算在「11」里"
    assert expect_met_by_text("金额变成「1」", ["1"]) == "1"


# ---------- 执行层 ----------

def test_new_text_confirms_the_expectation_without_asking_vision(fake_env):
    judge = _Judge({"worked": False, "confidence": 0.9, "why": "不该被问到"})
    res, _ = _tap(fake_env, judge, "进入通用页，出现「关于本机」", ["通用"], ["通用", "关于本机"])
    assert res.changed is True
    assert res.extra["expect_check"] == {"met": True, "by": "text", "matched": "关于本机"}
    assert judge.calls == 0 and "judged" not in res.extra and not res.hint


def test_key_that_was_already_on_screen_is_not_evidence(fake_env):
    """点「通用」这一行进「通用」页：「通用」点前就在屏上，它还在不证明到了 —— 交给看图。"""
    judge = _Judge({"worked": True, "confidence": 0.9, "why": "到了通用页"})
    res, _ = _tap(fake_env, judge, "进入「通用」页", ["通用", "通知"], ["通用", "关于本机"])
    assert judge.calls == 1
    assert res.extra["expect_check"] == {"met": True, "by": "vision"}


def test_text_miss_falls_back_to_vision_and_reports_off_track(fake_env):
    judge = _Judge({"worked": False, "confidence": 0.9, "why": "进的是通知页"})
    res, _ = _tap(fake_env, judge, "出现「关于本机」", ["通用"], ["通用", "通知"])
    assert res.extra["expect_check"] == {"met": False, "by": "vision"}
    assert res.extra["judged"]["on_change"] is True and res.extra["judged"]["worked"] is False
    assert "没达到预期" in res.hint and "通知页" in res.hint


def test_expectation_without_quotes_goes_straight_to_vision(fake_env):
    judge = _Judge({"worked": True, "confidence": 0.8, "why": "到了"})
    res, _ = _tap(fake_env, judge, "进入设置主界面", ["通用"], ["通用", "关于本机"])
    assert judge.calls == 1 and res.extra["expect_check"] == {"met": True, "by": "vision"}


def test_vision_not_answering_is_unverified_not_a_no(fake_env):
    res, _ = _tap(fake_env, _Judge(), "出现「关于本机」", ["通用"], ["通用", "通知"])
    assert res.extra["expect_check"] == {"met": None, "by": "unverified"}
    assert "judged" not in res.extra and not res.hint


def test_missing_expectation_is_recorded_and_costs_nothing(fake_env):
    judge = _Judge({"worked": False, "confidence": 0.9, "why": "x"})
    res, _ = _tap(fake_env, judge, None, ["通用"], ["通用", "通知"])
    assert res.extra["expect_check"] == {"met": None, "by": "missing"}
    assert judge.calls == 0


def test_unchanged_tap_with_expectation_uses_the_existing_negative_side_check(fake_env):
    judge = _Judge({"worked": False, "confidence": 0.9, "why": "两张图一样"})
    res, _ = _tap(fake_env, judge, "出现「关于本机」", ["通用"], ["通用"])
    assert res.changed is False
    assert res.extra["expect_check"] == {"met": False, "by": "vision"}
    assert "看图复核也说没生效" in res.hint


def test_the_check_sees_all_new_text_while_the_model_still_gets_the_capped_list(fake_env, monkeypatch):
    """同一个集合差入口：核对看全量，给模型的 appeared 仍按 TRANSITION_LIST_MAX 截断（行为不变）。"""
    monkeypatch.setattr(config, "TRANSITION_LIST_MAX", 2)
    judge = _Judge()
    res, _ = _tap(fake_env, judge, "出现「丁」", ["通用"], ["通用", "甲", "乙", "丙", "丁"])
    assert res.extra["appeared"] == ["甲", "乙"]
    assert res.extra["expect_check"]["by"] == "text" and judge.calls == 0


def test_text_confirmation_is_progress_not_off_track(fake_env):
    res, new = _tap(fake_env, _Judge(), "出现「关于本机」", ["通用"], ["通用", "关于本机"])
    g = ActionGuard()
    g.record_result(Action("tap", {"id": 1}, "r", "出现「关于本机」", "c1"), res, new)
    assert g.last_reason is None and g.no_progress == 0


@pytest.mark.parametrize("expect", [None, "出现「关于本机」"])
def test_non_tap_actions_get_no_expect_check(fake_env, expect):
    res, _ = _tap(fake_env, _Judge(), expect, ["通用"], ["通用", "关于本机"], name="scroll")
    assert "expect_check" not in res.extra


# ---------- 终审 2026-09-11：数字关键字要有数字边界（I1） ----------

def test_numeric_key_needs_digit_boundaries():
    """「13」不能算在「113」里、「100」不能算在「1100」里 —— 和 2026-09-09「金额 1 连点成 11」同一类。
    文字核对一旦误确认就不看图、不给 hint，历史表写 yes(文字)，是静默的错误肯定（CLAUDE.md §3）。"""
    assert expect_met_by_text("金额变成「13」", ["113"]) is None
    assert expect_met_by_text("金额变成「13」", ["1300"]) is None
    assert expect_met_by_text("金额变成「13」", ["-13.00"]) == "13"
    assert expect_met_by_text("金额变成「13」", ["13元"]) == "13"
    assert expect_met_by_text("出现「100」", ["1100"]) is None


def test_whitespace_is_stripped_including_fullwidth_spaces():
    assert expect_met_by_text("出现「关于　本机」", ["关于本机"]) == "关于本机"
    assert expect_met_by_text("出现「关于本机」", ["关于　本机"]) == "关于本机"
    assert expect_keys("出现「关于　本机」") == ["关于本机"]


# ---------- 终审 2026-09-11：画面变了这一侧，tap 问「是不是预期的样子」（I2） ----------

def test_changed_tap_asks_whether_the_screen_looks_as_expected(fake_env):
    """文字核对落空后，tap 问 judge.met_expectation，不问 did_action_work 的「这个动作生效了吗」——
    画面已经变了，按字面答「点中了、页面动了 = 生效」会放过错页面。"""
    judge = _Judge({"worked": True, "confidence": 0.9, "why": "到了关于本机页"})
    res, _ = _tap(fake_env, judge, "进入关于本机页，出现「软件更新」", ["通用"], ["通用", "关于本机"])
    assert judge.calls == 1
    p = judge.prompts[0]
    assert "进入关于本机页，出现「软件更新」" in p
    assert "预期的样子" in p and "这个动作生效了吗" not in p
    assert res.extra["expect_check"] == {"met": True, "by": "vision"}


def test_changed_tap_judged_keeps_the_keys_guard_and_twin_read(fake_env):
    """guard 的 off_track 判定、孪生分支的记账（record.off_track_of）读的都是 judged.worked / judged.on_change ——
    改键名会静默破坏它们。hint 文案也不变。"""
    judge = _Judge({"worked": False, "confidence": 0.8, "why": "进的是通知页"})
    res, _ = _tap(fake_env, judge, "出现「关于本机」", ["通用"], ["通用", "通知"])
    assert res.extra["judged"] == {"worked": False, "confidence": 0.8, "why": "进的是通知页", "on_change": True}
    assert res.hint == "画面变了，但看图复核说没达到预期：进的是通知页。先确认自己在哪。"
    assert res.extra["expect_check"] == {"met": False, "by": "vision"}


def test_changed_scroll_with_expect_still_asks_whether_the_action_worked(fake_env):
    """非 tap 动作行为不变：画面变了、带 expect，仍问 did_action_work。"""
    judge = _Judge({"worked": False, "confidence": 0.9, "why": "没滚到想要的地方"})
    res, _ = _tap(fake_env, judge, "看到更多设置项", ["通用"], ["通用", "关于本机"], name="scroll")
    assert judge.calls == 1 and "这个动作生效了吗" in judge.prompts[0]
    assert res.extra["judged"]["on_change"] is True and res.extra["judged"]["worked"] is False
    assert "没滚到想要的地方" in res.hint
    assert "expect_check" not in res.extra


def test_empty_quotes_give_no_keys():
    assert expect_keys("进入「」页面") == []
    assert expect_keys("出现「 」") == []
    assert expect_keys("出现「　」") == []
