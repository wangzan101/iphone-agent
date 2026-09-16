"""放弃或熔断之前，程序补看一次全屏（spec 2026-09-14 §3.5）：不看屏幕内容，只在两个程序自己知道的时刻补证据。"""
import json
import time
from dataclasses import replace

from iphone_agent.harness.loop import run_task
from iphone_agent.harness.runlog import RunLog
from iphone_agent.workspace import RunConfig, Workspace
from tests.conftest import CountingAsker
from tests.test_loop import INITIAL_SETTLE_FRAMES, ScriptedModel, isolated_store
from tests.test_loop_adopt import Labeler, run_with
from tests.test_screen_parse_policy import code_hits

FAILED = [("done", {"status": "failed", "result": "找不到"})]
TAPS = [[("tap", {"x": 10 + 35 * i, "y": 10})] for i in range(11)]     # 不同桶，画面都不变


def _recs(r):
    return RunLog.read_steps(r.run_dir)


def _meta(r):
    return json.loads((r.run_dir / "run.json").read_text(encoding="utf-8"))


def test_first_done_failed_buys_one_full_screen_look_and_the_task_goes_on(fake_env, tmp_path):
    r, dev, per, m = run_with(fake_env, [["通用"]] * 4, [FAILED, FAILED], tmp_path, asker=Labeler())
    assert r.end_reason == "done_failed" and len(m.seen) == 2 and r.steps == 2
    first = next(x for x in _recs(r) if "action" in x)
    assert first["result"]["error"] == "fallback_observe" and first["fallback"] == {"reason": "done_failed"}
    assert first["after_observation"]["perception"]["parse"] == "fallback"
    assert "补看了一次全屏，元素表已更新" in first["result"]["hint"]
    assert _meta(r)["perception"]["fallback_by"] == {"done_failed": 1, "no_progress": 0}


def test_a_failed_fallback_look_says_so(fake_env, tmp_path):
    r, *_ = run_with(fake_env, [["通用"]] * 4, [FAILED, FAILED], tmp_path, asker=CountingAsker())
    first = next(x for x in _recs(r) if "action" in x)
    assert "看全屏失败" in first["result"]["hint"] and "元素表已更新" not in first["result"]["hint"]


class WrongPageJudge(CountingAsker):
    """整屏解析和短标注一律没问成（同 CountingAsker）；只有「是不是预期的样子」（judge.EXPECT_PROMPT）答「不是」。"""

    def ask_json(self, prompt, images, max_tokens=None):
        super().ask_json(prompt, images, max_tokens)
        return {"met": False, "confidence": 0.8, "why": "进的是另一页"} if '"met"' in prompt else None


class FirstTapExpectsModel(ScriptedModel):
    """ScriptedModel 构造 Action 时 expect 一律是 None；这里只给第一步的 tap 补一句对不上的预期，
    后面的 tap 不带预期 —— 不问判官，guard 只按「见没见过」记 revisited。"""

    def decide(self, messages, image_size, tools=None, max_tokens=None):
        reply = super().decide(messages, image_size, tools, max_tokens)
        if len(self.seen) == 1:
            reply.actions = [replace(a, expect="进入「目标页」") if a.name == "tap" else a for a in reply.actions]
        return reply


# 两屏来回：每一步画面都真的变了，但没有进展 —— 模型在原地打转、找不到元素，正是兜底要对付的形状。
# ⚠ 不能用 TAPS（画面都不变）：每一下都是死点击，每 DEAD_TAPS_ALERT 次爬一次恢复阶梯，
#   兜底把熔断推到第 9 步正好是第三次爬梯，RECOVERY_MAX_RESTARTS=2 用完，任务以 device_error 结束，
#   第二次熔断没机会判。这个用例考的是「兜底之后熔断还在」，不是恢复阶梯。
# 第 1 步离开起始屏时落到的 B 屏是没见过的，按 guard 的判据本该算进展（第 7 步才熔断）；
#   所以第 1 步让看图复核说「没达到预期」（off_track，guard 记 wrong_page），B 记成见过但不算进展。
#   此后每一步都是回到见过的画面（revisited），dead_taps 每步清零，恢复阶梯不会被碰到。
LOOP = [["通用", "关于本机", "软件更新"], ["隐私与安全性", "定位服务", "跟踪"]]
LOOP_MOVES = {(0, "tap"): INITIAL_SETTLE_FRAMES + 1, (INITIAL_SETTLE_FRAMES + 1, "tap"): 0}


def test_first_stop_buys_one_look_then_three_more_unproductive_steps_end_the_task(action_env, tmp_path):
    dev, per, _ = action_env([LOOP[0]] * INITIAL_SETTLE_FRAMES + LOOP, LOOP_MOVES)
    per.asker = WrongPageJudge()
    m = FirstTapExpectsModel(TAPS)
    r = run_task("t", dev, per, m, tmp_path, run_config=RunConfig(screen_parse="on_demand"),
                 store=isolated_store(tmp_path), workspace=Workspace(tmp_path))
    assert r.end_reason == "no_progress" and r.steps == 9, "设回 WARN 后再 3 步在 == 处熔断，不会一路跑到 max_steps"
    prog = [x for x in _recs(r) if x.get("by") == "program"]
    assert len(prog) == 1 and prog[0]["action"]["name"] == "observe"
    assert prog[0]["fallback"] == {"reason": "no_progress"}
    assert prog[0]["after_observation"]["perception"]["parse"] == "fallback"
    assert "程序补看了一次全屏" in json.dumps(m.seen[6], ensure_ascii=False)
    assert _meta(r)["perception"]["fallback_by"] == {"done_failed": 0, "no_progress": 1}


def test_both_triggers_share_one_quota(fake_env, tmp_path):
    r, *_ = run_with(fake_env, [["a"]] * 80, [FAILED] + TAPS, tmp_path, asker=CountingAsker())
    assert r.end_reason == "no_progress" and r.steps == 7
    assert _meta(r)["perception"]["fallback_by"] == {"done_failed": 1, "no_progress": 0}


def test_off_mode_never_falls_back(fake_env, tmp_path):
    r, *_ = run_with(fake_env, [["a"]] * 80, TAPS, tmp_path, mode="off", asker=CountingAsker())
    assert r.end_reason == "no_progress" and r.steps == 6


def test_a_frame_that_was_already_looked_at_does_not_fall_back(fake_env, tmp_path):
    r, *_ = run_with(fake_env, [["a"]] * 80, TAPS, tmp_path, mode="always", asker=Labeler())
    assert r.end_reason == "no_progress" and r.steps == 6
    assert not [x for x in _recs(r) if x.get("by") == "program"]


def test_fallback_due_is_asked_in_exactly_one_place():
    hits = [h for h in code_hits(r"\bfallback_due\(") if not h.startswith("perceive/policy.py")]
    assert len(hits) == 1 and hits[0].startswith("harness/loop.py:"), hits


# ⚠ Fix round 1（评审发现 1）：兜底两条路都是 continue 回循环顶部，那里 steps >= max_steps /
#   超过时限的判断排在最前面。若兜底恰好发生在最后一步或时间已经用完，下一圈会立刻把这一步刚拿到的
#   done_failed / no_progress 连同模型的 result 一起吞成 max_steps / timeout，还白打一次整屏解析。
#   兜底因此还要问「兜底完还走得下去吗」：步数没到上限、时间也还够。

def test_done_failed_on_the_last_allowed_step_ends_clean_without_a_wasted_fallback(fake_env, tmp_path):
    r, dev, per, m = run_with(fake_env, [["通用"]] * 4, [FAILED, FAILED], tmp_path,
                              asker=Labeler(), max_steps=1)
    assert r.end_reason == "done_failed" and r.steps == 1 and r.done_result == "找不到"
    assert len(m.seen) == 1, "没有为了本来就打不完的兜底多问模型一次"
    rec = next(x for x in _recs(r) if "action" in x)
    assert "fallback" not in rec and rec["result"] == {"ok": True}


def test_no_progress_stop_on_the_last_allowed_step_ends_clean_without_a_wasted_fallback(fake_env, tmp_path):
    r, *_ = run_with(fake_env, [["a"]] * 80, TAPS, tmp_path, asker=CountingAsker(), max_steps=6)
    assert r.end_reason == "no_progress" and r.steps == 6
    assert not [x for x in _recs(r) if x.get("by") == "program"]


class SlowAtSixthDecision(FirstTapExpectsModel):
    """在熔断本该触发兜底的第 6 次决策前睡到明显超过 timeout_s，逼近『决定要不要兜底那一刻』
    时限已经用完（评审发现 1，时限分支——和步数分支走同一条预算判断）。"""

    def decide(self, messages, image_size, tools=None, max_tokens=None):
        if len(self.seen) == 5:
            time.sleep(0.5)
        return super().decide(messages, image_size, tools, max_tokens)


def test_no_progress_stop_near_timeout_ends_clean_without_a_wasted_fallback(action_env, tmp_path):
    """沿用第一次熔断兜底那个用例的画面（两屏来回，第 6 步会判 stop）；这里给一个刚好够跑到
    第 6 步、但不够再打一次整屏解析的 timeout_s，验证兜底真的被跳过而不是白打一次再被时限吞掉。"""
    dev, per, _ = action_env([LOOP[0]] * INITIAL_SETTLE_FRAMES + LOOP, LOOP_MOVES)
    per.asker = WrongPageJudge()
    m = SlowAtSixthDecision(TAPS)
    r = run_task("t", dev, per, m, tmp_path, run_config=RunConfig(screen_parse="on_demand"),
                 store=isolated_store(tmp_path), workspace=Workspace(tmp_path), timeout_s=0.6)
    assert r.end_reason == "no_progress" and r.steps == 6
    assert not [x for x in RunLog.read_steps(r.run_dir) if x.get("by") == "program"]


# ---------- 终审修（2026-09-15）----------
# ⚠ 发现 1：zoom 帧自己不解析（perception 为空），原来 fallback_due 一律把它算成「没拿到整屏解析」——
#   always 下 zoom 之后 done(failed) 也兜底，白打一次 20–60 秒的整屏解析（spec §3.1 不许）。
#   zoom 帧按它放大的那一帧判。
ZOOM = [("zoom", {"x1": 0, "y1": 0, "x2": 300, "y2": 300})]
OBSERVE = [("observe", {})]


def test_always_does_not_fall_back_on_a_zoom_of_a_parsed_frame(fake_env, tmp_path):
    r, *_ = run_with(fake_env, [["通用"]] * 6, [ZOOM, FAILED, FAILED], tmp_path, mode="always", asker=Labeler())
    assert r.end_reason == "done_failed" and r.steps == 2
    assert _meta(r)["perception"]["fallback_by"] == {"done_failed": 0, "no_progress": 0}


def test_on_demand_zoom_after_a_full_screen_look_does_not_fall_back(fake_env, tmp_path):
    r, *_ = run_with(fake_env, [["通用"]] * 8, [OBSERVE, ZOOM, FAILED, FAILED], tmp_path, asker=Labeler())
    assert r.end_reason == "done_failed" and r.steps == 3
    assert _meta(r)["perception"]["fallback_by"] == {"done_failed": 0, "no_progress": 0}


def test_on_demand_zoom_of_an_ocr_only_frame_still_falls_back(fake_env, tmp_path):
    r, *_ = run_with(fake_env, [["通用"]] * 8, [ZOOM, FAILED, FAILED], tmp_path, asker=Labeler())
    assert r.end_reason == "done_failed" and r.steps == 3
    assert _meta(r)["perception"]["fallback_by"] == {"done_failed": 1, "no_progress": 0}


def test_a_zoom_frame_carries_its_base_frames_outcome_but_is_never_labelled(fake_env):
    from iphone_agent.perceive import policy
    dev, per, _ = fake_env([["通用"]])
    per.asker = Labeler()
    with per.task_scope(None, "on_demand"):
        base = per.observe(dev.capture())
        z = per.zoom(base, (0, 0, 100, 100))
        zz = per.zoom(z, (0, 0, 50, 50))                  # 从 zoom 帧再 zoom：沿用最初那一帧
    assert z.perception == {} and z.zoom_base_vision == "off" and zz.zoom_base_vision == "off"
    assert not policy.wants_label("on_demand", z, True) and not policy.wants_label("on_demand", zz, True)
    assert policy.fallback_due("on_demand", zz, False, True)


# ⚠ 发现 5：第一次 done(failed) 换成兜底时，那次 done 的 remember / used_memories 没落盘，要告诉模型重带。
def test_the_fallback_hint_says_the_done_memories_were_not_saved(fake_env, tmp_path):
    for asker in (Labeler(), CountingAsker()):          # 看全屏成功、失败两种说法都要带
        r, *_ = run_with(fake_env, [["通用"]] * 4, [FAILED, FAILED], tmp_path / type(asker).__name__, asker=asker)
        hint = next(x for x in _recs(r) if "action" in x)["result"]["hint"]
        assert "remember / used_memories 没有保存" in hint and "重新带上" in hint


# ⚠ 发现 6：熔断兜底之后，【上一步之后】还在说模型上一步的变化；state 模式下它会被原样拼进下一次请求。
def test_after_the_no_progress_fallback_the_state_view_drops_the_old_transition(action_env, tmp_path):
    dev, per, _ = action_env([LOOP[0]] * INITIAL_SETTLE_FRAMES + LOOP, LOOP_MOVES)
    per.asker = WrongPageJudge()
    m = FirstTapExpectsModel(TAPS)
    r = run_task("t", dev, per, m, tmp_path, run_config=RunConfig(screen_parse="on_demand", context_mode="state"),
                 store=isolated_store(tmp_path), workspace=Workspace(tmp_path))
    assert _meta(r)["perception"]["fallback_by"] == {"done_failed": 0, "no_progress": 1}
    # 系统提示词里也提到【上一步之后】，所以认的是 transition.render 那一行「整屏：有变化 / 无变化」。
    assert "整屏：" in json.dumps(m.seen[5], ensure_ascii=False), "对照：平时这一段是在的"
    after_fb = json.dumps(m.seen[6], ensure_ascii=False)
    assert "程序补看了一次全屏" in after_fb and "整屏：" not in after_fb
