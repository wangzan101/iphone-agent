"""按运行阶段计时（spec 2026-09-14 §8.2）：每个调过模型的轮恰好一份 timing，等人的时间扣掉。"""
import json
import time

import pytest
from PIL import Image

from iphone_agent import config, timing
from iphone_agent.harness.runlog import RunLog
from iphone_agent.model.reply import ModelError
from tests.test_loop import run

DONE = [("done", {"status": "success", "result": "x"})]
TAP = [("tap", {"x": 10, "y": 10})]
HAND = [("handover", {"need": "登录"})]


# ---------- 计时器本身 ----------

def test_phases_accumulate_and_same_name_nesting_is_not_double_counted():
    t = timing.open_step()
    try:
        with timing.phase("settle"):
            with timing.phase("capture"):
                time.sleep(0.01)
            with timing.phase("settle"):
                time.sleep(0.01)
        with timing.phase("capture"):
            pass
    finally:
        timing.close_step()
    assert t.phases["settle"]["n"] == 1 and t.phases["capture"]["n"] == 2
    assert t.phases["settle"]["ms"] >= 15
    assert timing.current() is None
    with timing.phase("ocr"):          # 没有计时器：什么都不做，也不报错
        pass


def test_result_shape_and_waits_are_subtracted():
    t = timing.open_step()
    t.wait("pause", 0.05)
    r = t.result()
    timing.close_step()
    assert set(r) == {"step_ms", "model_ms", "exec_ms", "wait_ms", "phases", "parse_by"}
    assert r["wait_ms"] == {"pause": 50, "confirm": 0, "handover": 0}
    assert set(r["phases"]) == set(timing.PHASES) and r["step_ms"] >= 0


def test_judge_calls_are_timed():
    from iphone_agent.harness import judge

    class A:
        def ask_json(self, prompt, images, max_tokens=None):
            return None
    img = Image.new("RGB", (4, 4))
    t = timing.open_step()
    try:
        judge.did_action_work(img, img, "tap", None, A())
    finally:
        timing.close_step()
    assert t.phases["judge"]["n"] == 1


# ---------- 每种出口恰好一份 ----------

CASES = {
    "validation": ([["a"]] * 3, [[("tap", {"id": 99})], DONE], {}),
    "repeated": ([["a"]] * 6, [TAP, TAP, DONE], {}),
    "safety": ([["发送"]] * 4, [[("tap", {"id": 1})], DONE], {}),
    "recall_budget": ([["a"]] * 3, [[("recall", {"name": "x"})]] * (config.RECALL_PER_RUN + 1) + [DONE], {}),
    "empty_reply": ([["a"]] * 3, [[], DONE], {}),
    "multiple_calls": ([["a"]] * 3, [TAP + TAP, DONE], {}),
    "model_error": ([["a"]], [ModelError("x")], {}),
    "done": ([["a"]] * 3, [DONE], {}),
    "handover_none": ([["a"]] * 3, [HAND], {}),
    "handover_resumed": ([["a"]] * 4, [HAND, DONE], {"on_handover": lambda need, reason: ""}),
    "recall": ([["a"]] * 3, [[("recall", {"name": "nope"})], DONE], {}),
    "use_skill": ([["a"]] * 3, [[("use_skill", {"kind": "app", "name": "nope"})], DONE], {}),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_exit_carries_exactly_one_timing(fake_env, tmp_path, name):
    specs, script, kw = CASES[name]
    r, dev, m = run(fake_env, specs, script, tmp_path, **kw)
    recs = RunLog.read_steps(r.run_dir)
    assert sum("timing" in x for x in recs) == len(m.seen), [x.get("rejected") or x.get("action") for x in recs]
    assert timing.current() is None, "计时器不能漏到任务外面"


def test_an_execution_exception_still_carries_its_timing(fake_env, tmp_path, monkeypatch):
    from iphone_agent.harness.executor import Executor
    monkeypatch.setattr(Executor, "run", lambda self, a, o: (_ for _ in ()).throw(RuntimeError("boom")))
    r, dev, m = run(fake_env, [["a"]] * 3, [TAP], tmp_path)
    assert r.end_reason == "device_error"
    recs = RunLog.read_steps(r.run_dir)
    assert [x.get("error_phase") for x in recs if "timing" in x] == ["loop"]


def test_the_timer_does_not_leak_into_the_next_iteration(fake_env, tmp_path):
    r, dev, m = run(fake_env, [["a"], ["b"], ["c"], ["c"], ["c"]],
                    [TAP, [("tap", {"x": 200, "y": 400})], DONE], tmp_path)
    taps = [x for x in RunLog.read_steps(r.run_dir) if (x.get("action") or {}).get("name") == "tap"]
    assert [t["timing"]["phases"]["ocr"]["n"] for t in taps] == [1, 1], "每轮只数自己那一次观察"
    assert all(t["timing"]["exec_ms"] == t["exec_ms"] for t in taps)


def test_waiting_for_a_human_is_not_step_time(fake_env, tmp_path):
    waits = iter([0.3])

    def pause():
        time.sleep(next(waits, 0))
        return None
    r, dev, m = run(fake_env, [["a"]] * 3, [DONE], tmp_path, wait_if_paused=pause)
    t = next(x for x in RunLog.read_steps(r.run_dir) if "timing" in x)["timing"]
    assert t["wait_ms"]["pause"] >= 250 and t["step_ms"] < t["wait_ms"]["pause"]


def test_startup_is_timed_separately(fake_env, tmp_path):
    r, dev, m = run(fake_env, [["a"]] * 3, [DONE], tmp_path)
    st = json.loads((r.run_dir / "run.json").read_text(encoding="utf-8"))["startup_timing"]
    assert st["phases"]["ocr"]["n"] >= 1 and st["phases"]["settle"]["n"] == 1
