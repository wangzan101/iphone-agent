"""loop 收下的每个观察都先标注、再进孪生、再推给模型；最后一帧落盘（spec 2026-09-14 §5.2、§8.1、§10.1）。"""
import json

import pytest

from iphone_agent.harness.loop import run_task
from iphone_agent.harness.runlog import RunLog
from iphone_agent.twin.events import read_run_events
from iphone_agent.workspace import RunConfig, Workspace
from tests.test_loop import INITIAL_SETTLE_FRAMES, ScriptedModel, isolated_store

DONE = [("done", {"status": "success", "result": "x"})]


class Labeler:
    """整屏解析与短标注都回同一个标注（元素为空）；judge 的问题一律没问成。"""
    def __init__(self):
        self.prompts, self.last_outcome = [], None

    def ask_json(self, prompt, images, max_tokens=None):
        self.prompts.append(prompt)
        if "放在 screen 里" not in prompt:
            return None
        return {"screen": {"app": "设置", "name": "某屏", "same_as": None, "anchors": []}, "elements": []}


def run_with(fake_env, specs, script, tmp_path, *, mode="on_demand", asker=None, **kw):
    dev, per, frames = fake_env([specs[0]] * INITIAL_SETTLE_FRAMES + list(specs))
    per.asker = asker if asker is not None else Labeler()
    model = ScriptedModel(script)
    kw.setdefault("store", isolated_store(tmp_path))
    kw.setdefault("workspace", Workspace(tmp_path))
    r = run_task("t", dev, per, model, tmp_path, run_config=RunConfig(screen_parse=mode), **kw)
    return r, dev, per, model


def _meta(r):
    return json.loads((r.run_dir / "run.json").read_text(encoding="utf-8"))


def _actions(r):
    return [x for x in RunLog.read_steps(r.run_dir) if "action" in x]


def test_every_frame_the_loop_takes_in_is_labelled_before_the_twin_sees_it(fake_env, tmp_path, monkeypatch):
    seen = []

    class Spy:
        def __init__(self, *a, **k):
            pass

        def observe(self, snap):
            seen.append(("observe", snap.label_status))
            return {"owner": "system", "state": "unlabeled", "screen_id": None, "name": None}

        def after_action(self, action, result, after, ts):
            seen.append(("after", after.label_status if after else None))

        def position(self):
            return None

        def route(self, task):
            return None

        def summary(self):
            return {}
    monkeypatch.setattr("iphone_agent.twin.live.LiveTwin", Spy)
    r, dev, per, m = run_with(fake_env, [["通用"], ["关于本机"], ["关于本机"], ["关于本机"]],
                              [[("tap", {"id": 1})], DONE], tmp_path)
    assert r.end_reason == "done_success"
    assert seen and all(s == "ok" for _, s in seen), seen
    recs = _actions(r)
    assert all(x["observation"]["perception"]["label_by"] == "label" for x in recs)
    tap = recs[0]
    assert tap["after_observation"]["frame_file"] == recs[1]["observation"]["frame_file"]
    assert tap["after_observation"]["screen"] == recs[1]["observation"]["screen"]
    perception = _meta(r)["perception"]
    assert perception["label"]["adopt"]["ok"] == 2 and perception["parse_by"]["always"] == 0
    assert perception["observation_mismatch"] == 0


def test_the_frame_after_a_handover_is_labelled(fake_env, tmp_path):
    r, *_ = run_with(fake_env, [["登录"]] * 4, [[("handover", {"need": "登录"})], DONE], tmp_path,
                     on_handover=lambda need, reason: "")
    rec = next(x for x in _actions(r) if x["action"]["name"] == "handover")
    assert rec["after_observation"]["perception"]["label_by"] == "label"


def test_a_frame_swapped_in_by_ensure_connected_is_labelled(fake_env, tmp_path, monkeypatch):
    import iphone_agent.harness.loop as loop_mod
    swapped = {}

    def fake_connect(device, perceiver, obs=None):
        o = perceiver.observe(device.capture())
        swapped["id"] = o.observation_id
        return True, "", o
    monkeypatch.setattr(loop_mod, "ensure_connected", fake_connect)
    r, *_ = run_with(fake_env, [["通用"]] * 3, [DONE], tmp_path)
    first = _actions(r)[0]
    assert first["observation_id"] == swapped["id"]
    assert first["observation"]["perception"]["label_by"] == "label"


def test_a_frame_swapped_in_by_ensure_connected_is_the_frame_on_disk(fake_env, tmp_path, monkeypatch):
    """ensure_connected 换了一帧：第一条记录的 frame_file 指的是换进来的那一帧，而且它已落盘。"""
    import iphone_agent.harness.loop as loop_mod
    from iphone_agent.harness.runlog import frame_name
    swapped = {}

    def fake_connect(device, perceiver, obs=None):
        o = perceiver.observe(device.capture())
        swapped["frame_id"] = o.frame_id
        return True, "", o
    monkeypatch.setattr(loop_mod, "ensure_connected", fake_connect)
    r, *_ = run_with(fake_env, [["通用"]] * 3, [DONE], tmp_path)
    first = _actions(r)[0]
    assert first["observation"]["frame_file"] == frame_name(swapped["frame_id"])
    assert (r.run_dir / first["observation"]["frame_file"]).is_file()


@pytest.mark.parametrize("ending", ["max_steps", "stopped", "device_error", "no_progress"])
def test_the_last_frame_is_on_disk_and_replay_agrees_with_the_live_twin(fake_env, tmp_path, ending):
    specs = [["通用"], ["关于本机"], ["关于本机"], ["关于本机"]]
    script, kw = [[("tap", {"id": 1})], DONE], {}
    if ending == "max_steps":
        kw["max_steps"] = 1
    elif ending == "stopped":
        flags = iter([False, True])
        kw["should_stop"] = lambda: next(flags, True)
    elif ending == "device_error":
        script = [[("tap", {"id": 1})], RuntimeError("boom")]
    else:                   # 画面一直不变：第 6 步熔断，最后一步的 after 帧后面没有下一条记录
        specs = [["a"]] * 40
        script = [[("tap", {"x": 10 + 35 * i, "y": 10})] for i in range(10)]
    r, *_ = run_with(fake_env, specs, script, tmp_path, mode="always", **kw)
    assert r.end_reason == ending
    evs = read_run_events(r.run_dir)
    assert evs and evs[-1].name == "tap" and evs[-1].after is not None and evs[-1].after.label is not None
    twin = _meta(r)["twin"]
    assert twin["live"]["labeled"] == twin["record"]["labeled"] >= 2, "实时和重放对最后一帧的结论一致"


def test_observation_mismatch_is_recorded_when_the_loop_validates_another_object(fake_env, tmp_path):
    dev, per, _ = fake_env([["通用"]] * (INITIAL_SETTLE_FRAMES + 3))
    # 人为让「推给模型的」和「拿去校验的」不是同一个编号（spec §6.3 的内部 invariant）
    per.finalize = lambda o: setattr(o, "observation_id", o.observation_id + 1000)
    r = run_task("t", dev, per, ScriptedModel([DONE]), tmp_path, store=isolated_store(tmp_path),
                 workspace=Workspace(tmp_path))
    rec = _actions(r)[0]
    assert rec["observation_mismatch"]["sent"] + 1000 == rec["observation_mismatch"]["validated"]
    assert _meta(r)["perception"]["observation_mismatch"] == 1


def test_an_unchanged_frame_is_never_handed_back_as_the_new_observation(fake_env, tmp_path, monkeypatch):
    """画面没变时，executor / 剧本也不能把刚推给模型的那个 observation 当 new_obs 交回来：
    loop 的 adopt(after_obs) 在孪生的 try 外面，那一帧已经 finalized，ensure_label 抛 ObservationFrozen，
    整个任务按 device_error 结束（终审发现 7，2026-09-15）。这里把几类「画面不动」的动作都走一遍钉住。"""
    from iphone_agent.harness.executor import Executor
    from iphone_agent.harness.procedure import ProcedureRunner
    from tests.test_loop import _skill_store

    handed = []
    orig_ex, orig_pr = Executor.run, ProcedureRunner.run

    def ex_run(self, action, obs, *a, **k):
        res, new = orig_ex(self, action, obs, *a, **k)
        handed.append((action.name, obs, new))
        return res, new

    def pr_run(self, proc, args, obs, call_id, tool_name):
        res, new = orig_pr(self, proc, args, obs, call_id, tool_name)
        handed.append((tool_name, obs, new))
        return res, new
    monkeypatch.setattr(Executor, "run", ex_run)
    monkeypatch.setattr(ProcedureRunner, "run", pr_run)

    script = [[("route", {"scenario": "", "apps": ["settings"]})],
              [("tap", {"id": 1})], [("wait", {"seconds": 1})], [("scroll", {"direction": "down"})],
              [("observe", {})], [("zoom", {"x1": 0, "y1": 0, "x2": 300, "y2": 300})],
              [("settings__ios-version", {})], DONE]
    r, *_ = run_with(fake_env, [["设置", "通用"]] * 60, script, tmp_path, skill_store=_skill_store(tmp_path))
    assert r.end_reason == "done_success", (r.end_reason, r.error)
    names = {n for n, _, _ in handed}
    assert {"tap", "wait", "scroll", "observe", "zoom", "settings__ios-version"} <= names, names
    assert not [n for n, o, new in handed if new is o], "画面没变却把送出去的那一帧当新帧交回"
