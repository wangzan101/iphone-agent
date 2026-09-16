"""孪生接入主循环（spec §6）：每步留认屏结果、收尾记账、开关切换参考段来源；孪生坏了不影响任务。"""
import json

from iphone_agent.harness.runlog import RunLog
from iphone_agent.workspace import RunConfig
from tests.test_loop import run


class StubLive:
    calls: list = []

    def __init__(self, apps_dir, run_id, known_apps=frozenset()):
        StubLive.calls = []

    def observe(self, snap):
        return {"owner": "she-zhi", "state": "matched", "screen_id": "s_x", "name": "关于本机"}

    def after_action(self, action, result, after, ts):
        StubLive.calls.append(action["name"])

    def position(self):
        return "【位置】孪生说的"

    def route(self, task):
        return None

    def summary(self):
        return {"stub": True}


DONE = [("done", {"status": "success", "result": "x"})]


def test_each_action_record_carries_twin_recognition(fake_env, tmp_path):
    r, dev, m = run(fake_env, [["a", "b", "c"]] * 3, [[("tap", {"id": 1})], DONE], tmp_path)
    recs = [x for x in RunLog.read_steps(r.run_dir) if "action" in x]
    # 假环境的帧没有视觉标注：spec 2026-09-12 §4.3 下认屏状态是 unlabeled（不是旧的几何 skipped）。
    assert recs and all(x["twin"]["owner"] == "system" and x["twin"]["state"] == "unlabeled" for x in recs)


def test_run_json_has_twin_summary_and_config_flag(fake_env, tmp_path):
    r, dev, m = run(fake_env, [["a", "b", "c"]] * 2, [DONE], tmp_path)
    meta = json.loads((r.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["twin"]["record"]["runs"] == 1 and meta["twin"]["errors"] == []
    assert meta["config"]["twin_hints"] is True


def test_hints_come_from_twin_when_on_and_from_screenmap_when_off(fake_env, tmp_path, monkeypatch):
    monkeypatch.setattr("iphone_agent.twin.live.LiveTwin", StubLive)
    r, dev, m = run(fake_env, [["a", "b", "c"]] * 2, [DONE], tmp_path)
    assert "孪生说的" in json.dumps(m.seen, ensure_ascii=False)
    r2, dev2, m2 = run(fake_env, [["a", "b", "c"]] * 2, [DONE], tmp_path / "off",
                       run_config=RunConfig(twin_hints=False))
    assert "孪生说的" not in json.dumps(m2.seen, ensure_ascii=False)
    meta = json.loads((r2.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["config"]["twin_hints"] is False


def test_twin_failures_never_change_the_end_reason(fake_env, tmp_path, monkeypatch):
    class Broken(StubLive):
        def observe(self, snap):
            raise RuntimeError("boom")
    monkeypatch.setattr("iphone_agent.twin.live.LiveTwin", Broken)
    monkeypatch.setattr("iphone_agent.twin.record.record_run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    r, dev, m = run(fake_env, [["a", "b", "c"]] * 2, [DONE], tmp_path)
    assert r.end_reason == "done_success"
    meta = json.loads((r.run_dir / "run.json").read_text(encoding="utf-8"))
    assert any("boom" in e for e in meta["twin"]["errors"]) and "disk full" in meta["twin"]["record_error"]


def test_handover_tells_the_live_twin(fake_env, tmp_path, monkeypatch):
    monkeypatch.setattr("iphone_agent.twin.live.LiveTwin", StubLive)
    run(fake_env, [["a", "b", "c"]] * 4, [[("handover", {"need": "登录"})], DONE], tmp_path,
        on_handover=lambda need, reason: "")
    assert "handover" in StubLive.calls


def test_no_frame_action_then_home_icon_tap_gives_the_next_action_the_right_owner(fake_env, tmp_path, monkeypatch):
    """review finding：recall（这里用一个不存在的名字，validate_action 通过、executor 真的
    执行了它，只是没有新画面）之后紧跟着在主屏图标上 tap，不能让 tap 的归属推进被跳过——
    不然它之后每一条 record["twin"] 都挂着错的 owner="system"，【位置】【路线】也会说错话
    或不说话。用真实 LiveTwin（不 monkeypatch），跑到 done 看最后一条记录的 owner。

    2026-09-11（fbc1 用户决定）：点图标进 App 现在也要核过身份（result["identity"]["verified"]
    is True）才算数——这条测试本来就要测「核过身份的进入」，所以给它一个看图说「是」的
    is_target_app 桩，保持测试原来的目的（归属推进没被跳过），不是放宽断言。"""
    import iphone_agent.harness.judge as judge_mod
    monkeypatch.setattr(judge_mod, "is_target_app",
                        lambda image, name, asker: judge_mod.AppCheck(
                            is_app=True, confidence=1.0, seen=name, why="stub"))
    # ⚠ home 需要满足 executor.looks_like_home（≥3 个系统 App 标签），不然点图标进 App
    #   根本不会触发新的身份核对，测的就不是这条路了（2026-09-11 fbc1 用户决定）。
    home = ["设置", "微信", "相机", "标签", "页面"]
    detail = ["详情", "关于", "版本", "型号", "容量"]
    specs = [home, detail, detail, detail, detail]
    script = [[("recall", {"name": "does-not-exist"})],
              [("tap", {"id": 1, "target": "icon_above"})],
              DONE]
    r, dev, m = run(fake_env, specs, script, tmp_path)
    assert r.end_reason == "done_success"
    recs = [x for x in RunLog.read_steps(r.run_dir) if "action" in x]
    assert recs[0]["action"]["name"] == "recall" and recs[0]["result"]["error"] == "memory_not_found"
    assert recs[1]["action"]["name"] == "tap" and recs[1]["result"]["changed"] is True
    done_rec = next(x for x in recs if x["action"]["name"] == "done")
    # 有了修复：tap 在主屏图标上的归属推进（system → she-zhi）没被跳过。
    assert done_rec["twin"]["owner"] == "she-zhi"
    # 没有修复时会是这样（RED 证据，见 fix round 1 报告）：owner 一直卡在 "system"。


def test_location_hints_off_means_no_location_or_route_at_all(fake_env, tmp_path, monkeypatch):
    monkeypatch.setattr("iphone_agent.twin.live.LiveTwin", StubLive)
    r, dev, m = run(fake_env, [["a", "b", "c"]] * 2, [DONE], tmp_path,
                    run_config=RunConfig(location_hints=False))
    # System prompt always describes these sections (same in both arms), so only per-step messages are checked.
    seen = json.dumps([x for msgs in m.seen for x in msgs if x.get("role") != "system"], ensure_ascii=False)
    assert "孪生说的" not in seen and "【位置】" not in seen and "【路线】" not in seen
    meta = json.loads((r.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["config"]["location_hints"] is False
    recs = [x for x in RunLog.read_steps(r.run_dir) if "action" in x]
    assert all(x["hints"] == {"location": False, "route": False} for x in recs)


def test_hints_are_recorded_per_step(fake_env, tmp_path, monkeypatch):
    monkeypatch.setattr("iphone_agent.twin.live.LiveTwin", StubLive)
    r, dev, m = run(fake_env, [["a", "b", "c"]] * 2, [DONE], tmp_path)
    recs = [x for x in RunLog.read_steps(r.run_dir) if "action" in x]
    assert recs[0]["hints"] == {"location": True, "route": False}


def test_memory_off_injects_and_writes_nothing(fake_env, tmp_path):
    from tests.test_loop import isolated_store
    store = isolated_store(tmp_path)
    store.write("k", "d", "c", "runs/x", "done_success")
    r, dev, m = run(fake_env, [["a"]] * 3, [DONE], tmp_path, store=store, run_config=RunConfig(memory=False))
    texts = json.dumps(m.seen[0], ensure_ascii=False)
    assert "- k —" not in texts and "最近的运行" not in texts
    meta = json.loads((r.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["config"]["memory"] is False and meta["memory"]["injected"] is None


def test_identity_context_is_bound_during_the_run_and_released_after(fake_env, tmp_path):
    seen = []
    dev, per, frames = fake_env([["a", "b", "c"]] * 8)
    orig = per.observe

    def spy(frame, **kw):
        seen.append(per._identity)
        return orig(frame, **kw)
    per.observe = spy
    from iphone_agent.harness.loop import run_task
    from iphone_agent.workspace import Workspace
    from tests.test_loop import ScriptedModel, isolated_store
    run_task("t", dev, per, ScriptedModel([DONE]), tmp_path, store=isolated_store(tmp_path),
             workspace=Workspace(tmp_path))
    assert seen and all(x is not None for x in seen) and per._identity is None


def test_live_twin_gets_the_real_frame_file(fake_env, tmp_path, monkeypatch):
    got = []

    class Spy(StubLive):
        def observe(self, snap):
            got.append(snap.frame_file)
            return super().observe(snap)
    monkeypatch.setattr("iphone_agent.twin.live.LiveTwin", Spy)
    r, dev, m = run(fake_env, [["a", "b", "c"]] * 2, [DONE], tmp_path)
    assert got and all(f.startswith("frame_") and (r.run_dir / f).exists() for f in got)


def test_task_scope_is_entered_even_when_the_twin_fails_to_start(fake_env, tmp_path, monkeypatch):
    """spec 2026-09-14 §3.3：孪生初始化失败（ident_ctx=None）时，模式照样按本任务生效。"""
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("twin init")
    monkeypatch.setattr("iphone_agent.twin.live.LiveTwin", Boom)
    seen = []
    dev, per, frames = fake_env([["a", "b", "c"]] * 8)
    orig = per.observe

    def spy(frame, **kw):
        seen.append(per.mode)
        return orig(frame, **kw)
    per.observe = spy
    from iphone_agent.harness.loop import run_task
    from iphone_agent.workspace import Workspace
    from tests.test_loop import ScriptedModel, isolated_store
    r = run_task("t", dev, per, ScriptedModel([DONE]), tmp_path, store=isolated_store(tmp_path),
                 workspace=Workspace(tmp_path), run_config=RunConfig(screen_parse="on_demand"))
    assert seen and set(seen) == {"on_demand"} and per.mode == "always"
    meta = json.loads((r.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["config"]["screen_parse"] == "on_demand"
    assert any("twin init" in e for e in meta["twin"]["errors"])


def test_two_consecutive_runs_each_take_their_own_screen_parse_mode(fake_env, tmp_path):
    """controller ruling m5：同一个 Perceiver 连跑两个任务，各自的 screen_parse 模式都要生效，
    互不沾染（task_scope 每个任务都解除干净，spec 2026-09-14 §3.3）。"""
    seen = []
    dev, per, frames = fake_env([["a", "b", "c"]] * 8)
    orig = per.observe

    def spy(frame, **kw):
        seen.append(per.mode)
        return orig(frame, **kw)
    per.observe = spy
    from iphone_agent.harness.loop import run_task
    from iphone_agent.workspace import Workspace
    from tests.test_loop import ScriptedModel, isolated_store

    seen.clear()
    r1 = run_task("t1", dev, per, ScriptedModel([DONE]), tmp_path, store=isolated_store(tmp_path),
                  workspace=Workspace(tmp_path), run_config=RunConfig(screen_parse="always"))
    assert seen and set(seen) == {"always"}
    meta1 = json.loads((r1.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta1["config"]["screen_parse"] == "always"

    seen.clear()
    r2 = run_task("t2", dev, per, ScriptedModel([DONE]), tmp_path, store=isolated_store(tmp_path),
                  workspace=Workspace(tmp_path), run_config=RunConfig(screen_parse="off"))
    assert seen and set(seen) == {"off"}
    meta2 = json.loads((r2.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta2["config"]["screen_parse"] == "off"
    assert per.mode == "always"
