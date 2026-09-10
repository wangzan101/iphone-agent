from iphone_agent.cli.commands import Session, dispatch
from iphone_agent.skills import model as M
from iphone_agent.skills.store import SkillStore


def seed(tmp_path):
    store = SkillStore()          # cwd 已被 conftest 换到 tmp：.iphone/skills 与 skills 都在 tmp 下
    store.write_app(M.AppProfile("settings", "设置", "设置", "read", "draft", "2026-09-08", "系统设置"))
    store.write_procedure(M.Procedure(
        name="ios-version", app="settings", description="读 iOS 版本", params={}, returns=[], risk="write",
        status="draft", expect_source="single_run", provenance=M.Provenance(runs=["r1"], verified_count=1),
        steps=[M.Step(do="tap", target="通用", expect=("关于本机",)), M.Step(do="tap", target="关于本机", expect=())]))
    store.write_scenario(M.Scenario("daily", "记账", ("settings",), "read", "proposed", "1. …"))
    return store


def test_list_shows_status_marks(tmp_path, capsys):
    seed(tmp_path)
    assert dispatch(Session(), ["skill", "list"]) == 0
    out = capsys.readouterr().out
    assert "settings" in out and "草稿" in out and "ios-version" in out and "daily" in out and "proposed" in out


def test_list_marks_a_scenario_that_shadows_the_shared_layer(tmp_path, capsys):
    """遮蔽提示以前只有 App 有。场景被遮蔽而人看不见，是这个分支最贵的静默失败。"""
    from iphone_agent import config
    seed(tmp_path)
    shared = SkillStore(personal=config.SKILLS_DIR_SHARED, shared=config.SKILLS_DIR_SHARED)
    shared.write_scenario(M.Scenario("daily", "人审过的记账", ("settings",), "read", "manual", "人写的"))
    assert dispatch(Session(), ["skill", "list"]) == 0
    out = capsys.readouterr().out
    assert "遮蔽了结构层" in out and "记账" in out


def test_show_app_and_procedure(tmp_path, capsys):
    seed(tmp_path)
    assert dispatch(Session(), ["skill", "show", "settings"]) == 0
    assert "系统设置" in capsys.readouterr().out
    assert dispatch(Session(), ["skill", "show", "settings/ios-version"]) == 0
    out = capsys.readouterr().out
    assert "关于本机" in out and "r1" in out


def test_approve_app_then_procedure_then_scenario(tmp_path):
    store = seed(tmp_path)
    assert dispatch(Session(), ["skill", "approve", "settings"]) == 0
    assert store.load().app("settings").status == "verified"
    assert dispatch(Session(), ["skill", "approve", "settings/ios-version"]) == 0
    assert store.load().procedure("settings", "ios-version").status == "verified"
    assert dispatch(Session(), ["skill", "approve", "daily"]) == 0
    assert store.load().scenarios["daily"].status == "manual"
    assert dispatch(Session(), ["skill", "approve", "nope"]) == 1


def test_mv_rm_and_sync(tmp_path, capsys):
    store = seed(tmp_path)
    assert dispatch(Session(), ["skill", "mv", "settings/ios-version", "read-version"]) == 0
    assert store.load().procedure("settings", "read-version") is not None
    assert dispatch(Session(), ["skill", "rm", "settings/read-version"]) == 0
    assert store.load().procedures_of("settings") == []
    (tmp_path / "runs").mkdir()
    capsys.readouterr()
    assert dispatch(Session(), ["skill", "sync"]) == 0
    assert "0 个 App" in capsys.readouterr().out


def test_bad_usage_returns_2(tmp_path):
    assert dispatch(Session(), ["skill"]) == 0, "没有子命令等于 list"
    assert dispatch(Session(), ["skill", "what"]) == 2


def test_replay_indents_procedure_steps(tmp_path, capsys):
    from iphone_agent.harness.runlog import RunLog
    log = RunLog(tmp_path / "runs", task="t", model="m", config_snapshot={}, prompt_hash="h")
    log.step({"step": 1, "action": {"name": "settings__x", "args": {}}, "model": {"reason": "r"},
              "result": {"ok": True}})
    log.procedure_step({"parent_call_id": "c", "procedure": "settings__x", "step": 1,
                        "action": {"name": "tap", "args": {"id": 1}}, "result": {"ok": True, "changed": True},
                        "expect_ok": True})
    log.finish("done_success", 1, "success", "x", "m", {})
    assert dispatch(Session(), ["replay", str(log.dir)]) == 0
    out = capsys.readouterr().out
    assert "    ↳ tap" in out
