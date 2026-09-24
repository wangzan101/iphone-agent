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


def test_list_says_in_one_line_whether_procedure_tools_are_on(tmp_path, capsys, monkeypatch):
    """2026-09-17：剧本工具默认关。list 里一堆 ✓ 却永远没人调，人会以为坏了——
    说清现在给不给模型、怎么打开，别让人猜。"""
    seed(tmp_path)
    monkeypatch.delenv("IPHONE_SKILL_TOOLS", raising=False)
    assert dispatch(Session(), ["skill", "list"]) == 0
    out = capsys.readouterr().out
    assert "剧本工具：关" in out and "IPHONE_SKILL_TOOLS=on" in out
    monkeypatch.setenv("IPHONE_SKILL_TOOLS", "on")
    assert dispatch(Session(), ["skill", "list"]) == 0
    assert "剧本工具：开" in capsys.readouterr().out


def test_list_marks_a_content_bearing_draft_app_as_visible_but_risk_unconfirmed(tmp_path, capsys):
    """2026-09-17：seed() 的 settings 是 draft 但写了正文——对模型已经可见，risk 还没人确认过，
    这两层都要在 list 里说出来，不能只报一个「草稿」把「已经能读」这件事盖住。"""
    seed(tmp_path)
    assert dispatch(Session(), ["skill", "list"]) == 0
    out = capsys.readouterr().out
    assert "settings" in out and "对模型可见" in out and "risk 未确认" in out


def test_list_marks_an_empty_stub_app_as_not_visible(tmp_path, capsys):
    """extract.ensure_app 切出的空壳：没人写过一个字，list 要说清「不可见」，别跟有内容的草稿混着标。"""
    store = SkillStore()
    store.write_app(M.AppProfile("wechat", "微信", "微信", "write", "draft", "2026-09-08", body=""))
    assert dispatch(Session(), ["skill", "list"]) == 0
    out = capsys.readouterr().out
    assert "wechat" in out and "空壳" in out and "不可见" in out


def test_list_marks_a_scenario_that_shadows_the_shared_layer(tmp_path, capsys):
    """遮蔽提示以前只有 App 有。场景被遮蔽而人看不见，是这个分支最贵的静默失败。"""
    from iphone_agent import config
    seed(tmp_path)
    shared = SkillStore(personal=config.SKILLS_DIR_SHARED, shared=config.SKILLS_DIR_SHARED)
    shared.write_scenario(M.Scenario("daily", "人审过的记账", ("settings",), "read", "manual", "人写的"))
    assert dispatch(Session(), ["skill", "list"]) == 0
    out = capsys.readouterr().out
    assert "遮蔽了结构层" in out and "记账" in out


def test_list_marker_tells_the_truth_in_both_directions(tmp_path, capsys):
    """真机上两条剧本标着「可批准」，批准了也永远进不了工具列表（risk=write）——
    标记只看走通次数，不看批准之后到底能不能用。标记必须说实话。"""
    store = seed(tmp_path)
    p = store.read_procedure("settings", "ios-version")
    p.provenance.verified_count = 2
    store.write_procedure(p, overwrite=True)
    dispatch(Session(), ["skill", "list"])
    out = capsys.readouterr().out
    assert "可批准" not in out, "写类剧本批准了也不能自动调，不该标可批准"
    assert "write" in out and "手动" in out, "得说清卡在哪儿：它是写类的"

    store.write_procedure(M.Procedure(
        name="read-version", app="settings", description="读版本", params={}, returns=[], risk="read",
        status="draft", expect_source="single_run", provenance=M.Provenance(runs=["r1", "r2"], verified_count=2),
        steps=[M.Step(do="open_app", target="设置"),
               M.Step(do="tap", target="通用", effect="navigated")]))
    dispatch(Session(), ["skill", "list"])
    assert "可批准" in capsys.readouterr().out


def test_approve_says_out_loud_that_a_write_procedure_stays_manual_only(tmp_path, capsys):
    seed(tmp_path)
    assert dispatch(Session(), ["skill", "approve", "settings/ios-version"]) == 0
    out = capsys.readouterr().out
    assert "verified" in out and "手动" in out and "write" in out


def test_approve_recomputes_risk_from_the_steps(tmp_path):
    """批准只改 status，risk 仍按这条剧本自己的步骤算 —— 不能靠批准把写类洗成 read。"""
    store = seed(tmp_path)
    p = store.read_procedure("settings", "ios-version")
    p.risk = "read"                                  # 盘上写着 read，但步骤里的 tap 没有导航记录
    store.write_procedure(p, overwrite=True)
    dispatch(Session(), ["skill", "approve", "settings/ios-version"])
    assert store.load().procedure("settings", "ios-version").risk == "write"


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
