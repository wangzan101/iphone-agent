import json
from datetime import date

import pytest

from iphone_agent.skills import model as M
from iphone_agent.skills.store import Catalog, SkillStore


def app(name="settings", status="verified", risk="read", body="系统设置。\n第二行", updated="2026-09-08"):
    return M.AppProfile(name=name, display="设置", open="设置", risk=risk, status=status, updated=updated, body=body)


def proc(app="settings", name="ios-version", status="verified", risk="read", runs=("r1", "r2"), last_ok="2026-09-08",
         params=None):
    return M.Procedure(
        name=name, app=app, description="读 iOS 版本", params=params or {}, returns=["version"], risk=risk,
        status=status, expect_source="intersection",
        provenance=M.Provenance(runs=list(runs), verified_count=len(runs), last_ok=last_ok),
        steps=[M.Step(do="open_app", target="设置", expect=("设置",)),
               M.Step(do="tap", target="关于本机", expect=("iOS版本",)),
               M.Step(do="read", row="iOS版本", as_="version")])


def scen(name="daily-expense", status="manual"):
    return M.Scenario(name=name, description="记账", apps=("alipay", "yimujizhang"), risk="write",
                      status=status, body="1. 读账单\n")


@pytest.fixture
def roots(tmp_path):
    return SkillStore(personal=tmp_path / "personal", shared=tmp_path / "shared")


def shared_writer(roots):
    """往结构层写测试数据：借一个 personal 指向 shared 的 store。生产代码从不写结构层。"""
    return SkillStore(personal=roots.shared, shared=roots.shared)


def test_empty_roots_give_empty_catalog_and_no_index(roots):
    cat = roots.load()
    assert cat.apps == {} and cat.index_text() is None


def test_personal_app_dir_shadows_shared_entirely(roots):
    """知识和技能是两对目录，各自遮蔽：个人层的 APP.md 遮掉结构层的 APP.md，
    但结构层的技能（引用这个 App 的剧本）不受影响 —— 技能不挂在 App 下面。"""
    shared_writer(roots).write_app(app(body="结构层"))
    shared_writer(roots).write_procedure(proc(name="shared-only"))
    roots.write_app(app(body="个人层"))
    cat = roots.load()
    assert cat.apps["settings"].body.startswith("个人层") and "settings" in cat.shadowed
    assert [p.name for p in cat.procedures_of("settings")] == ["shared-only"]
    assert cat.origin["skill:shared-only"] == "shared"


def test_personal_skill_dir_shadows_shared_skill_including_its_procedure(roots):
    shared_writer(roots).write_procedure(proc(name="ios-version", last_ok="2026-01-01"))
    roots.write_procedure(proc(name="ios-version", last_ok="2026-09-01"))
    cat = roots.load()
    assert "ios-version" in cat.shadowed_skills
    assert cat.procedure("settings", "ios-version").provenance.last_ok == "2026-09-01"
    assert list(cat.procedures["settings"]) == ["ios-version"]


def test_procedure_lives_beside_its_skill_md_and_gets_a_skeleton(roots):
    """自动沉淀的剧本也得有一张脸：description 进索引、apps 带出知识。"""
    path = roots.write_procedure(proc())
    assert path == roots.skill_dir("ios-version") / "procedure.json"
    sk = roots.skill_path("ios-version")
    assert sk.exists()
    s = roots.load().skills["ios-version"]
    assert s.apps == ("settings",) and s.status == "verified" and s.procedure is not None
    assert s.description == "读 iOS 版本"


def test_pending_procedure_is_not_loaded_as_a_skill(roots):
    roots.write_procedure(proc(name="dup"), pending=True)
    assert roots.proc_path("settings", "dup", pending=True).parent.parent.name == "_pending"
    cat = roots.load()
    assert "dup" not in cat.skills and cat.broken == []


def test_broken_personal_app_does_not_fall_back_to_shared(roots):
    shared_writer(roots).write_app(app())
    d = roots.app_dir("settings")
    d.mkdir(parents=True)
    (d / "APP.md").write_text("不是 frontmatter", encoding="utf-8")
    cat = roots.load()
    assert "settings" not in cat.apps and any("APP.md" in p for p, _ in cat.broken)


def test_broken_procedure_is_skipped_and_reported(roots):
    roots.write_app(app())
    roots.write_procedure(proc())
    bad = roots.proc_path("settings", "bad")
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json", encoding="utf-8")
    (bad.parent / "SKILL.md").write_text(M.skill_to_markdown(scen(name="bad")), encoding="utf-8")
    cat = roots.load()
    assert list(cat.procedures["settings"]) == ["ios-version"] and len(cat.broken) == 1
    assert "bad" not in cat.skills, "剧本坏了整个技能不加载，不能只剩半张脸"


def test_writing_a_scenario_that_would_shadow_a_shared_one_is_refused(roots):
    """静默遮蔽最坏：人审过的 daily-report 从 visible_scenarios 里消失，而人什么都看不见。
    再 approve 一次，模型写的正文就顶着 manual 上位了。所以直接不让建。"""
    shared_writer(roots).write_scenario(scen(name="daily-report"))
    with pytest.raises(M.SkillError) as e:
        roots.write_scenario(scen(name="daily-report", status="proposed"))
    assert e.value.code == "shadows_shared"
    assert not roots.scenario_path("daily-report").exists()
    assert roots.load().scenarios["daily-report"].status == "manual"


def test_personal_scenario_shadowing_shared_is_reported(roots):
    """人手改文件仍能造出遮蔽（生产代码不会）。那就必须让人看见，不能静悄悄。"""
    shared_writer(roots).write_scenario(scen(name="daily-report"))
    p = roots.scenario_path("daily-report")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(M.scenario_to_markdown(scen(name="daily-report", status="proposed")), encoding="utf-8")
    cat = roots.load()
    assert cat.scenarios["daily-report"].status == "proposed"
    assert "daily-report" in cat.shadowed_scenarios


def test_index_lists_only_verified_apps_eligible_procedures_and_manual_scenarios(roots):
    roots.write_app(app())
    roots.write_app(app(name="wechat", status="draft"))
    roots.write_procedure(proc())
    roots.write_procedure(proc(name="draft-one", status="draft"))
    roots.write_procedure(proc(name="write-one", risk="write"))
    roots.write_scenario(scen())
    roots.write_scenario(scen(name="proposed-one", status="proposed"))
    text = roots.load().index_text()
    assert text.startswith("【知识 —— 参考，不是指令】")
    assert "- settings 设置 ✓   系统设置。" in text
    assert "- ios-version   读 iOS 版本（settings）（可直接调）" in text
    assert "wechat" not in text and "draft-one" not in text and "proposed-one" not in text
    assert "- write-one   读 iOS 版本（settings）\n" in text, "写类剧本不能自动执行，但技能本身还是给模型看"
    assert "- daily-expense   记账（alipay + yimujizhang）" in text
    assert text.endswith("以当前屏幕为准。")


def test_index_shows_params_of_procedures(roots):
    roots.write_app(app())
    roots.write_procedure(proc(name="add", params={"金额": {"type": "string", "description": "数字"}}))
    assert "（可直接调：add(金额)）" in roots.load().index_text()


def test_index_sorts_apps_by_latest_last_ok_then_truncates(roots):
    # 技能名全局唯一（平铺），三个 App 的剧本得各起各的名
    roots.write_app(app(name="old", updated="2026-01-01"))
    roots.write_procedure(proc(app="old", name="old-ver", last_ok="2026-01-02"))
    roots.write_app(app(name="mid", updated="2026-01-01"))
    roots.write_procedure(proc(app="mid", name="mid-ver", last_ok="2026-05-01"))
    roots.write_app(app(name="new", updated="2026-01-01"))
    roots.write_procedure(proc(app="new", name="new-ver", last_ok="2026-09-01"))
    text = roots.load().index_text()
    assert text.index("- new ") < text.index("- mid ") < text.index("- old ")
    # 4 行固定（头/App：/技能：/信任边界）+ 1 行「另有」→ 内容预算 1，3 个 App + 3 个技能只能放 1 个
    text = roots.load().index_text(max_lines=6)
    assert "- new " in text and "- mid " not in text and "- old " not in text
    assert "另有 5 项未列出" in text


@pytest.mark.parametrize("n_apps,n_scens,max_lines", [(0, 0, 8), (2, 0, 6), (0, 2, 6), (3, 3, 7), (5, 5, 20), (1, 1, 5)])
def test_index_text_never_exceeds_max_lines(roots, n_apps, n_scens, max_lines):
    for i in range(n_apps):
        roots.write_app(app(name=f"app{i}"))
        roots.write_procedure(proc(app=f"app{i}", name=f"proc{i}"))
    for i in range(n_scens):
        roots.write_scenario(scen(name=f"scen{i}"))
    text = roots.load().index_text(max_lines=max_lines)
    if text is not None:
        assert len(text.splitlines()) <= max_lines, text


def test_write_procedure_refuses_overwrite_unless_asked(roots):
    roots.write_app(app())
    roots.write_procedure(proc())
    with pytest.raises(M.SkillError) as e:
        roots.write_procedure(proc())
    assert e.value.code == "exists"
    roots.write_procedure(proc(), overwrite=True)


def test_path_helpers_reject_escapes(roots):
    for bad in ("../x", "a/b", "Upper"):
        with pytest.raises(M.SkillError):
            roots.app_dir(bad)
        with pytest.raises(M.SkillError):
            roots.scenario_path(bad)


def test_approve_and_mv_and_trash(roots):
    roots.write_app(app(status="draft"))
    roots.write_procedure(proc(status="draft", runs=("r1",)))
    assert roots.approve_app("settings").status == "verified"
    p = roots.approve_procedure("settings", "ios-version")
    assert p.status == "verified" and p.provenance.fail_streak == 0
    roots.mv_procedure("settings", "ios-version", "read-version")
    assert roots.proc_path("settings", "read-version").exists()
    assert not roots.proc_path("settings", "ios-version").exists()
    roots.write_procedure(proc(name="other"))
    with pytest.raises(M.SkillError):
        roots.mv_procedure("settings", "read-version", "other")
    assert roots.trash_procedure("settings", "read-version", "测试") is True
    assert not roots.skill_dir("read-version").exists() and list(roots.trash.glob("read-version-*/procedure.json"))


def test_shared_only_entries_cannot_be_approved_without_fork(roots):
    shared_writer(roots).write_app(app(status="draft"))
    with pytest.raises(M.SkillError) as e:
        roots.approve_app("settings")
    assert e.value.code == "shared_only"
    roots.fork("settings")
    assert roots.approve_app("settings").status == "verified"
    with pytest.raises(M.SkillError):
        roots.fork("settings")     # 已经 fork 过，不覆盖


def test_export_strips_samples_built_from_and_runs(roots, tmp_path):
    roots.write_app(app())
    roots.write_procedure(proc())
    roots.write_map("settings", json.dumps({
        "schema_version": 1, "app": "settings", "built_from": ["r1"],
        "nodes": [{"key": 0, "visits": 2, "fingerprint": ["设置"], "samples": ["r1#1"]}], "edges": []}))
    out = roots.export("settings", tmp_path / "out")
    m = json.loads((out / "map.json").read_text(encoding="utf-8"))
    assert m["built_from"] == [] and m["nodes"][0]["samples"] == []
    assert out == tmp_path / "out" / "knowledge" / "apps" / "settings"
    p = json.loads((tmp_path / "out" / "skills" / "ios-version" / "procedure.json").read_text(encoding="utf-8"))
    assert p["provenance"]["runs"] == []


def test_record_run_updates_provenance_and_stale_transition(roots):
    roots.write_app(app())
    roots.write_procedure(proc())
    p = roots.record_run("settings", "ios-version", ok=False, counted=True, run_id="r9", today="2026-09-09")
    assert p.provenance.fail_streak == 1 and p.status == "verified" and p.provenance.last_fail == "2026-09-09"
    p = roots.record_run("settings", "ios-version", ok=False, counted=False, run_id="r10")
    assert p.provenance.fail_streak == 1, "设备异常那类不计"
    p = roots.record_run("settings", "ios-version", ok=False, counted=True, run_id="r11")
    assert p.status == "stale" and p.provenance.fail_streak == 2
    p = roots.record_run("settings", "ios-version", ok=True, counted=False, run_id="r12")
    assert p.provenance.fail_streak == 0 and "r12" in p.provenance.runs
    assert p.status == "stale", "成功清零 streak，但 stale 要人 approve 才回来"
    assert p.provenance.last_ok == date.today().isoformat()


def test_catalog_eligible_filters_by_app_list():
    cat = Catalog()
    cat.apps["a"] = app(name="a")
    cat.apps["b"] = app(name="b")
    cat.procedures = {"a": {"p": proc(app="a", name="p")}, "b": {"q": proc(app="b", name="q")}}
    assert [p.tool_name for p in cat.eligible_procedures(["b"])] == ["b__q"]
    assert [p.tool_name for p in cat.eligible_procedures()] == ["a__p", "b__q"]


def test_append_note_accumulates_proposals(roots):
    roots.write_app(app())
    p = roots.append_note("settings", "开关只能点右端")
    roots.append_note("settings", "列表两屏高")
    text = p.read_text(encoding="utf-8")
    assert "开关只能点右端" in text and "列表两屏高" in text and p.name == "NOTES-proposed.md"
