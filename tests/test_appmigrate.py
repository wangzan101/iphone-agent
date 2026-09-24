"""旧拼音 id → 规范 id 的迁移（`iphone skill migrate-ids`）。

默认只说不做；动手之前先备份；目的地已经有东西时不许闷头覆盖。
"""
import json

import pytest

from iphone_agent import appmigrate
from iphone_agent.twin.screenfile import screen_id


def screen(app, first_event, *, name="首页", visits=1, runs=("r1",), to=None, anchors=None):
    d = {"schema": 2, "id": screen_id(app, first_event), "app": app,
         "created": "2026-09-01T00:00:00Z", "updated": "2026-09-01T00:00:00Z",
         "identity": {"name": name, "aliases": [], "anchors": anchors or {"通用": 1}},
         "transitions": [], "visits": visits, "status": "confirmed",
         "first_event": first_event, "applied_runs": list(runs), "human": {}, "revision": 1}
    if to is not None:
        d["transitions"] = [{"action": "tap", "target": "通用",
                             "outcome": {"sent": True, "effect": "navigated", "to": to},
                             "count": 2, "off_track": 0, "last": "r1", "evidence": ["r1:f1.png"]}]
    return d


def write_screen(root, app, d):
    p = root / app / "screens" / f"{d['id']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return p


def write_app_meta(root, app, display="设置", aliases=()):
    p = root / app / "screens" / "app.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": 2, "app": app, "display": display,
                             "aliases": list(aliases), "revision": 1}, ensure_ascii=False), encoding="utf-8")


def write_skill(skills, name, apps):
    d = skills / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f'---\nname: {name}\ndescription: 一条技能\napps: {json.dumps(apps, ensure_ascii=False)}\n'
        f'risk: read\nstatus: verified\n---\n正文\n', encoding="utf-8")
    return d


@pytest.fixture
def tree(tmp_path):
    apps, skills = tmp_path / "knowledge" / "apps", tmp_path / "skills"
    a = screen("she-zhi", "r1:f1.png")
    b = screen("she-zhi", "r1:f2.png", name="通用")
    a["transitions"] = screen("she-zhi", "r1:f1.png", to=b["id"])["transitions"]
    write_screen(apps, "she-zhi", a)
    write_screen(apps, "she-zhi", b)
    write_app_meta(apps, "she-zhi")
    write_screen(apps, "ji-zhang-ben", screen("ji-zhang-ben", "r2:f1.png", name="账单"))
    write_app_meta(apps, "ji-zhang-ben", display="记账本")
    write_skill(skills, "kan-ban-ben", ["she-zhi"])
    return apps, skills, a, b


# ---- 只说不做 ----

def test_dry_run_writes_nothing(tree):
    apps, skills, a, b = tree
    before = {p: p.read_bytes() for p in sorted(apps.rglob("*")) if p.is_file()}
    before |= {p: p.read_bytes() for p in sorted(skills.rglob("*")) if p.is_file()}
    plan = appmigrate.plan(apps, skills)
    text = appmigrate.format_plan(plan)
    assert "she-zhi" in text and "settings" in text
    assert {p: p.read_bytes() for p in sorted(apps.rglob("*")) if p.is_file()} | {
        p: p.read_bytes() for p in sorted(skills.rglob("*")) if p.is_file()} == before
    assert not list(apps.parent.parent.glob("backup-*"))


def test_plan_leaves_third_party_ids_alone(tree):
    apps, skills, _, _ = tree
    plan = appmigrate.plan(apps, skills)
    assert [(m.old_id, m.new_id) for m in plan.moves] == [("she-zhi", "settings")]


# ---- 真做 ----

def test_apply_renames_and_rewrites_ids_transitions_and_skills(tree):
    apps, skills, a, b = tree
    plan = appmigrate.plan(apps, skills)
    report = appmigrate.apply(plan)
    assert report.backup.is_dir() and (report.backup / "apps" / "she-zhi").is_dir()
    assert not (apps / "she-zhi").exists() and (apps / "settings" / "screens").is_dir()

    new_b = screen_id("settings", "r1:f2.png")
    files = {p.name for p in (apps / "settings" / "screens").glob("*.json")}
    assert files == {"app.json", f"{screen_id('settings', 'r1:f1.png')}.json", f"{new_b}.json"}
    d = json.loads((apps / "settings" / "screens" / f"{screen_id('settings', 'r1:f1.png')}.json")
                   .read_text(encoding="utf-8"))
    assert d["app"] == "settings" and d["id"] == screen_id("settings", "r1:f1.png")
    # 转移的目的地跟着改名走，否则整张图当场断开
    assert d["transitions"][0]["outcome"]["to"] == new_b
    meta = json.loads((apps / "settings" / "screens" / "app.json").read_text(encoding="utf-8"))
    assert meta["app"] == "settings" and "she-zhi" in meta["aliases"]
    # 技能的 apps[0] 就是剧本工具名 app__proc 的前半截：改了它，工具名自动跟着变
    assert json.loads((skills / "kan-ban-ben" / "SKILL.md").read_text(encoding="utf-8")
                      .split("apps: ")[1].split("\n")[0]) == ["settings"]


def test_a_destination_with_content_refuses_without_the_merge_flag(tree):
    apps, skills, a, b = tree
    write_screen(apps, "settings", screen("settings", "r9:f1.png", name="别人"))
    plan = appmigrate.plan(apps, skills)
    assert plan.needs_merge == ["settings"]
    with pytest.raises(appmigrate.MigrateRefused) as e:
        appmigrate.apply(plan)
    assert "settings" in str(e.value) and "--merge" in str(e.value)


def test_merging_sums_visits_and_keeps_both_sides_transitions(tree):
    apps, skills, a, b = tree
    # 目的地已有同一屏（同 first_event → 同 id），另有一条自己的转移和自己的 run
    dst = screen("settings", "r1:f1.png", visits=4, runs=("r7",), to="s_elsewhere")
    dst["identity"]["anchors"] = {"通用": 5, "隐私": 2}
    dst["identity"]["aliases"] = ["设置首页"]
    write_screen(apps, "settings", dst)
    plan = appmigrate.plan(apps, skills)
    appmigrate.apply(plan, merge=True)

    d = json.loads((apps / "settings" / "screens" / f"{screen_id('settings', 'r1:f1.png')}.json")
                   .read_text(encoding="utf-8"))
    assert d["visits"] == 5                                   # 1 + 4，不是覆盖
    assert sorted(d["applied_runs"]) == ["r1", "r7"]
    assert d["identity"]["anchors"] == {"通用": 6, "隐私": 2}
    assert d["identity"]["aliases"] == ["设置首页"]
    tos = {t["outcome"]["to"] for t in d["transitions"]}
    assert tos == {"s_elsewhere", screen_id("settings", "r1:f2.png")}


def test_merging_the_same_transition_sums_its_counts(tree):
    apps, skills, a, b = tree
    dst = screen("settings", "r1:f1.png", visits=2, to=screen_id("settings", "r1:f2.png"))
    write_screen(apps, "settings", dst)
    plan = appmigrate.plan(apps, skills)
    appmigrate.apply(plan, merge=True)
    d = json.loads((apps / "settings" / "screens" / f"{screen_id('settings', 'r1:f1.png')}.json")
                   .read_text(encoding="utf-8"))
    assert len(d["transitions"]) == 1 and d["transitions"][0]["count"] == 4


def test_nothing_to_do_is_said_out_loud(tmp_path):
    apps, skills = tmp_path / "knowledge" / "apps", tmp_path / "skills"
    apps.mkdir(parents=True)
    skills.mkdir(parents=True)
    plan = appmigrate.plan(apps, skills)
    assert not plan.moves and "没有" in appmigrate.format_plan(plan)


def test_cli_is_dry_run_by_default(tmp_path, monkeypatch, capsys):
    from iphone_agent.cli.commands import Session, dispatch
    apps = tmp_path / ".iphone" / "knowledge" / "apps"
    write_screen(apps, "tian-qi", screen("tian-qi", "r1:f1.png", name="今天"))
    write_app_meta(apps, "tian-qi", display="天气")
    monkeypatch.chdir(tmp_path)
    assert dispatch(Session(), ["skill", "migrate-ids"]) == 0
    out = capsys.readouterr().out
    assert "tian-qi" in out and "weather" in out and "--apply" in out
    assert (apps / "tian-qi").is_dir() and not (apps / "weather").exists()
