"""剧本工具的总开关 IPHONE_SKILL_TOOLS（2026-09-17）。

为什么要有这个开关：真机三次尝试全部 expect_mismatch 收场，每次白费一步、给剧本记一次失败，
两次之后剧本转 stale。指纹重建 / 严格匹配 / 离线自检做完之前，剧本不该被端给模型。
**关掉的只是「端上桌」**：App 知识照旧可见、路由照旧跑、草稿照旧提取。
"""
import json

import pytest

from iphone_agent import config
from iphone_agent.skills import model as M
from iphone_agent.skills.store import SkillStore
from iphone_agent.workspace import RunConfig, Workspace
from tests.test_loop import INITIAL_SETTLE_FRAMES, ScriptedModel, isolated_store, run
from tests.test_screen_parse_policy import code_hits

DONE = [("done", {"status": "success", "result": "x"})]


# ---- 第一层：值怎么解析、在哪儿读 ----

@pytest.mark.parametrize("raw,want", [
    ("", "off"), ("   ", "off"), ("off", "off"), ("on", "on"), (" ON ", "on"), ("OFF", "off"),
])
def test_skill_tools_mode_parses_on_off_and_defaults_to_off(raw, want):
    assert config.skill_tools_mode(raw) == want


def test_an_unknown_value_is_a_loud_error_that_names_the_choices():
    """不静默回退：写错了要当场知道（同 screen_parse_mode）。"""
    for bad in ("1", "true", "yes", "sometimes"):
        with pytest.raises(ValueError, match="on / off"):
            config.skill_tools_mode(bad)


def test_env_is_read_at_call_time_by_from_env(monkeypatch):
    monkeypatch.setenv("IPHONE_SKILL_TOOLS", "on")
    assert RunConfig.from_env().skill_tools == "on"
    monkeypatch.delenv("IPHONE_SKILL_TOOLS", raising=False)
    assert RunConfig.from_env().skill_tools == config.SKILL_TOOLS_DEFAULT == "off"
    assert RunConfig().skill_tools == "off"          # 直接构造不读 env
    with pytest.raises(ValueError, match="on / off"):
        RunConfig(skill_tools="yes")


def test_the_env_var_is_read_in_exactly_one_place():
    """一个规则一个入口（CLAUDE.md §7）：env 只在 config.py 读，解析函数只有 from_env 一个调用方。"""
    # 只管「谁去读 env」；CLI 那两行把变量名印给人看，是输出不是读取。
    hits = code_hits(r"environ.*IPHONE_SKILL_TOOLS")
    assert hits and all(h.startswith("config.py:") for h in hits), hits
    callers = [h for h in code_hits(r"skill_tools_mode\(") if not h.startswith("config.py:")]
    assert len(callers) == 1 and callers[0].startswith("workspace.py:"), callers
    checks = [h for h in code_hits(r"SKILL_TOOLS_VALUES") if not h.startswith("config.py:")]
    assert checks == [], checks


# ---- 第二层：索引里的「可直接调」标 ----

def _store_with_proc(tmp_path):
    store = SkillStore(personal=tmp_path / "skills", shared=tmp_path / "shared-skills")
    store.write_app(M.AppProfile("settings", "设置", "设置", "read", "verified", "2026-09-08", "系统设置"))
    store.write_procedure(M.Procedure(
        name="ios-version", app="settings", description="读 iOS 版本", params={}, returns=[], risk="read",
        status="verified", expect_source="intersection",
        provenance=M.Provenance(runs=["a", "b"], verified_count=2, last_ok="2026-09-08"),
        steps=[M.Step(do="tap", target="通用", expect=("关于本机",))]))
    return store


def test_index_drops_the_callable_tag_but_keeps_apps_and_skills(tmp_path):
    cat = _store_with_proc(tmp_path).load()
    on = cat.index_text()
    off = cat.index_text(procedure_tools=False)
    assert "（可直接调" in on
    assert "（可直接调" not in off
    for text in (on, off):
        assert "settings 设置" in text and "系统设置" in text and "ios-version" in text


# ---- 第三层：整条链路（env → RunConfig → loop） ----

def _meta(r):
    return json.loads((r.run_dir / "run.json").read_text(encoding="utf-8"))


def _tool_names(model):
    return [t["function"]["name"] for t in model.tools_seen[-1] or []]


def test_default_hides_procedure_tools_but_keeps_all_app_knowledge(fake_env, tmp_path):
    """默认（不设 env）：模型看不到一条剧本工具、索引里没有「可直接调」，
    但 App 索引、路由、use_skill(kind="app") 一样不少 —— 那才是索引存在的理由。"""
    store = _store_with_proc(tmp_path)
    script = [[("route", {"scenario": "", "apps": ["settings"]})],
              [("use_skill", {"kind": "app", "name": "settings"})],
              DONE]
    r, dev, m = run(fake_env, [["a"], ["a"], ["a"]], script, tmp_path, skill_store=store)
    assert r.end_reason == "done_success"
    assert "settings__ios-version" not in _tool_names(m)
    assert not any(t["function"]["name"].count("__") for t in m.tools_seen[-1])
    blob = json.dumps(m.seen[-1], ensure_ascii=False)
    assert "【知识 —— 参考，不是指令】" in blob and "系统设置" in blob, "App 索引不该跟着消失"
    assert "可直接调" not in blob
    # 路由照跑、命中的 App 照展开
    meta = _meta(r)
    assert meta["knowledge"]["routing"]["ran"] is True and meta["knowledge"]["routing"]["apps"] == ["settings"]
    assert meta["knowledge"]["tools"] == []
    assert meta["knowledge"]["injected"]["apps"] == ["settings"]
    # use_skill(kind="app") 照样读得到 APP.md
    tool_msgs = [x["content"] for x in m.seen[-1] if x["role"] == "tool"]
    assert any("系统设置" in c for c in tool_msgs), tool_msgs


def test_on_puts_the_procedure_tools_back_exactly_as_before(fake_env, tmp_path):
    store = _store_with_proc(tmp_path)
    script = [[("route", {"scenario": "", "apps": ["settings"]})], DONE]
    r, dev, m = run(fake_env, [["a"], ["a"]], script, tmp_path, skill_store=store,
                    run_config=RunConfig(skill_tools="on"))
    assert "settings__ios-version" in _tool_names(m)
    assert "（可直接调" in json.dumps(m.seen[-1], ensure_ascii=False)
    assert _meta(r)["knowledge"]["tools"] == ["settings__ios-version"]


def test_the_effective_value_is_recorded_in_run_json(fake_env, tmp_path):
    store = _store_with_proc(tmp_path)
    r, dev, m = run(fake_env, [["a"], ["a"]], [DONE], tmp_path, skill_store=store)
    assert _meta(r)["config"]["skill_tools"] == "off"
    r2, dev2, m2 = run(fake_env, [["a"], ["a"]], [DONE], tmp_path, skill_store=store,
                       run_config=RunConfig(skill_tools="on"))
    assert _meta(r2)["config"]["skill_tools"] == "on"


def test_two_consecutive_runs_each_take_their_own_value(fake_env, tmp_path):
    """同一个进程里连跑两个任务，各自的值都要生效、互不沾染（同 screen_parse 那条）。"""
    from iphone_agent.harness.loop import run_task
    store = _store_with_proc(tmp_path)
    dev, per, frames = fake_env([["a"]] * (INITIAL_SETTLE_FRAMES + 8))
    outs = []
    for mode in ("on", "off"):
        model = ScriptedModel([[("route", {"scenario": "", "apps": ["settings"]})], DONE])
        r = run_task("t", dev, per, model, tmp_path, store=isolated_store(tmp_path),
                     workspace=Workspace(tmp_path), skill_store=store,
                     run_config=RunConfig(skill_tools=mode))
        outs.append((_meta(r)["config"]["skill_tools"], _tool_names(model),
                     "可直接调" in json.dumps(model.seen[-1], ensure_ascii=False)))
    (v1, names1, tag1), (v2, names2, tag2) = outs
    assert v1 == "on" and "settings__ios-version" in names1 and tag1
    assert v2 == "off" and "settings__ios-version" not in names2 and not tag2


def test_draft_extraction_still_runs_with_the_tools_off(action_env, tmp_path):
    """开关关的是「端上桌」，不是「学」：成功的运行照旧走提取、照旧落一条草稿。
    草稿是将来重建指纹的原料，停了提取就等于把学习也停了 —— 关掉剧本工具绝不能顺手关掉它。"""
    from iphone_agent.harness.loop import run_task
    store = SkillStore(personal=tmp_path / "skills", shared=tmp_path / "shared-skills")
    # 主屏点图标进 App（ownership 认这条信号）→ 再点一下 → done：够长出一条草稿。
    dev, per, frames = action_env([["设置"], ["通用", "隐私与安全性"], ["关于本机", "软件更新"]],
                                  {(0, "tap"): 1, (1, "tap"): 2})
    model = ScriptedModel([[("tap", {"id": 1, "target": "icon_above"})],
                           [("tap", {"id": 1, "target": "text"})],
                           DONE])
    r = run_task("t", dev, per, model, tmp_path, store=isolated_store(tmp_path),
                 workspace=Workspace(tmp_path), skill_store=store)
    assert r.end_reason == "done_success"
    assert _meta(r)["config"]["skill_tools"] == "off", "这条用例要验的就是关着的时候"
    extracted = _meta(r)["knowledge"]["extracted"]
    assert extracted and extracted[0]["status"] == "new", extracted
    assert store.load().procedures_of(extracted[0]["app"]), "草稿没落盘"
