import json

import pytest

from iphone_agent.skills import model as m

APP_MD = """---
name: settings
display: 设置
open: 设置
risk: read
status: verified
updated: 2026-09-08
---
系统设置。列表很长。
"""

PROC = {
    "schema_version": 1, "name": "ios-version", "description": "读当前 iOS 版本号",
    "params": {}, "returns": ["version"], "risk": "read", "status": "verified",
    "expect_source": "intersection",
    "provenance": {"runs": ["r1", "r2"], "verified_count": 2, "last_ok": "2026-09-08",
                   "last_fail": None, "fail_streak": 0, "macos": "15.6.1"},
    "steps": [
        {"do": "open_app", "target": "设置", "expect": ["设置", "通用"]},
        {"do": "tap", "target": "通用", "how": "text", "expect": ["关于本机", "软件更新"]},
        {"do": "tap", "target": "关于本机", "find": {"direction": "down", "max_screens": 3},
         "expect": ["iOS版本", "型号名称"]},
        {"do": "read", "row": "iOS版本", "as": "version"},
    ],
}

SCEN_MD = """---
name: daily-expense
description: 把今天的支付记录记进一木记账
apps: ["alipay", "yimujizhang"]
risk: write
status: manual
---
1. 用 alipay/today-bills 读今天的支付记录。
2. 每一笔用 yimujizhang/add-expense(金额, 分类, 备注) 记一笔。
"""


def test_app_roundtrip_and_dirname_wins():
    app = m.app_from_markdown(APP_MD, dirname="settings")
    assert app.display == "设置" and app.risk == "read" and app.body.startswith("系统设置")
    renamed = m.app_from_markdown(APP_MD, dirname="settings-old")
    assert renamed.name == "settings-old", "name 以目录名为准，同记忆系统"
    assert m.app_from_markdown(m.app_to_markdown(app), "settings") == app


def test_app_rejects_bad_enum_and_missing_field():
    with pytest.raises(m.SkillError) as e:
        m.app_from_markdown(APP_MD.replace("risk: read", "risk: maybe"), "settings")
    assert e.value.code == "invalid_field"
    with pytest.raises(m.SkillError):
        m.app_from_markdown(APP_MD.replace("open: 设置\n", ""), "settings")


def test_procedure_roundtrip_and_tool_name():
    p = m.procedure_from_json(json.dumps(PROC), app="settings")
    assert p.tool_name == "settings__ios-version"
    assert p.steps[2].find == {"direction": "down", "max_screens": 3}
    assert p.steps[3].as_ == "version"
    again = m.procedure_from_json(m.procedure_to_json(p), app="settings")
    assert again.hash() == p.hash() and again.provenance.runs == ["r1", "r2"]


def test_procedure_rejects_unknown_do_returns_mismatch_and_undeclared_placeholder():
    bad = json.loads(json.dumps(PROC))
    bad["steps"][0]["do"] = "swipe"
    with pytest.raises(m.SkillError):
        m.procedure_from_json(json.dumps(bad), "settings")
    bad = json.loads(json.dumps(PROC))
    bad["returns"] = ["nope"]
    with pytest.raises(m.SkillError) as e:
        m.procedure_from_json(json.dumps(bad), "settings")
    assert e.value.code == "returns_mismatch"
    bad = json.loads(json.dumps(PROC))
    bad["steps"][1]["target"] = "{群名}"
    with pytest.raises(m.SkillError) as e:
        m.procedure_from_json(json.dumps(bad), "settings")
    assert e.value.code == "undeclared_param"


def test_step_hash_ignores_expect_but_not_text_or_how():
    a = m.procedure_from_json(json.dumps(PROC), "settings")
    b = json.loads(json.dumps(PROC))
    b["steps"][1]["expect"] = ["别的"]
    assert m.procedure_from_json(json.dumps(b), "settings").hash() == a.hash()
    c = json.loads(json.dumps(PROC))
    c["steps"][1]["how"] = "row_right"
    assert m.procedure_from_json(json.dumps(c), "settings").hash() != a.hash()


def test_scenario_roundtrip_and_apps_must_be_list():
    s = m.scenario_from_markdown(SCEN_MD, stem="daily-expense")
    assert s.apps == ("alipay", "yimujizhang") and s.status == "manual"
    assert m.scenario_from_markdown(m.scenario_to_markdown(s), "daily-expense") == s
    with pytest.raises(m.SkillError):
        m.scenario_from_markdown(SCEN_MD.replace('apps: ["alipay", "yimujizhang"]', "apps: alipay"), "daily-expense")


def test_scenario_risk_takes_the_highest_of_three_sources():
    assert m.scenario_risk("read", ["read"], "读一下就好") == "read"
    assert m.scenario_risk("read", ["write"], "读一下") == "write"
    assert m.scenario_risk("read", ["read"], "最后帮我付款") == "irreversible"


def test_auto_name_is_slug_plus_hash_and_never_too_long():
    name = m.auto_name("查 iOS 版本", "3f2a9c")
    assert name.startswith("cha-ios-ban-ben") and name.endswith("-3f2a")
    m.validate_name(name)
    long = m.auto_name("这是一个特别特别特别特别特别特别特别长的任务描述文本", "abcdef")
    assert len(long) <= 48


def test_validate_name_rejects_paths_and_uppercase():
    for bad in ("../x", "a/b", "Settings", "", "a\\b"):
        with pytest.raises(m.SkillError):
            m.validate_name(bad)
