from iphone_agent.harness.actions import Action
from iphone_agent.model.reply import ModelReply
from iphone_agent.skills import model as M
from iphone_agent.skills import route as R
from iphone_agent.skills.store import Catalog


def catalog():
    cat = Catalog()
    cat.apps["settings"] = M.AppProfile("settings", "设置", "设置", "read", "verified", "2026-09-08", "系统设置")
    cat.apps["yimujizhang"] = M.AppProfile("yimujizhang", "一木记账", "一木记账", "write", "verified", "2026-09-08", "记账")
    cat.apps["draft-app"] = M.AppProfile("draft-app", "草稿", "草稿", "write", "draft", "2026-09-08", "")
    cat.scenarios["daily-expense"] = M.Scenario("daily-expense", "记今天的账", ("alipay", "yimujizhang"), "write", "manual", "…")
    cat.scenarios["proposed"] = M.Scenario("proposed", "x", ("settings",), "read", "proposed", "…")
    return cat


def reply(scenario, apps, reason="因为", name="route"):
    a = Action(name, {"scenario": scenario, "apps": apps}, reason, None, "c1")
    return ModelReply(actions=[a], text="", model_version="m", usage={"prompt_tokens": 9}, latency_ms=12)


def test_messages_contain_index_task_and_name_hints():
    msgs = R.routing_messages("【知识】…", "把今天的花销记进一木记账，别动设置", catalog())
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == R.ROUTE_SYSTEM
    text = msgs[1]["content"][0]["text"]
    assert "【知识】…" in text and "任务：把今天的花销" in text
    assert "yimujizhang" in text and "settings" in text and "误命中" in text


def test_parse_keeps_only_names_that_exist_and_are_visible():
    r = R.parse_route(reply("daily-expense", ["settings", "draft-app", "nope"]), catalog())
    assert r.ran and r.scenario == "daily-expense" and r.apps == ["settings"]
    assert sorted(r.ignored) == ["app:draft-app", "app:nope"] and r.reason == "因为"
    assert r.usage == {"prompt_tokens": 9} and r.latency_ms == 12
    r = R.parse_route(reply("proposed", []), catalog())
    assert r.scenario is None and r.ignored == ["scenario:proposed"]


def test_parse_reports_missing_tool_call_and_bad_json():
    r = R.parse_route(ModelReply(actions=[], text="我觉得…", model_version="m", usage={}, latency_ms=1), catalog())
    assert r.ran and r.error and r.scenario is None and r.apps == []
    bad = ModelReply(actions=[Action("route", {"_parse_error": True}, "", None, "c")], text="", model_version="m",
                     usage={}, latency_ms=1)
    assert R.parse_route(bad, catalog()).error


def test_resolve_scenario_decides_apps_and_ignores_model_apps():
    cat = catalog()
    r = R.parse_route(reply("daily-expense", ["settings"]), cat)
    scen, expand, scope = R.resolve(r, cat)
    assert scen.name == "daily-expense"
    assert [a.name for a in expand] == ["yimujizhang"], "alipay 不在 catalog 里，只展开存在且 verified 的"
    assert scope == ["alipay", "yimujizhang"]


def test_resolve_apps_only_and_nothing():
    cat = catalog()
    scen, expand, scope = R.resolve(R.parse_route(reply("", ["yimujizhang", "settings"]), cat), cat)
    assert scen is None and [a.name for a in expand] == ["yimujizhang", "settings"] and scope == ["yimujizhang", "settings"]
    assert R.resolve(R.parse_route(reply("", []), cat), cat) == (None, [], None)
