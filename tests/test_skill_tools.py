from iphone_agent.harness.tools import TOOL_DEFS, tool_defs
from iphone_agent.skills import model as M
from iphone_agent.skills.store import Catalog
from iphone_agent.skills.tools import procedure_tool_defs, select_procedures, split_tool_name, tool_name


def proc(app="settings", name="ios-version", status="verified", risk="read", params=None):
    return M.Procedure(name=name, app=app, description="读 iOS 版本", params=params or {}, returns=["version"],
                       risk=risk, status=status, expect_source="intersection",
                       provenance=M.Provenance(runs=["r1", "r2"], verified_count=2, last_ok="2026-09-08"),
                       steps=[M.Step(do="tap", target="关于本机", expect=("iOS版本",)),
                              M.Step(do="read", row="iOS版本", as_="version")])


def test_tool_name_roundtrip():
    assert tool_name("settings", "ios-version") == "settings__ios-version"
    assert split_tool_name("settings__ios-version") == ("settings", "ios-version")
    assert split_tool_name("tap") is None


def test_procedure_function_def_has_params_reason_and_provenance_in_description():
    d = procedure_tool_defs([proc(params={"金额": {"type": "string", "description": "数字"}})])[0]["function"]
    assert d["name"] == "settings__ios-version"
    assert set(d["parameters"]["properties"]) == {"金额", "reason"}
    assert d["parameters"]["required"] == ["金额", "reason"]
    assert "走通 2 次" in d["description"] and "2026-09-08" in d["description"] and "version" in d["description"]


def test_select_procedures_scopes_by_app_or_caps_globally(monkeypatch):
    cat = Catalog()
    cat.procedures = {"a": {"p": proc(app="a", name="p")},
                      "b": {"q": proc(app="b", name="q"), "w": proc(app="b", name="w", risk="write")}}
    assert [p.tool_name for p in select_procedures(cat, ["b"])] == ["b__q"]
    assert [p.tool_name for p in select_procedures(cat, None)] == ["a__p", "b__q"]
    monkeypatch.setattr("iphone_agent.skills.tools.config.PROC_TOOLS_MAX", 1)
    assert select_procedures(cat, None) == []


def test_tool_defs_appends_procedures_and_has_use_skill():
    names = [t["function"]["name"] for t in tool_defs(True, [proc()])]
    assert "settings__ios-version" in names and "use_skill" in names
    assert [t["function"]["name"] for t in TOOL_DEFS] == [t["function"]["name"] for t in tool_defs()]


def test_tool_defs_without_coord_tap_drops_x_y():
    tap = next(t for t in tool_defs(False) if t["function"]["name"] == "tap")["function"]
    assert "x" not in tap["parameters"]["properties"] and "坐标" not in tap["description"]
    tap = next(t for t in tool_defs(True) if t["function"]["name"] == "tap")["function"]
    assert "x" in tap["parameters"]["properties"]
