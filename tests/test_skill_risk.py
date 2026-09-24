"""剧本的风险档描述**这条剧本自己做了什么**，不是它属于哪个 App。

真机上的死锁（2026-09-16）：15 条提取出来的剧本全是 `draft write` —— 风险跟着 App 走，
自动切出来的 App 永远是 draft，于是 `_risk_for` 一律给 `write`；而自动验证要求 `risk == "read"`，
人 `approve` 又只改 status 不改 risk。两条「学到的剧本变成工具」的路都被堵死，一次都没通过。
"""
import json

import pytest

from iphone_agent import riskwords
from iphone_agent.skills import extract as X
from iphone_agent.skills import model as M
from iphone_agent.skills.store import SkillStore
from tests.test_skill_extract import make_run, rec


def tap(step, texts, *, target_id=2, **res):
    """一次「点了、画面整屏换了」的点击：有新文字出现、不是局部变化 —— 导航。"""
    res.setdefault("appeared", ["新的一屏"])
    return rec(step, texts, "tap", {"id": target_id, "target": "text"}, **res)


LOOKUP = [
    rec(1, ["主屏", "设置"], "open_app", {"name": "设置"}),
    tap(2, ["设置", "通用", "隐私与安全性"]),
    tap(3, ["通用", "关于本机", "软件更新"]),
    rec(4, ["关于本机", "iOS版本", "18.3.1"], "done", {"status": "success", "result": "18.3.1"}),
]


@pytest.fixture
def store(tmp_path):
    return SkillStore(personal=tmp_path / "p", shared=tmp_path / "s")


# ---- 规则本身 ----

def test_navigation_only_procedure_is_read():
    steps = [M.Step(do="open_app", target="设置"),
             M.Step(do="tap", target="通用", effect="navigated"),
             M.Step(do="read", row="iOS版本", as_="version")]
    assert M.procedure_risk(steps) == "read"


def test_typing_makes_it_write():
    steps = [M.Step(do="open_app", target="备忘录"),
             M.Step(do="tap", target="新建", effect="navigated"),
             M.Step(do="type", text="你好")]
    assert M.procedure_risk(steps) == "write"


def test_tap_on_a_danger_word_is_at_least_write_and_never_stays_never():
    nav = {"effect": "navigated"}
    assert M.procedure_risk([M.Step(do="tap", target="删除", **nav)]) == "write"
    assert M.procedure_risk([M.Step(do="tap", target="转账给张三", **nav)]) == "irreversible"


def test_the_danger_word_list_is_not_copied_into_the_skills_layer():
    """CLAUDE.md §7：一个规则一个入口。词表只有 riskwords 一份，skills 里不许有第二份。"""
    import pathlib
    root = pathlib.Path(M.__file__).parent
    words = [w for w in ("删除", "发送", "转账", "purchase", "log out") if w in riskwords.DANGER.pattern
             or w in riskwords.NEVER.pattern]
    assert words, "取的样本词得真的在词表里，否则这条测试什么都没验"
    for f in root.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for w in words:
            assert w not in text, f"{f.name} 抄了一份危险词：{w}"


def test_tap_whose_effect_was_not_navigation_or_is_unknown_is_write():
    assert M.procedure_risk([M.Step(do="tap", target="飞行模式", effect="state_changed")]) == "write"
    assert M.procedure_risk([M.Step(do="tap", target="通用", effect="unknown")]) == "write"
    assert M.procedure_risk([M.Step(do="tap", target="通用")]) == "write", "没记录过效果 = 不知道"


def test_verified_app_profile_can_only_tighten_never_loosen():
    steps = [M.Step(do="open_app", target="支付宝"), M.Step(do="tap", target="账单", effect="navigated")]
    strict = M.AppProfile("alipay", "支付宝", "支付宝", "irreversible", "verified", "2026-09-16")
    assert M.procedure_risk(steps, strict) == "irreversible"
    loose = M.AppProfile("alipay", "支付宝", "支付宝", "read", "verified", "2026-09-16")
    assert M.procedure_risk([M.Step(do="type", text="x")], loose) == "write", "App 说 read 也放不低"
    draft = M.AppProfile("alipay", "支付宝", "支付宝", "irreversible", "draft", "2026-09-16")
    assert M.procedure_risk(steps, draft) == "read", "draft 的 App 档只是自动填的，不算声明"


# ---- 提取、自动晋级 ----

def test_lookup_procedure_extracts_as_read_and_auto_promotes_after_two_runs(store, tmp_path):
    runs = tmp_path / "runs"
    X.extract_from_run(store, make_run(runs, "r1", "查 iOS 版本", LOOKUP),
                       end_reason="done_success", audit_miss=[], procedures_used=[])
    p = store.load().procedures_of("settings")[0]
    assert p.risk == "read" and p.status == "draft"
    assert [s.effect for s in p.steps] == [None, "navigated", "navigated"]

    second = [LOOKUP[0],
              tap(2, ["设置", "通用", "隐私与安全性", "12:30"]),
              tap(3, ["通用", "关于本机", "软件更新", "AirDrop"]),
              rec(4, ["关于本机", "iOS版本", "18.3.1"], "done", {})]
    X.extract_from_run(store, make_run(runs, "r2", "查一下 iOS 版本", second),
                       end_reason="done_success", audit_miss=[], procedures_used=[])
    cat = store.load()
    p = cat.procedures_of("settings")[0]
    assert (p.status, p.risk) == ("verified", "read")
    assert cat.app("settings").status == "draft", "App 还没人批准，剧本照样能用：闸门是剧本自己的步骤"
    assert [x.tool_name for x in cat.eligible_procedures()] == [p.tool_name]


def test_procedure_with_typing_never_auto_promotes(store, tmp_path):
    runs = tmp_path / "runs"
    typing = [rec(1, ["主屏", "备忘录"], "open_app", {"name": "备忘录"}),
              tap(2, ["备忘录", "新建", "列表"]),
              rec(3, ["新建", "你好"], "type", {"text": "你好"}, verified="true"),
              rec(4, ["新建", "你好"], "done", {})]
    for r in ("r1", "r2"):
        X.extract_from_run(store, make_run(runs, r, "记一笔", typing),
                           end_reason="done_success", audit_miss=[], procedures_used=[])
    p = store.load().procedures_of("notes")[0]
    assert (p.status, p.risk) == ("draft", "write")
    assert store.load().eligible_procedures() == []


def test_switch_toggle_tap_extracts_as_write(store, tmp_path):
    """整屏没变、只有点击处变了（local_mad）—— 那是开关翻转，不是导航。"""
    runs = tmp_path / "runs"
    toggle = [rec(1, ["主屏", "设置"], "open_app", {"name": "设置"}),
              tap(2, ["设置", "通用", "蜂窝网络"]),
              rec(3, ["蜂窝网络", "飞行模式"], "tap", {"id": 2, "target": "row_right"}, local_mad=17.0),
              rec(4, ["蜂窝网络", "飞行模式"], "done", {})]
    for r in ("r1", "r2"):
        X.extract_from_run(store, make_run(runs, r, "开飞行模式", toggle),
                           end_reason="done_success", audit_miss=[], procedures_used=[])
    p = store.load().procedures_of("settings")[0]
    assert p.steps[-1].effect == "state_changed"
    assert (p.status, p.risk) == ("draft", "write")


def test_sync_recomputes_draft_risk_from_the_steps_themselves(store, tmp_path):
    runs = tmp_path / "runs"
    make_run(runs, "r1", "查 iOS 版本", LOOKUP)
    make_run(runs, "r2", "查 iOS 版本", LOOKUP)
    X.sync(store, runs)
    p = store.load().procedures_of("settings")[0]
    assert (p.status, p.risk) == ("verified", "read")


def test_old_stored_draft_without_effects_is_recomputed_not_downgraded(store, tmp_path):
    """盘上的老草稿（提取时还没有 effect 这一栏）：sync 按它**自己的步骤**重算 ——
    没有任何一步记录过效果，算出来就是 write，不会被悄悄降成 read。"""
    store.write_app(M.AppProfile("settings", "设置", "设置", "write", "draft", "2026-09-16"))
    store.write_procedure(M.Procedure(
        name="old-proc", app="settings", description="老草稿", params={}, returns=[], risk="write",
        status="draft", expect_source="single_run", provenance=M.Provenance(runs=["r0", "r9"], verified_count=2),
        steps=[M.Step(do="open_app", target="设置"), M.Step(do="tap", target="通用")]))
    (tmp_path / "runs").mkdir()
    X.sync(store, tmp_path / "runs")
    p = store.load().procedure("settings", "old-proc")
    assert (p.status, p.risk) == ("draft", "write")
    assert json.loads(store.proc_path("settings", "old-proc").read_text(encoding="utf-8"))["risk"] == "write"


def test_old_stored_draft_heals_when_the_same_path_is_extracted_again(store, tmp_path):
    """同一条路径再跑一次：合并时把新运行记下的效果补进老步骤，风险跟着重算。"""
    runs = tmp_path / "runs"
    X.extract_from_run(store, make_run(runs, "r1", "查 iOS 版本", LOOKUP),
                       end_reason="done_success", audit_miss=[], procedures_used=[])
    p = store.load().procedures_of("settings")[0]
    stripped = M.Procedure(**{**p.__dict__, "risk": "write", "status": "draft",
                              "steps": [M.Step(**{**s.__dict__, "effect": None}) for s in p.steps]})
    store.write_procedure(stripped, overwrite=True)
    X.extract_from_run(store, make_run(runs, "r2", "查 iOS 版本", LOOKUP),
                       end_reason="done_success", audit_miss=[], procedures_used=[])
    p = store.load().procedures_of("settings")[0]
    assert (p.status, p.risk) == ("verified", "read")


def test_step_effect_round_trips_but_does_not_change_the_hash():
    a = M.Step(do="tap", target="通用", expect=("关于本机",))
    b = M.Step(do="tap", target="通用", expect=("关于本机",), effect="navigated")
    assert M.step_hash([a]) == M.step_hash([b]), "效果是证据不是执行参数，不该变成另一条剧本"
    assert M.Step.from_json(b.to_json(), 1).effect == "navigated"
    with pytest.raises(M.SkillError):
        M.Step.from_json({"do": "tap", "target": "通用", "expect": [], "effect": "飞了"}, 1)
