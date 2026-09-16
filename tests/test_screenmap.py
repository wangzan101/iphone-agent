"""从运行日志拼屏幕图。

指纹稳不稳是可证伪的，已经量过（scripts/map_experiment.py，2026-09-08）：
222 次观察 / 26 个 run，指纹词数中位数 18，0 个空指纹，1/17 撞车。
这里测的是拼图逻辑本身。
"""
import json

from iphone_agent.memory.screenmap import ScreenMap, _semantic_target, build_from_runs


def obs(*texts):
    return {"elements": [{"id": i + 1, "text": t} for i, t in enumerate(texts)]}


def step(n, texts, action=None, ok=True):
    s = {"step": n, "observation": obs(*texts), "result": {"ok": ok}}
    if action:
        s["action"] = action
    return s


def tap(eid, target=None):
    a = {"id": eid}
    if target:
        a["target"] = target
    return {"name": "tap", "args": a}


def test_same_screen_twice_becomes_one_node_and_the_fingerprint_is_the_intersection():
    """内容会变（数字、时间），骨架不变（标题、按钮）—— 交集就是骨架。"""
    m = ScreenMap()
    m.ingest_run([step(1, ["设置", "通用", "关于本机", "01:23"]),
                  step(2, ["设置", "通用", "关于本机", "02:47"])], "r1")
    assert len(m.nodes) == 1
    n = m.nodes[0]
    assert n.visits == 2
    assert n.fingerprint == {"设置", "通用", "关于本机"}, n.fingerprint
    assert "01:23" not in n.fingerprint, "会变的东西不该进指纹"


def test_different_screens_are_different_nodes():
    m = ScreenMap()
    m.ingest_run([step(1, ["设置", "通用", "关于本机"]),
                  step(2, ["微信", "通讯录", "发现", "我"])], "r1")
    assert len(m.nodes) == 2


def test_tap_edges_carry_text_not_element_ids():
    """⚠ 元素编号只在**当次观察**里有效，跨运行毫无意义。
    边上必须是文字，否则这张图下次会把你指到别的东西上。"""
    m = ScreenMap()
    m.ingest_run([step(1, ["通用", "关于本机", "软件更新"], action=tap(2)),
                  step(2, ["关于本机", "iOS版本", "18.3.1"])], "r1")
    e = next(iter(m.edges.values()))
    assert e.target == "关于本机", f"边上存的是 {e.target!r}"
    assert e.action == "tap"


def test_unresolvable_id_gets_no_target_rather_than_a_number():
    """宁可这条边没有目标，也不要存一个下次会指向别的东西的数字。"""
    assert _semantic_target(tap(99), {1: "甲"}) is None
    assert _semantic_target(tap(1), {1: "甲"}) == "甲"
    assert _semantic_target(tap(1, "icon_above"), {1: "微信"}) == "微信@icon_above"


def test_identify_matches_by_fingerprint_subset():
    """指纹是当前屏文字的子集就算认出来 —— 内容多出来的部分不影响。"""
    m = ScreenMap()
    m.ingest_run([step(1, ["设置", "通用", "关于本机", "01:23"]),
                  step(2, ["设置", "通用", "关于本机", "02:47"])], "r1")
    hit = m.identify({"设置", "通用", "关于本机", "别的东西", "09:99"})
    assert hit is not None and hit.key == 0
    assert m.identify({"微信", "通讯录"}) is None


def test_a_screen_seen_once_is_not_yet_identifiable():
    """只见过一次，交集就是它自己 —— 那不是指纹，是一次快照，认它会误判。"""
    m = ScreenMap()
    m.ingest_run([step(1, ["设置", "通用", "01:23"])], "r1")
    assert m.identify({"设置", "通用", "01:23"}) is None
    assert m.stable_nodes() == []


def test_repeated_edges_are_counted_not_duplicated():
    m = ScreenMap()
    for i in range(3):
        m.ingest_run([step(1, ["通用", "关于本机"], action=tap(2)),
                      step(2, ["关于本机", "iOS版本"])], f"r{i}")
    assert len(m.edges) == 1
    assert next(iter(m.edges.values())).count == 3


def test_json_exports_all_nodes_while_stable_nodes_still_filters():
    """JSON 是**缓存**，得留住只见过一次的节点 ——
    丢了它们，重载之后这些屏永远到不了 visits==2，指纹就再也长不出来。
    过滤（谁算稳定）是读图时的事，`stable_nodes()` 仍然只给 visits >= 2 的。"""
    m = ScreenMap()
    m.ingest_run([step(1, ["设置", "通用", "关于本机", "a"]),
                  step(2, ["设置", "通用", "关于本机", "b"]),
                  step(3, ["只见过一次的屏"])], "r1")
    d = json.loads(m.to_json())
    assert len(d["nodes"]) == 2, "只见过一次的屏也要落盘"
    stable = m.stable_nodes()
    assert len(stable) == 1
    assert sorted(stable[0].fingerprint) == ["关于本机", "设置", "通用"]


def test_fingerprint_is_incremental_intersection():
    """增量交集 == 全量交集，但不再保留每次访问的原始集合。"""
    m = ScreenMap()
    m.ingest_run([step(1, ["设置", "通用", "12:01"]), step(2, ["设置", "通用", "12:02"])], "r1")
    m.ingest_run([step(1, ["设置", "通用", "12:03"])], "r2")
    n = m.nodes[0]
    assert n.fingerprint == {"设置", "通用"} and n.visits == 3
    assert not hasattr(m, "_sets")


def test_json_round_trip_keeps_all_nodes_and_ingested():
    m = ScreenMap()
    m.ingest_run([step(1, ["a", "b"])], "r1")          # visits=1，以前 to_json 会丢掉它
    m.ingested["r1"] = 123.0
    m2 = ScreenMap.from_json(m.to_json())
    assert len(m2.nodes) == 1 and m2.nodes[0].representative == {"a", "b"}
    assert m2.ingested == {"r1": 123.0}
    m2.ingest_run([step(1, ["a", "b", "c"])], "r2")
    assert m2.nodes[0].visits == 2 and m2.stable_nodes()


def _write_run(d, steps, end_reason="done_success"):
    """造一个**已结束**的 run 目录：steps.jsonl + 带 end_reason 的 run.json。"""
    d.mkdir(parents=True, exist_ok=True)
    (d / "steps.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in steps), encoding="utf-8")
    (d / "run.json").write_text(json.dumps({"end_reason": end_reason}), encoding="utf-8")


def test_build_from_runs_with_cache_only_ingests_new_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    cache = tmp_path / "screenmap.json"
    _write_run(runs / "20260101-000000-aaaa", [step(1, ["a", "b"]), step(2, ["a", "b"])])
    m1 = build_from_runs(runs, cache=cache)
    assert cache.exists() and m1.nodes[0].visits == 2
    calls = []
    orig = ScreenMap.ingest_run
    monkeypatch.setattr(ScreenMap, "ingest_run",
                        lambda self, steps, rid: calls.append(rid) or orig(self, steps, rid))
    _write_run(runs / "20260101-000001-bbbb", [step(1, ["a", "b"])])
    m2 = build_from_runs(runs, cache=cache)
    assert calls == ["20260101-000001-bbbb"]         # 老的没重喂
    assert m2.nodes[0].visits == 3


def test_build_from_runs_does_not_cache_a_run_still_going(tmp_path):
    """还在跑的 run 的 steps.jsonl 会继续被追加：一记进 ingested，
    下次 mtime 一变就**整个重喂**，visits 会重复计数。所以没结束就不记。"""
    runs = tmp_path / "runs"
    cache = tmp_path / "screenmap.json"
    _write_run(runs / "20260101-000000-aaaa", [step(1, ["a", "b"])])          # 跑完了
    live = runs / "20260101-000001-bbbb"
    _write_run(live, [step(1, ["a", "b"]), step(2, ["a", "b"])], end_reason=None)
    m1 = build_from_runs(runs, cache=cache)
    assert m1.nodes[0].visits == 3          # 没跑完的也照喂，图是最新的
    d = json.loads(cache.read_text(encoding="utf-8"))
    assert list(d["ingested"]) == ["20260101-000000-aaaa"], "没跑完的不该进缓存"
    assert d["nodes"][0]["visits"] == 1, "缓存里只该有跑完的那个 run 的账"
    # 这个 run 又追加了一步：整个重喂，visits 应该是 1 + 3 而不是 1 + 2 + 3
    _write_run(live, [step(1, ["a", "b"]), step(2, ["a", "b"]), step(3, ["a", "b"])],
               end_reason=None)
    m2 = build_from_runs(runs, cache=cache)
    assert m2.nodes[0].visits == 4


def test_build_from_runs_ignores_corrupt_cache(tmp_path):
    runs = tmp_path / "runs"
    cache = tmp_path / "screenmap.json"
    _write_run(runs / "20260101-000000-aaaa", [step(1, ["a", "b"])])
    cache.write_text("{not json")
    m = build_from_runs(runs, cache=cache)
    assert len(m.nodes) == 1


def test_build_from_runs_without_cache_touches_no_disk(tmp_path):
    """cache=None 时行为跟以前完全一样：全量重建，不落盘。"""
    runs = tmp_path / "runs"
    _write_run(runs / "20260101-000000-aaaa", [step(1, ["a", "b"]), step(2, ["a", "b"])])
    m = build_from_runs(runs)
    assert m.nodes[0].visits == 2
    assert sorted(p.name for p in (runs / "20260101-000000-aaaa").iterdir()) == [
        "run.json", "steps.jsonl"]
    assert list(tmp_path.iterdir()) == [runs]


# --- 寻路 ---

def test_route_finds_a_multi_hop_path():
    m = ScreenMap()
    for i in range(2):
        m.ingest_run([step(1, ["主屏", "设置图标"], action=tap(2)),
                      step(2, ["设置", "通用", "隐私"], action=tap(2)),
                      step(3, ["通用", "关于本机", "软件更新"], action=tap(2)),
                      step(4, ["关于本机", "iOS版本", "序列号"])], f"r{i}")
    from iphone_agent.memory.screenmap import find_nodes, route
    dst = find_nodes(m, "iOS版本")[0]
    src = find_nodes(m, "设置图标")[0]
    r = route(m, src.key, dst.key)
    assert r is not None and len(r) == 3, r
    assert [e.target for e in r] == ["设置图标", "通用", "关于本机"], [e.target for e in r]


def test_route_returns_none_when_there_is_no_path():
    m = ScreenMap()
    for i in range(2):
        m.ingest_run([step(1, ["甲屏", "甲"]), step(2, ["乙屏", "乙"])], f"r{i}")
    from iphone_agent.memory.screenmap import route
    assert route(m, 0, 1) is None, "两个节点之间没有带目标的边，不该编一条出来"


def test_route_skips_edges_without_a_target():
    """没有语义目标的边没法复述给模型，路上有这么一步就等于断了。"""
    m = ScreenMap()
    for i in range(2):
        m.ingest_run([step(1, ["甲屏"], action={"name": "tap", "args": {"x": 1, "y": 2}}),
                      step(2, ["乙屏"])], f"r{i}")
    from iphone_agent.memory.screenmap import route
    assert route(m, 0, 1) is None


def _rec(texts, action=None, ok=True, changed=True, extra_result=None, kind=None):
    s = {"step": 1, "observation": obs(*texts), "result": {"ok": ok, "changed": changed, **(extra_result or {})}}
    if action:
        s["action"] = action
    if kind:
        s["kind"] = kind
    return s


def test_ownership_walk_follows_open_app_home_spotlight_and_switcher():
    from iphone_agent.memory.screenmap import SYSTEM, UNKNOWN, ownership
    recs = [
        _rec(["主屏"], {"name": "open_app", "args": {"name": "设置"}}),          # 观察时是 system；打开后归 she-zhi
        _rec(["设置", "通用"], {"name": "tap", "args": {"id": 2}}),
        _rec(["通用"], {"name": "key", "args": {"name": "app_switcher"}}),
        _rec(["多任务"], {"name": "tap", "args": {"id": 1}}),
        _rec(["别的 App"], {"name": "key", "args": {"name": "home"}}),
        _rec(["主屏"], {"name": "open_app", "args": {"name": "wechat"}}, changed=False),   # 没打开，归属不变
        _rec(["主屏"], {"name": "open_app", "args": {"name": "微信"}}),
        _rec(["微信", "通讯录"], {"name": "done", "args": {}}),
    ]
    assert ownership(recs) == [SYSTEM, "she-zhi", "she-zhi", UNKNOWN, UNKNOWN, SYSTEM, SYSTEM, "wei-xin"]


def test_apps_seen_maps_id_to_original_open_name():
    from iphone_agent.memory.screenmap import apps_seen
    recs = [_rec(["主屏"], {"name": "open_app", "args": {"name": "记账本"}}), _rec(["记账"])]
    assert apps_seen(recs) == {"ji-zhang-ben": "记账本"}


# --- 主屏图标点开一个 App，也算「进入」（真机 44 个「从未进过任何 App」里 10 个是这样丢的）---

def test_ownership_icon_tap_from_home_enters_the_app_for_the_next_record():
    """本条记录观察时仍是 system——动作改变的是**之后**的归属。"""
    from iphone_agent.memory.screenmap import SYSTEM, ownership
    recs = [
        _rec(["主屏", "日历"], tap(2, "icon_above")),      # id 2 -> "日历"
        _rec(["日历", "今天"]),
    ]
    assert ownership(recs) == [SYSTEM, "ri-li"]


def test_ownership_icon_tap_that_did_not_change_the_screen_does_not_enter():
    from iphone_agent.memory.screenmap import SYSTEM, ownership
    recs = [
        _rec(["主屏", "小红书"], tap(2, "icon_above"), changed=False),
        _rec(["主屏", "小红书"]),
    ]
    assert ownership(recs) == [SYSTEM, SYSTEM]


def test_ownership_tapping_the_spotlight_search_box_does_not_enter_an_app():
    """target=text 点「Q 搜索」框——真实数据里要排除掉的那个 case，不是进 App。"""
    from iphone_agent.memory.screenmap import SYSTEM, ownership
    recs = [
        _rec(["主屏", "Q 搜索"], tap(2, "text")),
        _rec(["Q 搜索", "最近搜索"]),
    ]
    assert ownership(recs) == [SYSTEM, SYSTEM]


def test_ownership_icon_above_tap_inside_an_app_does_not_change_ownership():
    """底部 tab 栏有时也用 icon_above 点——已经在 App 里了，不该被当成又进了一次。"""
    from iphone_agent.memory.screenmap import ownership
    recs = [
        _rec(["主屏", "设置"], {"name": "open_app", "args": {"name": "设置"}}),
        _rec(["设置", "通用"], tap(2, "icon_above")),      # id 2 -> "通用"，已经在设置里
        _rec(["通用", "隐私"]),
    ]
    assert ownership(recs) == ["system", "she-zhi", "she-zhi"]


def test_ownership_icon_above_tap_with_unresolvable_id_does_not_change_ownership():
    from iphone_agent.memory.screenmap import SYSTEM, ownership
    recs = [
        _rec(["主屏", "日历"], tap(99, "icon_above")),     # id 99 在这一步的观察里不存在
        _rec(["主屏", "日历"]),
    ]
    assert ownership(recs) == [SYSTEM, SYSTEM]


def test_apps_seen_picks_up_icon_tap_label_as_display_name():
    from iphone_agent.memory.screenmap import apps_seen
    recs = [_rec(["主屏", "文件"], tap(2, "icon_above")), _rec(["文件", "最近项目"])]
    assert apps_seen(recs) == {"wen-jian": "文件"}


def test_apps_seen_keeps_the_open_app_name_over_a_later_icon_label():
    """同一个 App id，open_app 先写入的名字不能被后面的图标标签覆盖——先来者赢。"""
    from iphone_agent.memory.screenmap import apps_seen
    recs = [
        _rec(["主屏"], {"name": "open_app", "args": {"name": " 小红书"}}),
        _rec(["主屏", "小红书"], tap(2, "icon_above")),    # 同一个 App（同一个 slug），标签更干净
        _rec(["小红书", "首页"]),
    ]
    assert apps_seen(recs) == {"xiao-hong-shu": " 小红书"}


def test_procedure_step_records_are_ingested_like_steps():
    m = ScreenMap()
    m.ingest_run([
        _rec(["通用", "关于本机"], {"name": "settings__ios-version", "args": {}}),
        _rec(["通用", "关于本机"], {"name": "tap", "args": {"id": 2}}, kind="procedure_step"),
        _rec(["关于本机", "iOS版本"], {"name": "done", "args": {}}),
    ], "r1")
    assert any(e.action == "tap" and e.target == "关于本机" for e in m.edges.values()), "子记录的边也要进图"


def test_subgraph_keeps_only_that_apps_stable_nodes_and_edges_between_them():
    from iphone_agent.memory.screenmap import appmap_json
    m = ScreenMap()
    for r in ("r1", "r2"):
        m.ingest_run([
            _rec(["主屏", "设置"], {"name": "open_app", "args": {"name": "设置"}}),
            _rec(["设置", "通用", "隐私"], {"name": "tap", "args": {"id": 2, "target": "row_right"}}),
            _rec(["通用", "关于本机", "软件更新"], {"name": "key", "args": {"name": "home"}}),
            _rec(["主屏", "设置"], {"name": "done", "args": {}}),
        ], r)
    sub = m.subgraph("she-zhi")
    fps = [n.fingerprint for n in sub.nodes]
    assert {"设置", "通用", "隐私"} in fps and {"通用", "关于本机", "软件更新"} in fps
    assert not any("主屏" in fp for fp in fps)
    assert all(e.src in {n.key for n in sub.nodes} and e.dst in {n.key for n in sub.nodes} for e in sub.edges.values())
    j = __import__("json").loads(appmap_json(sub, "she-zhi", ["r1", "r2"]))
    assert j["schema_version"] == 1 and j["app"] == "she-zhi" and j["built_from"] == ["r1", "r2"]
    tap = next(e for e in j["edges"] if e["action"] == "tap")
    assert tap["target"] == "通用" and tap["how"] == "row_right"


def test_node_app_is_majority_owner():
    m = ScreenMap()
    m.ingest_run([_rec(["主屏"], {"name": "open_app", "args": {"name": "设置"}}),
                  _rec(["设置", "通用"], {"name": "done", "args": {}})], "r1")
    m.ingest_run([_rec(["主屏"], {"name": "open_app", "args": {"name": "设置"}}),
                  _rec(["设置", "通用"], {"name": "done", "args": {}})], "r2")
    node = next(n for n in m.nodes if "通用" in n.fingerprint)
    assert m.node_app(node.key) == "she-zhi" and m.app_names == {"she-zhi": "设置"} and m.runs == {"r1", "r2"}
