"""记账核心（spec §4.3、§5）：逐事件记访问、推归属、记转移；同一个 run 不重复计数。"""
import json

from iphone_agent.memory.screenmap import app_id_for
from iphone_agent.twin.record import RecordStats, TwinState, record_run, replay_run, simulate
from iphone_agent.workspace import Workspace
from tests.twin_fixtures import (
    SET,
    VERIFIED,
    cand,
    home_obs,
    id_of,
    screen_obs,
    settings_run,
    sid_of,
    step,
    write_run,
)


def mem():
    return TwinState(None, stats=RecordStats())


def by_name(state, app="she-zhi"):
    return {s.name: s for s in state.screens_of(app)}


def _same(frame, rows=("关于本机", "软件更新", "存储")):
    """动作后还是 r1 那一屏「通用」：模型看图答 same_as=1，候选就是 r1 在 f2.png 建出的那一屏。"""
    return screen_obs(frame, "通用", "设置", rows, same_as=1,
                      candidates=(cand(1, SET, sid_of(SET, "r1", "f2.png"), "通用", "设置"),))


def test_settings_run_builds_three_screens_and_a_navigation_chain(tmp_path):
    st = mem()
    replay_run(st, write_run(tmp_path, "r1", settings_run()))
    s = by_name(st)
    assert set(s) == {"设置", "通用", "关于本机"}
    assert all(x.visits == 1 and x.status == "provisional" for x in s.values())
    (t,) = s["设置"].transitions.values()
    assert (t.action, t.target, t.effect, t.to) == ("tap", "通用", "navigated", s["通用"].id)
    assert s["关于本机"].anchors == {"关于本机": 1, "iOS版本": 1}


def test_second_run_confirms_screens_and_counts_edges_twice(tmp_path):
    st = mem()
    replay_run(st, write_run(tmp_path, "r1", settings_run()))
    replay_run(st, write_run(tmp_path, "r2", settings_run(t0=100, first="r1")))
    s = by_name(st)
    assert len(st.screens_of("she-zhi")) == 3
    assert s["通用"].visits == 2 and s["通用"].status == "confirmed"
    assert next(iter(s["设置"].transitions.values())).count == 2


def test_same_run_twice_changes_nothing(tmp_path):
    st = mem()
    d = write_run(tmp_path, "r1", settings_run())
    replay_run(st, d)
    before = {x.id: x.to_json() for x in st.screens_of("she-zhi")}
    replay_run(st, d)
    assert {x.id: x.to_json() for x in st.screens_of("she-zhi")} == before
    assert st.stats.skipped_applied > 0


def test_unverified_open_app_records_nothing(tmp_path):
    recs = settings_run()
    recs[0]["result"] = {"ok": True, "changed": True}              # 7322f80 之前的留档没有 identity
    st = mem()
    replay_run(st, write_run(tmp_path, "r1", recs))
    assert st.screens_of("she-zhi") == [] and st.stats.skipped == 4


def _one_screen_run(action, args=None, result=None, after_obs=None):
    """在「通用」屏上做一个动作，after_obs 是动作后那一帧（None = 没有下一帧）。"""
    f1 = home_obs("f1.png")
    f2 = screen_obs("f2.png", "通用", "设置", rows=("关于本机", "软件更新", "存储"))
    recs = [step(1, f1, "open_app", {"name": "设置"}, VERIFIED, after="f2.png"),
            step(2, f2, action, args, result, after="f3.png")]
    if after_obs is not None:
        recs.append(step(3, after_obs, "done", {"status": "success"}))
    return recs


def _only_edge(tmp_path, recs):
    st = mem()
    replay_run(st, write_run(tmp_path, "r1", recs))
    return next(iter(by_name(st)["通用"].transitions.values()))


def test_effect_none_when_screen_did_not_change(tmp_path):
    t = _only_edge(tmp_path, _one_screen_run("tap", {"id": 3}, {"ok": True, "changed": False}, _same("f3.png")))
    assert (t.target, t.effect, t.to) == ("关于本机", "none", None)


def test_effect_state_changed_when_same_screen_but_changed(tmp_path):
    same = _same("f3.png", rows=("关于本机", "软件更新", "存储", "新行"))
    assert _only_edge(tmp_path, _one_screen_run("tap", {"id": 3}, None, same)).effect == "state_changed"


def test_effect_left_app_even_when_after_is_skipped(tmp_path):
    t = _only_edge(tmp_path, _one_screen_run("key", {"name": "home"}, None, home_obs("f3.png")))
    assert (t.target, t.effect) == ("home", "left_app")


def test_effect_unknown_when_after_missing_or_not_sent(tmp_path):
    assert _only_edge(tmp_path, _one_screen_run("tap", {"id": 3})).effect == "unknown"
    t = _only_edge(tmp_path / "b", _one_screen_run("tap", {"id": 3}, {"ok": False, "error": "x"}))
    assert (t.sent, t.effect) == (False, "unknown")


def test_off_track_is_counted_from_flattened_judged(tmp_path):
    other = screen_obs("f3.png", "存储空间", "通用")
    t = _only_edge(tmp_path, _one_screen_run(
        "tap", {"id": 3}, {"ok": True, "changed": True, "judged": {"on_change": True, "worked": False}}, other))
    assert t.effect == "navigated" and t.off_track == 1


def test_type_never_stores_what_was_typed(tmp_path):
    t = _only_edge(tmp_path, _one_screen_run("type", {"text": "我的密码123"}, None, _same("f3.png")))
    assert t.target is None and "我的密码" not in str(t.to_json())


def test_tap_on_vision_element_or_unknown_id_records_no_edge(tmp_path):
    vis = {"text": "放大镜图标", "confidence": 0, "box": [0, 0, 1, 1], "center": [500, 400], "source": "vision"}
    f2 = screen_obs("f2.png", "通用", "设置", rows=("关于本机", "软件更新", "存储"), extra=(vis,))
    recs = [step(1, home_obs("f1.png"), "open_app", {"name": "设置"}, VERIFIED, after="f2.png"),
            step(2, f2, "tap", {"id": id_of(f2, "放大镜图标")}, after="f3.png"),
            step(3, screen_obs("f3.png", "搜索", "通用"), "tap", {"id": 99}, after="f4.png"),
            step(4, screen_obs("f4.png", "搜索", "通用"), "done")]
    st = mem()
    replay_run(st, write_run(tmp_path, "r1", recs))
    assert all(not s.transitions for s in st.screens_of("she-zhi"))


def test_observation_after_zoom_is_not_recognized_as_a_screen(tmp_path):
    f2 = screen_obs("f2.png", "关于本机", "通用", rows=("iOS版本", "型号", "容量"))
    zoomed = screen_obs("f2z.png", None, None, rows=("iOS版本 18.3.1", "型号名称", "容量 128GB", "可用"))
    recs = [step(1, home_obs("f1.png"), "open_app", {"name": "设置"}, VERIFIED, after="f2.png"),
            step(2, f2, "zoom", {"box": [0, 300, 624, 700]}, after="f2z.png"),
            step(3, zoomed, "done", {"status": "success"})]
    st = mem()
    replay_run(st, write_run(tmp_path, "r1", recs))
    assert [s.name for s in st.screens_of("she-zhi")] == ["关于本机"]


def test_new_screen_found_as_after_is_created_once_and_first_visit_keeps_it_provisional(tmp_path):
    st = mem()
    replay_run(st, write_run(tmp_path, "r1", settings_run()))
    s = by_name(st)["通用"]
    assert s.id == sid_of(SET, "r1", "f3.png"), "after 帧建的屏，id 用它自己那一帧（run:frame_file）"
    assert s.status == "provisional" and s.visits == 1


def test_id_collision_gets_a_salted_id(tmp_path):
    from iphone_agent.twin.screenfile import Screen, screen_id
    st = mem()
    squatter = Screen(id=sid_of(SET, "r", "f9.png"), app=SET, first_event="别的事件", created="c",
                      updated="u", name="占位")
    st.screens_of(SET)
    st._apps[SET][squatter.id] = squatter
    s = st.create(SET, "r:f9.png", "新屏", 0.0)
    assert s.id == screen_id(SET, "r:f9.png", 1) and st.stats.collisions == 1
    assert st.create(SET, "r:f9.png", "新屏", 0.0) is s, "同一帧再建一次：沿用，不再加盐"


def test_simulate_maps_every_event_to_owner_state_and_screen(tmp_path):
    write_run(tmp_path, "r1", settings_run())
    sim = simulate(Workspace(tmp_path))
    assert sim.by_event["r1:1"] == ("system", "skipped", None)
    owner, state, sid = sim.by_event["r1:4"]
    # 第一次来的屏，模型 same_as=null → 认屏态是 new（旧几何规则按标题认成 matched）；落到的还是那一屏
    assert owner == "she-zhi" and state == "new" and sim.state.get("she-zhi", sid).name == "关于本机"
    assert sid == sid_of(SET, "r1", "f4.png")


def test_unfinished_runs_are_not_simulated(tmp_path):
    write_run(tmp_path, "r1", settings_run(), finished=False)
    assert simulate(Workspace(tmp_path)).by_event == {}


def test_finished_run_with_invalid_utf8_steps_does_not_raise(tmp_path):
    """坏编码等于「没有」（不变式 5），不该让 simulate 炸（同 events.read_run_events 的规矩）。"""
    d = write_run(tmp_path, "r1", settings_run())
    with (d / "steps.jsonl").open("ab") as f:
        f.write(b"\xff\xfe not valid utf-8\n")
    sim = simulate(Workspace(tmp_path))
    assert sim.by_event == {} and sim.state.screens_of("she-zhi") == []


def names(st, app=SET):
    return {s.name for s in st.screens_of(app)}


def test_screens_are_named_by_the_label_and_ids_come_from_run_and_frame(tmp_path):
    st = TwinState(None, stats=RecordStats())
    replay_run(st, write_run(tmp_path, "r0", settings_run()))
    assert names(st) == {"设置", "通用", "关于本机"}
    s = st.get(SET, sid_of(SET, "r0", "f3.png"))
    assert s.name == "通用" and s.anchors == {"通用": 1, "关于本机": 1}
    # 4 帧各认一次（after 帧在 apply 里认过，下一个事件的 before 沿用，不再计数）：
    # f1 系统（skipped，归属来自标注）、f2 核过身份的进入、f3 f4 标注
    x = st.stats
    assert (x.labeled, x.new, x.screens_created, x.skipped, x.matched) == (4, 3, 3, 1, 0)
    assert (x.owner_from_entry, x.owner_from_label, x.owner_from_tracker) == (1, 3, 0)


def test_second_run_matches_through_same_as_and_confirms(tmp_path):
    st = TwinState(None, stats=RecordStats())
    replay_run(st, write_run(tmp_path, "r0", settings_run()))
    replay_run(st, write_run(tmp_path, "r1", settings_run(t0=100, first="r0")))
    assert len(st.screens_of(SET)) == 3
    assert all(s.status == "confirmed" for s in st.screens_of(SET))
    assert st.stats.candidate_unresolved == 0


def test_same_as_with_another_name_adds_an_alias(tmp_path):
    st = TwinState(None, stats=RecordStats())
    replay_run(st, write_run(tmp_path, "r0", settings_run()))
    recs = settings_run(t0=100, first="r0")
    recs[2]["observation"]["screen"]["name"] = "通用设置"
    replay_run(st, write_run(tmp_path, "r1", recs))
    assert st.get(SET, sid_of(SET, "r0", "f3.png")).aliases == {"通用设置"}
    assert st.stats.aliases_added == 1


def test_same_as_pointing_nowhere_is_counted_not_created(tmp_path):
    recs = settings_run()
    obs = recs[2]["observation"]
    obs["screen"]["same_as"] = 1
    obs["screen_candidates"] = [cand(1, SET, "s_ghost", "关于本机")]
    st = TwinState(None, stats=RecordStats())
    replay_run(st, write_run(tmp_path, "r0", recs))
    assert st.stats.candidate_unresolved == 1                   # f3 只认一次


def test_frames_without_labels_record_nothing(tmp_path):
    recs = settings_run()
    for r in recs:
        r["observation"].pop("screen", None)
        r["observation"].get("perception", {})["screen"] = "failed"
    st = TwinState(None, stats=RecordStats())
    replay_run(st, write_run(tmp_path, "r0", recs))
    assert st.screens_of(SET) == [] and st.stats.unlabeled == 4 and st.stats.label_failed == 4


def test_same_name_without_same_as_is_a_second_screen(tmp_path):
    st = TwinState(None, stats=RecordStats())
    replay_run(st, write_run(tmp_path, "r0", settings_run()))
    replay_run(st, write_run(tmp_path, "r1", settings_run(t0=100)))       # 模型一次都没挑中候选
    assert len(st.screens_of(SET)) == 6 and st.stats.duplicate_names == 3


def test_label_naming_an_unknown_app_is_unknown(tmp_path):
    f1 = screen_obs("f1.png", "聊天", rows=("张三", "李四", "王五"), app="微信")
    recs = [step(1, f1, "tap", {"id": id_of(f1, "张三")}, after="f2.png"),
            step(2, screen_obs("f2.png", "张三", rows=("你好", "在吗", "好的"), app="微信"), "done", {})]
    st = TwinState(None, stats=RecordStats())
    replay_run(st, write_run(tmp_path, "r0", recs))
    assert st.stats.owner_unknown_app == 2 and st.all_apps() == []


def test_label_of_a_layout_app_is_accepted(tmp_path):
    f1 = screen_obs("f1.png", "聊天", rows=("张三", "李四", "王五"), app="微信")
    st = TwinState(None, stats=RecordStats(), known_apps=frozenset({app_id_for("微信")}))
    replay_run(st, write_run(tmp_path, "r0", [step(1, f1, "done", {})]))
    assert names(st, app_id_for("微信")) == {"聊天"} and st.stats.owner_from_label == 1


def test_unsure_app_with_same_as_takes_the_candidate_app(tmp_path):
    st = TwinState(None, stats=RecordStats())
    replay_run(st, write_run(tmp_path, "r0", settings_run()))
    f = screen_obs("g1.png", "通用", "设置", ("关于本机", "软件更新", "存储"), app="不确定", same_as=1,
                   candidates=(cand(1, SET, sid_of(SET, "r0", "f3.png"), "通用"),))
    replay_run(st, write_run(tmp_path, "r1", [step(1, f, "done", {})]))
    assert st.stats.owner_from_same_as == 1
    assert st.get(SET, sid_of(SET, "r0", "f3.png")).visits == 2


def test_verified_entry_learns_the_app_name_alias_and_later_frames_follow_it(tmp_path):
    aid = app_id_for("App Store")
    f1 = home_obs("f1.png")
    f2 = screen_obs("f2.png", "今天", rows=("游戏", "App", "搜索"), app="应用商店")
    f3 = screen_obs("f3.png", "游戏", rows=("热门", "新品", "排行"), app="应用商店")
    recs = [step(1, f1, "open_app", {"name": "App Store"}, VERIFIED, after="f2.png"),
            step(2, f2, "tap", {"id": id_of(f2, "游戏")}, after="f3.png"),
            step(3, f3, "done", {"status": "success", "result": "x"})]
    ws = Workspace(tmp_path)
    record_run(ws, write_run(tmp_path, "r1", recs))
    assert {s.name for s in TwinState(ws.twin_apps).screens_of(aid)} == {"今天", "游戏"}
    meta = json.loads((ws.twin_apps / aid / "screens" / "app.json").read_text(encoding="utf-8"))
    assert meta["aliases"] == ["应用商店"] and meta["display"]


def test_one_label_name_aliased_to_two_apps_is_unknown(tmp_path):
    st = TwinState(None, stats=RecordStats())
    st.learn_alias("商店", "app-a")
    st.learn_alias("商店", "app-b")
    assert st.resolve_app("商店") == "unknown" and st.stats.app_alias_conflict == 1


def test_unsure_label_after_a_verified_entry_builds_no_screen(tmp_path):
    """spec §4.2 最后一行：核过身份进了设置，但这一帧模型说「不确定」、又没挑候选 → 不建屏。"""
    f2 = screen_obs("f2.png", "设置", None, ("通用", "隐私", "电池"), app="不确定")
    recs = [step(1, home_obs("f1.png"), "open_app", {"name": "设置"}, VERIFIED, after="f2.png"),
            step(2, f2, "done", {})]
    st = mem()
    replay_run(st, write_run(tmp_path, "r0", recs))
    assert st.screens_of(SET) == [] and st.stats.new == 0 and st.stats.skipped == 2     # f1 主屏 + f2
    assert st.stats.owner_from_entry == 1


def test_a_known_app_name_cannot_become_another_apps_alias():
    st = TwinState(None, stats=RecordStats(), known_apps=frozenset({SET}))
    st.learn_alias("设置", "bei-wang-lu")
    assert st.stats.app_aliases_added == 0 and st.stats.app_alias_conflict == 1
    assert st.resolve_app("设置") == SET and st.stats.app_alias_conflict == 1


def test_known_app_slug_aliased_elsewhere_resolves_to_unknown():
    st = mem()
    st.learn_alias("设置", "bei-wang-lu")                     # 学的时候设置还不认识
    st.create(SET, "r:f.png", "设置", 0.0)                    # 之后设置自己成了已知 App
    assert st.resolve_app("设置") == "unknown" and st.stats.app_alias_conflict == 1


def test_one_mislabelled_entry_cannot_hijack_a_real_app_name(tmp_path):
    """评审 probe B：进备忘录（核过身份）那一帧模型把 App 标成「设置」→ 学到 设置→备忘录。
    之后正常的设置 run 不能把「通用」「关于本机」记到备忘录名下。"""
    ws = Workspace(tmp_path)
    notes = app_id_for("备忘录")
    n2 = screen_obs("f2.png", "备忘录", rows=("购物清单", "会议记录", "读书笔记"), app="设置")
    r1 = [step(1, home_obs("f1.png"), "open_app", {"name": "备忘录"}, VERIFIED, after="f2.png"),
          step(2, n2, "done", {})]
    record_run(ws, write_run(tmp_path, "r1", r1))
    stats = record_run(ws, write_run(tmp_path, "r2", settings_run(t0=100)))
    assert not names(TwinState(ws.twin_apps), notes) & {"通用", "关于本机"}
    assert stats.app_alias_conflict == 2                        # r2 的 f3、f4 各一次（每帧只认一次）
    sim = simulate(ws)
    assert all(owner != notes for k, (owner, _, _) in sim.by_event.items() if k.startswith("r2"))


def test_verified_entry_frame_mislabelled_as_the_app_just_left_is_one_screen_of_the_entered_app(tmp_path):
    """评审 P5：在设置里 open_app 备忘录（核过身份），入口帧「文件夹」被标成「设置」。
    这一帧在 apply 里作为 after 认过一次（取走了核过身份的进入），又作为下一个事件的 before 再认一次，
    第二次退到标注 → 一帧两屏（备忘录下 0 次访问的幻影 + 设置下的「文件夹」），之后在备忘录里的
    点击记成 设置 … left_app —— 进入核过身份也照样错归（闸门 B）。"""
    notes = app_id_for("备忘录")
    f2 = screen_obs("f2.png", "设置", None, ("通用", "隐私", "电池"))
    f3 = screen_obs("f3.png", "文件夹", None, ("备忘录", "最近删除", "共享"), app="设置")
    f4 = screen_obs("f4.png", "备忘录", "文件夹", ("购物清单", "会议记录", "读书笔记"), app="备忘录")
    recs = [step(1, home_obs("f1.png"), "open_app", {"name": "设置"}, VERIFIED, after="f2.png"),
            step(2, f2, "open_app", {"name": "备忘录"}, VERIFIED, after="f3.png"),
            step(3, f3, "tap", {"id": id_of(f3, "备忘录")}, after="f4.png"),
            step(4, f4, "done", {"status": "success"})]
    st = TwinState(None, stats=RecordStats(), known_apps=frozenset({SET, notes}))    # 两个都在主屏布局表里
    replay_run(st, write_run(tmp_path, "r1", recs))
    assert names(st) == {"设置"} and names(st, notes) == {"文件夹", "备忘录"}
    folder = by_name(st, notes)["文件夹"]
    assert folder.id == sid_of(notes, "r1", "f3.png") and folder.visits == 1
    (t,) = folder.transitions.values()
    assert (t.target, t.effect, t.to) == ("备忘录", "navigated", by_name(st, notes)["备忘录"].id)
    assert st.stats.screens_created == 3


def test_system_labelled_dialog_after_a_verified_entry_builds_no_screen_and_no_edge(tmp_path):
    """spec §4.2 最后一行、评审 P1：open_app 设置（核过身份）后第一帧是权限弹窗、标「系统」→
    不建屏、不记转移（旧代码建出一屏 设置「允许访问位置」，还进了【位置】）。"""
    dlg = screen_obs("f2.png", "允许访问位置", None, ("允许", "不允许", "仅一次"), app="系统")
    f3 = screen_obs("f3.png", "设置", None, ("通用", "隐私", "电池"))
    recs = [step(1, home_obs("f1.png"), "open_app", {"name": "设置"}, VERIFIED, after="f2.png"),
            step(2, dlg, "tap", {"id": id_of(dlg, "允许")}, after="f3.png"),
            step(3, f3, "done", {"status": "success"})]
    st = TwinState(None, stats=RecordStats(), known_apps=frozenset({SET}))
    replay_run(st, write_run(tmp_path, "r1", recs))
    assert names(st) == {"设置"} and all(not s.transitions for s in st.screens_of(SET))
    assert st.stats.owner_from_entry == 1


def test_entered_name_missing_from_the_layout_becomes_an_alias_of_the_labelled_layout_app(tmp_path):
    """评审 P2（spec §5.1 第 1 条、§5.2）：布局表里叫「App Store」，模型 open_app("应用商店")（核过身份），
    各帧标「App Store」。进入的名字不是已知 App、标注的是 → 以标注的 App 为准，把 open_app 的名字
    学成它的别名。旧代码：入口帧归 ying-yong-shang-dian、之后归 app-store，点「游戏」记成 left_app。"""
    aid = app_id_for("App Store")
    f2 = screen_obs("f2.png", "今天", rows=("游戏", "App", "搜索"), app="App Store")
    f3 = screen_obs("f3.png", "游戏", rows=("热门", "新品", "排行"), app="App Store")
    recs = [step(1, home_obs("f1.png"), "open_app", {"name": "应用商店"}, VERIFIED, after="f2.png"),
            step(2, f2, "tap", {"id": id_of(f2, "游戏")}, after="f3.png"),
            step(3, f3, "done", {"status": "success", "result": "x"})]
    st = TwinState(None, stats=RecordStats(), known_apps=frozenset({aid}))
    replay_run(st, write_run(tmp_path, "r1", recs))
    assert st.all_apps() == [aid] and names(st, aid) == {"今天", "游戏"}
    edges = [t for s in st.screens_of(aid) for t in s.transitions.values()]
    assert [(t.target, t.effect) for t in edges] == [("游戏", "navigated")]
    assert st.resolve_app("应用商店") == aid
    assert (st.stats.app_aliases_added, st.stats.app_alias_conflict, st.stats.owner_from_entry) == (1, 0, 1)
    assert st.dirty_meta == {aid}, "进入的名字不该另长一份 app.json"


def _live_stats(recs) -> dict:
    """照主循环喂实时孪生：第一帧 observe；每个动作 after_action，有新帧就再 observe 它（push_obs）。
    done 在主循环里不走 _twin_after，这里也不喂。"""
    from iphone_agent.twin.events import Snapshot
    from iphone_agent.twin.live import LiveTwin
    live = LiveTwin(None, "r1")
    live.observe(Snapshot.from_observation(recs[0]["observation"]))
    for r in recs:
        if r["action"]["name"] == "done":
            break
        after = Snapshot.from_observation(r.get("after_observation"))
        live.after_action(r["action"], r["result"], after, r["ts"])
        if after is not None:
            live.observe(after)
    return live.summary()


def _replay_stats(tmp_path, recs) -> dict:
    st = mem()
    replay_run(st, write_run(tmp_path, "r1", recs))
    return st.stats.to_dict()


def _ends_on_a_tap(tap_args):
    """主屏 → open_app 设置 → 在「设置」上点一下，任务就此结束（max_steps / 停止 / 出错）：最后一帧只有 after_observation。"""
    f1, f2 = home_obs("f1.png"), screen_obs("f2.png", "设置", None, ("通用", "隐私", "电池"))
    f3 = screen_obs("f3.png", "通用", "设置", ("关于本机", "软件更新", "存储"))
    return [step(1, f1, "open_app", {"name": "设置"}, VERIFIED, after="f2.png", after_observation=f2),
            step(2, f2, "tap", tap_args, after="f3.png", after_observation=f3)]


def test_replay_recognizes_the_last_after_frame_like_the_live_twin(tmp_path):
    """⚠ 2026-09-15（spec 按需看图 §8.1）：按坐标点击时 apply 不认 after 帧，最后一个动作之后又没有下一个事件；
    实时在 push_obs 里认过这一帧，重放也要认一次，两边计数一致。"""
    recs = _ends_on_a_tap({"x": 10, "y": 10})
    live, replay = _live_stats(recs), _replay_stats(tmp_path, recs)
    assert replay["labeled"] == 3, "三帧都带标注，最后那一帧也算"
    assert {k: v for k, v in replay.items() if k != "runs"} == {k: v for k, v in live.items() if k != "runs"}


def test_replay_does_not_count_the_last_after_frame_twice_when_apply_already_recognized_it(tmp_path):
    recs = _ends_on_a_tap({"id": 1})             # 点「通用」：apply 认过 after 帧、建了屏
    live, replay = _live_stats(recs), _replay_stats(tmp_path, recs)
    assert replay["labeled"] == 3 and replay["new"] >= 1
    assert {k: v for k, v in replay.items() if k != "runs"} == {k: v for k, v in live.items() if k != "runs"}
