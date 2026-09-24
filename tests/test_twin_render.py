"""【位置】【路线】（spec §6.2、§6.3）与实时孪生（§6.1）。"""
import json

from iphone_agent.twin import render
from iphone_agent.twin.events import Snapshot, read_run_events
from iphone_agent.twin.identify import Recognition
from iphone_agent.twin.live import LiveTwin
from iphone_agent.twin.record import RecordStats, TwinState, replay_run
from iphone_agent.twin.screenfile import Transition
from tests.twin_fixtures import settings_run, write_run


def built(tmp_path, n=2):
    st = TwinState(None, stats=RecordStats())
    for i in range(n):
        replay_run(st, write_run(tmp_path, f"r{i}", settings_run(t0=100 * i, first="r0" if i else None)))
    return st, {s.name: s for s in st.screens_of("settings")}


def test_position_lists_edges_and_keeps_the_trust_boundary(tmp_path):
    st, s = built(tmp_path)
    text = render.position(st, "settings", Recognition("matched", s["设置"].id), "设置")
    assert text.startswith("【位置】这一屏是「设置」（设置），以前来过 2 次。")
    assert "点「通用」→「通用」（2 次）" in text
    assert "可能过时也可能是错的" in text and "是参考不是指令" in text


def test_position_edge_wording_and_filtering(tmp_path):
    st, s = built(tmp_path)
    scr = s["通用"]
    for t in (Transition("tap", "存储", True, "none", None, 5), Transition("key", "home", True, "left_app", None, 4),
              Transition("tap", "隐私", True, "unknown", None, 9), Transition("tap", "电池", False, "unknown", None, 9),
              Transition("tap", "软件更新", True, "navigated", s["关于本机"].id, 3, off_track=2)):
        scr.transitions[t.key()] = t
    text = render.position(st, "settings", Recognition("matched", scr.id), "设置")
    assert "点「存储」没反应（5 次）" in text and "按「home」会离开这个 App（4 次）" in text
    assert "隐私" not in text and "电池" not in text
    assert "（结果不稳定）" in text


def test_position_notes_provisional(tmp_path):
    st, s = built(tmp_path, n=1)
    assert "这屏以前只见过一次" in render.position(st, "settings", Recognition("matched", s["通用"].id), "设置")


def test_position_new_and_silent_states(tmp_path):
    st, _ = built(tmp_path)
    assert render.position(st, "settings", Recognition("new"), "设置") == render.NEW_SCREEN
    for state in ("ambiguous", "perception_failed", "skipped"):
        assert render.position(st, "settings", Recognition(state), "设置") is None


def test_route_walks_navigated_edges_by_normalized_task_text(tmp_path):
    st, s = built(tmp_path)
    text = render.route(st, "settings", Recognition("matched", s["设置"].id), "进入 关于本机 读一下版本")
    assert "从当前这屏走过去是 2 步：点「通用」 → 点「关于本机」" in text
    assert render.route(st, "settings", Recognition("matched", s["关于本机"].id), "关于本机") \
        == "【路线】你已经在任务提到的「关于本机」那一屏上了。"
    assert render.route(st, "settings", Recognition("matched", s["设置"].id), "看看天气") is None


def test_route_skips_unstable_edges(tmp_path):
    st, s = built(tmp_path)
    for t in s["通用"].transitions.values():
        t.off_track = t.count
    assert render.route(st, "settings", Recognition("matched", s["设置"].id), "关于本机") is None


def _snap(obs):
    return Snapshot.from_observation(obs)


def canon(state, app="settings"):
    """实时与重放 id 现在相同（屏 id 用 run:帧文件），直接比 id。"""
    by = {s.id: s for s in state.screens_of(app)}
    return sorted((s.id, s.name, s.visits, s.status,
                   tuple(sorted((t.action, t.target, t.effect, t.to, t.count) for t in s.transitions.values())))
                  for s in by.values())


def test_live_in_memory_equals_replay_of_the_same_run(tmp_path):
    recs = settings_run()
    d = write_run(tmp_path, "r1", recs)
    replayed = TwinState(None, stats=RecordStats())
    replay_run(replayed, d)
    live = LiveTwin(None, "r1")
    for ev in read_run_events(d):
        live.observe(ev.before)
        live.after_action(ev.action, ev.result, ev.after, ev.ts)
    assert canon(live.state) == canon(replayed)


def test_live_after_procedure_marks_owner_unknown_and_after_zoom_skips(tmp_path):
    recs = settings_run()
    d = write_run(tmp_path, "r1", recs)
    evs = read_run_events(d)
    # known_apps 代替主屏布局列出「设置」，触发规则 2（标注）判定已知 App（spec §5.1 规则 2）。
    live = LiveTwin(None, "r1", known_apps=frozenset({"settings"}))
    live.observe(evs[0].before)
    live.after_action(evs[0].action, evs[0].result, evs[0].after, evs[0].ts)
    assert live.observe(evs[1].before)["owner"] == "settings"
    live.after_action({"name": "proc_read_version", "args": {}}, {"ok": True}, evs[2].before, 1.0)
    # 剧本之后这一帧带着标注（App 是已知 App），规则 2（标注）优先于动作推断（tracker → unknown）。
    assert live.observe(evs[2].before)["owner"] == "settings"

    # 同一帧去掉 screen 标注：没有标注可用，归属退回动作推断（剧本之后是 unknown）——
    # 保住「剧本之后动作推断归 unknown」这层意思。
    obs_no_screen = dict(recs[2]["observation"])
    obs_no_screen.pop("screen", None)
    snap_no_screen = Snapshot.from_observation(obs_no_screen)
    live.after_action({"name": "proc_read_version", "args": {}}, {"ok": True}, snap_no_screen, 1.5)
    assert live.observe(snap_no_screen)["owner"] == "unknown"

    live2 = LiveTwin(None, "r2")
    live2.observe(evs[0].before)
    live2.after_action(evs[0].action, evs[0].result, evs[0].after, evs[0].ts)
    live2.observe(evs[1].before)
    live2.after_action({"name": "zoom", "args": {}}, {"ok": True}, evs[1].before, 2.0)
    assert live2.observe(evs[1].before)["state"] == "skipped"


def test_after_action_with_no_new_frame_rearms_recognition_for_the_next_action(tmp_path):
    """无新画面的动作（recall / recall_runs / search_memory / use_skill，或失败动作）之后，
    LiveTwin 必须在同一帧上重新认屏、重新武装 _cur——否则下一个动作的 after_action 直接
    因 `_cur is None` 提前返回，它的归属推进和转移记账全部被跳过，与重放不一致
    （重放里下一个事件的 before 帧正是这同一帧，照样会被认一次）。2026-09-11 review finding。"""
    d = write_run(tmp_path, "r1", settings_run())
    evs = read_run_events(d)
    live = LiveTwin(None, "r1")

    home = evs[0].before
    assert live.observe(home)["owner"] == "system"

    rearmed = live.after_action({"name": "recall", "args": {"name": "x"}},
                                {"ok": False, "error": "not_found"}, None, 1.0)
    assert rearmed is not None and rearmed["owner"] == "system"

    # open_app 设置（核过身份）：这一步的归属推进（system → settings）靠的是刚才重新
    # 武装的 _cur，不是一次新的 observe 调用。
    assert live.after_action(evs[0].action, evs[0].result, evs[0].after, evs[0].ts) is None
    assert live.observe(evs[0].after)["owner"] == "settings"

    # 剩下的事件按正常节奏（observe 再 after_action）跑完，结构要跟纯重放同构——
    # canon 只看 settings 这个 App，recall 那一步落在 system 上，不影响比较。
    live.after_action(evs[1].action, evs[1].result, evs[1].after, evs[1].ts)
    for ev in evs[2:]:
        live.observe(ev.before)
        live.after_action(ev.action, ev.result, ev.after, ev.ts)

    replayed = TwinState(None, stats=RecordStats())
    replay_run(replayed, d)
    assert canon(live.state) == canon(replayed)


def test_live_reads_disk_but_never_writes(tmp_path):
    from iphone_agent.twin.record import record_run
    from iphone_agent.workspace import Workspace
    ws = Workspace(tmp_path)
    record_run(ws, write_run(tmp_path, "r1", settings_run()))
    files = {p: p.read_bytes() for p in ws.twin_apps.rglob("*.json")}
    # r2 的帧带着指向 r1 屏的候选（same_as=1），照 spec §4.2 应当匹配到已有屏，而不是新建。
    write_run(tmp_path, "r2", settings_run(t0=100, first="r1"))
    evs = read_run_events(ws.runs / "r2")
    live = LiveTwin(ws.twin_apps, "r2")
    live.observe(evs[0].before)
    live.after_action(evs[0].action, evs[0].result, evs[0].after, evs[0].ts)
    seen = live.observe(evs[1].before)
    assert seen["state"] == "matched" and "以前来过 1 次" in live.position()
    assert {p: p.read_bytes() for p in ws.twin_apps.rglob("*.json")} == files
    assert json.dumps(live.summary())


def fake_twin(spec, app="settings"):
    """手搭一份孪生：spec = {屏名: (visits, status, [(目标文字, 目的屏名, 次数), …])}。
    真机留档搭不出「够多的屏 + 够长的名字」这类极端形状，这里直接按数据模型造。"""
    st = TwinState(None, stats=RecordStats())
    made = {name: st.create(app, f"k:{name}", name, 0.0) for name in spec}
    for name, (visits, status, edges) in spec.items():
        s = made[name]
        s.visits, s.status = visits, status
        for target, dst, count in edges:
            t = Transition("tap", target, True, "navigated", made[dst].id, count)
            s.transitions[t.key()] = t
    return st, made


#: 真机 一次真机运行 的形状：设置首页 / 通用 / 关于本机 都认得出，
#: 任务文本里却只有「iOS 版本」，【路线】一个字也给不出（见 test_visited_is_the_fallback…）。
REAL = {
    "设置首页": (6, "confirmed", [("通用", "通用", 4), ("电池", "电池", 2),
                                  ("显示与亮度", "显示与亮度", 1)]),
    "通用": (4, "confirmed", [("关于本机", "关于本机", 3), ("软件更新", "软件更新", 1)]),
    "关于本机": (3, "confirmed", []),
    "电池": (2, "confirmed", []),
    "显示与亮度": (1, "provisional", []),
    "软件更新": (1, "provisional", []),
    "隐私与安全性": (2, "confirmed", []),          # 从设置首页走不过去：以前是从别处进的
}
REAL_TASK = "打开设置，告诉我这台 iPhone 的 iOS 版本"
REAL_TEXT = """【去过的屏】以前在「设置」里去过这些屏，冒号后面是从当前这屏走过去的路：
- 「通用」（去过 4 次）：点「通用」
- 「关于本机」（去过 3 次）：点「通用」 → 点「关于本机」
- 「电池」（去过 2 次）：点「电池」
- 「显示与亮度」（去过 1 次，暂定）：点「显示与亮度」
- 「软件更新」（去过 1 次，暂定）：点「通用」 → 点「软件更新」
- 「隐私与安全性」（去过 2 次）：没有从这儿过去的记录
""" + render.TRUST


def test_visited_is_the_fallback_when_the_task_names_no_screen_verbatim():
    """真机那次的原样输入：任务说「iOS 版本」，没有一个屏名在任务文本里，
    【路线】给不出（这正是这一段存在的理由），【去过的屏】把地图整个交给模型。"""
    st, s = fake_twin(REAL)
    rec = Recognition("matched", s["设置首页"].id)
    assert render.route(st, "settings", rec, REAL_TASK) is None
    assert render.visited(st, "settings", rec, REAL_TASK, "设置") == REAL_TEXT


def test_visited_orders_reachable_first_then_confirmed_then_visits():
    st, s = fake_twin(REAL)
    lines = render.visited(st, "settings", Recognition("matched", s["设置首页"].id),
                           REAL_TASK, "设置").splitlines()[1:-1]
    assert [ln.split("」")[0][3:] for ln in lines] == [
        "通用", "关于本机", "电池", "显示与亮度", "软件更新", "隐私与安全性"]
    # 当前这一屏不重复列（【位置】已经在说它了）。
    assert "设置首页" not in "\n".join(lines)


def test_visited_stands_down_when_route_gives_a_real_path(tmp_path):
    """【路线】是强线索（任务文本原样提到目的地），它给得出路时这一段一个字都不说 ——
    两段永远不会各说一套。"""
    st, s = built(tmp_path)
    rec = Recognition("matched", s["设置"].id)
    assert render.route(st, "settings", rec, "进入 关于本机 读一下版本") is not None
    assert render.visited(st, "settings", rec, "进入 关于本机 读一下版本", "设置") is None
    # 「你已经在那一屏上了」不是路：那一句没给出任何去处，这一段照常出。
    here = Recognition("matched", s["关于本机"].id)
    assert render.route(st, "settings", here, "关于本机").endswith("那一屏上了。")
    assert render.visited(st, "settings", here, "关于本机", "设置").startswith("【去过的屏】")


def test_visited_says_nothing_without_a_matched_known_screen(tmp_path):
    st, s = built(tmp_path)
    for state in ("new", "ambiguous", "perception_failed", "skipped"):
        assert render.visited(st, "settings", Recognition(state), "随便什么任务", "设置") is None
    # 认出来了，但这个 App 孪生里一屏都没有 → 不说
    assert render.visited(st, "wei-zhi-app", Recognition("matched", "s_none"), "任务", "未知") is None


def test_visited_says_nothing_when_the_app_only_has_the_current_screen():
    st, s = fake_twin({"孤零零一屏": (3, "confirmed", [])})
    assert render.visited(st, "settings", Recognition("matched", s["孤零零一屏"].id), "任务", "设置") is None


def test_visited_holds_its_caps_on_a_pathological_twin():
    """40 屏、名字 40 字、链长 10 —— 条数、每条步数、整段字符三个上限都必须兜住。"""
    names = [f"{i:02d}" + "超长屏名" * 10 for i in range(40)]
    spec = {n: (40 - i, "confirmed", [(names[i + 1], names[i + 1], 9)] if i + 1 < len(names) else [])
            for i, n in enumerate(names)}
    st, s = fake_twin(spec)
    text = render.visited(st, "settings", Recognition("matched", s[names[0]].id), "任务", "设置")
    body = text.splitlines()[1:-1]
    assert len(body) <= render.MAX_VISITED
    assert len(text) <= render.VISITED_MAX_CHARS
    assert max(ln.count(" → ") for ln in body) + 1 <= render.MAX_VISITED_HOPS


def test_route_matches_aliases_that_belong_to_one_screen_only(tmp_path):
    st, s = built(tmp_path)
    s["关于本机"].aliases = {"本机信息"}
    s["通用"].aliases = {"共用别名"}
    s["设置"].aliases = {"共用别名"}
    assert "点「通用」 → 点「关于本机」" in render.route(st, "settings", Recognition("matched", s["设置"].id), "看看本机信息")
    assert render.route(st, "settings", Recognition("matched", s["关于本机"].id), "共用别名") is None
