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
    return st, {s.name: s for s in st.screens_of("she-zhi")}


def test_position_lists_edges_and_keeps_the_trust_boundary(tmp_path):
    st, s = built(tmp_path)
    text = render.position(st, "she-zhi", Recognition("matched", s["设置"].id), "设置")
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
    text = render.position(st, "she-zhi", Recognition("matched", scr.id), "设置")
    assert "点「存储」没反应（5 次）" in text and "按「home」会离开这个 App（4 次）" in text
    assert "隐私" not in text and "电池" not in text
    assert "（结果不稳定）" in text


def test_position_notes_provisional(tmp_path):
    st, s = built(tmp_path, n=1)
    assert "这屏以前只见过一次" in render.position(st, "she-zhi", Recognition("matched", s["通用"].id), "设置")


def test_position_new_and_silent_states(tmp_path):
    st, _ = built(tmp_path)
    assert render.position(st, "she-zhi", Recognition("new"), "设置") == render.NEW_SCREEN
    for state in ("ambiguous", "perception_failed", "skipped"):
        assert render.position(st, "she-zhi", Recognition(state), "设置") is None


def test_route_walks_navigated_edges_by_normalized_task_text(tmp_path):
    st, s = built(tmp_path)
    text = render.route(st, "she-zhi", Recognition("matched", s["设置"].id), "进入 关于本机 读一下版本")
    assert "从当前这屏走过去是 2 步：点「通用」 → 点「关于本机」" in text
    assert render.route(st, "she-zhi", Recognition("matched", s["关于本机"].id), "关于本机") \
        == "【路线】你已经在任务提到的「关于本机」那一屏上了。"
    assert render.route(st, "she-zhi", Recognition("matched", s["设置"].id), "看看天气") is None


def test_route_skips_unstable_edges(tmp_path):
    st, s = built(tmp_path)
    for t in s["通用"].transitions.values():
        t.off_track = t.count
    assert render.route(st, "she-zhi", Recognition("matched", s["设置"].id), "关于本机") is None


def _snap(obs):
    return Snapshot.from_observation(obs)


def canon(state, app="she-zhi"):
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
    live = LiveTwin(None, "r1", known_apps=frozenset({"she-zhi"}))
    live.observe(evs[0].before)
    live.after_action(evs[0].action, evs[0].result, evs[0].after, evs[0].ts)
    assert live.observe(evs[1].before)["owner"] == "she-zhi"
    live.after_action({"name": "proc_read_version", "args": {}}, {"ok": True}, evs[2].before, 1.0)
    # 剧本之后这一帧带着标注（App 是已知 App），规则 2（标注）优先于动作推断（tracker → unknown）。
    assert live.observe(evs[2].before)["owner"] == "she-zhi"

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

    # open_app 设置（核过身份）：这一步的归属推进（system → she-zhi）靠的是刚才重新
    # 武装的 _cur，不是一次新的 observe 调用。
    assert live.after_action(evs[0].action, evs[0].result, evs[0].after, evs[0].ts) is None
    assert live.observe(evs[0].after)["owner"] == "she-zhi"

    # 剩下的事件按正常节奏（observe 再 after_action）跑完，结构要跟纯重放同构——
    # canon 只看 she-zhi 这个 App，recall 那一步落在 system 上，不影响比较。
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


def test_route_matches_aliases_that_belong_to_one_screen_only(tmp_path):
    st, s = built(tmp_path)
    s["关于本机"].aliases = {"本机信息"}
    s["通用"].aliases = {"共用别名"}
    s["设置"].aliases = {"共用别名"}
    assert "点「通用」 → 点「关于本机」" in render.route(st, "she-zhi", Recognition("matched", s["设置"].id), "看看本机信息")
    assert render.route(st, "she-zhi", Recognition("matched", s["关于本机"].id), "共用别名") is None
