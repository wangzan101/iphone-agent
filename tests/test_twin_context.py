"""本次任务的认屏上下文（spec 2026-09-12 §3.3）。"""
from iphone_agent.perceive.screen import Candidate
from iphone_agent.twin.context import SCREEN_CANDIDATES_MAX, ScreenIdentityContext
from iphone_agent.twin.events import read_run_events
from iphone_agent.twin.live import LiveTwin
from iphone_agent.twin.record import RecordStats, TwinState, replay_run
from iphone_agent.twin.screenfile import Screen
from tests.twin_fixtures import SET, settings_run, sid_of, write_run


def live_with(*screens):
    lt = LiveTwin(None, "r9")
    for s in screens:
        lt.state.screens_of(s.app)
        lt.state._apps[s.app][s.id] = s
    return lt


def scr(app, sid, name, anchors):
    return Screen(sid, app, "k", "t", "t", name, anchors=dict.fromkeys(anchors, 1))


def test_candidates_are_screens_sharing_an_anchor_ranked_by_hits_then_app_then_id():
    lt = live_with(scr("b", "s_2", "B2", ["文件夹"]), scr("a", "s_1", "A1", ["文件夹", "编辑"]),
                   scr("a", "s_0", "A0", ["文件夹"]), scr("c", "s_9", "无关", ["别的"]))
    got = ScreenIdentityContext(lt).candidates(["文件夹", "编辑 ", "iCloud"])
    assert [(c.index, c.app_id, c.screen_id) for c in got] == [(1, "a", "s_1"), (2, "a", "s_0"), (3, "b", "s_2")]
    assert all(isinstance(c, Candidate) for c in got)


def test_candidates_are_capped():
    lt = live_with(*[scr("a", f"s_{i:02d}", f"屏{i}", ["共用"]) for i in range(SCREEN_CANDIDATES_MAX + 5)])
    assert len(ScreenIdentityContext(lt).candidates(["共用"])) == SCREEN_CANDIDATES_MAX


def test_resolve_and_learn_go_through_the_live_state():
    lt = LiveTwin(None, "r9")
    ctx = ScreenIdentityContext(lt)
    assert ctx.resolve_app("应用商店") == "ying-yong-shang-dian"
    ctx.learn_app_alias("应用商店", "app-store")
    assert ctx.resolve_app("应用商店") == "app-store" and lt.state.stats.app_aliases_added == 1


def test_live_ids_equal_replay_ids(tmp_path):
    d0 = write_run(tmp_path, "r0", settings_run())
    d1 = write_run(tmp_path, "r1", settings_run(t0=100, first="r0"))
    replayed = TwinState(None, stats=RecordStats())
    replay_run(replayed, d0)
    replay_run(replayed, d1)
    live_state = TwinState(None, stats=RecordStats())
    replay_run(live_state, d0)
    live = LiveTwin(None, "r1")
    live.state = live_state
    live.replayer.state = live_state
    live_state.begin_run("r1")
    for ev in read_run_events(d1):
        live.observe(ev.before)
        live.after_action(ev.action, ev.result, ev.after, ev.ts)
    assert {s.id for s in live_state.screens_of(SET)} == {s.id for s in replayed.screens_of(SET)} \
        == {sid_of(SET, "r0", f) for f in ("f2.png", "f3.png", "f4.png")}
    assert live_state.stats.candidate_unresolved == 0
