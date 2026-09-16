"""写盘与重建（spec §4.4、§4.5）：唯一写入口、崩溃可重跑、重建幂等且保住人写的。"""
import copy
import json
import multiprocessing as mp

import pytest

from iphone_agent.twin import record as R
from iphone_agent.workspace import Workspace
from tests.twin_fixtures import (
    VERIFIED,
    home_obs,
    id_of,
    screen_obs,
    settings_run,
    step,
    tree_bytes,
    write_run,
)


def notes_run(t0=0.0):
    f2 = screen_obs("f2.png", "备忘录", rows=("购物清单", "会议记录", "读书笔记"), app="备忘录")
    recs = [step(1, home_obs("f1.png"), "open_app", {"name": "备忘录"}, VERIFIED, after="f2.png"),
            step(2, f2, "done", {"status": "success"})]
    for r in recs:
        r["ts"] += t0
    return recs


def test_record_run_writes_screens_and_is_idempotent(tmp_path):
    ws = Workspace(tmp_path)
    d = write_run(tmp_path, "r1", settings_run())
    st = R.record_run(ws, d)
    files = tree_bytes(ws.twin_apps)
    assert len([k for k in files if k.startswith("she-zhi/screens/s_")]) == 3 and st.screens_created == 3
    assert "she-zhi/screens/app.json" in files
    R.record_run(ws, d)
    assert tree_bytes(ws.twin_apps) == files


def test_unfinished_run_is_not_recorded_until_it_finishes(tmp_path):
    ws = Workspace(tmp_path)
    d = write_run(tmp_path, "r1", settings_run(), finished=False)
    assert R.record_run(ws, d).unfinished == 1 and not ws.twin_apps.exists()
    (d / "run.json").write_text(json.dumps({"end_reason": "done_success"}), encoding="utf-8")
    assert R.record_run(ws, d).screens_created == 3


def test_rebuild_twice_is_byte_identical_and_equals_incremental(tmp_path):
    ws = Workspace(tmp_path)
    d1 = write_run(tmp_path, "r1", settings_run())
    d2 = write_run(tmp_path, "r2", settings_run(t0=100, first="r1"))
    R.record_run(ws, d1)
    R.record_run(ws, d2)
    incremental = tree_bytes(ws.twin_apps)
    R.rebuild(ws)
    first = tree_bytes(ws.twin_apps)
    R.rebuild(ws)
    assert tree_bytes(ws.twin_apps) == first == incremental


def test_rebuild_twice_is_byte_identical_including_app_json_aliases(tmp_path):
    """spec §8.4 机制第 4 条：屏文件和带别名的 app.json 重建两遍逐字节相同，并等于增量记账。"""
    ws = Workspace(tmp_path)
    f2 = screen_obs("f2.png", "今天", rows=("游戏", "App", "搜索"), app="应用商店")
    f3 = screen_obs("f3.png", "游戏", rows=("热门", "新品", "排行"), app="应用商店")
    store = [step(1, home_obs("f1.png"), "open_app", {"name": "App Store"}, VERIFIED, after="f2.png"),
             step(2, f2, "tap", {"id": id_of(f2, "游戏")}, after="f3.png"),
             step(3, f3, "done", {"status": "success"})]
    R.record_run(ws, write_run(tmp_path, "r1", store))
    R.record_run(ws, write_run(tmp_path, "r2", settings_run(t0=100)))
    incremental = tree_bytes(ws.twin_apps)
    assert json.loads(incremental["app-store/screens/app.json"])["aliases"] == ["应用商店"]
    R.rebuild(ws)
    first = tree_bytes(ws.twin_apps)
    R.rebuild(ws)
    assert tree_bytes(ws.twin_apps) == first == incremental


def test_rebuild_keeps_human_fields_and_reports_orphans(tmp_path):
    ws = Workspace(tmp_path)
    write_run(tmp_path, "r1", settings_run())
    notes = write_run(tmp_path, "r2", notes_run(t0=100))
    R.rebuild(ws)
    kept = next(ws.twin_apps.glob("she-zhi/screens/s_*.json"))
    d = json.loads(kept.read_text(encoding="utf-8"))
    d["human"] = {"note": "人写的"}
    kept.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    gone = next(ws.twin_apps.glob("bei-wang-lu/screens/s_*.json"))
    g = json.loads(gone.read_text(encoding="utf-8"))
    g["human"] = {"no_go": True}
    gone.write_text(json.dumps(g, ensure_ascii=False), encoding="utf-8")
    for p in notes.iterdir():
        p.unlink()
    notes.rmdir()
    st = R.rebuild(ws)
    assert json.loads(kept.read_text(encoding="utf-8"))["human"] == {"note": "人写的"}
    orphans = json.loads((tmp_path / ".iphone" / "knowledge" / "twin-orphans.json").read_text(encoding="utf-8"))
    assert orphans[f"bei-wang-lu/{gone.stem}"] == {"no_go": True} and st.orphans == 1


def test_failed_rebuild_leaves_the_old_twin(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    write_run(tmp_path, "r1", settings_run())
    R.rebuild(ws)
    before = tree_bytes(ws.twin_apps)
    monkeypatch.setattr(R, "_replay", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        R.rebuild(ws)
    assert tree_bytes(ws.twin_apps) == before
    assert not (tmp_path / ".iphone" / "knowledge" / ".twin-rebuild").exists()


def test_corrupt_screen_file_is_quarantined_not_overwritten(tmp_path):
    ws = Workspace(tmp_path)
    bad = ws.twin_apps / "she-zhi" / "screens" / "s_bad.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json", encoding="utf-8")
    st = R.record_run(ws, write_run(tmp_path, "r1", settings_run()))
    assert st.corrupt == 1 and not bad.exists()
    assert (bad.parent / ".corrupt" / "s_bad.json").read_text(encoding="utf-8") == "{not json"


@pytest.mark.parametrize("crash_after", [0, 1, 2])
def test_crash_mid_write_then_rerun_equals_clean_run(tmp_path, monkeypatch, crash_after):
    clean = Workspace(tmp_path / "clean")
    R.record_run(clean, write_run(tmp_path / "clean", "r1", settings_run()))
    ws = Workspace(tmp_path / "crash")
    d = write_run(tmp_path / "crash", "r1", settings_run())
    real, n = R.write_json, {"i": 0}

    def flaky(*a, **k):
        if n["i"] == crash_after:
            raise KeyboardInterrupt            # 模拟进程被杀：不是 OSError，flush 接不住
        n["i"] += 1
        return real(*a, **k)
    monkeypatch.setattr(R, "write_json", flaky)
    with pytest.raises(KeyboardInterrupt):
        R.record_run(ws, d)
    monkeypatch.setattr(R, "write_json", real)
    R.record_run(ws, d)
    assert tree_bytes(ws.twin_apps) == tree_bytes(clean.twin_apps)


def test_lock_timeout_queues_the_run_and_next_record_catches_up(tmp_path):
    ws = Workspace(tmp_path)
    d1 = write_run(tmp_path, "r1", settings_run())
    d2 = write_run(tmp_path, "r2", notes_run(t0=100))
    with R.twin_lock(ws):
        st = R.record_run(ws, d1, lock_wait_s=0.1)
    assert st.lock_timeout == 1 and not ws.twin_apps.exists()
    R.record_run(ws, d2)
    assert list(ws.twin_apps.glob("she-zhi/screens/s_*.json")) and list(ws.twin_apps.glob("bei-wang-lu/screens/s_*.json"))


def test_pending_queue_keeps_undrained_ids_on_mid_drain_failure(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    write_run(tmp_path, "r1", settings_run())
    write_run(tmp_path, "r2", notes_run(t0=100))
    d3 = write_run(tmp_path, "r3", settings_run(t0=200, first="r1"))
    R._write_pending(ws, ["r1", "r2"])
    real = R._record_run_locked

    def flaky(ws_, run_dir, stats, state=None):
        if run_dir.name == "r2":
            raise RuntimeError("boom")
        return real(ws_, run_dir, stats, state)
    monkeypatch.setattr(R, "_record_run_locked", flaky)
    with pytest.raises(RuntimeError):
        R.record_run(ws, d3)
    assert R._read_pending(ws) == ["r2"]
    assert list(ws.twin_apps.glob("she-zhi/screens/s_*.json"))          # r1 已经记进去了


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_swap_in_rolls_back_completely_on_any_rename_failure(tmp_path, monkeypatch, k):
    ws = Workspace(tmp_path)
    write_run(tmp_path, "r1", settings_run())
    write_run(tmp_path, "r2", notes_run(t0=100))
    R.rebuild(ws)
    before = tree_bytes(ws.twin_apps)
    write_run(tmp_path, "r3", settings_run(t0=200, first="r1"))          # 换个新 run，重建会真的改文件
    real, n = R._rename, {"i": -1}

    def flaky(src, dst):
        n["i"] += 1
        if n["i"] == k:
            raise OSError("boom")
        return real(src, dst)
    monkeypatch.setattr(R, "_rename", flaky)
    with pytest.raises(OSError):
        R.rebuild(ws)
    assert tree_bytes(ws.twin_apps) == before
    assert not list(ws.twin_apps.glob("*/.screens-old"))


def _record_in_child(root, run_id):
    from iphone_agent.twin import record
    from iphone_agent.workspace import Workspace as W
    record.record_run(W(root), W(root).runs / run_id)


def test_two_processes_recording_at_once_are_serialized(tmp_path):
    write_run(tmp_path, "r1", settings_run())
    write_run(tmp_path, "r2", notes_run(t0=100))
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=_record_in_child, args=(tmp_path, r)) for r in ("r1", "r2")]
    for p in ps:
        p.start()
    for p in ps:
        p.join(60)
    assert all(p.exitcode == 0 for p in ps)
    ws = Workspace(tmp_path)
    assert list(ws.twin_apps.glob("she-zhi/screens/s_*.json"))
    assert list(ws.twin_apps.glob("bei-wang-lu/screens/s_*.json"))
    together = tree_bytes(ws.twin_apps)
    R.rebuild(ws)                                  # 两个 run 碰的是不同 App：顺序无关，必须和重建一致
    assert tree_bytes(ws.twin_apps) == together


def test_needs_initial_rebuild(tmp_path):
    ws = Workspace(tmp_path)
    assert not R.needs_initial_rebuild(ws)
    write_run(tmp_path, "r1", settings_run())
    assert R.needs_initial_rebuild(ws)
    R.rebuild(ws)
    assert not R.needs_initial_rebuild(ws)


def test_schema1_and_future_files_are_skipped_not_quarantined(tmp_path):
    ws = Workspace(tmp_path)
    d = ws.twin_apps / "she-zhi" / "screens"
    d.mkdir(parents=True)
    (d / "s_old.json").write_text(json.dumps({"schema": 1, "id": "s_old"}), encoding="utf-8")
    (d / "s_new.json").write_text(json.dumps({"schema": 3, "id": "s_new"}), encoding="utf-8")
    st = R.TwinState(ws.twin_apps, writable=True)
    assert st.screens_of("she-zhi") == []
    assert st.stats.stale_schema == 1 and st.stats.unsupported_schema == 1 and st.stats.corrupt == 0
    assert (d / "s_old.json").exists() and (d / "s_new.json").exists() and not (d / ".corrupt").exists()


@pytest.mark.parametrize("meta", [{"schema": 3, "app": "she-zhi", "future": True},
                                  {"app": "she-zhi", "display": "设置", "aliases": ["系统设置"]}])
def test_unsupported_app_meta_survives_a_run_that_enters_the_app(tmp_path, meta):
    """spec §7：app.json 同屏文件一样，不支持的 schema 不隔离、也不覆盖（可能是更新的程序写的）。"""
    ws = Workspace(tmp_path)
    d = ws.twin_apps / "she-zhi" / "screens"
    d.mkdir(parents=True)
    raw = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    (d / "app.json").write_bytes(raw)
    stats = R.record_run(ws, write_run(tmp_path, "r1", settings_run()))
    assert len(list(d.glob("s_*.json"))) == 3                  # 确实进了设置、记了屏
    assert (d / "app.json").read_bytes() == raw
    assert stats.unsupported_schema >= 1 and stats.corrupt == 0 and not (d / ".corrupt").exists()


def test_rebuild_marker_is_v2(tmp_path):
    ws = Workspace(tmp_path)
    write_run(tmp_path, "r1", settings_run())
    (ws.dot / "knowledge").mkdir(parents=True)
    (ws.dot / "knowledge" / ".twin-rebuilt").touch()          # 旧标记不算数：升级后要重建一次
    assert R.needs_initial_rebuild(ws)
    R.rebuild(ws)
    assert (ws.dot / "knowledge" / ".twin-rebuilt-v2").exists() and not R.needs_initial_rebuild(ws)


def test_replay_does_not_depend_on_where_the_label_came_from(tmp_path):
    """不变式守卫（spec 2026-09-14 §10.1 孪生回归）：标注来自短标注还是整屏解析，重放结论一致；重建两遍逐字节相同。

    重放本来就不读 perception 里的 vision / parse / label_by，所以这条一开始就该绿 —— 它守的是
    「以后别让重放按标注出处分叉」，不是本次改动的失败用例（那条在 test_loop_adopt 的最后一帧一致）。
    """
    def as_short_label(recs):
        out = copy.deepcopy(recs)
        for r in out:
            p = r["observation"].get("perception")
            if p:
                p.update({"vision": "off", "parse": None, "label_by": "label"})
        return out
    a, b = Workspace(tmp_path / "a"), Workspace(tmp_path / "b")
    R.record_run(a, write_run(tmp_path / "a", "r1", settings_run()))
    R.record_run(b, write_run(tmp_path / "b", "r1", as_short_label(settings_run())))
    assert tree_bytes(a.twin_apps) and tree_bytes(a.twin_apps) == tree_bytes(b.twin_apps)
    R.rebuild(b)
    first = tree_bytes(b.twin_apps)
    R.rebuild(b)
    assert tree_bytes(b.twin_apps) == first
