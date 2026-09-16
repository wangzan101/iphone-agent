"""按需看图的真机 A/B 装配（spec 2026-09-14 §10.3）：只测装配，不碰设备。"""
import json
from pathlib import Path
from types import SimpleNamespace

from iphone_agent.eval import ab
from iphone_agent.eval.verify import Task
from iphone_agent.workspace import Workspace
from tests._resolved import fake_resolved


def _task(tid, prompt, answer=True):
    checks = [{"type": "screen_reached", "texts": ["x"]}]
    if answer:
        checks.append({"type": "answer_contains", "any_of": ["y"]})
    return Task.from_dict({"id": tid, "prompt": prompt, "strength": "medium", "checks": checks})


def test_run_ab_follows_ABBAAB_pins_the_config_and_adds_a_no_coord_layer_for_hard_tasks(tmp_path):
    calls = []

    def runner(prompt, dev, per, model, runs_root, **kw):
        rc = kw["run_config"]
        calls.append((prompt, rc.screen_parse, model.resolved.model.allow_coord_tap,
                      rc.memory, rc.twin_hints, kw["confirm"]))
        d = Path(runs_root) / f"r{len(calls)}"
        d.mkdir(parents=True)
        (d / "run.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(run_dir=d)
    session = SimpleNamespace(workspace=Workspace(tmp_path / "src"), dev=None, per=None,
                              model=SimpleNamespace(resolved=fake_resolved(allow_coord_tap=True),
                                                    decide=lambda *a, **k: None))
    res = ab.run_ab(session, [_task("r1", "p1", answer=False)], [_task("h1", "p2")], 3, tmp_path / "ws",
                    runner=runner, verifier=lambda t, d: SimpleNamespace(status="pass", reasons=[]),
                    log=lambda *_: None)
    assert [c[1] for c in calls if c[0] == "p1"] == ["always", "on_demand", "on_demand", "always", "always", "on_demand"]
    assert {(r["task"], r["layer"]) for r in res["runs"]} == {("r1", "as_is"), ("h1", "as_is"), ("h1", "no_coord")}
    assert sorted({c[2] for c in calls if c[0] == "p2"}) == [False, True]
    assert all(c[3] is False and c[4] is True and c[5] is None for c in calls)
    assert session.model.resolved.model.allow_coord_tap is True, "不改用户的模型档案"


def test_new_hard_tasks_must_have_an_answer_check_before_the_ab():
    assert ab.ready(_task("h", "p")) is None
    assert "answer_contains" in ab.ready(_task("h", "p", answer=False))


def test_run_metrics_reads_timing_perception_and_taps(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "run.json").write_text(json.dumps({
        "steps": 3, "startup_timing": {"step_ms": 900},
        "perception": {"fallback_by": {"done_failed": 1, "no_progress": 0}, "parse_by": {"model": 1},
                       "label": {"adopt": {"ok": 2}}},
        "taps": {"by_id": 1, "by_coord": 1, "coord_unclassified": 1}}), encoding="utf-8")
    recs = [{"step": 1, "action": {"name": "observe"}, "timing": {"step_ms": 30000}},
            {"step": 2, "action": {"name": "zoom"}, "timing": {"step_ms": 4000}},
            {"step": 3, "action": {"name": "tap"}, "validation": "stale_element_id", "timing": {"step_ms": 8000}},
            {"step": 3, "by": "program", "action": {"name": "observe"}}]
    (d / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    m = ab.run_metrics(d)
    assert (m["steps"], m["step_ms_p50"], m["step_ms_p90"]) == (3, 8000, 30000)
    assert (m["observe"], m["zoom"], m["stale_element_id"]) == (1, 1, 1), "程序发的 observe 不算模型的"
    assert m["taps"]["coord_unclassified"] == 1 and m["fallback_by"]["done_failed"] == 1
