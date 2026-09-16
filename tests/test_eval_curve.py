"""学习曲线评测（spec 2026-09-12 §8）。"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from iphone_agent.eval import curve
from iphone_agent.eval.verify import Task
from iphone_agent.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_abba_order():
    assert [curve.order(i) for i in (1, 2, 3, 4)] == [("on", "off"), ("off", "on"), ("on", "off"), ("off", "on")]


def test_workspaces_are_fresh_and_only_copy_the_layout(tmp_path):
    src = Workspace(tmp_path / "src")
    (src.twin_device).mkdir(parents=True)
    (src.twin_device / "layout.json").write_text('{"pages": []}', encoding="utf-8")
    (src.memory_dir).mkdir(parents=True)
    (src.memory_dir / "k.md").write_text("x", encoding="utf-8")
    wss = curve.prepare_workspaces(src, tmp_path / "ws")
    assert set(wss) == {"on", "off"}
    for w in wss.values():
        assert (w.twin_device / "layout.json").read_text(encoding="utf-8") == '{"pages": []}'
        assert not w.memory_dir.exists() and not w.twin_apps.exists()


def fake_run_dir(root: Path, name: str, hints: bool) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"steps": 3, "started_at": 10.0, "ended_at": 40.5,
                                            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                                            "twin": {"record": {"matched": 2}}}), encoding="utf-8")
    recs = [{"step": 1, "action": {"name": "tap"}, "observation": {"screen": {"name": "设置"}},
             "hints": {"location": hints, "route": False}},
            {"kind": "procedure_step", "action": {"name": "tap"}, "observation": {}},
            {"step": 2, "action": {"name": "done"}, "observation": {}, "hints": {"location": hints, "route": hints}}]
    (d / "steps.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    return d


def test_run_metrics(tmp_path):
    m = curve.run_metrics(fake_run_dir(tmp_path, "r", True))
    assert m == {"steps": 3, "tokens": 120, "wall_s": 30.5, "location_steps": 2, "route_steps": 1,
                 "start_screen": "设置", "twin": {"matched": 2}}


def test_run_curve_alternates_arms_and_isolates_conditions(tmp_path, monkeypatch):
    from iphone_agent import config
    # ⚠ 2026-09-12 终审 FR#4：twin_hints 默认读 IPHONE_TWIN_HINTS；关掉时开组会退回旧 screenmap 提示，
    #   两组差的就不止孪生了。评测必须自己钉死，不能跟着环境变量走。
    monkeypatch.setattr(config, "TWIN_HINTS", False)
    calls, twin_hints = [], []

    def runner(prompt, dev, per, model, runs_root, *, workspace, skill_store, confirm, run_config):
        arm = workspace.root.name
        twin_hints.append(run_config.twin_hints)
        calls.append((prompt, arm, run_config.location_hints, run_config.memory,
                      Path(skill_store.personal).is_relative_to(workspace.root), confirm))
        return SimpleNamespace(run_dir=fake_run_dir(tmp_path / "runs", f"{prompt}-{arm}-{len(calls)}", arm == "on"))

    def verifier(task, run_dir):
        return SimpleNamespace(status="pass", reasons=[])
    t = Task("curve-x", "题", ({"type": "screen_reached", "texts": ["a"]},))
    session = SimpleNamespace(dev=None, per=None, model=None, workspace=Workspace(tmp_path / "src"))
    res = curve.run_curve(session, [t], reps=2, root=tmp_path / "ws", runner=runner, verifier=verifier, log=lambda *_: None)
    assert [(a, lh, mem, iso, c) for _, a, lh, mem, iso, c in calls] == [
        ("on", True, False, True, None), ("off", False, False, True, None),
        ("off", False, False, True, None), ("on", True, False, True, None)]
    assert twin_hints == [True] * 4
    assert [(r["rep"], r["arm"], r["status"]) for r in res["runs"]] == [
        (1, "on", "pass"), (1, "off", "pass"), (2, "off", "pass"), (2, "on", "pass")]
    lines = curve.table(res)
    assert lines[0].startswith("| 题 | 遍 |") and any("curve-x" in x for x in lines[2:])
    p = curve.save(res, tmp_path / "results")
    assert p.name.startswith("curve-") and p.with_suffix(".md").exists()


def test_twin_label_candidates_script_runs_on_duplicate_named_screens(tmp_path):
    """脚本改用 ocr_texts + 同名分叉后仍能跑通并产出候选对（note 4(e)）。

    第二遍 settings_run() 不传 first=，同名屏被重复建出来，制造「同名分叉」样本；
    脚本在导入时执行，用子进程隔离，避免污染本进程状态。
    """
    from tests.twin_fixtures import settings_run, write_run

    ws_root = tmp_path / "ws"
    write_run(ws_root, "run-1", settings_run())
    write_run(ws_root, "run-2", settings_run(t0=100.0))

    proc = subprocess.run(
        [sys.executable, "scripts/twin_label_candidates.py", str(ws_root)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert "pairs" in out and "owners" in out
    assert len(out["pairs"]) >= 1
