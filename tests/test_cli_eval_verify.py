"""m9（终审）：iphone eval verify 打印和退出码 —— ok=None（待审）不算失败、不打 ✗、不让退出码非零。"""
import json
from types import SimpleNamespace

from iphone_agent.cli import commands as C
from tests.test_verify import GOOD, TASK, _run, _step


def _write_task(tmp_path, task=TASK):
    d = tmp_path / "evalset" / "tasks"
    d.mkdir(parents=True)
    (d / f"{task['id']}.json").write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")


def test_review_only_run_prints_daishen_and_exits_zero(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_task(tmp_path)
    tap = GOOD[:1] + [_step(2, "tap", {"x": 10, "y": 10}, ["通用"])] + GOOD[1:]
    run_dir = _run(tmp_path / "evalset", "a", "done_success", "18.3.1", tap)
    rc = C.cmd_eval(SimpleNamespace(), ["verify", TASK["id"], str(run_dir)])
    out = capsys.readouterr().out
    assert "review" in out
    assert "待审" in out and "✗" not in out
    assert rc == 0


def test_a_real_failure_still_prints_a_cross_and_exits_nonzero(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_task(tmp_path)
    run_dir = _run(tmp_path / "evalset", "a", "done_success", "26.6.1", GOOD)
    rc = C.cmd_eval(SimpleNamespace(), ["verify", TASK["id"], str(run_dir)])
    out = capsys.readouterr().out
    assert "fail" in out and "✗" in out
    assert rc == 1


def test_review_never_masks_a_real_failure_in_the_cli_exit_code(monkeypatch, capsys, tmp_path):
    """review 和真失败同时出现（同一次运行）：不能因为有 review 就把失败盖成 0。"""
    monkeypatch.chdir(tmp_path)
    _write_task(tmp_path)
    tap = GOOD[:1] + [_step(2, "tap", {"x": 10, "y": 10}, ["通用"])] + GOOD[1:]
    run_dir = _run(tmp_path / "evalset", "a", "done_success", "26.6.1", tap)
    rc = C.cmd_eval(SimpleNamespace(), ["verify", TASK["id"], str(run_dir)])
    out = capsys.readouterr().out
    assert "fail" in out
    assert rc == 1
