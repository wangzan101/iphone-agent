"""闸门（spec §8.1）：标「不同」却认成同一屏 = 错合，必须 0；归到错误的具体 App，必须 0。"""
from iphone_agent.twin.bench import run_bench
from iphone_agent.twin.record import simulate
from iphone_agent.twin.report import report_text
from iphone_agent.workspace import Workspace
from tests.twin_fixtures import settings_run, write_run


def sim(tmp_path):
    write_run(tmp_path, "r1", settings_run())
    write_run(tmp_path, "r2", settings_run(t0=100, first="r1"))
    return simulate(Workspace(tmp_path))


def test_pairs_count_wrong_merges_and_splits(tmp_path):
    s = sim(tmp_path)
    pairs = [{"a": "r1:2", "b": "r2:2", "label": "same"},          # 两次都是「设置」
             {"a": "r1:2", "b": "r1:3", "label": "different"},     # 设置 vs 通用
             {"a": "r1:3", "b": "r2:3", "label": "different"}]     # 故意标错：同一屏标成不同 → 记错合
    r = run_bench(s, pairs, [])
    assert r.wrong_merge == [("r1:3", "r2:3")] and r.wrong_split == [] and not r.passed


def test_owner_truth(tmp_path):
    s = sim(tmp_path)
    r = run_bench(s, [], [{"event": "r1:2", "app": "she-zhi"}, {"event": "r1:1", "app": "system"},
                          {"event": "r1:3", "app": "bei-wang-lu"}])
    assert r.wrong_owner == [("r1:3", "she-zhi", "bei-wang-lu")] and not r.passed


def test_unknown_owner_is_allowed_but_unresolved_events_still_fail_the_gate(tmp_path):
    """2026-09-11 final review：闸门原先无视 unresolved，「没跑完」能悄悄混进「通过」。
    找不到事件（留档缺行/没跑完）必须让闸门不通过，而不只是被报告。"""
    s = sim(tmp_path)
    r = run_bench(s, [{"a": "r9:1", "b": "r1:2", "label": "same"}], [])
    assert not r.passed and r.unresolved == ["r9:1"]


def test_zero_concrete_owner_coverage_is_not_passed(tmp_path):
    """归属全是 system/unknown 时，闸门没有真正核过任何一个具体 App —— 不能算通过。"""
    s = sim(tmp_path)
    r = run_bench(s, [], [{"event": "r1:1", "app": "system"}])
    assert r.wrong_owner == [] and r.concrete_owner == 0 and not r.passed


def test_zero_resolved_different_pairs_is_not_passed(tmp_path):
    """没有任何一对「different」标注双方都落到了具体屏 —— 闸门 A 也没被真正核验过。"""
    s = sim(tmp_path)
    r = run_bench(s, [], [{"event": "r1:2", "app": "she-zhi"}])
    assert r.wrong_owner == [] and r.concrete_owner == 1 and r.resolved_different == 0 and not r.passed


def test_bench_passes_with_nonzero_concrete_coverage(tmp_path):
    s = sim(tmp_path)
    pairs = [{"a": "r1:2", "b": "r1:3", "label": "different"}]
    owners = [{"event": "r1:2", "app": "she-zhi"}, {"event": "r1:3", "app": "she-zhi"}]
    r = run_bench(s, pairs, owners)
    assert r.passed and r.concrete_owner == 2 and r.resolved_different == 1


def test_report_lists_apps_and_state_counts(tmp_path):
    write_run(tmp_path, "r1", settings_run())
    text = report_text(Workspace(tmp_path))
    assert "认屏：认出" in text and "归属：核身份进入" in text
    assert "设置（she-zhi）：3 屏" in text and "关于本机" in text


def test_cli_bench_resolves_evalset_relative_to_cwd(tmp_path, monkeypatch, capsys):
    """labels are repo-committed files; runs are workspace data.
    Verify that bench loads labels from cwd's evalset/twin/, not workspace.root.
    """
    from types import SimpleNamespace

    from iphone_agent.cli.commands import cmd_twin

    # Create workspace at a different location (no evalset)
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()

    # Create evalset/twin in cwd (tmp_path)
    (tmp_path / "evalset" / "twin").mkdir(parents=True)
    (tmp_path / "evalset" / "twin" / "owners.json").write_text(
        '[{"event": "r9:1", "app": "system"}]'
    )
    (tmp_path / "evalset" / "twin" / "pairs.json").write_text("[]")

    # Change to tmp_path where evalset exists
    monkeypatch.chdir(tmp_path)

    # Create minimal session pointing to workspace without evalset
    session = SimpleNamespace(workspace=Workspace(ws_dir))

    # Before the original fix, this would return 2 (no labels found).
    # After it, labels are found in cwd — but the workspace here has no runs, so the
    # one owner event ("r9:1") never resolves and (2026-09-11 final review) an
    # unresolved event now fails the gate: result is 1, not 0. What this test cares
    # about — labels were found in cwd, not workspace.root — is that we did NOT hit
    # the "no labels at all" branch (return 2 / "没有标注" message).
    result = cmd_twin(session, ["bench"])

    assert result == 1, f"cmd_twin returned {result} but expected 1 (found labels, gate fails on unresolved event)"

    # Should find the labels in cwd's evalset/twin, not report "no labels"
    captured = capsys.readouterr()
    assert "没有标注" not in captured.err, f"Unexpected error message in stderr: {captured.err}"

    # Should NOT print "没有标注" message
    captured = capsys.readouterr()
    assert "没有标注" not in captured.err, f"Unexpected error message in stderr: {captured.err}"
