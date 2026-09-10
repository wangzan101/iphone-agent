"""`iphone memory list` —— 看得见、改得掉是这个项目的既有承诺。

三档标记（✓已验证/○未验证/✗来自失败运行）原来只在 recap 里用；CLI 这里以前
自己另算一套只有 ✓/✗ 两档的逻辑，跟 recap 看到的对不上。这里改成同一个
`recap.mark_for`，人在 CLI 上看到的和模型在 recall 里看到的必须是同一件事。
"""
from iphone_agent.cli.commands import COMMANDS, Session
from iphone_agent.memory import MemoryStore
from iphone_agent.workspace import Workspace


def test_list_shows_three_way_mark_kind_and_usage_counts(capsys, tmp_path):
    ws = Workspace(tmp_path)
    store = MemoryStore(ws.memory_dir)
    store.write("verified-one", "已验证过的知识", "身体", "runs/a", "done_success")
    store.update_usage("verified-one", "success", "runs/a")
    store.write("unverified-two", "还没被用过", "身体", "runs/b", "done_success",
                kind="playbook")
    store.write("bad-three", "来自失败运行", "身体", "runs/c", "done_failed")

    rc = COMMANDS["memory"](Session(ws), ["list"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "✓已验证" in out and "verified-one" in out
    assert "○未验证" in out and "unverified-two" in out and "playbook" in out
    assert "✗来自失败运行" in out and "bad-three" in out
    # 用量：verified-one 成功用过 1 次、失败 0 次
    assert "1/0" in out


def test_list_uses_recap_mark_for_not_a_second_implementation(tmp_path, capsys):
    """回归护栏：CLI 不能自己另算一套两档 ✓/✗ —— 那会让 manual 来源
    （没有 used_success 也该是 ✓已验证）在 CLI 上被误判成 ✗。"""
    ws = Workspace(tmp_path)
    store = MemoryStore(ws.memory_dir)
    store.write("manual-one", "人手写的", "身体", "manual", "manual")

    rc = COMMANDS["memory"](Session(ws), ["list"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "✓已验证" in out and "manual-one" in out
