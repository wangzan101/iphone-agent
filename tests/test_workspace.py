"""工作区与单次运行配置。

为什么要有它：路径（runs/、.iphone/memory、.iphone/config.toml）原来是几个模块
各自的模块级常量，全相对进程 cwd；运行模式在 import 时读一次 env。于是同一个进程
里跑两台设备、两个会话、或者 window/state 两种模式的对照实验都做不了（docs/20 D3/D7/D8）。
Workspace 把路径收成一个显式对象，RunConfig 把「这一次怎么跑」收成一个参数。
"""
from pathlib import Path

from iphone_agent.workspace import RunConfig, Workspace


def test_workspace_paths_hang_off_root(tmp_path):
    w = Workspace(tmp_path)
    assert w.runs == tmp_path / "runs"
    assert w.dot == tmp_path / ".iphone"
    assert w.memory_dir == tmp_path / ".iphone" / "memory"
    assert w.config_file == tmp_path / ".iphone" / "config.toml"
    assert w.screenmap_cache == tmp_path / ".iphone" / "screenmap.json"


def test_workspace_default_reads_env_at_call_time(tmp_path, monkeypatch):
    """⚠ 在**调用时**读 env，不在 import 时 —— 否则测试和同进程多工作区都改不动它。"""
    monkeypatch.setenv("IPHONE_WORKSPACE", str(tmp_path))
    assert Workspace.default().root == tmp_path
    monkeypatch.delenv("IPHONE_WORKSPACE")
    assert Workspace.default().root == Path.cwd()


def test_run_config_from_env_and_overrides(monkeypatch):
    monkeypatch.setenv("IPHONE_USE_CONTEXT", "state")
    rc = RunConfig.from_env()
    assert rc.context_mode == "state"
    rc2 = RunConfig.from_env(max_steps=7)
    assert rc2.max_steps == 7 and rc2.timeout_s is None
    from iphone_agent import config
    assert rc2.effective_timeout_s() == 7 * config.SECONDS_PER_STEP


def test_run_config_default_context_mode_comes_from_config(monkeypatch):
    """env 没设时退回 config.CONTEXT_MODE —— 那个常量仍是「默认值来源」。"""
    from iphone_agent import config
    monkeypatch.delenv("IPHONE_USE_CONTEXT", raising=False)
    monkeypatch.setattr(config, "CONTEXT_MODE", "state")
    assert RunConfig.from_env().context_mode == "state"


def test_run_config_max_steps_read_at_call_time(monkeypatch):
    from iphone_agent import config
    monkeypatch.setattr(config, "MAX_STEPS", 3)
    assert RunConfig.from_env().max_steps == 3


def test_explicit_timeout_wins_over_seconds_per_step():
    assert RunConfig(max_steps=100, timeout_s=5.0).effective_timeout_s() == 5.0


def test_run_config_default_max_steps_read_at_call_time_not_import_time(monkeypatch):
    """⚠ `max_steps` 的默认值不能在 import 时绑死（Minor 3）：
    monkeypatch `config.MAX_STEPS` 之后，不传参数构造的 `RunConfig()` 也要跟着变。"""
    from iphone_agent import config
    monkeypatch.setattr(config, "MAX_STEPS", 3)
    assert RunConfig().max_steps == 3


def test_run_config_twin_hints_from_env_uses_single_parser(monkeypatch):
    """⚠ 2026-09-11 final review：IPHONE_TWIN_HINTS 原来在 config.py 和 workspace.py 各解析
    一遍，两处对 .strip() 已经不一致（CLAUDE.md §7 一个规则一个入口）。
    现在唯一入口是 config.env_flag，两头都要认得 " OFF " 这种带空白的值。"""
    monkeypatch.setenv("IPHONE_TWIN_HINTS", " OFF ")
    assert RunConfig.from_env().twin_hints is False
    monkeypatch.setenv("IPHONE_TWIN_HINTS", "on")
    assert RunConfig.from_env().twin_hints is True
    monkeypatch.delenv("IPHONE_TWIN_HINTS", raising=False)
    from iphone_agent import config
    assert RunConfig.from_env().twin_hints == config.TWIN_HINTS


def test_env_flag_strips_lowercases_and_falls_back_to_default(monkeypatch):
    from iphone_agent.config import env_flag
    monkeypatch.delenv("SOME_TEST_FLAG", raising=False)
    assert env_flag("SOME_TEST_FLAG", True) is True
    assert env_flag("SOME_TEST_FLAG", False) is False
    monkeypatch.setenv("SOME_TEST_FLAG", " OFF ")
    assert env_flag("SOME_TEST_FLAG", True) is False
    monkeypatch.setenv("SOME_TEST_FLAG", "on")
    assert env_flag("SOME_TEST_FLAG", False) is True
    monkeypatch.setenv("SOME_TEST_FLAG", "")
    assert env_flag("SOME_TEST_FLAG", True) is True
