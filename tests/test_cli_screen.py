"""CLI 的观察和任务同一条链路，输出写明模式（spec 2026-09-14 §7）。"""
from types import SimpleNamespace

from iphone_agent.cli import commands as C
from iphone_agent.driver.timing import IOS_TIMING, Timing
from iphone_agent.perceive.screen import PROMPT_SCREEN
from tests.conftest import FakeDevice, frame_with_text, perceiver_for
from tests.test_observe_tool import Screens


def _session():
    frames = [frame_with_text(["通用"], 1)]
    per = perceiver_for(frames)
    per.asker = Screens()
    return SimpleNamespace(dev=FakeDevice(frames), per=per)


def test_iphone_screen_is_a_full_screen_look_and_says_the_mode(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IPHONE_SCREEN_PARSE", "on_demand")
    s = _session()
    assert C.cmd_screen(s, []) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert "on_demand" in first and "整屏看过" in first
    assert s.per.asker.prompts[0].startswith(PROMPT_SCREEN) and s.per.mode == "always"


def test_other_cli_observations_run_in_the_task_mode(monkeypatch, capsys):
    monkeypatch.setenv("IPHONE_SCREEN_PARSE", "on_demand")
    fast = Timing(min_wait_ms=0, poll_ms=10, stable_span_ms=20, settle_max_ms=1000)
    monkeypatch.setattr("iphone_agent.driver.timing.IOS_TIMING", dict.fromkeys(IOS_TIMING, fast))
    s = _session()
    C.cmd_scroll(s, ["down"])
    assert "整屏解析模式 on_demand" in capsys.readouterr().err, "模式行走 stderr，stdout 只留结果"
    assert s.per.asker.prompts == [], "on_demand 下手动动作的观察只跑 OCR"
