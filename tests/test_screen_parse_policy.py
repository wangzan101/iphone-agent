"""整屏解析模式与唯一决策入口（spec 2026-09-14 §3、§8.4）。"""
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from iphone_agent import config
from iphone_agent.driver.geometry import Frame, Rect
from iphone_agent.perceive import policy
from iphone_agent.perceive.observe import Perceiver
from iphone_agent.perceive.ocr import RawBox
from iphone_agent.workspace import RunConfig

PKG = Path(__file__).resolve().parents[1] / "iphone_agent"


def code_hits(pattern: str) -> list[str]:
    """iphone_agent/ 下命中 pattern 的非注释行（"相对路径:行号"）。grep 测试守「一个规则一个入口」。"""
    rx = re.compile(pattern)
    out = []
    for p in sorted(PKG.rglob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            if rx.search(line) and not line.lstrip().startswith("#"):
                out.append(f"{p.relative_to(PKG).as_posix()}:{i}")
    return out


class Counting:
    def __init__(self, data=None):
        self.calls, self.data = 0, data

    def ask_json(self, prompt, images, max_tokens=None):
        self.calls += 1
        return self.data


def frame(fid=1):
    return Frame(Image.new("RGB", (600, 1200), "white"), 600, 1200, Rect(0, 0, 300, 600), 0.0, fid)


def per_with(asker):
    return Perceiver(ocr=lambda im: [RawBox("通用", 0.99, 0.1, 0.8, 0.3, 0.04)], asker=asker)


def obs_with(**perception):
    return SimpleNamespace(perception=perception)


@pytest.mark.parametrize("raw,mode", [
    ("", "always"), ("  ", "always"), ("always", "always"), ("on_demand", "on_demand"),
    (" ON_DEMAND ", "on_demand"), ("off", "off"), ("on", "always"), ("1", "always"),
    ("true", "always"), ("TRUE", "always"), ("0", "off"), ("false", "off"),
])
def test_screen_parse_mode_maps_old_and_new_values(raw, mode):
    assert config.screen_parse_mode(raw) == mode


def test_unknown_value_is_an_error_that_lists_the_choices():
    with pytest.raises(ValueError, match="always / on_demand / off"):
        config.screen_parse_mode("sometimes")


def test_env_is_read_at_call_time_by_from_env(monkeypatch):
    monkeypatch.setenv("IPHONE_SCREEN_PARSE", "on_demand")
    assert RunConfig.from_env().screen_parse == "on_demand"
    monkeypatch.delenv("IPHONE_SCREEN_PARSE", raising=False)
    assert RunConfig.from_env().screen_parse == policy.DEFAULT_MODE == "always"
    assert RunConfig().screen_parse == "always"          # 直接构造不读 env（计划 R4）
    with pytest.raises(ValueError):
        RunConfig(screen_parse="sometimes")
    assert not hasattr(config, "SCREEN_PARSE"), "旧开关已删，唯一入口是 config.screen_parse_mode"


@pytest.mark.parametrize("mode,requested,fallback,has_asker,want", [
    ("always", False, False, True, "always"),
    ("always", True, False, True, "model"),
    ("always", False, True, True, "fallback"),
    ("on_demand", False, False, True, None),
    ("on_demand", True, False, True, "model"),
    ("on_demand", False, True, True, "fallback"),
    ("on_demand", True, True, True, "fallback"),          # fallback 优先于 requested
    ("off", False, False, True, None),
    ("off", True, False, True, None),
    ("off", False, True, True, None),
    ("always", False, False, False, None),                # 没有 asker：三种模式都只跑 OCR
    ("on_demand", True, True, False, None),
])
def test_parse_trigger_decision_table(mode, requested, fallback, has_asker, want):
    assert policy.parse_trigger(mode, requested, fallback, has_asker) == want


@pytest.mark.parametrize("mode,obs,has_asker,want", [
    ("on_demand", obs_with(ocr=3, label_by=None), True, True),
    ("on_demand", obs_with(ocr=3, label_by="parse"), True, False),   # 整屏解析已带（或已尝试）标注
    ("on_demand", obs_with(ocr=3, label_by="label"), True, False),   # 已标过：幂等
    ("on_demand", obs_with(), True, False),                          # zoom 观察：perception 为空
    ("on_demand", obs_with(ocr=3, label_by=None), False, False),
    ("always", obs_with(ocr=3, label_by=None), True, False),
    ("off", obs_with(ocr=3, label_by=None), True, False),
])
def test_wants_label(mode, obs, has_asker, want):
    assert policy.wants_label(mode, obs, has_asker) is want


@pytest.mark.parametrize("mode,vision,used,has_asker,want", [
    ("on_demand", "off", False, True, True),
    ("on_demand", "failed", False, True, True),
    ("always", "failed", False, True, True),              # always：只在解析失败的帧上触发 = 再试一次
    ("always", "ok", False, True, False),
    ("on_demand", "empty", False, True, False),
    ("on_demand", "off", True, True, False),              # 每个任务至多一次
    ("off", "off", False, True, False),
    ("on_demand", "off", False, False, False),
])
def test_fallback_due(mode, vision, used, has_asker, want):
    assert policy.fallback_due(mode, obs_with(vision=vision), used, has_asker) is want


def test_bad_mode_is_rejected_by_every_decision():
    for call in (lambda: policy.parse_trigger("x", False, False, True),
                 lambda: policy.wants_label("x", obs_with(), True),
                 lambda: policy.fallback_due("x", obs_with(), False, True)):
        with pytest.raises(ValueError):
            call()


def test_perceiver_follows_the_scope_mode_and_records_why():
    a = Counting()
    per = per_with(a)
    o = per.observe(frame())                              # 不在任何 scope 里：DEFAULT_MODE
    assert a.calls == 1 and o.perception["mode"] == "always" and o.perception["parse"] == "always"
    assert o.perception["label_by"] == "parse"
    with per.task_scope(None, "on_demand"):
        o = per.observe(frame(2))
        assert a.calls == 1 and o.perception["parse"] is None and o.perception["vision"] == "off"
        assert o.perception["label_by"] is None and o.perception["mode"] == "on_demand"
        o = per.observe(frame(3), requested=True)
        assert a.calls == 2 and o.perception["parse"] == "model"
    with per.task_scope(None, "off"):
        o = per.observe(frame(4), requested=True, fallback=True)
        assert a.calls == 2 and o.perception["parse"] is None
    assert per.mode == "always"


def test_no_asker_is_written_as_no_asker():
    o = per_with(None).observe(frame())
    assert o.perception["parse"] == "no_asker" and o.perception["vision"] == "off"


def test_task_scope_restores_identity_and_mode_even_on_exception():
    per = per_with(None)
    ctx = object()
    with pytest.raises(RuntimeError), per.task_scope(ctx, "on_demand"):
        assert per.mode == "on_demand" and per._identity is ctx
        raise RuntimeError("x")
    assert per.mode == "always" and per._identity is None
    with pytest.raises(ValueError), per.task_scope(None, "sometimes"):
        pass


def test_env_var_is_read_in_exactly_one_place():
    hits = code_hits(r"IPHONE_SCREEN_PARSE")
    assert hits and all(h.startswith("config.py:") for h in hits), hits
    callers = [h for h in code_hits(r"screen_parse_mode\(") if not h.startswith("config.py:")]
    assert len(callers) == 1 and callers[0].startswith("workspace.py:"), callers


def test_the_mode_is_validated_in_one_place():
    """模式合法性只在 policy.check_mode 里判（controller ruling m7，CLAUDE.md §7；终审 minor 3）。"""
    hits = code_hits(r"not in (policy\.)?MODES\b")
    assert hits and all(h.startswith("perceive/policy.py:") for h in hits), hits
