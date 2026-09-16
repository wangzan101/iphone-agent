"""observe 改义为「看全屏」：重新截一帧再整屏解析（spec 2026-09-14 §4.1、§9）。"""
from iphone_agent.harness.executor import Executor
from iphone_agent.perceive.screen import PROMPT_SCREEN
from tests.test_ensure_label import Asker
from tests.test_executor import act


class Screens:
    """整屏解析回一个视觉项（右上角的加号图标）；别的问题一律没问成。"""
    def __init__(self):
        self.prompts, self.last_outcome = [], None

    def ask_json(self, prompt, images, max_tokens=None):
        self.prompts.append(prompt)
        if prompt.startswith(PROMPT_SCREEN):
            return {"screen": {"app": "设置", "name": "通用", "same_as": None, "anchors": []},
                    "elements": [{"kind": "icon", "label": "加号", "box": [900, 120, 980, 160]}]}
        return None


def test_observe_tool_recaptures_and_parses_the_full_screen(fake_env):
    dev, per, _ = fake_env([["通用"], ["通用", "关于本机"]])
    per.asker = Screens()
    with per.task_scope(None, "on_demand"):
        obs = per.observe(dev.capture())
        res, new = Executor(dev, per).run(act("observe"), obs)
    assert new is not obs and new.frame_id != obs.frame_id, "看全屏要新截一帧，不解析模型刚才看到的那一帧"
    assert new.perception["parse"] == "model" and new.perception["label_by"] == "parse"
    assert any(e.source == "vision" and e.text == "加号" for e in new.elements)
    assert res.ok and res.changed is None and not res.hint


def test_observe_tool_in_off_mode_says_parsing_is_off(fake_env):
    dev, per, _ = fake_env([["通用"]])
    per.asker = Screens()
    with per.task_scope(None, "off"):
        res, _ = Executor(dev, per).run(act("observe"), per.observe(dev.capture()))
    assert "整屏解析已关闭" in res.hint and per.asker.prompts == []


def test_the_header_tells_the_model_which_list_it_got(fake_env):
    from iphone_agent.perceive.elements import HEAD_FULL, HEAD_OCR_ONLY
    dev, per, _ = fake_env([["通用"], ["通用"]])
    per.asker = Screens()
    with per.task_scope(None, "on_demand"):
        obs = per.observe(dev.capture())
        _, new = Executor(dev, per).run(act("observe"), obs)
    assert obs.elements_text.splitlines()[0].endswith(HEAD_OCR_ONLY)
    assert new.elements_text.splitlines()[0].endswith(HEAD_FULL)
    per.asker = Asker(None)
    with per.task_scope(None, "on_demand"):
        _, failed = Executor(dev, per).run(act("observe"), per.observe(dev.capture()))
    assert failed.elements_text.splitlines()[0].endswith(HEAD_OCR_ONLY), "解析失败不能说整屏看过（计划 R12）"


def test_a_failed_full_screen_look_says_so_instead_of_passing_ocr_off_as_complete(fake_env):
    dev, per, _ = fake_env([["通用"]])
    per.asker = Asker(None)
    with per.task_scope(None, "on_demand"):
        res, new = Executor(dev, per).run(act("observe"), per.observe(dev.capture()))
    assert new.perception["vision"] == "failed"
    assert res.hint.startswith("看全屏失败（transport：boom）") and "只有 OCR" in res.hint
