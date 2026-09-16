"""短标注：进入 loop 的观察各标一次（spec 2026-09-14 §5、§6.3、§8.1、§8.3）。"""
import hashlib

import pytest
from PIL import Image

from iphone_agent.driver.geometry import Frame, Rect
from iphone_agent.harness.runlog import observation_snapshot
from iphone_agent.model.reply import ModelError
from iphone_agent.model.vision import VisionAsker
from iphone_agent.perceive.observe import ObservationFrozen, Perceiver
from iphone_agent.perceive.ocr import RawBox
from iphone_agent.perceive.screen import (
    PROMPT_ELEMENTS,
    PROMPT_LABEL,
    PROMPT_SCREEN,
    Candidate,
    label_prompt,
    label_screen,
    screen_prompt,
)
from tests.test_screen_parse_policy import code_hits

GOOD = {"screen": {"app": "设置", "name": "通用", "same_as": None, "anchors": ["通用"]}}


class Asker:
    """按顺序吐回复；回复是 None 时像 VisionAsker 一样留下 last_outcome。"""
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []
        self.last_outcome = None

    def ask_json(self, prompt, images, max_tokens=None):
        self.prompts.append(prompt)
        data = self.replies.pop(0) if self.replies else None
        self.last_outcome = None if data is not None else {"kind": "transport", "detail": "boom"}
        return data


class Ctx:
    def __init__(self, cands):
        self.cands, self.seen = list(cands), []

    def candidates(self, texts):
        self.seen.append(sorted(texts))
        return self.cands


def frame(fid=1, color="white"):
    return Frame(Image.new("RGB", (600, 1200), color), 600, 1200, Rect(0, 0, 300, 600), 0.0, fid)


def per_with(asker):
    return Perceiver(ocr=lambda im: [RawBox("通用", 0.99, 0.1, 0.8, 0.3, 0.04)], asker=asker)


# ---------- 提示词 ----------

def test_existing_prompts_are_byte_identical_so_the_vision_cache_survives():
    assert hashlib.sha256(PROMPT_ELEMENTS.encode("utf-8")).hexdigest() == \
        "f1470183d336b93d12d9b79a5d7680d360135e3ee82febabb04263657e3b5ac9"
    assert hashlib.sha256(PROMPT_SCREEN.encode("utf-8")).hexdigest() == \
        "8ddbce17d3df18a95c20b7822327bb2b8cd5c2b1b5748dd2d1c4dbdfa1971ace"
    assert len(PROMPT_SCREEN) == 920


def test_label_prompt_shares_the_screen_text_and_asks_for_the_label_only():
    anchors_line = next(ln for ln in PROMPT_SCREEN.splitlines() if ln.startswith("- anchors"))
    assert anchors_line in PROMPT_LABEL and "「系统」" in PROMPT_LABEL and "same_as" in PROMPT_LABEL
    assert "把元素列出来" not in PROMPT_LABEL and '"elements"' not in PROMPT_LABEL
    cands = [Candidate(1, "she-zhi", "设置", "s_1", "通用")]
    assert label_prompt(()) == PROMPT_LABEL
    assert label_prompt(cands) == PROMPT_LABEL + screen_prompt(cands)[len(PROMPT_SCREEN):]


def test_label_screen_without_an_asker_is_off():
    """controller ruling m4：五种标注状态里的 off 也要覆盖到（label_screen 是唯一入口）。"""
    assert label_screen(Image.new("RGB", (4, 4)), None) == (None, "off")


# ---------- VisionAsker.last_outcome ----------

class T:
    def __init__(self, *items):
        self.items, self.last_finish_reason, self.last_max_tokens = list(items), None, 100

    def ask(self, messages, max_tokens=None):
        it = self.items.pop(0)
        if isinstance(it, BaseException):
            raise it
        text, self.last_finish_reason = it
        return text


@pytest.mark.parametrize("item,kind", [
    (ModelError("503"), "transport"), (OSError("reset"), "exception"),
    (("没有 JSON", "length"), "truncated"), (("没有 JSON", "stop"), "unparsable"),
    (('[{"a":1},{"b', "length"), "partial"), (('{"screen": {}}', "stop"), None),
])
def test_each_failure_branch_has_its_own_kind(item, kind):
    a = VisionAsker(T(item))
    a.ask_json("q", [Image.new("RGB", (4, 4))])
    assert (a.last_outcome or {}).get("kind") == kind


def test_outcome_is_cleared_at_the_start_of_every_call():
    a = VisionAsker(T(ModelError("503"), ('{"screen": {}}', "stop")))
    a.ask_json("q", [Image.new("RGB", (4, 4))])
    assert a.last_outcome["kind"] == "transport"
    a.ask_json("q", [Image.new("RGB", (4, 4))])
    assert a.last_outcome is None and a.last_error, "last_error 只留给 stats() 汇总，不代表这一次"
    assert VisionAsker(None).ask_json("q", []) is None


def test_parse_error_belongs_to_its_own_frame():
    per = per_with(Asker(None, {"elements": []}))
    first, second = per.observe(frame(1)), per.observe(frame(2))
    assert first.perception["parse_error"] == {"kind": "transport", "detail": "boom"}
    assert second.perception["parse_error"] is None and second.perception["vision"] == "empty"
    assert per.stats()["vision_failures"]["transport"] == 1


# ---------- ensure_label ----------

def test_on_demand_frame_gets_one_short_label_with_real_candidates():
    a = Asker({"screen": {**GOOD["screen"], "same_as": 1}})
    per = per_with(a)
    ctx = Ctx([Candidate(1, "she-zhi", "设置", "s_1", "通用")])
    with per.task_scope(ctx, "on_demand"):
        o = per.observe(frame())
        assert a.prompts == [], "on_demand 的观察本身只跑 OCR"
        per.ensure_label(o, "adopt")
        per.ensure_label(o, "adopt")                         # 幂等
    assert len(a.prompts) == 1 and a.prompts[0].startswith(PROMPT_LABEL) and "1. 设置 / 通用" in a.prompts[0]
    assert ctx.seen == [["通用"]], "候选由这帧的 OCR 文字挑出"
    assert o.screen.same_as == 1 and [c.screen_id for c in o.screen_candidates] == ["s_1"]
    p = o.perception
    assert (p["screen"], p["label_by"], p["label_error"]) == ("ok", "label", None)
    snap = observation_snapshot(o, "f.png")
    assert snap["screen"]["name"] == "通用" and snap["screen_candidates"][0]["screen_id"] == "s_1"
    assert per.stats()["label"]["adopt"]["ok"] == 1


@pytest.mark.parametrize("reply,status", [(GOOD, "ok"), ([], "missing"), ({"screen": "x"}, "invalid"),
                                          (None, "failed")])
def test_every_label_status_lands_in_the_frame_and_the_stats(reply, status):
    per = per_with(Asker(reply))
    with per.task_scope(None, "on_demand"):
        o = per.observe(frame())
        per.ensure_label(o, "identity")
    assert o.perception["screen"] == status
    assert observation_snapshot(o, "f.png")["perception"]["screen"] == status
    assert per.stats()["label"]["identity"][status] == 1
    assert (o.perception["label_error"] is not None) == (status == "failed")


@pytest.mark.parametrize("mode", ["always", "off"])
def test_other_modes_never_short_label(mode):
    a = Asker(GOOD, GOOD)
    per = per_with(a)
    with per.task_scope(None, mode):
        o = per.observe(frame())
        n = len(a.prompts)
        per.ensure_label(o, "adopt")
    assert len(a.prompts) == n, "always 的标注来自整屏解析（失败也不补）；off 不标"


def test_zoom_and_no_asker_frames_are_not_labelled():
    a = Asker(GOOD)
    per = per_with(a)
    with per.task_scope(None, "on_demand"):
        base = per.observe(frame())
        z = per.zoom(base, (0, 0, 300, 300))
        per.ensure_label(z, "adopt")
    assert not any(p.startswith(PROMPT_LABEL) for p in a.prompts)
    plain = per_with(None)
    with plain.task_scope(None, "on_demand"):
        o = plain.observe(frame())
        plain.ensure_label(o, "adopt")
    assert o.screen is None and o.perception["label_by"] is None


def test_finalized_frames_cannot_be_rewritten():
    per = per_with(Asker(GOOD))
    with per.task_scope(None, "on_demand"):
        o = per.observe(frame())
        per.finalize(o)
        assert o.finalized is True
        with pytest.raises(ObservationFrozen):
            per.ensure_label(o, "adopt")
        with pytest.raises(ValueError):
            per.ensure_label(per.observe(frame(2)), "whatever")


def test_exact_duplicate_frames_and_empty_candidates_are_only_measured():
    per = per_with(Asker(GOOD, GOOD, GOOD))
    with per.task_scope(None, "on_demand"):
        for fid, color in ((1, "white"), (2, "white"), (3, "black")):
            per.ensure_label(per.observe(frame(fid, color)), "adopt")
    st = per.stats()
    assert st["label_dup_exact"] == 1 and st["label"]["adopt"]["ok"] == 3 and st["candidates_empty"] == 3


def test_each_task_scope_starts_with_fresh_stats_and_its_own_mode():
    per = per_with(Asker(GOOD, {"elements": []}))
    with per.task_scope(None, "on_demand"):
        per.ensure_label(per.observe(frame()), "adopt")
    assert per.stats()["mode"] == "on_demand" and per.stats()["label"]["adopt"]["ok"] == 1, \
        "scope 关掉之后 stats 仍读得到（计划 R14）"
    with per.task_scope(None, "always"):
        per.observe(frame(2))
    st = per.stats()
    assert st["mode"] == "always" and st["label"]["adopt"]["ok"] == 0 and st["parse_by"]["always"] == 1


# ---------- 唯一入口 ----------

def test_decisions_are_only_asked_inside_the_perceiver():
    hits = [h for h in code_hits(r"\b(parse_trigger|wants_label)\(") if not h.startswith("perceive/policy.py")]
    assert hits and all(h.startswith("perceive/observe.py:") for h in hits), hits


def test_nobody_outside_the_perceiver_writes_perception_fields():
    pat = (r"\.(screen|screen_candidates|perception|zoom_base_vision)\s*=[^=]|\.perception\[[^\]]*\]\s*=[^=]"
           r"|\.perception\.(update|setdefault)\(")
    hits = code_hits(pat)
    assert all(h.startswith("perceive/observe.py:") for h in hits), hits
