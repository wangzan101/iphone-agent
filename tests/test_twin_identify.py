"""认屏（spec 2026-09-12 §4.2）：按标注认，程序只核对。"""
import pytest

from iphone_agent.memory.screenmap import SYSTEM, UNKNOWN
from iphone_agent.perceive.screen import ScreenLabel
from iphone_agent.twin.identify import Recognition, grounded, identify, norm_text, ocr_texts, plain_app_id
from iphone_agent.twin.screenfile import Screen


class St:
    def __init__(self, *screens):
        self.by = {(s.app, s.id): s for s in screens}

    def get(self, app, sid):
        return self.by.get((app, sid))


LIST = Screen("s_list", "bei-wang-lu", "r0:f2.png", "t", "t", "iCloud全部备忘录", anchors={"文件夹": 1})
C1 = ({"index": 1, "app_id": "bei-wang-lu", "screen_id": "s_list", "name": "iCloud全部备忘录"},)
TEXTS = frozenset({"iCloud全部备忘录", "文件夹", "编辑"})


def L(name="iCloud全部备忘录", same_as=None, app="备忘录", anchors=("iCloud全部备忘录", "文件夹", "不在屏上")):
    return ScreenLabel(app, name, same_as, tuple(anchors))


@pytest.mark.parametrize("raw,want", [
    ("＜ 返回", "返回"), ("iOS 18.3.1", "iOS#.#.#"), ("  ", ""), ("•••", ""), ("〈设置", "设置")])
def test_norm_text(raw, want):
    assert norm_text(raw) == want


def test_ocr_texts_skip_vision_elements_and_normalize():
    els = [{"text": "＜ 设置", "center": [1, 1], "source": "ocr"},
           {"text": "返回箭头", "center": [1, 1], "source": "vision"},
           {"text": "旧留档视觉", "center": [1, 1], "confidence": 0}]
    assert ocr_texts(els) == frozenset({"设置"})


def test_grounded_keeps_only_anchors_seen_by_ocr():
    assert grounded(("iCloud全部备忘录", "文件夹", "不在屏上", "文件夹"), TEXTS) == ("iCloud全部备忘录", "文件夹")


def test_plain_app_id():
    assert plain_app_id("系统") == SYSTEM
    assert plain_app_id("不确定") is None and plain_app_id(" ") is None
    assert plain_app_id("设置") == "she-zhi"


def test_same_as_matches_and_learns_alias():
    r = identify(L("全部备忘录", same_as=1), "bei-wang-lu", TEXTS, C1, St(LIST))
    assert r == Recognition("matched", "s_list", "iCloud全部备忘录", ("iCloud全部备忘录", "文件夹"), "全部备忘录")
    same = identify(L(same_as=1), "bei-wang-lu", TEXTS, C1, St(LIST))
    assert same.state == "matched" and same.alias is None


def test_same_as_to_missing_screen_is_unresolved():
    assert identify(L(same_as=1), "bei-wang-lu", TEXTS, C1, St()).state == "candidate_unresolved"


def test_same_as_to_another_app_is_a_conflict():
    assert identify(L(same_as=1, app="设置"), "she-zhi", TEXTS, C1, St(LIST)).state == "label_conflict"


def test_no_same_as_is_a_new_screen_even_with_the_same_name():
    r = identify(L(), "bei-wang-lu", TEXTS, C1, St(LIST))
    assert r == Recognition("new", None, "iCloud全部备忘录", ("iCloud全部备忘录", "文件夹"))


def test_system_and_unknown_owner_without_same_as_are_skipped():
    for owner in (SYSTEM, UNKNOWN):
        assert identify(L(app="系统"), owner, TEXTS, (), St()).state == "skipped"


def test_unsure_app_without_same_as_is_skipped_even_with_a_real_owner():
    """spec §4.2 最后一行：「不确定」又没有 same_as → 不建屏。归属是核过身份的进入给的也一样。"""
    for app in ("不确定", " "):
        assert identify(L(app=app), "bei-wang-lu", TEXTS, C1, St(LIST)) == Recognition("skipped")


def test_system_label_without_same_as_is_skipped_even_with_a_real_owner():
    """spec §4.2 最后一行：标「系统」又没有 same_as → 不建屏。归属是核过身份的进入给的也一样（评审 P1：
    open_app 设置 之后第一帧是权限弹窗）。"""
    assert identify(L(app="系统"), "bei-wang-lu", TEXTS, C1, St(LIST)) == Recognition("skipped")
