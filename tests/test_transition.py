from iphone_agent.perceive.observe import Perceiver
from iphone_agent.perceive.ocr import RawBox
from iphone_agent.perceive.transition import Transition, render, transition
from tests.conftest import frame_with_text


def _obs(texts, frame_id):
    frame, _ = frame_with_text(texts, frame_id)
    # y 从 0.8 起、每项降 0.03：起点必须离屏幕顶部够远，否则第一项换算成
    # 像素后落进状态栏裁剪区（config.STATUS_BAR_CROP_RATIO=0.11），会被
    # Observation.text_set 悄悄丢掉，跟这里要测的增减逻辑无关却污染结果。
    boxes = [RawBox(text=t, confidence=0.9, x=0.1, y=0.8 - i * 0.03, w=0.3, h=0.05)
             for i, t in enumerate(texts)]
    per = Perceiver(ocr=lambda img: boxes)
    return per.observe(frame)


def test_added_and_removed_follow_element_order():
    b = _obs(["Wi-Fi", "蓝牙", "通用"], 1)
    a = _obs(["通用", "关于本机", "软件更新"], 2)
    t = transition(b, a, tap_px=None)
    assert t.added == ("关于本机", "软件更新")
    assert t.removed == ("Wi-Fi", "蓝牙")
    assert t.local_changed is None


def test_lists_are_capped_but_totals_kept():
    b = _obs([f"old{i}" for i in range(20)], 1)
    a = _obs([f"new{i}" for i in range(20)], 2)
    t = transition(b, a, tap_px=None, list_max=3)
    assert len(t.added) == 3 and t.added_total == 20
    assert len(t.removed) == 3 and t.removed_total == 20


def _obs_status_bar_text(text, frame_id):
    # RawBox.y 是 Vision 坐标（原点在左下角，spec §5.1: y1=(1-y-h)*H）。
    # y=0.95, h=0.03 换算成像素后 y1=16 y2=40（400x800 帧），落在状态栏裁剪区内
    # （config.STATUS_BAR_CROP_RATIO=0.11 → 阈值 88px），即 Observation.text_set
    # 按约定会丢掉的那类文字（时钟、电量）。
    frame, _ = frame_with_text([text], frame_id)
    boxes = [RawBox(text=text, confidence=0.9, x=0.4, y=0.95, w=0.2, h=0.03)]
    per = Perceiver(ocr=lambda img: boxes)
    return per.observe(frame)


def test_status_bar_only_change_yields_no_added_removed():
    # 状态栏文字（时钟）从 9:41 变成 9:42，但 text_set 本该把它过滤掉——
    # 不该退回全量 elements，否则这种每分钟都有的变化会污染 added/removed。
    b = _obs_status_bar_text("9:41", 1)
    a = _obs_status_bar_text("9:42", 2)
    assert b.text_set == set() and a.text_set == set()
    t = transition(b, a, tap_px=None)
    assert t.added == ()
    assert t.removed == ()
    assert t.added_total == 0
    assert t.removed_total == 0


def test_render_mentions_local_only_when_tap_given():
    t = Transition(changed=False, local_changed=True, added=(), removed=(),
                   added_total=0, removed_total=0)
    s = render(t)
    assert "点击位置附近：有变化" in s and "新出现的文字：无" in s
    t2 = Transition(changed=True, local_changed=None, added=("a",), removed=("b", "c"),
                    added_total=1, removed_total=7)
    s2 = render(t2)
    assert "点击位置附近" not in s2
    assert "[a]" in s2 and "共 7 项，只列前 2 项" in s2
