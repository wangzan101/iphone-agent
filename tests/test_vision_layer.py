"""看图那一路：屏幕解析、融合、判定，以及它接进执行层之后的行为。

这一层的存在理由是 2026-09-09 那个 bug：候选栏 OCR 读得好好的，只因为 y 不在
屏幕下部 30% 就被几何规则判成「不存在」，接着 toggle_ime 把一个**正确的**中文
输入法切走。所以下面最要紧的一组测试不是「视觉能不能找到候选栏」，而是
**找到之后有没有管住那只手**。
"""

import pytest
from PIL import Image

from iphone_agent.harness import judge
from iphone_agent.harness.actions import Action
from iphone_agent.harness.executor import Executor
from iphone_agent.model.vision import VisionAsker, extract_json
from iphone_agent.perceive.elements import Element, fuse
from iphone_agent.perceive.screen import ScreenItem, parse_screen
from tests.test_executor import _PAD, _ime_env, act


class FakeAsker:
    """按顺序吐预设回复。没有回复了就返回 None（模拟调用失败）。"""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def ask_json(self, prompt, images, max_tokens=None):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else None


# ---------- JSON 挖掘：模型不会老老实实只给 JSON ----------

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_code_fence():
    """模型很爱裹代码围栏，哪怕你明说了别裹。"""
    assert extract_json('```json\n[{"kind":"icon"}]\n```') == [{"kind": "icon"}]


def test_extract_json_with_preamble():
    """「好的，这是结果：{...}」—— 截首尾括号那一层就是为它准备的。"""
    assert extract_json('好的，这是结果：{"found": true} 希望有帮助') == {"found": True}


def test_extract_json_gives_up_quietly():
    """挖不出来返回 None，不抛 —— 调用方要的是降级，不是崩。"""
    assert extract_json("我觉得屏幕上有个按钮") is None
    assert extract_json("") is None


def test_asker_never_raises_when_the_endpoint_blows_up():
    """端点炸了只记账，不往上抛。视觉是增强，不是命脉。"""
    class Boom:
        def ask(self, *a, **k):
            raise RuntimeError("connection reset")

    asker = VisionAsker(Boom())
    assert asker.ask_json("问题", [Image.new("RGB", (10, 10))]) is None
    assert asker.failures == 1 and "connection reset" in asker.last_error


def test_asker_off_costs_nothing():
    class NeverCalled:
        def ask(self, *a, **k):
            raise AssertionError("关掉了还调？")

    asker = VisionAsker(NeverCalled(), enabled=False)
    assert asker.ask_json("问题", [Image.new("RGB", (10, 10))]) is None
    assert asker.calls == 0


# ---------- 屏幕解析 ----------

def _img(w=600, h=1200):
    return Image.new("RGB", (w, h), "white")


def test_parse_screen_converts_normalized_to_pixels():
    asker = FakeAsker([{"kind": "icon", "label": "返回箭头", "box": [0, 0, 100, 50]}])
    items = parse_screen(_img(600, 1200), asker)
    assert len(items) == 1
    assert items[0].box == (0, 0, 60, 60), items[0].box
    assert items[0].kind == "icon"


def test_parse_screen_survives_one_bad_item():
    """一条坏的不该带走一整屏。"""
    asker = FakeAsker([
        {"kind": "icon", "label": "好的", "box": [0, 0, 100, 100]},
        {"kind": "icon", "label": "没有 box"},
        {"kind": "icon", "label": "box 坏了", "box": ["a", "b", "c", "d"]},
        {"kind": "icon", "label": "零面积", "box": [50, 50, 50, 50]},
    ])
    items = parse_screen(_img(), asker)
    assert [i.label for i in items] == ["好的"]


def test_parse_screen_unwraps_a_dict():
    """模型有时候包一层 {"elements": [...]}。"""
    asker = FakeAsker({"elements": [{"kind": "button", "label": "完成",
                                     "box": [0, 0, 100, 100]}]})
    assert [i.label for i in parse_screen(_img(), asker)] == ["完成"]


def test_parse_screen_swaps_flipped_corners():
    """两个角给反了，排一下序比拒掉它有用。"""
    asker = FakeAsker([{"kind": "icon", "label": "反的", "box": [200, 200, 100, 100]}])
    items = parse_screen(_img(600, 1200), asker)
    assert items[0].box == (60, 120, 120, 240), items[0].box


def test_parse_screen_without_asker_is_empty_not_an_error():
    assert parse_screen(_img(), None) == []


def test_parse_screen_when_the_call_fails():
    """⚠ 返回空**不等于**这屏没东西可点，只等于这一路没结果。"""
    assert parse_screen(_img(), FakeAsker()) == []


# ---------- 融合 ----------

def _el(text, box):
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    return Element(id=0, text=text, confidence=1.0, box=box, center=(cx, cy))


def test_fuse_keeps_ocr_text_and_borrows_the_kind():
    """文字以 OCR 为准，kind 由视觉补。视觉的转录没有 OCR 准。"""
    ocr = [_el("完成", (100, 100, 200, 140))]
    vis = [ScreenItem(kind="button", label="完成按钮", box=(102, 98, 198, 142))]
    out = fuse(ocr, vis)
    assert len(out) == 1
    assert out[0].text == "完成", "视觉的措辞盖掉了 OCR 的文字"
    assert out[0].kind == "button" and out[0].source == "both"


def test_fuse_adds_what_ocr_could_not_see():
    """图标、无字按钮 —— 这才是这一层存在的理由。"""
    ocr = [_el("设置", (100, 100, 200, 140))]
    vis = [ScreenItem(kind="icon", label="返回箭头", box=(10, 10, 50, 50))]
    out = fuse(ocr, vis)
    assert len(out) == 2
    icon = [e for e in out if e.source == "vision"][0]
    assert icon.text == "返回箭头" and icon.kind == "icon"


def test_fuse_lets_each_ocr_element_be_claimed_once():
    """两个视觉框都压在同一个 OCR 元素上时，第二个应当自成一个元素，
    而不是把第一个的 kind 覆盖掉。"""
    ocr = [_el("完成", (100, 100, 200, 140))]
    vis = [ScreenItem(kind="button", label="a", box=(100, 100, 200, 140)),
           ScreenItem(kind="icon", label="b", box=(100, 100, 200, 140))]
    out = fuse(ocr, vis)
    assert len(out) == 2
    assert out[0].kind == "button" and out[0].source == "both"


def test_fuse_carries_switch_state():
    ocr = [_el("飞行模式", (100, 100, 200, 140))]
    vis = [ScreenItem(kind="switch", label="飞行模式", box=(100, 100, 200, 140), state="off")]
    assert fuse(ocr, vis)[0].state == "off"


# ---------- 判定 ----------

def test_find_candidate_bar_parses_per_item_boxes():
    bar = judge.find_candidate_bar(_img(600, 1200), "nihao", FakeAsker(
        {"found": True, "items": [{"text": "你好", "box": [100, 200, 200, 240]},
                                  {"text": "你号", "box": [210, 200, 310, 240]}]}))
    assert bar.found and [c.text for c in bar.items] == ["你好", "你号"]
    assert bar.items[0].center == (90, 264), bar.items[0].center


def test_find_candidate_bar_accepts_bare_strings_but_cannot_click_them():
    """模型偷懒只给词不给 box：能拿来核对有没有目标，但点不了。"""
    bar = judge.find_candidate_bar(_img(), "nihao",
                                   FakeAsker({"found": True, "items": ["你好"]}))
    assert bar.found and bar.items[0].center is None


def test_not_found_and_no_answer_are_different_things():
    """⚠ 这两个绝不能合并：前者是模型看过了说没有（切输入法才有依据），
    后者是没问成（这时候什么都不该断定）。"""
    said_no = judge.find_candidate_bar(_img(), "nihao",
                                       FakeAsker({"found": False, "why": "英文模式"}))
    assert said_no is not None and said_no.found is False and said_no.why == "英文模式"
    assert judge.find_candidate_bar(_img(), "nihao", FakeAsker()) is None
    assert judge.find_candidate_bar(_img(), "nihao", None) is None


def test_claims_a_bar_but_names_no_words_is_no_answer():
    """说找到了却给不出词 —— 当成没结果，别拿它做决定。"""
    assert judge.find_candidate_bar(_img(), "nihao",
                                    FakeAsker({"found": True, "items": []})) is None


def test_did_action_work_clamps_confidence():
    eff = judge.did_action_work(_img(), _img(), "tap(x=1,y=2)", "打开设置",
                                FakeAsker({"worked": True, "confidence": 5, "why": "开了"}))
    assert eff.worked and eff.confidence == 1.0


def test_action_describe_leaves_out_the_models_own_reasoning():
    """拿 reason 去问「这个动作生效了吗」，等于让判官先读一遍辩方陈词。"""
    a = Action(name="tap", args={"x": 10, "y": 20}, reason="我觉得这里能点开设置",
               expect="打开设置", call_id="c1")
    assert a.describe() == "tap(x=10，y=20)"
    assert "我觉得" not in a.describe()


# ---------- 接进执行层之后：2026-09-09 那个 bug 的回归 ----------

def _no_bar_env(fake_env):
    """OCR 里没有任何候选栏形状的东西 —— 几何规则必然找不到。

    真机上对应的场景是：候选栏在屏幕上部（贴光标），当时的 y 下限只看下部 30%。
    这里不去复刻那个 y 下限（它已经删了），直接构造「几何这一路交白卷」。
    """
    return _ime_env(fake_env, "nihao", after_pick=["你好"])


def test_vision_finds_the_bar_and_the_ime_must_not_be_toggled(fake_env):
    """**这是今天那个 bug 的回归。**

    几何规则漏了候选栏，视觉找到了 —— 键盘本来就是中文的。这时候切输入法
    等于把一个正确的状态毁掉，重打更没候选，最后报 ime_not_chinese。
    """
    dev, per, _ = _no_bar_env(fake_env)
    obs = per.observe(dev.capture())
    asker = FakeAsker({"found": True,
                       "items": [{"text": "你好", "box": [100, 200, 200, 240]}]})
    res, _new = Executor(dev, per, asker=asker).run(act("type", text="你好"), obs)
    assert ("toggle_ime",) not in dev.calls, (
        f"视觉都说候选栏在了还去切输入法 —— 这正是那个 bug。实际动作：{dev.calls}")
    assert res.ok, f"候选栏找到了却没点成：{res.error} / {res.hint}"
    assert res.extra["picked"] == "你好"


def test_vision_sees_the_bar_but_not_the_target_is_candidate_not_found(fake_env):
    """候选栏在、只是没有目标词 —— 这是 candidate_not_found，不是 ime_not_chinese。

    两句话给模型指的路正好相反：前者「换个词再试」有意义，后者明说了「换词没用」。
    """
    dev, per, _ = _no_bar_env(fake_env)
    obs = per.observe(dev.capture())
    asker = FakeAsker({"found": True,
                       "items": [{"text": "泥好", "box": [100, 200, 200, 240]}]})
    res, _new = Executor(dev, per, asker=asker).run(act("type", text="你好"), obs)
    assert res.error == "candidate_not_found", res.error
    assert ("toggle_ime",) not in dev.calls
    assert res.extra["candidates"] == ["泥好"]


def test_vision_confirms_there_is_no_bar_so_toggling_is_earned(fake_env):
    """模型看过了说确实没有 —— 这时候切输入法是有依据的，老行为该照走。"""
    dev, per, _ = _no_bar_env(fake_env)
    obs = per.observe(dev.capture())
    asker = FakeAsker({"found": False, "why": "键盘是英文模式"})
    Executor(dev, per, asker=asker).run(act("type", text="你好"), obs)
    assert ("toggle_ime",) in dev.calls, "模型都说没有候选栏了，该切一次试试"


def test_no_asker_behaves_exactly_like_before(fake_env):
    """降级路必须一直留着：没有 asker 时行为和 2026-09-09 之前一模一样。"""
    dev, per, _ = _no_bar_env(fake_env)
    obs = per.observe(dev.capture())
    Executor(dev, per).run(act("type", text="你好"), obs)
    assert ("toggle_ime",) in dev.calls


def test_a_failed_vision_call_does_not_change_the_outcome(fake_env):
    """问不成 = 什么都不该断定 = 退回老行为。"""
    dev, per, _ = _no_bar_env(fake_env)
    obs = per.observe(dev.capture())
    Executor(dev, per, asker=FakeAsker()).run(act("type", text="你好"), obs)
    assert ("toggle_ime",) in dev.calls


# ---------- 接进执行层之后：动作是否生效 ----------

def test_judge_can_overturn_a_no_change_verdict(fake_env):
    """像素判据说没变化，看图复核说生效了 —— changed 要翻过来。

    判错的代价不止是一句错话：changed=False 会被记成一次无进展，
    攒到 NO_PROGRESS_STOP 就把整个任务终止了。
    """
    dev, per, frames = fake_env([["同一屏"], ["同一屏"]])
    obs = per.observe(dev.capture())
    asker = FakeAsker({"worked": True, "confidence": 0.9, "why": "开关从关变成了开"})
    res, _ = Executor(dev, per, asker=asker).run(act("tap", x=10, y=10), obs)
    assert res.changed is True
    assert res.extra["judged"]["why"] == "开关从关变成了开"
    assert not res.hint, "都判定生效了还发「可能没点中」的提示，模型会照错的走"


def test_judge_agreeing_keeps_the_hint_and_adds_its_reason(fake_env):
    dev, per, _ = fake_env([["同一屏"], ["同一屏"]])
    obs = per.observe(dev.capture())
    asker = FakeAsker({"worked": False, "confidence": 0.8, "why": "两张图完全一样"})
    res, _ = Executor(dev, per, asker=asker).run(act("tap", x=10, y=10), obs)
    assert res.changed is False
    assert "两张图完全一样" in res.hint


def test_a_real_change_is_never_second_guessed(fake_env):
    """只在否定侧问。有变化就是有变化，不用花钱复核。"""
    dev, per, _ = fake_env([["第一屏"], ["完全不同的第二屏"]])
    obs = per.observe(dev.capture())
    asker = FakeAsker({"worked": False, "confidence": 1.0, "why": "不该问我"})
    res, _ = Executor(dev, per, asker=asker).run(act("tap", x=10, y=10), obs)
    assert res.changed is True
    assert asker.prompts == [], "有变化还去复核，白花钱"


@pytest.mark.parametrize("bad", [None, {"nope": 1}, [1, 2, 3], "文字"])
def test_a_garbage_answer_never_flips_the_verdict(fake_env, bad):
    dev, per, _ = fake_env([["同一屏"], ["同一屏"]])
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per, asker=FakeAsker(bad)).run(act("tap", x=10, y=10), obs)
    assert res.changed is False
    assert "judged" not in res.extra


def _committed_env(fake_env):
    """字上屏了，但前后两屏差别小到低于阈值 —— 真机上中文输入就长这样。

    实测 hamming=1、text_diff=1，阈值是 2 / 8，两个都不到。
    """
    first = _PAD + ["Q 搜索"]
    typing = _PAD + ["1你好", "2 你号", "Q 搜索"]
    picked = _PAD + ["你好", "Q 搜索"]
    return fake_env([first] + [typing] * 4 + [picked] * 4)


def test_a_committed_chinese_word_is_not_reported_as_no_change(fake_env):
    """中文上屏了却报「没变化」—— 2026-09-09 真机实测的浪费，钉住它。

    'nihao' 变成 '你好' 只贡献 hamming=1、text_diff=6，两个都低于阈值（2 / 8）。
    当时模型据此以为没输入成功，去点了撤销、又重打一遍，白花两步。
    这条路不走 _after，所以复核必须在 _commit_candidate 里自己做。
    """
    dev, per, _ = _committed_env(fake_env)
    obs = per.observe(dev.capture())
    asker = FakeAsker({"worked": True, "confidence": 0.95, "why": "输入框里出现了「你好」"})
    res, _new = Executor(dev, per, asker=asker).run(act("type", text="你好"), obs)
    assert res.ok and res.extra["picked"] == "1你好"
    assert res.changed is True, "字都上屏了还报没变化，模型会以为白打了"
    assert res.extra["judged"]["why"] == "输入框里出现了「你好」"


def test_a_committed_word_without_an_asker_keeps_the_old_verdict(fake_env):
    """降级路：没有 asker 时照旧按像素判据报，不假装。"""
    dev, per, _ = _committed_env(fake_env)
    obs = per.observe(dev.capture())
    res, _new = Executor(dev, per).run(act("type", text="你好"), obs)
    assert res.ok and "judged" not in res.extra
    assert res.changed is False


# ---------- 手：回车与退格。driver 早就能做，工具面一直没接 ----------

def test_return_is_reachable_from_the_tool_surface(fake_env):
    """回车的 keycode 从第一天就在 KEY 表里，enter() 也写好了，但工具面从没接过它
    —— 全项目零调用点。后果是模型完全没有换行 / 确认 / 发送的手段：真机上它想在
    备忘录里另起一行，只能反复点文本区，9 步熔断。那不是笨，是手里没这把工具。
    """
    from iphone_agent.driver.injector import KEY
    from iphone_agent.harness.actions import validate_action

    dev, per, _ = fake_env([["一屏"], ["另一屏"]])
    obs = per.observe(dev.capture())
    action = validate_action(Action("key", {"name": "return"}, "换行", None, "c1"), obs)
    res, _ = Executor(dev, per).run(action, obs)
    assert res.ok
    assert ("key", "return") in dev.calls, dev.calls
    # 回车**不带 cmd** —— home/app_switcher/spotlight 是 cmd+数字，它不是。
    from iphone_agent.driver.device import Device
    assert Device._KEY_COMBOS["return"] == (KEY["return"], [])
    assert Device._KEY_COMBOS["home"][1] == [KEY["cmd"]]


def test_the_tool_surface_actually_offers_return_and_erase():
    """光在 driver 里能做不算数 —— 模型看到的是工具表。enter() 就是这么被埋了一整个项目的。"""
    from iphone_agent.harness.tools import tool_defs

    defs = {t["function"]["name"]: t["function"] for t in tool_defs(True)}
    assert "erase" in defs
    key_enum = defs["key"]["parameters"]["properties"]["name"]["enum"]
    assert "return" in key_enum, key_enum


def test_erase_deletes_exactly_what_it_was_asked_to(fake_env):
    from iphone_agent.harness.actions import validate_action

    dev, per, _ = fake_env([["有字"], ["没字"]])
    obs = per.observe(dev.capture())
    action = validate_action(Action("erase", {"count": 3}, "删错字", None, "c1"), obs)
    res, _ = Executor(dev, per).run(action, obs)
    assert res.ok and ("backspace", 3) in dev.calls


@pytest.mark.parametrize("bad", [0, -1, 101, 1.5, True, "3", None])
def test_erase_refuses_a_bad_count_instead_of_clamping(bad):
    """⚠ 这里**不能夹住**。模型说「退 500 个」时它想的是「清空」，
    一夹变成退 100 个，删掉的就是它没打算删的正文，而且没有回头路。
    """
    from iphone_agent.harness.actions import ValidationError, validate_action

    with pytest.raises(ValidationError):
        validate_action(Action("erase", {"count": bad}, "", None, "c1"), None)


def test_vision_splits_candidates_that_ocr_glued_together(fake_env):
    """OCR 把几个候选粘成一个元素时，看图的那一方能拆开。

    2026-09-09 真机：打 'ming'，OCR 吐出 ['1名', '2明 3命4鸣']。剥掉编号是
    '明 3命4鸣'，不等于 '明' —— 只认完全相等（'设置'/'摄制' 差一个音，点错没有
    回头路），所以整体放弃，报 candidate_not_found。模型试了两次都卡在这儿。
    而它们在屏幕上本来就是分开的三个词，视觉能给出各自的坐标。
    """
    dev, per, _ = _ime_env(fake_env, "1名", "2明 3命4鸣", after_pick=["明"])
    obs = per.observe(dev.capture())
    asker = FakeAsker({"found": True, "items": [
        {"text": "名", "box": [100, 200, 160, 240]},
        {"text": "明", "box": [170, 200, 230, 240]},
        {"text": "命", "box": [240, 200, 300, 240]},
    ]})
    res, _new = Executor(dev, per, asker=asker).run(act("type", text="明"), obs)
    assert res.ok, f"视觉都把候选拆开了还是没点成：{res.error}"
    assert res.extra["picked"] == "明"


def test_glued_candidates_still_fail_cleanly_when_the_target_is_absent(fake_env):
    """拆开之后确实没有目标 —— 照旧报 candidate_not_found，但把拆开后的候选交回去，
    比交一串粘着的有用。"""
    dev, per, _ = _ime_env(fake_env, "1名", "2明 3命4鸣", after_pick=["名"])
    obs = per.observe(dev.capture())
    asker = FakeAsker({"found": True, "items": [
        {"text": "名", "box": [100, 200, 160, 240]},
        {"text": "命", "box": [240, 200, 300, 240]},
    ]})
    res, _new = Executor(dev, per, asker=asker).run(act("type", text="明"), obs)
    assert res.error == "candidate_not_found"
    assert res.extra["candidates"] == ["名", "命"], res.extra["candidates"]


# ---------- 工具结果要说「变成了什么」，不只是「变了」 ----------

def test_tool_result_says_what_appeared_not_just_that_something_changed(fake_env):
    """2026-09-09 真机（记一笔账）：点了数字 1，金额从 '0.00' 变成 '1'，
    工具只回 {"changed": true}。模型没看出来，把「输入第一位数字1」又执行一遍，
    金额成了 '11'，此后 6 步全在收拾烂摊子，最后熔断 —— **坐标一次都没点错**。

    答案本来就在手边（OCR 文字集合差），以前只在 context_mode=state 时注入，
    默认的 window 模式算了就扔。
    """
    dev, per, _ = fake_env([["金额 0.00", "键盘 1"], ["金额 1", "键盘 1"]])
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("tap", x=10, y=10), obs)
    assert res.extra.get("appeared") == ["金额 1"], res.extra
    assert res.extra.get("disappeared") == ["金额 0.00"], res.extra


def test_no_change_means_no_appeared_key_at_all(fake_env):
    """屏幕没变就不要塞两个空列表进去 —— extra 里每个键都要有话说。"""
    dev, per, _ = fake_env([["同一屏"], ["同一屏"]])
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("tap", x=10, y=10), obs)
    assert "appeared" not in res.extra and "disappeared" not in res.extra


def test_the_log_records_where_each_element_came_from():
    """source 总是写。漏了它，日志里就分不出哪些元素来自视觉那一路 ——
    排查真机失败时我因此误判过一次「视觉层没生效」。"""
    from types import SimpleNamespace

    from iphone_agent.harness.runlog import observation_snapshot
    from iphone_agent.perceive.elements import Element

    els = [Element(id=1, text="完成", confidence=1.0, box=(0, 0, 10, 10), center=(5, 5)),
           Element(id=2, text="返回箭头", confidence=0.0, box=(0, 0, 10, 10), center=(5, 5),
                   kind="icon", source="vision"),
           Element(id=3, text="飞行模式", confidence=1.0, box=(0, 0, 10, 10), center=(5, 5),
                   kind="switch", state="off", source="both")]
    obs = SimpleNamespace(window_rect=SimpleNamespace(x=0, y=0, w=1, h=1),
                          width_px=10, height_px=10, ahash=0, elements=els)
    rows = observation_snapshot(obs, "f.png")["elements"]
    assert [r["source"] for r in rows] == ["ocr", "vision", "both"]
    assert "kind" not in rows[0], "没有 kind 的元素不该塞一个空字段"
    assert rows[1]["kind"] == "icon"
    assert rows[2]["state"] == "off"


# ---------- 截断：可见，但不丢已经说出来的部分 ----------

def test_extract_json_rescues_a_truncated_array():
    """2026-09-09 真机：屏幕解析输出 1000+ token 撞上 profile 的 max_tokens=1024，
    JSON 断在半截，整屏视觉贡献归零 —— 而前面 40 个元素其实是好的。
    半个列表远好过一个空列表。"""
    cut = ('[{"kind":"icon","label":"返回箭头","box":[1,2,3,4]},'
           '{"kind":"button","label":"支出","box":[5,6,7,8]},'
           '{"kind":"button","label":"标')
    got = extract_json(cut)
    assert isinstance(got, list) and len(got) == 2, got
    assert got[1]["label"] == "支出"


def test_a_truncated_reply_is_reported_as_truncated_not_as_gibberish():
    """截断和「模型答非所问」是两种病，说法必须分开：前者加额度，后者改 prompt。
    以前两者都报「回复里没有可解析的 JSON」，指错了方向。"""
    class Truncating:
        last_finish_reason = "length"
        last_max_tokens = 1024

        def ask(self, *a, **k):
            return "这里没有任何 JSON"

    asker = VisionAsker(Truncating())
    assert asker.ask_json("问题", [Image.new("RGB", (10, 10))]) is None
    assert "截断" in asker.last_error and "1024" in asker.last_error
    assert asker.truncations == 1
    assert asker.stats()["truncations"] == 1


def test_a_rescued_truncation_still_says_the_result_is_partial():
    """抢救成功也要说一声 —— 否则会以为这屏就这么点东西可点。"""
    class Truncating:
        last_finish_reason = "length"
        last_max_tokens = 1024

        def ask(self, *a, **k):
            return '[{"kind":"icon","label":"返回","box":[1,2,3,4]},{"kind":"but'

    asker = VisionAsker(Truncating())
    got = asker.ask_json("问题", [Image.new("RGB", (10, 10))])
    assert isinstance(got, list) and len(got) == 1
    assert asker.failures == 0, "抢救出来了就不算失败"
    assert "截断" in asker.last_error


def test_full_screen_parsing_is_on_because_ocr_misses_things():
    """整屏解析默认开：它是任务成功的直接条件，不是锦上添花。

    回归场景（合成名称）：「示例银行」列表项被 OCR 漏掉，只有 vision 能补全。
    我一度以它慢为由默认关掉、改用 zoom 按需放大，那是错的：
    zoom 解决「看不清」，不解决「不知道有」。
    """
    from iphone_agent import config

    assert config.SCREEN_PARSE is True
    assert not hasattr(config, "SCREEN_PARSE_MAX_TOKENS"), \
        "输出长度不该由我们拍一个数；不传 max_tokens 就是不限制"


def test_ask_does_not_cap_output_by_default():
    """⚠ 不传 max_tokens = 这个字段整个不出现 = 不限制。

    一度默认拿 profile 的 m.max_tokens 顶上，而那个数是给「回一个工具调用」定的，
    屏幕解析列几十个元素一列就超，JSON 断在半句话、整屏归零 ——
    而那一屏里「示例银行」只有视觉读得到，OCR 没有它。这会导致任务失败。
    """
    from types import SimpleNamespace

    from iphone_agent.model.transports.chat_completions import ChatCompletionsTransport

    seen = {}

    class Completions:
        @staticmethod
        def create(**kw):
            seen.clear()
            seen.update(kw)
            return SimpleNamespace(model_dump=lambda: {
                "choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}]})

    t = ChatCompletionsTransport.__new__(ChatCompletionsTransport)
    t.resolved = SimpleNamespace(
        provider=SimpleNamespace(omit_params=(), extra_body={}),
        model=SimpleNamespace(id="m", max_tokens=1024, extra_body={}))
    t._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions))
    t.last_finish_reason = None
    t.last_max_tokens = None

    t.ask([{"role": "user", "content": "hi"}])
    assert "max_tokens" not in seen, f"默认就不该限制，实际发了 {seen.get('max_tokens')}"

    t.ask([{"role": "user", "content": "hi"}], max_tokens=200)
    assert seen["max_tokens"] == 200, "调用方明确要限制时才限制"


def test_ocr_text_inside_a_button_fuses_into_one_element():
    """⚠ 这是 IoU 度量的死穴，也是换成包含关系的理由。

    OCR 的「完成」文字框套在视觉的按钮框里面，面积只有它的 8% —— IoU ≈ 0.08，
    永远够不到 0.3，于是同一个按钮被拆成两个元素、两个编号。
    2026-09-09 真机实测：'+'(OCR) 和「加号图标按钮」(视觉) 中心只差 13px 却没合并。
    **每一个带文字的按钮都会中招**，不是个例。
    """
    from iphone_agent.perceive.elements import _iou

    text_box = (100, 110, 140, 130)        # 「完成」两个字
    button_box = (60, 90, 220, 160)        # 整个按钮
    assert _iou(text_box, button_box) < 0.3, "前提：IoU 确实够不到阈值"

    ocr = [_el("完成", text_box)]
    vis = [ScreenItem(kind="button", label="完成按钮", box=button_box)]
    out = fuse(ocr, vis)
    assert len(out) == 1, [(e.text, e.source) for e in out]
    assert out[0].text == "完成" and out[0].kind == "button" and out[0].source == "both"


def test_far_apart_boxes_still_do_not_fuse():
    """包含关系不能松到把邻居也吞进来。"""
    ocr = [_el("完成", (100, 110, 140, 130))]
    vis = [ScreenItem(kind="button", label="取消按钮", box=(300, 90, 460, 160))]
    assert len(fuse(ocr, vis)) == 2


def test_the_full_screen_switch_does_not_disable_zoom(monkeypatch, fake_env):
    """config.SCREEN_PARSE 只管「每次观察都整屏解析」。

    一度把它做在 VisionAsker.enabled 上，结果关掉整屏解析，连 zoom 的局部解析
    和各种看图复核一起废了 —— 而那些正是要留着的。两件事的成本差 6 倍。
    """
    from iphone_agent import config
    from iphone_agent.perceive import observe as observe_mod

    calls = []
    monkeypatch.setattr(observe_mod, "parse_screen",
                        lambda img, asker: calls.append(img.size) or [])
    monkeypatch.setattr(config, "SCREEN_PARSE", False)

    dev, per, _ = fake_env([["一屏"]])
    per.asker = object()          # 视觉能力可用
    obs = per.observe(dev.capture())
    assert calls == [], "整屏解析该被开关拦住"

    per.zoom(obs, (0, 0, 200, 400))
    assert len(calls) == 1, "zoom 的局部解析不该被同一个开关拦住"
