import pytest

from iphone_agent import config
from iphone_agent.harness.messages import (
    ELEMENTS_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    MessageLog,
    build_state_parts,
    build_state_text,
    middle_truncate,
)


def build(n):
    log = MessageLog()
    log.system("sys")
    for i in range(1, n + 1):
        log.user_observation([("obs_elements", f"observation #{i}\n[1] a")], image_b64=f"img{i}")
        log.assistant_tool_call(f"c{i}", "tap", '{"id":1}', "")
        log.tool_result(f"c{i}", "x" * 1000)
    return log


def test_raw_keeps_everything():
    raw = build(3).raw()
    imgs = [m for m in raw if m["role"] == "user" and any(p.get("type") == "image_url" for p in m["content"])]
    assert len(imgs) == 3


def test_windowed_keeps_last_image_only():
    w = build(3).windowed(keep_images=1, keep_elements=1)
    users = [m for m in w if m["role"] == "user"]
    assert len(users) == 3
    assert all(p["type"] == "text" for p in users[0]["content"])
    assert IMAGE_PLACEHOLDER in users[0]["content"][-1]["text"]
    assert ELEMENTS_PLACEHOLDER in users[0]["content"][0]["text"]
    assert any(p.get("type") == "image_url" for p in users[-1]["content"])
    assert "observation #3" in users[-1]["content"][0]["text"]


def test_older_tool_output_truncated_but_the_latest_is_not():
    """两档预算：最新那条结果模型正要据以行动，砍掉它等于让它盲着走下一步；
    更老的它已经用过了，留着只是历史。"""
    w = build(3).windowed(max_tool_chars=500, max_latest_tool_chars=20000)
    tools = [m for m in w if m["role"] == "tool"]
    assert len(tools) == 3
    assert all(len(m["content"]) <= 520 for m in tools[:-1])
    assert len(tools[-1]["content"]) == 1000


def test_a_long_latest_tool_result_is_still_capped():
    """给足不等于不设限：超过最新那一档也照样截，免得一条结果撑爆上下文。"""
    log = MessageLog()
    log.tool_result("c1", "x" * 30000)
    assert len(log.windowed()[-1]["content"]) == config.LATEST_TOOL_RESULT_MAX_CHARS + 1   # +1 省略号


def test_assistant_message_has_tool_calls_shape():
    w = build(1).windowed()
    a = [m for m in w if m["role"] == "assistant"][0]
    assert a["tool_calls"][0]["id"] == "c1"
    assert a["tool_calls"][0]["function"]["name"] == "tap"


# ---- 状态报告视图（docs/19 §2）----

def test_freeze_rejects_prefix_edits():
    log = MessageLog()
    log.system("sys")
    log.user_text("任务：t", seg="task")
    log.freeze()
    with pytest.raises(RuntimeError):
        log.user_text("late", seg="task")
    with pytest.raises(RuntimeError):
        log.system("late")


def test_prefix_hash_is_stable_across_appends_after_freeze():
    """freeze() 之后追加的 observation/assistant/tool 消息不进 prefix_hash 的
    计算范围——它们本来就不是「前缀」（docs/19 §2），hash 只该盯着前缀本身。"""
    log = MessageLog()
    log.system("sys")
    log.user_text("任务：t", seg="task")
    log.freeze()
    h0 = log.prefix_hash()
    log.user_observation([("obs_elements", "E")], "img")       # 冻结后仍可追加（这是事实）
    log.assistant_tool_call("c1", "tap", "{}", "")
    assert log.prefix_hash() == h0, "追加非前缀消息不该改变前缀 hash"


def test_prefix_hash_changes_if_someone_edits_the_prefix_through_raw():
    """freeze() 不是真不可变：raw() 暴露内部 list。这条测试记录这个事实，
    并证明 hash 能抓到它——这正是 prefix_drift 存在的理由。"""
    log = MessageLog()
    log.system("sys")
    log.user_text("任务：t", seg="task")
    log.freeze()
    h0 = log.prefix_hash()
    log.raw()[0]["content"] = "tampered"
    assert log.prefix_hash() != h0


def test_freeze_records_frozen_prefix_hash():
    """freeze() 拍下的 frozen_prefix_hash 就是当时的 prefix_hash()——loop.py
    靠这两个值的对比判断 prefix_drift，这里钉住它们在 freeze 那一刻相等。"""
    log = MessageLog()
    log.system("sys")
    log.user_text("任务：t", seg="task")
    assert log.frozen_prefix_hash is None       # freeze 之前没有意义
    log.freeze()
    assert log.frozen_prefix_hash == log.prefix_hash()


def test_state_view_is_prefix_plus_one_user_message():
    log = build(3)          # raw 里有 3 张图、3 次 tool_call
    log.freeze()
    v = log.state_view("STATE", image_b64="cur")
    assert [m["role"] for m in v] == ["system", "user"]
    assert v[-1]["content"][0]["text"] == "STATE"
    assert v[-1]["content"][1]["image_url"]["url"].endswith("cur")
    assert not any(m["role"] in ("tool", "assistant") for m in v)


def test_middle_truncate():
    s = "a" * 100 + "b" * 100
    t = middle_truncate(s, 50)
    assert t.startswith("a" * 25) and t.endswith("b" * 25) and "已省略 150 字" in t
    assert middle_truncate("short", 50) == "short"


def test_build_state_text_sections_in_order():
    s = build_state_text(history="【到目前为止】\n（还没有动作）", memory=None, last_result=None,
                         transition=None, elements_text="observation #1\n[1] a", step=1, max_steps=30)
    assert s.index("【到目前为止】") < s.index("【当前屏幕】") < s.index("第 1/30 步")
    assert "【备忘】" not in s and "【上一步结果】" not in s
    # transition 自带【上一步之后】标题（transition.render 的输出），这里照它的形状给。
    s2 = build_state_text(history="【到目前为止】\nH", memory="M", last_result="R",
                          transition="【上一步之后】\nT",
                          elements_text="E", step=2, max_steps=30)
    order = ["【到目前为止】", "【备忘】", "【上一步结果】", "【上一步之后】", "【当前屏幕】", "第 2/30 步"]
    idx = [s2.index(k) for k in order]
    assert idx == sorted(idx)


def test_build_state_parts_names_and_order():
    parts = build_state_parts(history="【到目前为止】\nH", memory="M", last_result="R",
                              transition="【上一步之后】\nT", elements_text="E",
                              step=2, max_steps=30)
    assert [seg for seg, _ in parts] == [
        "state_history", "state_memo", "state_last_result",
        "state_transition", "state_elements", "state_step"]


def test_build_state_parts_skips_absent_sections():
    parts = build_state_parts(history="H", memory=None, last_result=None,
                              transition=None, elements_text="E", step=1, max_steps=30)
    assert [seg for seg, _ in parts] == ["state_history", "state_elements", "state_step"]


def test_build_state_text_is_exactly_the_join_of_parts():
    """签名和返回值不变是这次重构的全部要求：现有 9 条测试一条都不该改。"""
    kw = {"history": "【到目前为止】\nH", "memory": "M", "last_result": "R",
              "transition": "【上一步之后】\nT", "elements_text": "E", "step": 2, "max_steps": 30}
    assert build_state_text(**kw) == "\n\n".join(t for _, t in build_state_parts(**kw))


import json
from pathlib import Path

from iphone_agent.harness.messages import INTERNAL_KEYS, to_wire

GOLDEN_DIR = Path(__file__).parent / "data"


def build_all_shapes() -> MessageLog:
    """覆盖 to_wire 必须处理的全部消息形状（spec §2.1 的四条路径 + 常规）。

    v1 曾断言「windowed/state_view 一律剥掉下划线键」，Codex 核出是错的：
    保留的 image_url 原样 append 不剥、非 list 的 content 走 continue 不剥、
    system 只是浅拷贝不剥。这个 log 就是按那三条漏网路径构造的。
    """
    log = MessageLog()
    log.system("sys")                                   # 顶层不剥的路径
    log.user_text("prefix-1", seg="task")
    log.user_text("prefix-2", seg="task")
    log.user_observation([("obs_elements", "obs #1\n[1] a")], image_b64="img1")   # 会被滑窗替换成占位符
    log.assistant_tool_call("c1", "tap", '{"id":1}', "thinking")   # 非 list content
    log.tool_result("c1", "y" * 50)                      # 非 list content
    log.user_observation([("obs_elements", "obs #2\n[2] b")], image_b64="img2")   # 保留的 image_url
    return log


def _dump(view) -> str:
    return json.dumps(view, sort_keys=True, ensure_ascii=False, indent=2)


def test_wire_payload_matches_the_golden_snapshot():
    """硬约束的度量衡：发出去的字节永远等于实施前的字节（Global Constraints 第 1 条）。

    这份快照在任务 1 生成，那时代码里还没有 _seg。之后每加一处标记，
    这条测试都必须保持绿；它一红就说明有标记漏出网了。
    """
    log = build_all_shapes()
    cases = [("wire_window_golden.json", log.windowed()),
             ("wire_state_golden.json", log.state_view("state text", "imgS"))]
    for name, view in cases:
        expected = (GOLDEN_DIR / name).read_text(encoding="utf-8").rstrip("\n")
        assert _dump(to_wire(view)) == expected, name


def test_wire_payload_has_no_internal_keys_at_any_depth():
    log = build_all_shapes()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                assert k not in INTERNAL_KEYS and not k.startswith("_"), k
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(to_wire(log.windowed()))
    walk(to_wire(log.state_view("s", "i")))


# ---- 任务 6：视图打 _seg（spec §8 R4）----

def test_every_wire_part_has_a_segment_upstream():
    """带标记的视图里，每个 part 都要有 _seg —— 漏标的会进 unknown 段，
    而 unknown 出现在汇总里就是「有东西没被计量」的信号（spec §8 R4）。"""
    log = build_all_shapes()
    for view in (log.windowed(), log.state_view("state text", "imgS")):
        for m in view:
            c = m.get("content")
            if isinstance(c, list):
                for p in c:
                    assert "_seg" in p, (m.get("role"), p)
            else:
                # system / assistant / tool 的 content 不是 list，段名打在顶层 —— 
                # 只查 list 分支的话，删掉 tool_result 的 _seg 不会让任何测试变红。
                assert "_seg" in m, m.get("role")


def test_state_view_marks_each_state_section_separately():
    """段名要逐段可查，但协议结构不能变（否则黄金快照会红）——所以段信息
    挂在合并后 text part 的 `_obs_parts` 上，而不是拆成多个 top-level part。"""
    log = MessageLog()
    log.system("sys")
    log.user_text("任务：t", seg="task")
    log.freeze()
    parts = build_state_parts(history="H", memory="M", last_result=None,
                              transition=None, elements_text="E", step=1, max_steps=30)
    view = log.state_view("\n\n".join(t for _, t in parts), "img", parts=parts)
    state_msg = view[-1]
    text_part = next(p for p in state_msg["content"] if p.get("type") == "text")
    image_part = next(p for p in state_msg["content"] if p.get("type") == "image_url")
    segs = [seg for seg, _ in text_part["_obs_parts"]]
    # 合并即协议不变的根据：分段挂在旁边，模型看到的字节仍是 join 出来的那一段。
    assert text_part["text"] == "\n\n".join(t for _, t in parts)
    # 顺序，不只是包含 —— 段序错了 token 也会记错段。
    assert segs == [s for s, _ in parts]
    # 协议结构没被拆成 N 个 part。这是「一个 text part + 一个 image part」
    # 这条核心设计决策在 parts= 分支上唯一能直接失败的断言点。
    assert len(state_msg["content"]) == 2
    assert image_part["_seg"] == "image"


def test_state_view_with_parts_is_byte_identical_to_build_state_text():
    """生产（loop.py）永远走 parts= 分支，黄金快照走的却是 parts=None 分支。
    「发出去的字节一字不变」这条硬约束在真实路径上必须有一条自己的钉子：
    parts= 产出的 text part 内容要与 build_state_text(同样 kwargs) 逐字符相等。"""
    from iphone_agent.harness.messages import build_state_text
    log = MessageLog()
    log.system("sys")
    log.user_text("任务：t", seg="task")
    log.freeze()
    kwargs = {"history": "H", "memory": "M", "last_result": "R", "transition": "T",
                  "elements_text": "E", "step": 3, "max_steps": 30}
    parts = build_state_parts(**kwargs)
    expected = build_state_text(**kwargs)
    view = log.state_view(expected, "img", parts=parts)
    text_part = next(p for p in view[-1]["content"] if p.get("type") == "text")
    assert text_part["text"] == expected


def test_state_view_rejects_state_text_that_diverged_from_parts():
    """state_text 曾是个「被计算、被传入、然后被丢弃」的哑参数：两个 builder
    一旦分叉没人会发现。现在它是断言。"""
    log = MessageLog()
    log.system("sys")
    log.freeze()
    parts = build_state_parts(history="H", memory=None, last_result=None,
                              transition=None, elements_text="E", step=1, max_steps=30)
    with pytest.raises(ValueError, match="分叉"):
        log.state_view("完全不一样的文本", "img", parts=parts)


def test_to_wire_does_not_mutate_the_view_it_is_given():
    """loop 里只在传给 decide 的位置转换：`view` 是带标记的原件（后面的计量层
    按它记账），`to_wire(view)` 是发出去的那份。to_wire 一旦就地剥，原件就没了。"""
    log = build_all_shapes()
    for view in (log.windowed(), log.state_view("state text", "imgS")):
        before = _dump(view)
        to_wire(view)
        assert _dump(view) == before


# ---- 任务 6 第二轮：windowed() 里 _obs_parts 的去留（回归网，spec §8 R4）----

def test_windowed_drops_obs_parts_when_truncated_but_keeps_it_when_not():
    """下一个任务要把观察拆成 obs_elements/obs_location/obs_route 三段。window 模式下
    要是 `_obs_parts` 丢了，这三段会被压回顶层单一 _seg —— 两段的 token 量算到
    第三段头上，静默错账，日志上看不出来。

    build(3) 配 keep_elements=1（默认）：前两条 elements part 会被滑窗截断（文本换成
    「首行 + 占位符」，原分段不再对应实际文本，_obs_parts 必须丢弃），最后一条不截断
    （_obs_parts 必须原样带过去）。"""
    w = build(3).windowed(keep_elements=1)
    users = [m for m in w if m["role"] == "user"]
    assert len(users) == 3

    truncated_el = next(p for p in users[0]["content"]
                        if p.get("type") == "text" and ELEMENTS_PLACEHOLDER in p.get("text", ""))
    assert "_obs_parts" not in truncated_el

    kept_el = next(p for p in users[-1]["content"] if p.get("type") == "text")
    assert "_obs_parts" in kept_el
    assert kept_el["_obs_parts"] == [("obs_elements", "observation #3\n[1] a")]


def test_windowed_leaves_seg_missing_when_upstream_forgot_to_tag_it():
    """漏标的 elements part 必须继续缺 `_seg` 键，不能被写进一个 `_seg: None`——
    那会让「每个 part 都有 _seg」的防线在漏标时也放行。"""
    log = MessageLog()
    log.system("sys")
    log._msgs.append({"role": "user", "content": [
        {"type": "text", "text": "obs #1\n[1] a", "_kind": "elements"},
        {"type": "image_url", "_seg": "image", "_px": [0, 0],
         "image_url": {"url": "data:image/png;base64,img1"}},
    ]})
    w = log.windowed(keep_elements=1)
    el = next(p for p in w[-1]["content"] if p.get("type") == "text")
    assert "_seg" not in el
