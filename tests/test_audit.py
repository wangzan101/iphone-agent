"""事后审计的纯逻辑。

2026-09-08 一晚出了两次假成功，两次都是人肉看截图才发现的。
"""
from iphone_agent.harness.audit import (
    RunFacts,
    fabricated_values,
    named_targets,
    never_reached,
)


def facts(result, *screen):
    return RunFacts("r1", "t", result, set(screen))


def test_a_value_that_was_never_on_screen_is_flagged():
    """屏幕上从没出现过的数字，模型只能是编的 —— 精确性由构造保证。"""
    assert fabricated_values(facts("iOS 版本是 17.2.1", "设置", "18.3.1")) == ["17.2.1"]


def test_a_value_that_was_on_screen_is_not_flagged():
    assert fabricated_values(facts("iOS 版本是 18.3.1", "iOS版本", "18.3.1>")) == []


def test_it_matches_inside_longer_screen_text():
    """OCR 常把值和别的字挤在一个元素里（'18.3.1>'、'已使用 81.86 GB'）。"""
    assert fabricated_values(facts("已用 81.86 GB", "iPhone 储存空间 81.86 GB / 128 GB")) == []


def test_single_digits_are_not_treated_as_values():
    """一位数噪声太大 —— '共 3 笔' 里的 3 不该拿去核对。"""
    assert fabricated_values(facts("共 3 笔账", "记账")) == []


def test_only_bracketed_words_count_as_named_targets():
    """⚠ 不加这条的话，「不要修改任何设置」里的「设置」会被当成目标 ——
    实测这一个误判就能把审计结果整个淹掉。"""
    assert named_targets("打开设置，进入「关于本机」，不要修改任何设置项") == ["关于本机"]
    assert named_targets("打开设置，找到 iOS 的系统版本号") == []


def test_the_known_false_success_would_not_be_caught_by_value_check():
    """诚实记录这个检查抓不到什么：那次报了 iOS 26.6.1（真值 18.3.1），
    而「更新到 iOS 26.6.1」确实在屏幕上 —— **假成功不是幻觉，是读错了地方**。"""
    assert fabricated_values(facts("iOS 26.6.1", "更新到iOS 26.6.1", "设置")) == []


# --- 「有没有到过该到的屏」 ---
#
# 用两条真实轨迹验证过（runs/20260908-022551-5204 假成功、024924-eca3 真成功）：
# 前者标为可疑、后者放行。下面是同样形状的合成用例。

def _map(*runs):
    from iphone_agent.memory.screenmap import ScreenMap
    m = ScreenMap()
    for i, steps in enumerate(runs):
        m.ingest_run(steps, f"r{i}")
    return m


def _obs(*texts):
    return {"elements": [{"id": i + 1, "text": t} for i, t in enumerate(texts)]}


def _s(n, texts, action=None):
    d = {"step": n, "observation": _obs(*texts), "result": {"ok": True}}
    if action:
        d["action"] = action
    return d


def _tap(eid):
    return {"name": "tap", "args": {"id": eid}}


TRIP = [_s(1, ["通用", "关于本机", "软件更新"], _tap(2)),
        _s(2, ["关于本机", "iOS版本", "18.3.1"])]


def test_a_run_that_never_reached_the_named_screen_is_flagged():
    m = _map(TRIP, TRIP)          # 跑两遍，两个节点都有指纹
    f = RunFacts("bad", "进入「关于本机」读版本号", "iOS 26.6.1", set())
    detour = [_s(1, ["通用", "关于本机", "软件更新"])]     # 停在通用页，没进去
    assert never_reached(m, f, detour) == ["关于本机"]


def test_a_run_that_did_reach_it_is_not_flagged():
    m = _map(TRIP, TRIP)
    f = RunFacts("good", "进入「关于本机」读版本号", "18.3.1", set())
    assert never_reached(m, f, TRIP) == []


def test_an_unidentifiable_destination_is_not_audited():
    """⚠ 目的地本身认不出来的不能拿来审计 —— 那时「没到过」其实是「认不出来」。
    漏了这一条，第一版把 5 次正确的运行标成了假成功。"""
    once = [_s(1, ["主屏", "天气图标"], _tap(2)), _s(2, ["天气", "示例城区", "28°"])]
    m = _map(once)                # 只跑一遍：目的地只到过 1 次，不可识别
    f = RunFacts("x", "打开「天气」App 读温度", "28°", set())
    assert never_reached(m, f, []) == []


def test_a_task_without_brackets_is_not_audited():
    m = _map(TRIP, TRIP)
    f = RunFacts("x", "打开设置，找到 iOS 的系统版本号", "26.6.1", set())
    assert never_reached(m, f, []) == []
