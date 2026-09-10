"""两次观察之间变了什么 —— 给模型的是**答案**，不是上一张图。

设计说明：一度想每步给模型两张图（动作前 + 动作后）让它自己比。撤回了：
图序会混、看到细微差别会反复纠结、图片开销翻倍。模型要的是「开关翻了没、
字进去没、滚对方向没」这些答案，而程序已经在算 —— 整屏 did_change、
点击处局部 MAD（就是为开关加的）、OCR 文字集合差（算了但从没给模型看过）。
这里把三样拼成一段文字。UI-TARS 叫它 state transition captioning，
那是让模型写；我们用 OCR 差集机械生成，零成本、无图序问题。
"""
from __future__ import annotations

from dataclasses import dataclass

from iphone_agent import config
from iphone_agent.perceive.change import did_change, local_mad


@dataclass(frozen=True)
class Transition:
    changed: bool
    local_changed: bool | None
    added: tuple[str, ...]
    removed: tuple[str, ...]
    added_total: int
    removed_total: int


def transition(before, after, tap_px: tuple[int, int] | None, list_max: int = 15) -> Transition:
    ch = did_change(before, after)
    local = None
    if tap_px is not None:
        local = local_mad(before.image, after.image, *tap_px) >= config.LOCAL_CHANGED_MAD
    # 不退回全量 elements：text_set 是特意去掉状态栏文字的（见 elements.py
    # build_observation 里的注释 —— 时钟/电量每分钟都变，会污染变化判定）。
    # 真实屏幕上非状态栏文字为空是合法状态，不是「没填」，退回反而把状态栏
    # 文字重新混进 added/removed。
    bset = before.text_set
    aset = after.text_set
    # 顺序按元素列表（上到下、左到右），不按集合 —— 模型读到的顺序要和它看到的屏幕一致。
    added_all = [e.text for e in after.elements if e.text in aset - bset]
    removed_all = [e.text for e in before.elements if e.text in bset - aset]
    added = tuple(dict.fromkeys(added_all))       # 去重保序
    removed = tuple(dict.fromkeys(removed_all))
    return Transition(changed=ch.changed, local_changed=local,
                      added=added[:list_max], removed=removed[:list_max],
                      added_total=len(added), removed_total=len(removed))


def _fmt(items: tuple[str, ...], total: int) -> str:
    if not items:
        return "无"
    s = "[" + ", ".join(items) + "]"
    if total > len(items):
        s += f"（共 {total} 项，只列前 {len(items)} 项）"
    return s


def render(t: Transition) -> str:
    head = "整屏：" + ("有变化" if t.changed else "无变化")
    if t.local_changed is not None:
        head += "    点击位置附近：" + ("有变化" if t.local_changed else "无变化")
    return "\n".join(["【上一步之后】", head,
                      "新出现的文字：" + _fmt(t.added, t.added_total),
                      "消失的文字：" + _fmt(t.removed, t.removed_total)])
