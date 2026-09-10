"""准入与验收共用的屏幕导航助手。

## 为什么单独一个文件

这些函数原来在 `driver_gate.py` 和 `acceptance.py` 里各写了一份。
2026-09-07 踩到：用户主屏幕第一页没有「设置」，我给准入加了翻页查找，
**却没同步到验收** —— 验收五次全挂在「找不到设置」，一次真正的测试都没跑。

同一份逻辑存两份、只修一份，是这类脚本最容易犯的错。放这里让它们没法再分叉。

这些都是**已通过准入的原语的组合**，本身不含新机制：
翻页用 scroll、点图标用 tap target=icon_above、返回用 tap 到箭头元素。
"""
from __future__ import annotations

import time

from iphone_agent.driver.geometry import Frame, image_to_screen
from iphone_agent.driver.timing import IOS_TIMING
from iphone_agent.harness.actions import ICON_ABOVE_LABEL_RATIO
from iphone_agent.harness.settle import settle
from iphone_agent.perceive.hashing import ahash

CHEVRONS = "＜<‹〈❮◀️"
BACK_MAX_DEPTH = 8

# ⚠ 横向翻页之后必须多等一下再点，否则点击会落到错的地方。
#   2026-09-07 实测：翻到有「设置」的那一页，观察到标签在 (104,894)，
#   **2 秒后再观察它仍在 (104,894)**（页面确实已经静止），可是紧接着点就打开了
#   另一个 App（记账应用），等 2 秒再点才打开设置。连续三次复现。
#   也就是说这不是「页面还在滑」，是翻页手势结束后有一小段时间点击会被路由到别处。
#   settle 判不出来，因为画面已经不变了。
#   这条同时解释了：验收第 1 次「翻遍主屏各页都没有设置图标」，
#   以及更早那次「本该开设置却开了 Gemini」—— 都是翻页之后太快动手。
AFTER_PAGE_FLIP_S = 2.0


def look(dev, per):
    return per.observe(dev.capture())


def frame_of(o) -> Frame:
    return Frame(o.image, o.width_px, o.height_px, o.window_rect, 0.0, o.frame_id)


def settle_a_bit(dev, kind="key"):
    settle(dev, dev.capture(), IOS_TIMING[kind], ahash)


def find(o, *texts, exact=False):
    for t in texts:
        for e in o.elements:
            if (e.text.strip() == t) if exact else (t in e.text):
                return e
    return None


def find_back(o):
    """左上角的返回控件。

    ⚠ 判据必须包含**箭头字符**，不能只看「位置靠左上 + 文字短」——
    2026-09-07 踩到：设置首页上「飞行模式」满足位置和长度条件，被当成返回控件
    连点了六次（那一行还带开关，误触后果是真的）。
    实测 OCR 会把它读成 "<"、"＜返回"、"＜设置" 等形态，所以按箭头字符找。
    """
    cands = [e for e in o.elements
             if e.center[1] < o.height_px * 0.22
             and e.center[0] < o.width_px * 0.35
             and any(c in e.text for c in CHEVRONS)]
    return min(cands, key=lambda e: e.center[0]) if cands else None


def find_across_pages(dev, per, text, max_pages=5, on_page=None):
    """在主屏幕各页之间往右翻找一个 App 标签。

    返回 (observation, element)，找不到则 element 为 None。
    右滑看下一页（下一页在右侧），这一条已在准入里验证过。
    """
    o = look(dev, per)
    el = find(o, text, exact=True)
    if el is not None:
        return o, el
    for i in range(max_pages):
        dev.scroll("right", "page")
        settle_a_bit(dev, "scroll")
        time.sleep(AFTER_PAGE_FLIP_S)      # 见文件顶部 AFTER_PAGE_FLIP_S 的实测记录
        o = look(dev, per)
        el = find(o, text, exact=True)
        if el is not None:
            if on_page:
                on_page(i + 1)
            return o, el
    return o, None


def tap_icon_above(dev, o, label):
    """点 App 图标 —— 图标中心在标签中心上方约 3 倍文字高处。

    ⚠ 点标签本身在 iOS 上**没有任何反应**，必须点图标（准入里有专门的对照项）。
    偏移量由代码算，不让模型估坐标。
    """
    lift = ICON_ABOVE_LABEL_RATIO * (label.box[3] - label.box[1])
    x, y = label.center[0], max(0, label.center[1] - lift)
    dev.tap(*image_to_screen(x, y, frame_of(o)))
    settle_a_bit(dev, "tap")


def tap_element(dev, o, el):
    dev.tap(*image_to_screen(*el.center, frame_of(o)))
    settle_a_bit(dev, "tap")


def back_to_root(dev, per, max_depth=BACK_MAX_DEPTH) -> bool:
    """一路点返回，直到左上角没有返回控件。到不了首页返回 False。"""
    for _ in range(max_depth):
        o = look(dev, per)
        back = find_back(o)
        if back is None:
            return True
        tap_element(dev, o, back)
    return False
