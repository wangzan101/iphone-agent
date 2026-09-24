"""一步剧本「该到的屏」由哪些证据认定。**全项目只有这一份规则**（CLAUDE.md §7）。

⚠ 2026-09-16 真机上的一次运行：一条 2026-09-07 学到的剧本，第 1 步就
  `expect_mismatch` 废掉 —— 指纹是
  `["今 ④5", "设置", "蜂窝网络", "向右箭头", "个人热点", "电池图标（绿色电池）", "电池", "VPN图标（蓝色地球仪）"]`。
  8 个词里 4 个永远对不上：`今 ④5` 是状态栏时钟的 OCR 乱码，每分钟都变；
  `向右箭头` / `…图标（…）` 只有整屏解析那一路才给得出，而日常按需看图（on_demand），
  观察里只有 OCR。核对要 90% 覆盖，于是**当年在 always 模式下提的每一条剧本都一跑就废** ——
  学到的剧本一步也省不下来。这不是某个文件坏了，是这一类文件全坏了（CLAUDE.md §1）。

规矩：进指纹的证据必须**跨时间稳定、且在任何解析模式下都一样**：

1. **只要 OCR 那一路读到的行**（`is_ocr_row`）。视觉那一路给无字图标起的名字每次措辞都不同
   （同一个撤销按钮先后叫「撤销图标（向左弯曲箭头）」「撤销按钮（…）」），而且 on_demand 下根本不出现。
2. **状态栏那一条裁掉**（`below_status_bar`）。时钟、电量每分钟都变；裁剪几何只有
   `config.STATUS_BAR_CROP_RATIO` 这一条，aHash、`Observation.text_set` 与这里共用 ——
   这个项目为「一条规矩两个入口」栽过五次（`perceive/elements.py` 的注释）。

两个方向都按这一条办，老文件才会立刻自愈：提取/合并时**建**指纹按它过滤；
执行时**比**指纹（`harness/procedure.py`）也按它算 —— 盘上写着的、这条规则如今不认的词
直接不算数，而不是把整条剧本拖垮。
"""
from __future__ import annotations

from iphone_agent import config

# 指纹少于这么多个稳定锚点，就不足以说明「到了该到的屏」：这一步标「指纹弱」。
# 也是执行时的核对门槛（见 matches）。
WEAK_EXPECT = 4


def is_ocr_row(e: dict) -> bool:
    """这一行元素是不是 OCR 那一路读到的（source 为 ocr / both）。唯一入口：孪生身份、
    任务判定器与剧本指纹共用（CLAUDE.md §7；`twin/identify.py` 从这里 import 回去）。
    旧留档没有 source：confidence == 0 的是视觉元素（2026-09-11：新留档 1797 个视觉元素全为 0，
    OCR/融合 7475 个里 1 个）。"""
    src = e.get("source")
    return src in ("ocr", "both") or (src is None and e.get("confidence") != 0)


def below_status_bar(center_y: float, height: int) -> bool:
    """这一行在状态栏下面吗。状态栏裁剪只有 `config.STATUS_BAR_CROP_RATIO` 这一条几何。"""
    return center_y > height * config.STATUS_BAR_CROP_RATIO


def stable_texts(elements, height: int) -> list[str]:
    """一帧的元素（留档里的字典形式）→ 可以进指纹的文字，保序去重。"""
    out: list[str] = []
    for e in elements or []:
        if not is_ocr_row(e):
            continue
        c = e.get("center")
        if not isinstance(c, (list, tuple)) or len(c) != 2 or not below_status_bar(c[1], height):
            continue
        t = str(e.get("text") or "").strip()
        if t and t not in out:
            out.append(t)
    return out


def is_weak(expect) -> bool:
    """过滤之后锚点太少 = 这一步的指纹说明不了「到了哪」。CLI 的「指纹弱」和 weak_steps 共用它。"""
    return len(expect) < WEAK_EXPECT


def matches(expect, texts) -> tuple[bool, list[str]]:
    """到没到该到的屏 +（这一刻屏上还在的锚点）。`texts` 传 `Observation.text_set` ——
    它就是上面这条规则应用在当前这一帧上的结果（OCR、状态栏以下），所以**这条规则如今不认的词
    永远命中不了**，老文件里的状态栏乱码和视觉名字自动不算数。

    ⚠ 门槛是「至少 WEAK_EXPECT 个锚点还在」，不是 `screenmap.COVERAGE` 的 90% 覆盖。
      覆盖率那条是给认屏用的：在几十个候选屏里挑一个，宁可认不出也不能认错。这里问的是另一个
      问题 —— 「我是不是在这条剧本期望的那一屏上」，只有是/否两个答案。按覆盖率算，指纹里多记
      一个词就把门槛抬高一分，于是**指纹越长越脆**：老文件里 8 个词有 4 个是废的就直接判死，
      而这 4 个废词恰恰是这条规则要扔掉的。锚点的条数本来就是抽样（EXPECT_WORDS 只取前 8 个），
      不该当成「这一屏必须一字不差」。4 个稳定锚点同时出现在一屏上，已经是很强的肯定信号；
      锚点本来就不足 4 个的（指纹弱那类）则一个都不能少。
    """
    if not expect:
        return True, []
    seen = set(texts)
    got = [a for a in expect if a in seen]
    return len(got) >= min(len(expect), WEAK_EXPECT), got
