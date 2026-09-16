"""认屏（spec 2026-09-12 §4）：纯程序，不调模型，不写文件。

「这一屏叫什么」「是不是候选里的某一屏」「属于哪个 App」是语义判断，来自视觉标注；这里只做核对。
⚠ 2026-09-12：旧版用头部带（0.11–0.18H）/ 返回前缀「<」/ 弱屏覆盖率认屏。2026-09-11 真机：备忘录
  大标题落在 0.184H（头部带之下）、返回键是图标 OCR 读不到、提醒事项首页没有导航标题 —— 列表页
  大多「未命名」。每修一处就多一条规则，是补丁跑步机（CLAUDE.md §1、§2），整套删掉。
⚠ 没有标注就不认（spec §4.3，codex 评审第 9 条）：锚点「稳定」不等于「有区分度」——「编辑」「完成」
  跨屏跨 App 共用；「只有一屏满足」还取决于孪生当时有多全，不是可靠的肯定结论。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from iphone_agent.memory.screenmap import SYSTEM, UNKNOWN, app_id_for
from iphone_agent.perceive.screen import SYSTEM_WORD, UNSURE_WORD

_BACK_PREFIX = "<〈‹"             # NFKC 之后「＜」已经是「<」


def _raw(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text or "").split())


def norm_text(s: str) -> str:
    """认屏、锚点落地、记转移目标、匹配任务文本共用的唯一归一化入口（开发原则 §7）。
    NFKC → 去空白 → 去开头返回箭头 → 连续数字换成 # → 没有字母/数字类字符的丢掉。"""
    t = _raw(s).lstrip(_BACK_PREFIX)
    t = re.sub(r"\d+", "#", t)
    if not any(unicodedata.category(c)[0] in "LN" for c in t):
        return ""
    return t


def is_ocr_row(e: dict) -> bool:
    """这一行元素是不是 OCR 那一路读到的（source 为 ocr / both）。唯一入口：孪生身份与任务判定器共用（CLAUDE.md §7）。
    旧留档没有 source：confidence == 0 的是视觉元素（2026-09-11：新留档 1797 个视觉元素全为 0，OCR/融合 7475 个里 1 个）。"""
    src = e.get("source")
    return src in ("ocr", "both") or (src is None and e.get("confidence") != 0)


def ocr_elements(elements) -> list[dict]:
    """只要 OCR 那一路的文字。⚠ 视觉解析给无字图标起的名字每次不一样（同一个撤销按钮先后叫
    「撤销图标（向左弯曲箭头）」「撤销按钮（…）」），不能进身份。旧留档没有 source：
    confidence == 0 的就是视觉元素（2026-09-11：新留档 1797 个视觉元素全为 0，OCR/融合 7475 个里 1 个）。"""
    out = []
    for e in elements:
        if not is_ocr_row(e):
            continue
        c = e.get("center")
        if not str(e.get("text") or "").strip() or not isinstance(c, (list, tuple)) or len(c) != 2:
            continue
        out.append(e)
    return out


def ocr_texts(elements) -> frozenset[str]:
    """这一帧 OCR 读到的文字（归一化）。锚点落地、挑候选都拿它比。"""
    return frozenset(t for t in (norm_text(str(e.get("text") or "")) for e in ocr_elements(elements)) if t)


def grounded(anchors, texts: frozenset[str]) -> tuple[str, ...]:
    """标注锚点里 OCR 真看得到的那些（归一化、去重、保序）。模型编的字进不来。"""
    out: list[str] = []
    for a in anchors:
        n = norm_text(a)
        if n and n in texts and n not in out:
            out.append(n)
    return tuple(out)


def plain_app_id(label_app: str) -> str | None:
    """标注 App 名 → App id，不查别名（查别名的是 TwinState.resolve_app）。系统 → SYSTEM；不确定 → None。"""
    t = (label_app or "").strip()
    if t == SYSTEM_WORD:
        return SYSTEM
    if not t or t == UNSURE_WORD:
        return None
    return app_id_for(t)


@dataclass(frozen=True)
class Recognition:
    state: str        # matched | new | skipped | unlabeled | label_conflict | candidate_unresolved
    screen_id: str | None = None
    name: str | None = None
    anchors: tuple[str, ...] = ()
    alias: str | None = None          # same_as 认出、而这次叫法不同：要加进别名集合


def identify(label, owner: str, texts: frozenset[str], candidates: tuple[dict, ...], state) -> Recognition:
    """spec §4.2 的表。label 已过 parse_screen_label：same_as 一定在候选范围内。"""
    anchors = grounded(label.anchors, texts)
    if label.same_as is not None:
        c = candidates[label.same_as - 1]
        if str(c.get("app_id") or "") != owner:
            return Recognition("label_conflict")
        s = state.get(owner, str(c.get("screen_id") or ""))
        if s is None:
            # 按 spec §4.1 屏 id 实时与重放一致，这里不该发生；发生了就是 bug，必须看得见。
            return Recognition("candidate_unresolved")
        alias = label.name if label.name not in s.names() else None
        return Recognition("matched", s.id, s.name, anchors, alias)
    if owner in (SYSTEM, UNKNOWN):
        return Recognition("skipped")
    # ⚠ 2026-09-12：spec §4.2 最后一行——「系统」或「不确定」又没有 same_as → 不建屏、不记转移。只看 owner 不够：
    #   owner 可能来自核过身份的进入 / tracker（评审 probe A：open_app 设置 之后一帧标「不确定」，
    #   旧代码照样建出一屏「设置」；终审 P1：open_app 设置 之后第一帧是权限弹窗、标「系统」，建出一屏
    #   设置「允许访问位置」，带着 navigated 转移进了【位置】）。模型说这帧是系统界面、或者自己都说不准
    #   是哪个 App，就不是一个可以新建的 App 屏。
    if plain_app_id(label.app) in (None, SYSTEM):
        return Recognition("skipped")
    # same_as 为 null：新建。同 App 已有同名屏也新建（宁可分开，不可合错），计数交给 TwinState.create。
    return Recognition("new", None, label.name, anchors)
