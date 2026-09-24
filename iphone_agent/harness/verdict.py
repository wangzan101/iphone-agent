"""一个动作做完，成了没有：成了 / 没成 / 不知道（spec 2026-09-18）。

⚠ 2026-09-18 两个现场（一次真机运行的第 12 步、一次真机运行的第 16 步）：
  点歪进了别的 App 的登录页、打「18」只上屏「8」，出事那一步核对机制都在、都跑了、都给出了「不知道」
  （expect_check.by=missing、verified=unknown），然后都按 ok 放行 —— 模型读到的是没人解释过的裸字段，
  守卫把它记成进展。09-12 之后执行成功的点击 61% 是这种「没人核过」的。

规矩（CLAUDE.md §2、§3）：
· 写结论只有一个入口：Executor._conclude。这里只管**读**和**说**。
· of() 先认结果里自带的 verdict；没有（旧留档）才从旧字段推导。推导分支不给新代码用。
· 像素判据说「没变」且没人复核 → not_met/pixels，不是 unknown：它一直就是守卫「无进展」的依据，
  改成 unknown 会让没有看图的一方时 no_progress 和 dead_taps 全部失灵（计划 R7）。

⚠ 2026-09-18（最终评审 I5）：**「读结论」目前只收口了三家**，别以为全项目都经过这里 ——
  guard.record_result、history.row_from_record、eval/taps.py 走 of()；而
  · twin/record.py 的 off_track_of 仍直接读 judged.on_change / judged.worked
  · skills/extract.py 仍读 result["verified"] == "true"
  · harness/procedure.py 仍读 res.extra["verified"]
  这三家靠 Executor._conclude 把旧字段**一字不差**地照写撑着：改 _conclude 的旧字段就会静默改它们的行为。
  把它们也收口到 of() 是下一稿的范围（CLAUDE.md §7）。
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from iphone_agent.harness.actions import BUILTIN_ACTION_NAMES, screen_neutral

# ⚠ 2026-09-22（终审 M-5）：只借 LEVELS 这个常量当白名单，perceive/screen.py 不反过来导入 harness，
#   没有循环导入（screen.py 只依赖 dataclasses / collections.abc）。
from iphone_agent.perceive.screen import LEVELS

STATES = ("met", "not_met", "unknown", "n/a")


@dataclass(frozen=True)
class Verdict:
    state: str
    by: str
    why: str = ""
    seen: str = ""

    def to_json(self) -> dict:
        d = {"state": self.state, "by": self.by}
        if self.why:
            d["why"] = self.why
        if self.seen:
            d["seen"] = self.seen
        return d

    @staticmethod
    def from_json(d) -> Verdict | None:
        if not isinstance(d, dict) or d.get("state") not in STATES:
            return None
        return Verdict(d["state"], str(d.get("by") or ""), str(d.get("why") or ""), str(d.get("seen") or ""))


def _by_judged(result: dict) -> Verdict | None:
    j = result.get("judged")
    if not isinstance(j, dict) or not isinstance(j.get("worked"), bool):
        return None
    return Verdict("met" if j["worked"] else "not_met", "vision", why=str(j.get("why") or ""))


def _by_pixels(result: dict) -> Verdict:
    return Verdict("met", "pixels") if result.get("changed") else Verdict("not_met", "pixels")


def of(name: str, args: dict, result: dict) -> Verdict:
    if not result.get("ok"):
        return Verdict("n/a", "failed")
    if name in ("done", "handover") or screen_neutral(SimpleNamespace(name=name, args=args or {})):
        return Verdict("n/a", "neutral")
    if name not in BUILTIN_ACTION_NAMES:
        return Verdict("n/a", "procedure")
    own = Verdict.from_json(result.get("verdict"))
    if own is not None:
        return own
    # ---- 以下只服务旧留档 ----
    ident = result.get("identity") if isinstance(result.get("identity"), dict) else None
    if name == "tap":
        if ident is not None and ident.get("verified") is False:
            return Verdict("not_met", "identity", seen=str(ident.get("seen") or ""))
        if ident is not None and ident.get("verified") is True:
            return Verdict("met", "identity", seen=str(ident.get("seen") or ""))
        if result.get("changed") is False:
            return _by_judged(result) or Verdict("not_met", "pixels")
        ec = result.get("expect_check") if isinstance(result.get("expect_check"), dict) else {}
        if ec.get("met") is True:
            return Verdict("met", str(ec.get("by") or "text"), seen=str(ec.get("matched") or ""))
        if ec.get("met") is False:
            return Verdict("not_met", "vision", why=str((result.get("judged") or {}).get("why") or ""))
        # ⚠ 2026-09-18（最终评审 C1）：变了 + 看图说没达到预期、**没有** expect_check ——
        #   e00fe7c 引入 expect_check 之前那段旧留档的形状。不接过来的话，「进的是通知页」这类
        #   已经判出来的失败在历史表里退成 unknown，证据白丢（CLAUDE.md §3）。
        #   只取 not_met：worked is True 不能接 —— 「低于指纹分辨率 + 没写 expect」的新出口
        #   写的是 unknown/missing，接过来 _aligned 会打架（计划 R1）。
        j = _by_judged(result)
        if j is not None and j.state == "not_met":
            return j
        return Verdict("unknown", "missing" if ec.get("by") == "missing" else "unverified")
    if name == "type":
        # ⚠ 2026-09-18（Task 4 评审）：点候选现在也走 _conclude_type，结果里带 verified —— 有它就按下面
        #   那段通用映射走（true→met/text，wrong→not_met/readback，……），不再走 judged 那条旧推导，
        #   否则选候选的结果会推出 unknown，跟真正写进去的 met/not_met 打架（计划 R1）。
        #   没有 verified 键才是 09-18 之前的旧留档，走原来那条推导。
        if "picked" in result and "verified" not in result:
            j = _by_judged(result)
            return j if (j is not None and j.state == "not_met") else Verdict("unknown", "unverified")
        v = result.get("verified")
        if v == "true":
            return Verdict("met", "text")
        if v == "false":
            return Verdict("not_met", "text")
        if v == "wrong":
            return Verdict("not_met", "readback")
        return Verdict("unknown", "unverified")
    if name == "open_app":
        if ident is not None and isinstance(ident.get("verified"), bool):
            return Verdict("met" if ident["verified"] else "not_met", "identity", seen=str(ident.get("seen") or ""))
        return Verdict("unknown", "unverified")
    if name == "switch_ime":
        return Verdict("unknown", "unverified")
    if name in ("scroll_until", "collect"):
        if result.get("never_moved"):
            return Verdict("unknown", "never_moved", why="一次都没滚动起来，不能说到底了")
        if name == "scroll_until" and not result.get("found"):
            return Verdict("not_met", "text")
        return Verdict("met", "text")
    return _by_judged(result) or _by_pixels(result)       # scroll / key / erase


_LANDED = {
    "home": "打开后停在「{name}」，是这个 App 的首页。",
    "top": "打开后停在「{name}」，是顶层标签页，不是首页。",
    "inner": "打开后停在「{name}」，是从别的页点进来的内页，不是首页；任务要从首页开始的话先退回去。",
    None: "打开后停在「{name}」，看不出是不是首页。",
}

_UNKNOWN_WHY = {
    "missing": "你没写 expect，系统无从核对",
    "unverified": "看图核对没问成",
    "never_moved": "一次都没滚动起来，不能说到底了",
    # 这一条只由 Executor._run_once 的出口兜底写：会改画面的动作 ok 了却没人下结论（漏网的出口）。
    "unconcluded": "执行层没有给这个动作下结论",
}


def say(v: Verdict, name: str, result: dict, quiet: bool = False) -> str | None:
    """放在结果 JSON 前面给模型看的那句话。met / n/a 不说；已经有 hint 说过的不重复说。"""
    if v.state == "unknown":
        if quiet and v.by != "missing":
            return None
        s = f"这一步**没有核实**（{v.why or _UNKNOWN_WHY.get(v.by, v.by)}）。别当它成了：先看当前屏幕确认，再继续。"
        tapped = result.get("tapped")
        if name == "tap" and isinstance(tapped, dict) and tapped.get("hits") == []:
            s += "这一下没有落在任何已知元素上。"
        return s
    if v.state == "not_met":
        if v.by == "readback":
            return f"想打的字没有按原样落进去：输入框里现在是「{v.seen}」。" if v.seen else "想打的字没有落进输入框。"
        if v.by == "identity" and name == "tap":
            return f"打开的不是你要的那个 App，现在是「{v.seen}」。" if v.seen else "打开的不是你要的那个 App。"
    # ⚠ 2026-09-22（spec 打开App落点 §3.2）：打开 App 开成了原来什么都不说。iOS 回到的是上次离开的那一屏，
    #   模型要知道自己是不是在首页 —— on_demand 下它平时不看图，只能由这句话告诉它。只说看图看到的，不掺孪生。
    if v.state == "met" and name == "open_app" and isinstance(result.get("landed"), dict):
        landed = result["landed"]
        if not landed.get("name"):
            # ⚠ 2026-09-22（终审 M-3）：label == "off" 是没看图（on_demand 下这一步本就不看），
            #   不是「看了说不出来」——CLAUDE.md §3「没做」不等于「做失败」。别的状态
            #   （failed/missing/invalid）是真的看了、答坏了，照旧说一句「没看成」提醒别当成已经确认。
            if landed.get("label") == "off":
                return None
            return "打开后没看成停在哪一页。"
        # ⚠ 2026-09-22（终审 M-5）：level 只信 LEVELS 里的三个值——手改过的留档可能塞进一个列表
        # 这类不可哈希的东西，_LANDED.get 直接用它当 key 会 TypeError。先用 in（对 tuple 不需要哈希）
        # 筛一遍，筛不过就当没答，跟 None 走同一句话。
        level = landed.get("level")
        level = level if level in LEVELS else None
        return _LANDED.get(level, _LANDED[None]).format(name=landed["name"])
    return None
