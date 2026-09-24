"""熔断（spec M11）：同屏同动作拒绝；无进展 3 警告 / 6 终止。observe/wait 不计数也不清零。

「同屏」「见过的画面」里的「画面」= Observation.state_key（aHash + 文字集合），不是裸 aHash ——
2026-09-09 真机：裸 aHash 把 App 键盘上每一位数字的输入都判成「回到见过的画面」，见 hashing.state_key。
"""
from __future__ import annotations

import json

from iphone_agent import config
from iphone_agent.harness import verdict as verdict_mod
from iphone_agent.harness.actions import NO_PROGRESS_EXEMPT, Action, screen_neutral

BUCKET_RATIO = 0.05


def signature(action: Action, screen_hash: int, width: int, height: int) -> str:
    args = dict(action.args)
    if action.name == "tap":
        bx = max(1, int(width * BUCKET_RATIO)); by = max(1, int(height * BUCKET_RATIO))
        args = {"bx": int(args["x"]) // bx, "by": int(args["y"]) // by}
    return f"{screen_hash:x}|{action.name}|{json.dumps(args, sort_keys=True, ensure_ascii=False)}"


class ActionGuard:
    def __init__(self):
        self._executed: set[str] = set()
        self._seen_screens: set[int] = set()
        self.no_progress = 0
        # 无进展有两种，给模型的说法必须分开 —— 2026-09-08 真机踩到：
        # 模型在底部 tab 之间来回点，每一步画面都真的变了，熔断判「无进展」判对了，
        # 但告诉它的是「没有检测到画面变化」。于是它以为**没点中**，
        # 换了三个不同的图标继续试 —— 在错误的假设上白烧六步。
        self.last_reason: str | None = None      # "revisited" | "unchanged"
        # 连续几次点击**完全没有反应**。2026-09-08 撞到过一个真实状态：
        # 镜像窗口滚轮还灵、`key` 还灵，**点击整个不生效** —— SkyLight 后台点、
        # CGEvent 前台点、HID 全局点全试了，指针位置对、按键没卡、权限齐、
        # App 重启过，都没用（见 docs/15）。
        # 这时模型收到的是「没检测到变化，可能没点中」，于是它换目标、换方式、
        # 一路试到熔断 —— 20 步全废在一个它根本改变不了的事情上。
        # 2026-09-10 起攒到 DEAD_TAPS_ALERT 不再提示模型，loop 直接走恢复阶梯（recovery.py）。
        self.dead_taps = 0
        # 连续几步「做了、画面也变了，但没人核实成没成」。⚠ 2026-09-18，见 record_outcome 里那段注释。
        # 和 no_progress 是两个计数：没核实的一步既不清零也不加一，只往这里攒。
        self.unverified_streak = 0

    def record_screen(self, screen_hash: int) -> None:
        self._seen_screens.add(screen_hash)

    def check_repeat(self, action: Action, screen_hash: int, w: int, h: int) -> bool:
        if screen_neutral(action) or action.name == "done":
            return False
        return signature(action, screen_hash, w, h) in self._executed

    def record_executed(self, action: Action, screen_hash: int, w: int, h: int) -> None:
        if screen_neutral(action) or action.name == "done":
            return
        self._executed.add(signature(action, screen_hash, w, h))

    def record_result(self, action: Action, res, new_obs, *, count_unverified: bool = True) -> str | None:
        """动作跑完后记一笔。loop 和剧本各调一次，别自己拼参数 —— 「变化在指纹分辨率之下」这条规矩只在这里。

        ⚠ `changed=True` 有两种来路：整屏判据（aHash 汉明 / 文字集合差）说变了，
          或者整屏说没变、**局部 MAD 或看图复核**说变了（executor._after）。后一种意味着
          变化小到 state_key 都看不见（开关翻转：汉明 0、文字差 0、局部 MAD 17）——
          这时指纹没资格说「这画面见过」，不能记成 revisited。一个判据只能在它的分辨率之内下否定结论。

        ⚠ 2026-09-18（Task 7 评审发现 1、2）：`count_unverified=False` 给**自己核对过**的调用方。
          剧本的内部动作从不写 expect（procedure.py 造 Action 那处恒传 None），结论于是结构性地恒是
          unknown/missing；但剧本每一步都用 `fingerprint.matches(step.expect, ...)` 核对过，对不上就
          expect_mismatch 中止 —— 说它「没人核实成没成」是错的。不排除的话有三个后果：
          进剧本时的 no_progress 在整条剧本里再也清不了零（回放到新画面本该清零），
          n=4 时两步没进展就撞 STOP、剧本背一次不该背的失败；新的 "unverified" 会盖掉
          guard_verdict 里的 warn（procedure.py 那几行是按 warn → stop 单调升级写的）；
          还会白白花掉任务唯一那份兜底额度。
        """
        if not res.ok:
            return None
        judged = res.extra.get("judged") or {}
        # ⚠ judged 有两个来路（executor._after）：整屏说没变、看图说变了（低于指纹分辨率）；
        #   或整屏说变了、看图核对有没有达到预期（on_change=True，第三态）。后者的变化指纹看得见，
        #   不算 below_resolution；它 worked=False 就是 off_track。
        on_change = bool(judged.get("on_change"))
        fine = bool(res.changed) and ("local_mad" in res.extra or (bool(judged) and not on_change))
        # ⚠ 2026-09-18（最终评审 I3）：「低于指纹分辨率 + 写了 expect」那条路（executor._after 的
        #   `⚠ 不写 hint` 那一支）第二问答「没达到预期」时，judged 按设计留的是**第一问**
        #   （这个动作生效了吗）的答案、不带 on_change —— 只看 on_change 的旧判据读不到那个否定，
        #   这一步会被记成进展并把无进展计数清零，正是 off_track 当初要堵的那个洞。
        #   结论同时写在 expect_check.met 里，两处一起认。
        ec = res.extra.get("expect_check") or {}
        off_track = bool(res.changed) and ((on_change and judged.get("worked") is False) or ec.get("met") is False)
        key = new_obs.state_key if new_obs is not None else None
        # ⚠ 2026-09-18：「这一步到底成没成」只经 verdict.of() 读（计划 R7、CLAUDE.md §7），
        #   守卫只取其中一位：是不是 unknown（没人核实）。not_met 照旧走上面的 off_track。
        unverified = False
        if count_unverified:
            v = verdict_mod.of(action.name, action.args, {"ok": True, "changed": res.changed, **res.extra})
            unverified = v.state == "unknown"
        return self.record_outcome(action, bool(res.changed), key,
                                   below_resolution=fine, off_track=off_track,
                                   unverified=unverified)

    def record_outcome(self, action: Action, changed: bool,
                       new_screen_hash: int | None = None, below_resolution: bool = False,
                       off_track: bool = False, unverified: bool = False) -> str | None:
        """`changed` 只说画面动了，不说有进展。`new_screen_hash` 是 Observation.state_key。
        `below_resolution`：变化是局部/看图判出来的，指纹看不见 —— 按进展算，不查见没见过。

        2026-09-07 验收第 5 次：主屏 App 页 ⇄ 小组件页来回七步，每一步画面都真的变了，
        于是这里每一步都清零，两道熔断都没拦住。回到一个**见过的**画面是原地打转，
        不是进展——所以进展的判据是「到了一个没见过的画面」。
        `new_screen_hash` 为 None 时（没拿到新观察）退回只看 `changed`。

        `off_track`：画面变了，但看图复核说**没达到模型自己写的预期**（第三态）。
        2026-09-09 读 Mobile-Agent-E / mirroir 的代码才意识到：误点进一个**新的**错页面
        恰好满足上面「变了且是新画面」，会把计数洗干净 —— 误点这类失败在这里是看不见的。
        错页面照样记进 `_seen_screens`（它确实见过了，再回来算 revisited），但不算进展。

        `unverified`：这一步的结论是「不知道」（没人核实成没成，verdict.state == "unknown"）。
        只拦「清零」这一件事：本来要判成进展的那一步改成谁也不算，并往 `unverified_streak` 攒一笔。
        """
        if action.name in NO_PROGRESS_EXEMPT or screen_neutral(action):
            return None
        if changed and new_screen_hash is not None:
            new_screen = below_resolution or new_screen_hash not in self._seen_screens
            self._seen_screens.add(new_screen_hash)
        else:
            new_screen = changed and new_screen_hash is None
        if changed and new_screen and not off_track:
            self.dead_taps = 0
            if unverified:
                # ⚠ 2026-09-18（一次真机运行的第 12 步）：点歪进了微博登录页，没写 expect →
                #   没人核对 →「变了且是新画面」→ 计数清零。错页面被记成了进展，同一个坐标后面又点了两轮。
                #   没核实的一步不能当进展（不清零），也没有证据说它在打转（不加一）。
                # ⚠ unknown **只在这个分支里**生效：下面「没变 / 回到见过的画面 / 偏航」照旧加一。
                #   写成函数开头提前返回的话，没有看图的一方时每个「变了」的点击都是 unknown，
                #   revisited 这道熔断整个就被拆了。
                # ⚠ 只在**正好等于**阈值时报一次 —— 和 no_progress 同一个形状（grant_fallback 的注释）。
                self.unverified_streak += 1
                return "unverified" if self.unverified_streak == config.UNVERIFIED_WARN else None
            self.no_progress = 0
            self.last_reason = None
            self.unverified_streak = 0
            return None
        # ⚠ 2026-09-18（最终评审 I2）：streak 要的是**严格连续**，spec C2 原文是「任何 met/not_met 清零」。
        #   原来这一段（没变 / 回到见过的画面 / 偏航）压根不碰它，于是「没核实 → 核实过的 revisited →
        #   没核实 → 没核实」被数成连续三步：白报一次「连续没核实」，还花掉任务唯一那份兜底额度。
        #   没核实的一步照旧不清零（它正是要攒的那一种）。
        if not unverified:
            self.unverified_streak = 0
        self.last_reason = "wrong_page" if (changed and off_track) else ("revisited" if changed else "unchanged")
        self.no_progress += 1
        if action.name == "tap":
            self.dead_taps = 0 if changed else self.dead_taps + 1
        if self.no_progress == config.NO_PROGRESS_STOP:
            return "stop"
        if self.no_progress == config.NO_PROGRESS_WARN:
            return "warn"
        return None

    def grant_fallback(self) -> None:
        """程序补看了一次全屏（spec 2026-09-14 §3.5），给模型再留 NO_PROGRESS_STOP - NO_PROGRESS_WARN 步。

        ⚠ 必须设回 WARN，不能清零也不能不管：record_outcome 只在计数**正好等于**阈值时报 stop，
          计数越过去就再也不会等于 —— 熔断就被拆了（loop.py 剧本那段注释说的是同一个坑）。
        """
        self.no_progress = config.NO_PROGRESS_WARN


NO_PROGRESS_SAYS = {
    "revisited": ("你回到了**之前已经到过的画面** —— 动作是生效的（画面确实变了），"
                  "但你在原地打转。不要再点同一批入口；换一条完全不同的路，"
                  "或者 done(failed) 说清楚你找不到什么。"),
    "unchanged": ("连续几步**画面都没变** —— 多半是没点中，或者那个位置本来就没反应。"
                  "换目标或换方式，不要原样重复。"),
    "wrong_page": ("动作是生效的（画面变了），但看图复核说**没达到你写的预期** —— 你进了不该进的页面。"
                   "先确认自己现在在哪（必要时返回上一页），别在这个页面上接着做。"),
}


# spec 2026-09-14 §3.4：元素表默认只有 OCR，「列表里找不到」不等于「屏上没有」。每一种无进展都补这一句。
NOT_IN_LIST = "目标不在元素列表里，就 zoom 那一块或 observe 看全屏。"


def no_progress_hint(reason: str | None) -> str:
    """给模型的说法。⚠ 两种无进展必须分开说，见 ActionGuard.last_reason 的注释。"""
    return NO_PROGRESS_SAYS.get(reason or "", NO_PROGRESS_SAYS["unchanged"]) + " " + NOT_IN_LIST


