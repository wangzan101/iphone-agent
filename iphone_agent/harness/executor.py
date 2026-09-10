"""执行已校验的动作：activate/窗口核对 → 注入 → 稳定等待 → did_change → hint（spec §6.1、§6.4）。"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from itertools import pairwise

from pypinyin import lazy_pinyin

from iphone_agent import config
from iphone_agent.driver.device import WindowMoved
from iphone_agent.driver.geometry import OutOfWindow, image_to_screen
from iphone_agent.driver.injector import ActivateFailed
from iphone_agent.driver.timing import IOS_TIMING
from iphone_agent.harness import judge, recap
from iphone_agent.harness.actions import Action, icon_y_above_label, screen_neutral
from iphone_agent.harness.settle import settle
from iphone_agent.perceive import transition as tr
from iphone_agent.perceive.change import did_change, local_mad
from iphone_agent.perceive.elements import Observation
from iphone_agent.perceive.hashing import ahash
from iphone_agent.twin.layout import infer_page, norm_label, same_page

HINTS = {
    "tap": "未检测到明显变化：可能没点中，也可能该处本就无反应。换目标或换方式，不要原样重复。",
    "scroll": "未检测到变化：可能已到底/到顶，或这个方向上没有可滚的内容。",
    "key": "未检测到变化。",
    "erase": "未检测到变化：可能光标前已经没有字符了，也可能输入框根本没聚焦（先 tap 它）。",
    "open_app": "未检测到变化：App 可能没打开。",
}
TYPE_REGION_RADIUS = 0.12  # 以最近一次 tap 为中心，取图像高度 12% 的区域做回读


@dataclass
class ToolResult:
    ok: bool
    error: str | None = None
    changed: bool | None = None
    hamming: int | None = None
    text_diff: int | None = None
    settled: bool | None = None
    hint: str | None = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = {"ok": self.ok}
        for k in ("error", "changed", "hamming", "text_diff", "settled", "hint"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        d.update(self.extra)
        return json.dumps(d, ensure_ascii=False)


# 打字没落进去时说什么。⚠ 只陈述事实，不劝模型「先 tap 输入框」或「别再试了」——
# 2026-09-08 批跑：34 次打字全没落，每次都收到「先 tap 输入框聚焦」，模型 tap→type 循环白烧几十步；
# 09-09 改成第三次说「别试了」，模型照样换 20 种方式试。设备故障不该交给模型：
# 有恢复阶梯（recovery.py）时它根本看不到这句；没有阶梯时它看到的也只是事实。
TYPING_NOT_LANDED = "按键没有落进输入框（这次是 {what!r}）：键盘通道可能失效了。"


# 还在 Spotlight 里的标志。open_app 开没开成看它，不看画面变没变；自动探索（eval/explore.py）
# 开完 App 的身份核对也用它 —— 一个规则一个入口。
SPOTLIGHT_MARKERS = ("最佳搜索结果", "在App中搜索", "在 App中搜索", "Siri建议", "Siri 建议",
                     "Top Hit", "Top Hits", "Search in Apps", "Siri Suggestions")


def in_spotlight(obs) -> bool:
    return any(m.casefold() in e.text.casefold() for e in obs.elements for m in SPOTLIGHT_MARKERS)


# 主屏幕的特征文字：一屏里出现好几个，说明在主屏（或跳出 App 回到了主屏）。
HOME_MARKERS = (("设置", "Settings"), ("备忘录", "Notes"), ("计算器", "Calculator"),
                ("时钟", "Clock"), ("照片", "Photos"), ("日历", "Calendar"), ("相机", "Camera"),
                ("Safari",), ("App Store",), ("微信", "WeChat"))
HOME_MARKERS_MIN = 3


def looks_like_home(obs) -> bool:
    texts = {e.text.casefold() for e in obs.elements}
    # Neither a bilingual label nor repeated mentions of one app count as several icons.
    matched = set()
    for text in sorted(texts):
        for index, aliases in enumerate(HOME_MARKERS):
            if index not in matched and any(alias.casefold() in text for alias in aliases):
                matched.add(index)
                break
    return len(matched) >= HOME_MARKERS_MIN


# 布局表直达没开成的原因，固定就这几个（写进最终结果的 extra["layout_miss"]，eval/verify.py 按它计数）：
#   no_hit             表里没有这个 App
#   page_out_of_range  表里的页码不在 1..HOME_PAGES_MAX（表被写坏 / 手改），不翻
#   label_missing      翻到了那一页，当前帧上找不到这个标签（布局过时）
#   out_of_window      标签上方的图标算出来在窗口外
#   still_home         点完还在 Spotlight / 主屏 / 同一页网格上：没开成
#   error              孪生自己的逻辑抛了异常（表坏了）
LAYOUT_MISSES = ("no_hit", "page_out_of_range", "label_missing", "out_of_window", "still_home", "error",
                 "wrong_app")


def _still_on_same_home_page(before, after) -> bool:
    """点完之后还认得出点之前那一页主屏的网格 = 没开成。

    `looks_like_home` 的特征词全是系统 App：一页全是第三方 App 的主屏上点空了，点完的画面
    既不像 Spotlight 也不像主屏，直达会报 ok=True, via=layout 假成功（2026-09-10 终审 M2）。
    这里不靠特征词，靠「点前点后是不是同一组标签排成的同一页网格」。
    """
    a = infer_page(after.elements, after.width_px, after.height_px)
    if a is None:
        return False
    b = infer_page(before.elements, before.width_px, before.height_px)
    return b is not None and same_page(a.labels, b.labels)


# 回主屏第一页要按几次 home：第一次退出前台 App（回到的是退出前那一页，不一定是第一页），
# 后续几次才把系统翻回第一页（真机实测；抄自 `_open_app_from_home` 原有注释的原意）。
HOME_PRESSES = 3


def return_to_first_home_page(dev, frame):
    """回主屏第一页：连按 HOME_PRESSES 次 home，每次之后 settle，返回最后一次 settle 到的帧。

    一个规则一个入口（项目开发约定）：`_open_app_from_layout`、`_open_app_from_home`、
    `twin/scan.scan_home` 都要回主屏第一页，统一调这里，别再各写一份「按 3 次 home」。

    每轮传给 settle 的帧选哪个都不影响行为：它的 `before` 从不被读（见 harness/settle.py 的 docstring）。
    """
    for _ in range(HOME_PRESSES):
        dev.key("home")
        frame, _ = settle(dev, frame, IOS_TIMING["key"], ahash)
    return frame


def spotlight_query(name: str) -> str:
    """App 名 → 往 Spotlight 里打的那串字母。

    · 纯英文：原样。
    · 纯中文：整个转拼音（'一木记账' → 'yimujizhang'）。Spotlight 按拼音匹配 App 名，
      2026-09-08 实测干净的搜索框里打这串，它直接出现在「最佳搜索结果」。
    · **中英混合：只用英文部分**（'Safari 浏览器' → 'Safari'）。
      2026-09-09 评测集跑出来的：lazy_pinyin 把整个名字转成 'Safari liulanqi'，
      Spotlight 搜不到；而截图证明打 'safari' 就出「最佳搜索结果 Safari 浏览器」。
      英文部分本身就是唯一标识，拼音只会把它搅浑。
    """
    if name.isascii():
        return name
    latin = "".join(ch for ch in name if ch.isascii() and (ch.isalnum() or ch == " ")).strip()
    if latin:
        return latin
    return "".join(lazy_pinyin(name))


class Executor:
    def __init__(self, device, perceiver, store=None, runs_root=None, catalog=None,
                 asker=None, recovery=None, layout=None):
        self.dev = device
        self.per = perceiver
        # 恢复阶梯（harness/recovery.py）。None = 不救，验证失败就原样报错 —— 测试和纯观察的场合用。
        # 设备的事是系统的事：有 recovery 时，通道失效对模型是透明的，它只看到成功或 device_error。
        self.recovery = recovery
        # 设备层孪生的布局表（twin/layout.Layout）。None = 没有，open_app 直接走 Spotlight。
        # 它只是参考：给候选页和格子，标签在当前帧上找、坐标用当前帧算、开没开成看身份（设计说明）。
        self.layout = layout
        # 会看图的那一方。只在硬规则给出**否定结论**时才问它 —— 见 harness/judge.py。
        # None 就是全部退回老行为，一行都不差。
        self.asker = asker
        self.store = store
        self.runs_root = runs_root
        self.catalog = catalog
        self._last_tap_px: tuple[int, int] | None = None
        # 输入法当前是不是中文。⚠ 这是**信念不是事实** —— 没有任何 API 能查，
        # 只能从证据推：打一串拼音字母，出候选栏就是中文，没候选就是英文。
        # None = 还不知道。切换之后必须重新验证，不能想当然地翻转。
        self._ime_chinese: bool | None = None

    def run(self, action: Action, obs: Observation) -> tuple[ToolResult, Observation | None]:
        """跑一个动作；验证说通道失效且有恢复阶梯，就爬梯重做。模型看不到中间过程。"""
        res, new = self._run_once(action, obs)
        channel = self._channel_failure(action, res)
        if channel is None or self.recovery is None:
            return res, new
        box: dict = {}

        def redo() -> bool:
            # 重做前重新观察：镜像可能重启过，帧序号和窗口都是新的。screen 本身没变（手机没重启）。
            cur = self.per.observe(self.dev.capture())
            r, n = self._run_once(action, cur)
            box["r"], box["n"] = r, n
            return r.ok and self._channel_failure(action, r) is None

        self.recovery.climb(channel, redo)          # 全失败会抛 DeviceChannelDead，loop 记 device_error
        return box["r"], box["n"]

    @staticmethod
    def _channel_failure(action: Action, res: ToolResult) -> str | None:
        """这个结果是不是「通道失效」而不是「模型点错了」。只认执行层自己验证出来的否定。"""
        if action.name == "type" and (res.error == "keystrokes_not_landing"
                                      or res.extra.get("verified") == "false"):
            return "keyboard"
        if action.name == "open_app" and res.error == "app_not_found" and res.extra.get("typed") is False:
            return "keyboard"
        return None

    def _run_once(self, action: Action, obs: Observation) -> tuple[ToolResult, Observation | None]:
        n = action.name
        if n == "done":
            return ToolResult(ok=True), None
        if n == "recall":
            body = self.store.read(action.args["name"]) if self.store else None
            if body is None:
                # 不做模糊匹配 —— 猜错比查不到坏。
                return ToolResult(ok=False, error="memory_not_found",
                                  hint=f"没有名叫 {action.args['name']!r} 的记录"), None
            return ToolResult(ok=True, extra={"content": body}), None
        if n == "recall_runs":
            lines, _skipped = recap.run_summaries(self.runs_root, action.args["limit"])
            return ToolResult(ok=True, extra={"runs": lines}), None
        if n == "use_skill":
            body = self._skill_text(action.args["kind"], action.args["name"])
            if body is None:
                return ToolResult(ok=False, error="skill_not_found",
                                  hint=f"索引里没有 {action.args['kind']} {action.args['name']!r}"), None
            return ToolResult(ok=True, extra={"content": body}), None
        if n == "search_memory":
            # 索引超过 top-K 被截断时，这是模型找到「没被列出来的那些」的唯一入口
            # ——不在索引里不等于没有，只是没被注入。跟 recall 一样查文字、不改画面。
            hits = []
            if self.store is not None:
                hits = [{"name": e.name, "description": e.description,
                        "mark": recap.mark_for(e), "score": score}
                       for e, score in self.store.search(action.args["query"],
                                                          config.SEARCH_MEMORY_TOPK)]
            return ToolResult(ok=True, extra={"hits": hits}), None
        if n == "observe":
            new = self.per.observe(self.dev.capture())
            return ToolResult(ok=True, changed=None), new
        if n == "zoom":
            # 只看不动。changed=None 而不是 False：屏幕**本来就不该**变，
            # 报 False 会被 guard 记成一次无进展，攒够就熔断 —— 而它明明在好好干活。
            a = action.args
            box = (a["x1"], a["y1"], a["x2"], a["y2"])
            new = self.per.zoom(obs, box)
            inside = sum(1 for e in new.elements
                         if box[0] <= e.center[0] <= box[2] and box[1] <= e.center[1] <= box[3])
            return ToolResult(ok=True, changed=None,
                              extra={"zoomed": list(box), "elements_in_region": inside}), new
        if n == "wait":
            import time
            time.sleep(float(action.args["seconds"]))
            return self._after(action, obs, extra={})
        try:
            self.dev.ensure_frame_valid(self._frame_of(obs))
            extra = self._inject(action, obs)
        except WindowMoved as e:
            new = self.per.observe(self.dev.capture())
            return ToolResult(ok=False, error="window_moved", hint=f"窗口移动了，已重新观察：{e}"), new
        except ActivateFailed as e:
            return ToolResult(ok=False, error="activate_failed", hint=str(e)), None
        except OutOfWindow as e:
            return ToolResult(ok=False, error="out_of_window", hint=str(e)), None
        except Exception as e:  # driver 其他异常
            return ToolResult(ok=False, error="device_error", hint=f"{type(e).__name__}: {e}"), None
        if "_done" in extra:            # 中文输入、光标键：_inject 已经自己观察并下了结论
            return extra["_done"], extra["_obs"]
        if n in ("open_app", "scroll_until", "collect"):
            return extra  # 这几个自己内部循环，观察与结果都自己出
        return self._after(action, obs, extra)

    # ---- 注入 ----
    def _frame_of(self, obs: Observation):
        """用观察时的矩形重建 Frame：坐标换算与窗口核对都只认观察时那一组。"""
        from iphone_agent.driver.geometry import Frame
        return Frame(obs.image, obs.width_px, obs.height_px, obs.window_rect, 0.0, obs.frame_id)

    def _inject(self, action: Action, obs: Observation):
        n, a = action.name, action.args
        if n == "tap":
            frame = self._frame_of(obs)
            sx, sy = image_to_screen(a["x"], a["y"], frame)
            self.dev.tap(sx, sy)
            self._last_tap_px = (a["x"], a["y"])
            return {"screen": [sx, sy]}
        if n == "scroll":
            self.dev.scroll(a["direction"], a["amount"]); return {}
        if n in ("scroll_until", "collect"):
            return self._scroll_scan(action, obs)
        if n == "switch_ime":
            self.dev.toggle_ime()
            self._ime_chinese = None      # 切完是什么模式要重新验，不能假设翻转成功
            return {}
        if n == "erase":
            self.dev.backspace(a["count"]); return {}
        if n == "key":
            self.dev.key(a["name"])
            if screen_neutral(action):
                # 光标键：画面本来就不该变（光标不可见）。像 observe 一样报 changed=None，
                # 不做变化判定、不问看图复核、不进熔断 —— 判 False 会让模型以为没按中而重按。
                frame, _ = settle(self.dev, self._frame_of(obs), IOS_TIMING["key"], ahash)
                return {"_done": ToolResult(ok=True, changed=None,
                                            hint="光标移动看不见，接着 type / erase / key(return) 就行。"),
                        "_obs": self.per.observe(frame)}
            return {}
        if n == "type":
            if not a["text"].isascii():
                return self._type_via_ime(a["text"], obs)
            baseline = self._region_texts(obs)
            baseline_all = set(obs.text_set)
            self.dev.type(a["text"])
            # ⚠ 中文模式下打 ASCII，字母会变成**拼音待上屏**，英文根本打不出来 ——
            #   而且是静默失败：动作报成功、屏幕也变了（候选栏冒出来了）。
            #   混合输入必须管这一侧，不能只管中文那一侧。
            frame, _ = settle(self.dev, self._frame_of(obs), IOS_TIMING["type"], ahash)
            after = self.per.observe(frame)
            if self._candidates(after, a["text"]):
                self._ime_chinese = True
                self.dev.backspace(len(a["text"]))     # 只退自己刚打的
                self.dev.toggle_ime()
                self.dev.type(a["text"])
                self._ime_chinese = False
            else:
                self._ime_chinese = False
            return {"_baseline": baseline, "_baseline_all": baseline_all, "_target": a["text"]}
        if n == "open_app":
            return self._open_app(a["name"], obs)
        raise ValueError(n)

    # ---- 动作后 ----
    def _after(self, action: Action, before: Observation, extra: dict):
        frame_before = self._frame_of(before)
        frame, settled = settle(self.dev, frame_before, IOS_TIMING[action.name], ahash)
        new = self.per.observe(frame)
        ch = did_change(before, new)
        changed = ch.changed
        local = None
        if action.name == "tap" and self._last_tap_px is not None and not changed:
            # ⚠ 整屏判据看不见小控件。2026-09-08 实测：开关翻转两次，
            #   aHash 汉明都是 0、文本差都是 0，而点击处周围的局部 MAD 是 17.0 / 16.98；
            #   点空白处三次局部 MAD 都是 0.0。
            #   不看局部的后果不是「少报一次变化」：模型点了开关、工具说没变化，
            #   它以为没点中就再点一次 —— 又翻回去了，熔断还把这算成无进展。
            local = local_mad(before.image, new.image, *self._last_tap_px)
            changed = local >= config.LOCAL_CHANGED_MAD
        res = ToolResult(ok=True, changed=changed, hamming=ch.hamming,
                         text_diff=ch.text_diff, settled=settled)
        if local is not None:
            res.extra["local_mad"] = round(local, 2)
        if action.name == "type":
            extra_target = extra.get("_target", "")
            verified = self._verify_type(new, extra.pop("_baseline"), extra.pop("_target"),
                                         extra.pop("_baseline_all", None))
            res.extra["verified"] = verified
            if verified == "false":
                res.hint = TYPING_NOT_LANDED.format(what=extra_target)
        elif not changed and action.name in HINTS:
            # ---- 像素判据说「没变化」：看图复核一次 ----
            # 原来到这里就直接下结论了，而两个判据都是纯像素的（aHash 汉明 + 文本集合差），
            # 在判一件语义的事。判错的代价不止是一句错话：`changed=False` 会被 guard
            # 记成一次无进展，攒到 NO_PROGRESS_WARN=3 就开始警告、6 就终止整个任务。
            # 很多看起来「在原地打转」的运行，其实是**判官说它在原地打转**。
            #
            # 只在否定侧问 —— 有变化就是有变化，不用花钱复核。
            eff = judge.did_action_work(before.image, new.image, action.describe(),
                                        action.expect, self.asker)
            if eff is not None and eff.worked:
                res.changed = True
                res.extra["judged"] = {"worked": True, "confidence": eff.confidence,
                                       "why": eff.why}
                # ⚠ 不写 hint：HINTS 那句是「可能没点中，换个目标」，而这里刚判定它**生效了**。
                #   两句一起发出去，模型会照那句错的走。
            else:
                res.hint = HINTS[action.name]
                if eff is not None:
                    res.extra["judged"] = {"worked": False, "confidence": eff.confidence,
                                           "why": eff.why}
                    # 模型看过两张图说没生效 —— 这比汉明距离有说服力得多，交给模型当线索。
                    res.hint += f"（看图复核也说没生效：{eff.why}）"
        elif changed and action.expect and action.name in HINTS:
            # ---- 像素判据说「变了」：变了 ≠ 做对了 ----
            # 上面那条「只在否定侧问」的规矩管的是**动作生效没**；这里问的是另一件事：
            # **生效了，但去的是不是对的地方**。误点进一个错页面正是「变了」——
            # 以前一律当成功，guard 还把它记成进展（guard.py record_outcome 的注释）。
            # 只在模型给了 expect 时问：没有预期就没有可核对的东西，也省下这次调用。
            # 2026-09-09 统计 runs/：1113 步里模型自评字段一次没填过，expect 只有 12% 的步带 ——
            # 所以这条信号现在覆盖有限；tools.py 里 expect 的描述该改成「换页动作请填」，等 token 重标一起改。
            # on_change=True 让 guard.record_result 分得清这次 judged 是哪一路来的（那边的注释）。
            eff = judge.did_action_work(before.image, new.image, action.describe(),
                                        action.expect, self.asker)
            if eff is not None:
                res.extra["judged"] = {"worked": eff.worked, "confidence": eff.confidence,
                                       "why": eff.why, "on_change": True}
                if not eff.worked:
                    res.hint = f"画面变了，但看图复核说没达到预期：{eff.why}。先确认自己在哪。"
        # ---- 不光说「变了」，还要说**变成了什么** ----
        # changed=True 只是一个布尔。模型得自己从几十个元素里认出哪一格变了 ——
        # 2026-09-09 真机（记一笔账）：点了数字 1，金额从 '0.00' 变成 '1'，
        # 工具回的是 {"ok":true,"changed":true,"hamming":3,"text_diff":5}。
        # 模型没看出来，把「输入金额13的第一位数字1」又执行了一遍，金额成了 '11'，
        # 此后 6 步全在收拾这个烂摊子，最后熔断。**坐标一次都没点错。**
        #
        # 答案本来就在手边：transition 用 OCR 文字集合差机械生成，零成本、无图序问题。
        # 它以前只在 context_mode=state 时注入（loop.py），而默认是 window ——
        # 算出来了，然后扔掉。放进 extra 就落在「上一步结果」里，两种模式都看得到。
        #
        # ⚠ tap_px 传 None：local_mad 是像素级循环，上面真正需要它时已经算过一次了，
        #   这里再算一遍纯属白烧 CPU（而且结果上面已经用掉了）。
        t = tr.transition(before, new, None, config.TRANSITION_LIST_MAX)
        if t.added:
            res.extra["appeared"] = list(t.added)
        if t.removed:
            res.extra["disappeared"] = list(t.removed)
        res.extra.update({k: v for k, v in extra.items() if not k.startswith("_")})
        return res, new


    # ---- 中文输入：驱动 iOS 自己的输入法 ----
    # 候选编号只有一位（1-9）。原来写 \d+ 会把 '10000' 整串当编号吃掉，
    # 剥完剩空串，于是分不出「编号 1 的候选」和「一个叫 10000 的东西」。
    _CAND_PREFIX = re.compile(r"^\s*([1-9])\s*")
    _CJK = re.compile(r"[\u4e00-\u9fff]")

    def _candidates(self, obs: Observation, near_text: str | None = None) -> list:
        """输入法候选栏。靠**几何**认，不靠「以数字开头」。

        ⚠ 原来的规则是「在下部 20% + 以数字开头 + 不是最底下那一行（那是输入框）」。
        2026-09-08 探针实测它有两种塌法，而且两种都发生过：

        1. **OCR 没读到输入框时，候选栏自己成了最底行**，被「不是最底下那一行」
           整个排除掉，返回的是 '97字节·JavaScript·2025/8/9' 这种垃圾
           —— 它以数字开头、又在下部，规则全中。
        2. **Spotlight 的搜索结果里凡是以数字开头的都混进来**（'10000'、'2026/8/25'）。
           之前能用只是因为真候选也在里面，是巧合。

        现在认三个特征，都来自实测（截图 runs/ime-q*.png）：

        · **一条水平带**：候选的 y 几乎相同（实测跨度 2px，容差取图像高 1%）
        · **编号从 1 开始、按 x 从左到右递增**（一位数）
        · **带里至少有一个汉字候选** —— 这条把 '10000'/'2026/8/25' 那种纯数字行挡掉

        多条带都满足时，取**离 near_text（刚打进去的那串拼音）最近**的一条。

        ⚠ 这里原来写的是「取最低的一条：候选栏永远贴着键盘」，还配了一个
          `CAND_MIN_Y_RATIO = 0.70` 只在屏幕下部 30% 找带。**两条都是错的**，
          2026-09-09 真机坐实：备忘录里打 'nihao'，OCR 好好地读到了
          '1 你好'/'2'/'3你还'，y=318/1388=**0.229** —— 第一步就被那个下限筛没了。
          于是判定「不在中文模式」，反手 toggle_ime 把好好的中文模式切走，
          重打更没候选，最后报 ime_not_chinese。App 内容框 8 次尝试 8 次全挂，
          而 9 次成功**全部**发生在 Spotlight —— 唯一符合那个假设的场景。

          真实规律是：**候选栏贴的是光标，不是键盘**。Spotlight 的输入框恰好在
          屏幕底部、候选栏落在键盘上方，那是特例。

          也不能改成「取拼音下方最近的一条」—— 两个场景方向正好相反：
          Spotlight 里拼音在候选栏**下面**（'Q shezhi'@1283，候选栏@1218），
          备忘录里拼音在候选栏**上面**（'nihao'@257，候选栏@318）。
          所以用**绝对距离**，不看方向。

        near_text 给不出、或者 OCR 没读到它时，退回取最低的一条 —— 那是老行为，
        在 Spotlight 上仍然对。

        实测形态：'1设置'、'2 摄制'、'1 关于本机'。偶尔一个 OCR 元素里挤两个候选
        （'2 关于 3关羽'），那种只能整体点，点到哪个不确定 —— 所以调用方只认
        「剥掉编号后正好等于目标」的元素，不做包含匹配。
        """
        if not obs.elements:
            return []
        H = obs.height_px
        tol = max(1, round(H * config.CAND_BAND_TOL_RATIO))
        rows = sorted(obs.elements, key=lambda e: e.center[1])   # 全屏找带，不设 y 下限
        bands: list[list] = []
        for e in rows:
            if bands and e.center[1] - bands[-1][0].center[1] <= tol:
                bands[-1].append(e)
            else:
                bands.append([e])

        ok: list[tuple[int, list]] = []      # (这条带的 y, 带里的候选元素)
        for band in bands:
            numbered = []
            for e in band:
                m = self._CAND_PREFIX.match(e.text)
                if m:
                    numbered.append((int(m.group(1)), e))
            numbered.sort(key=lambda pair: pair[1].center[0])
            nums = [n for n, _ in numbered]
            if len(nums) < 2 or nums[0] != 1:
                continue
            if any(b <= a for a, b in pairwise(nums)):   # 编号必须递增
                continue
            if not any(self._CJK.search(self._CAND_PREFIX.sub("", e.text)) for _, e in numbered):
                continue
            ok.append((band[0].center[1], [e for _, e in numbered]))
        if not ok:
            return []
        anchors = self._anchor_ys(obs, near_text)
        if anchors:
            return min(ok, key=lambda t: min(abs(t[0] - y) for y in anchors))[1]
        return max(ok, key=lambda t: t[0])[1]           # 没锚点：退回取最低的一条

    def _anchor_ys(self, obs: Observation, near_text: str | None) -> list[int]:
        """屏幕上那串拼音落在哪些 y 上 —— 用来在多条候选带里挑对的那条。

        空格照 OCR 的心情来（'Q shezhi' vs 'Qshezhi'），所以两边都去掉空格再比。
        匹配不上很正常（拼音串可能被 OCR 漏读或和别的字粘在一起），返回空就是了，
        调用方会退回老行为。
        """
        key = (near_text or "").lower().replace(" ", "")
        if not key:
            return []
        return [e.center[1] for e in obs.elements
                if key in e.text.lower().replace(" ", "")]

    def _type_via_ime(self, text: str, obs: Observation):
        """中文走 iOS 自己的输入法：发拼音 → 看候选 → 点选。

        为什么必须这样：镜像只转发 keycode、不读事件里的 unicode 载荷（`设计说明`），
        汉字没有 keycode 打不出来；而粘贴这条路在本环境实测不通（`设计说明`）。
        剩下唯一能用的就是让 iOS 的输入法自己把拼音转成汉字。

        ⚠ 只认「剥掉编号后正好等于目标」的候选，不做包含匹配 ——
        候选里 '设置' 和 '摄制' 只差一个音，包含匹配会点错，而点错了没有回头路
        （字已经上屏）。找不到就如实报告并把候选交回去，让模型自己决定下一步。
        """
        pinyin = "".join(lazy_pinyin(text))
        if not pinyin.isascii() or not pinyin:
            return {"_done": ToolResult(
                ok=False, error="pinyin_failed",
                hint=f"{text!r} 转不出可打的拼音（转出来是 {pinyin!r}）"), "_obs": None}
        if self._ime_chinese is False:
            # 已经知道在英文模式了，先切再打 —— 省掉一次注定失败的尝试。
            # 只在**确知**时这么做；不知道（None）就照打，让证据说话。
            self.dev.toggle_ime()
            self._ime_chinese = None
        self.dev.type(pinyin)
        frame, settled = settle(self.dev, self._frame_of(obs), IOS_TIMING["type"], ahash)
        after_typing = self.per.observe(frame)

        cands = self._candidates(after_typing, pinyin)
        hit = next((e for e in cands
                    if self._CAND_PREFIX.sub("", e.text).strip() == text), None)
        shown = [e.text for e in cands]
        if not cands and pinyin.lower() not in "".join(
                e.text for e in after_typing.elements).lower():
            # ⚠ 候选栏空**而且拼音字母也没出现在屏幕上** = 按键根本没进去，
            #   不是"输入法模式不对"。这两种说法差很远：模式不对可以切一下，
            #   而按键进不去时切几次都没用。
            #   2026-09-08 批跑踩到：34 次 Spotlight 打字全部没落进去，
            #   而当时报的是「字母是直接上屏的」—— 屏幕上一个字母都没有。
            self.dev.backspace(len(pinyin))
            frame, settled = settle(self.dev, frame, IOS_TIMING["type"], ahash)
            new = self.per.observe(frame)
            return {"_done": ToolResult(
                ok=False, error="keystrokes_not_landing",
                hint=TYPING_NOT_LANDED.format(what=pinyin),
                extra={"pinyin": pinyin}), "_obs": new}
        # ---- 几何规则说「没有候选栏」时，**先看图，再动手** ----
        # 2026-09-09：这里原来直接就 toggle_ime 了。而几何规则会漏 —— 候选栏贴的是光标，
        # 它可以出现在屏幕任何高度，当时的 y 下限只看下部 30%。于是一个**正确的**中文
        # 输入法被切走，重打更没候选，报 ime_not_chinese。App 内容框 8 次尝试 8 次全挂。
        #
        # 现在：切之前问一次会看图的那一方。三种回答三条路，一条都不能合并 ——
        #   found=True → 几何规则漏了，键盘本来就是中文的，**绝不能切**
        #   found=False → 模型看过了确实没有，切才有依据（老行为）
        #   None        → 没问成，什么都不该断定，退回老行为
        vis = judge.find_candidate_bar(after_typing.image, pinyin, self.asker) if not cands else None
        if vis is not None and vis.found:
            self._ime_chinese = True
            shown = [c.text for c in vis.items]
            pick = next((c for c in vis.items if c.text == text and c.center), None)
            if pick is not None:
                return self._commit_candidate(pick.center, after_typing, frame, obs,
                                              pinyin, pick.text, shown)
            # 候选栏在、只是没有目标词 —— 这是 candidate_not_found，**不是** ime_not_chinese。
            # 前者「换个词再试」有意义，后者明说了「换词没用」。说错了模型就走错路。
            return self._candidate_missing(text, pinyin, shown, obs, frame)
        if not cands:
            # 候选栏空 = 键盘不在中文模式。切一次再试一遍 —— 这是模型自己也会做的事，
            # 没必要为此浪费一步和一次模型调用。
            #
            # ⚠ 这里**不能加「信念不是英文才重试」这种条件**。第一版写成
            #   `if not cands and self._ime_chinese is not False`，结果恰恰在
            #   「明知道在英文模式」时跳过了切换 —— 逻辑正好反了。
            #   真机上表现为：打完 safari 再打中文必然失败（2026-09-08 实测）。
            #   决定要不要切的是**刚拿到的证据**（候选栏空不空），不是旧信念。
            self.dev.backspace(len(pinyin))
            self.dev.toggle_ime()
            self.dev.type(pinyin)
            frame, settled = settle(self.dev, frame, IOS_TIMING["type"], ahash)
            after_typing = self.per.observe(frame)
            cands = self._candidates(after_typing, pinyin)
            hit = next((e for e in cands
                        if self._CAND_PREFIX.sub("", e.text).strip() == text), None)
            shown = [e.text for e in cands]
        self._ime_chinese = bool(cands)
        if not cands:
            # ⚠ **候选栏是空的**和**候选里没有目标**是两种病，说法必须分开。
            #   空的意味着 iOS 键盘根本不在中文模式 —— 拼音字母是直接上屏的，
            #   再换个词、再试一次都没有用。2026-09-08 实测撞到过：
            #   打 'shezhi' 搜索框显示 'Q shezhi — 设置'，一个候选都没有。
            #   输入法模式是个隐藏的全局量（设计说明），什么时候会变还没查清。
            self.dev.backspace(len(pinyin))     # 只撤销自己打的那几个，绝不多删
            frame, settled = settle(self.dev, frame, IOS_TIMING["type"], ahash)
            new = self.per.observe(frame)
            ch_e = did_change(obs, new)
            return {"_done": ToolResult(
                ok=False, error="ime_not_chinese", changed=ch_e.changed,
                hamming=ch_e.hamming, text_diff=ch_e.text_diff,
                hint=f"打了拼音 {pinyin!r}，但**一个候选都没有** —— iOS 键盘不在中文模式，"
                     f"字母是直接上屏的。已经把它们退掉了。**换个词再试没有用**："
                     f"要么这一步改用不需要中文的做法（比如 Spotlight 直接搜拼音字母），"
                     f"要么 done(failed) 说明「输入法不在中文模式」让人来切一下。",
                extra={"pinyin": pinyin}), "_obs": new}
        if hit is None:
            # ---- 候选栏在，但里面没有目标：**再看一次图，可能是 OCR 把候选粘住了** ----
            # 2026-09-09 真机：打 'ming'，OCR 吐出 ['1名', '2明 3命4鸣'] —— 后面三个候选
            # 粘成了一个元素。剥掉编号是 '明 3命4鸣'，不等于 '明'，于是整体放弃。
            # 而它们在屏幕上是三个分开的词，看图的那一方能拆开、还能给出各自的坐标。
            #
            # ⚠ 仍然只在否定侧问：这里的否定结论是「候选里没有我要的」。
            #   前面那次复核只在几何交白卷时问过，这条路没问过，不重复。
            vis2 = judge.find_candidate_bar(after_typing.image, pinyin, self.asker)
            if vis2 is not None and vis2.found:
                pick = next((c for c in vis2.items if c.text == text and c.center), None)
                if pick is not None:
                    return self._commit_candidate(pick.center, after_typing, frame, obs,
                                                  pinyin, pick.text,
                                                  [c.text for c in vis2.items])
                shown = [c.text for c in vis2.items]     # 交回拆开后的，比粘住的有用
            return self._candidate_missing(text, pinyin, shown, obs, frame)
        return self._commit_candidate(hit.center, after_typing, frame, obs,
                                      pinyin, hit.text, shown)

    def _commit_candidate(self, center_px, ref: Observation, frame, obs: Observation,
                          pinyin: str, picked: str, shown: list[str]) -> dict:
        """点中一个候选，让字上屏。center_px 是**图像像素**坐标。

        两条路共用：几何规则找到的（Element.center），和看图找到的（Cand.center）。
        坐标换算必须用**打完拼音那一次观察**的窗口矩形（ref），不是任务开始时那个。
        """
        sx, sy = image_to_screen(*center_px, self._frame_of(ref))
        self.dev.tap(sx, sy)
        frame, settled = settle(self.dev, frame, IOS_TIMING["tap"], ahash)
        new = self.per.observe(frame)
        ch = did_change(obs, new)
        changed = ch.changed
        extra = {"pinyin": pinyin, "picked": picked, "candidates": shown}
        if not changed:
            # ⚠ 这条路**不走 _after**，所以那边的看图复核管不到它 —— 得在这儿自己问一次。
            #   而中文输入恰恰是最容易被像素判据看漏的：2026-09-09 真机实测，
            #   'nihao' 变成 '你好' 只贡献 hamming=1、text_diff=6，两个都低于阈值（2 / 8）。
            #   那一跑里模型因此以为没输入成功，去点了撤销、又重打一遍，白花两步。
            eff = judge.did_action_work(obs.image, new.image,
                                        f"用输入法选中候选 {picked!r}，让 {picked!r} 上屏",
                                        f"{picked} 出现在输入框里", self.asker)
            if eff is not None:
                extra["judged"] = {"worked": eff.worked, "confidence": eff.confidence,
                                   "why": eff.why}
                changed = eff.worked
        return {"_done": ToolResult(
            ok=True, changed=changed, hamming=ch.hamming, text_diff=ch.text_diff,
            settled=settled, extra=extra), "_obs": new}

    def _candidate_missing(self, text: str, pinyin: str, shown: list[str],
                           obs: Observation, frame) -> dict:
        """候选栏在，但里面没有正好等于目标的那一项。

        ⚠ changed 必须如实报。以前这里不设，默认 None，而 guard 里
          `if changed and ...` 对 None 是假 —— 于是「打了拼音、候选栏冒出来了」
          被记成一次无进展。逐字输入三个字就自己撞上 NO_PROGRESS_WARN。
        ⚠ 必须把自己刚打的拼音退掉，否则下一次输入会**叠在残留上**：
          连着两次 'shezhi'，候选变成「则设置设置」（2026-09-08 实测）。
          select_all_and_delete 做不到这件事 —— 它清不掉待上屏缓冲区。
        """
        self.dev.backspace(len(pinyin))
        frame, settled = settle(self.dev, frame, IOS_TIMING["type"], ahash)
        after_typing = self.per.observe(frame)
        ch_t = did_change(obs, after_typing)
        return {"_done": ToolResult(
            ok=False, error="candidate_not_found", changed=ch_t.changed,
            hamming=ch_t.hamming, text_diff=ch_t.text_diff,
            hint=f"打了拼音 {pinyin!r}，候选里没有**正好**是 {text!r} 的一项"
                 f"（只认完全相同，'设置' 和 '摄制' 差一个音，点错没有回头路）。"
                 f"字还没上屏。**候选栏就在下一张截图上，编号 1、2、3 排成一横排 —— "
                 f"认得哪个是你要的就直接 tap 它**；元素列表里的文字可能被 OCR 读错"
                 f"（「一」常被读成「—」），以截图为准。都不是就点候选栏最右边的 ∨ 展开更多，"
                 f"或者换更短的一段重打。",
            extra={"pinyin": pinyin, "candidates": shown}), "_obs": after_typing}

    def _region_texts(self, obs: Observation) -> set[str]:
        if self._last_tap_px is None:
            return set()
        cx, cy = self._last_tap_px
        r = obs.height_px * TYPE_REGION_RADIUS
        return {e.text for e in obs.elements if abs(e.center[1] - cy) <= r}

    def _verify_type(self, new: Observation, baseline: set[str], target: str,
                     baseline_all: set[str] | None = None) -> str:
        """打进去了没。先看上次 tap 附近那一带，再看整屏 —— 两处都各只在自己的分辨率内下结论。

        ⚠ 原来没有 tap 过就直接 "unknown"。2026-09-10：Spotlight 里打字（前面没有 tap，键盘是
          key(spotlight) 唤出来的）通道死了也报 unknown，恢复阶梯永远触发不了 —— 验证太弱，
          阶梯就是摆设。整屏判据和 open_app 的 _query_landed 同一个道理：只认输入前没有、
          输入后才出现的文字；整屏文字集合一字不差才说 false。"""
        if any(target in t for t in baseline):
            return "unknown"
        if self._last_tap_px is not None:
            now = self._region_texts(new)
            if any(target in t for t in now):
                return "true"
            region_unchanged = now == baseline
        else:
            region_unchanged = True
        if baseline_all is None:
            return "unknown"
        if any(target in t for t in new.text_set - baseline_all):
            return "true"
        if region_unchanged and new.text_set == baseline_all:
            return "false"
        return "unknown"

    # ---- 高阶滚动：内部循环滚动+观察，对外只是一个动作 ----
    def _scroll_scan(self, action: Action, obs: Observation):
        """scroll_until / collect 共用的「滚一屏 → 等稳 → 观察」循环。

        为什么要有它：单步 scroll 滚一屏就要一次模型调用（约 3.5 秒）并吃掉一步预算，
        20 屏的列表直接把 MAX_STEPS 烧光 —— 长列表任务因此根本做不了。
        把 N 次滚动压进一步是这两个工具的**全部**价值，所以它们不重推中间观察，
        只在结束时交回最后（scroll_until 命中时是命中的那一次）观察。

        ⚠ 「到底」不另立判据：用的就是单步 scroll 判 changed 的 did_change
        （阈值 aHash 2 / 文本 8，已在真机标定）。
        同一件事在两处有两个理解，是这个项目已经踩过三次的坑。

        ⚠ 但要**连续 2 次**没变化才算到底，不是一次。「画面没变」是有歧义的：
        既可能是真到底了，也可能是这一次滚动没生效。一次就收工，遇上一次丢失的滚动
        就会把半个列表当成全部交出去 —— 而调用方无从分辨。
        「几次都没变化才下结论」和熔断的无进展判据是同一个形状。
        同理，每屏之间的等待用 settle（IOS_TIMING["scroll"]，与单步 scroll 同一档），
        不自己 sleep —— 观察必须落在静止的画面上。

        ⚠ 熔断照常：这两个工具改变画面，changed / 新观察都照常返回，
        guard 的进展判据（到了没见过的画面）不需要为它们开特例。
        """
        n, a = action.name, action.args
        direction, target = a["direction"], a.get("text")
        frame = self._frame_of(obs)
        cur = obs
        screens = 0
        settled = None          # 一屏都没滚时它无意义，留 None 让 to_json 省掉
        reached_end = False
        still = 0               # 连续几屏没变化 —— 见下面为什么不能只看一次
        # ⚠ 「一次都没动过」和「滚到底了」是两回事，必须分开报。
        #   2026-09-08 真机（runs/example-run）：在通用页 scroll_until
        #   找「关于本机」，滚了两次画面纹丝不动，工具报「滚不动了（到底/到顶）」。
        #   模型于是相信关于本机不在通用里，改去设置首页找，把「更新到 iOS 26.6.1」
        #   这条**可更新版本**当成当前版本报了 done(success) —— 设备版本与报告不一致。
        #   误导性的诊断直接造出了一次假成功。
        any_change = False
        truncated = False
        lines: list[str] = []
        seen: set[str] = set()
        chars = 0
        if n == "collect":
            chars, truncated = self._absorb(lines, seen, obs, chars)
        # 先看当前这一屏：目标本来就在眼前时不该白滚一屏。
        hit = self._find_text(obs, target) if target else None

        while hit is None and not truncated and screens < config.SCROLL_MAX_SCREENS:
            try:
                self.dev.scroll(direction, "page")
                # 每屏之间用 settle 等稳，不自己 sleep：下一次观察必须落在静止的画面上。
                frame, settled = settle(self.dev, frame, IOS_TIMING["scroll"], ahash)
                new = self.per.observe(frame)
            except Exception as e:
                # 中途炸了不能只报个错就完：已经滚过的那几屏把画面挪走了，模型手里那份
                # 观察已经过期，拿它去点就是点在一个不存在的画面上 —— 这一类错这个项目栽过。
                # 一屏都没滚成时没这个问题，照旧交给 run() 的统一处理（那里返回 None 观察）。
                if screens == 0:
                    raise
                extra = {"screens": screens} | ({"lines": lines} if n == "collect" else {})
                return (ToolResult(ok=False, error="device_error",
                                   hint=f"滚到第 {screens + 1} 屏时出错：{type(e).__name__}: {e}",
                                   extra=extra), cur)
            screens += 1
            changed = did_change(cur, new).changed
            cur = new
            if n == "collect":
                chars, truncated = self._absorb(lines, seen, new, chars)
            if target:
                hit = self._find_text(new, target)
            any_change = any_change or changed
            still = still + 1 if not changed else 0
            if still >= config.SCROLL_END_CONFIRMATIONS:
                reached_end = True
                break

        ch = did_change(obs, cur)   # 对外报告的是这一步的净效果，与 open_app 一致
        extra: dict = {"screens": screens, "reached_end": reached_end}
        never_moved = screens > 0 and not any_change
        if never_moved:
            extra["never_moved"] = True
        stuck = (f"滚了 {screens} 次，**画面一次都没变** —— 这不是到底了，是根本没滚动起来。"
                 "可能这块区域不可滚动、被浮层挡住，或者内容本来就只有一屏。"
                 "别再往这个方向滚了，换个做法。")
        hint = None
        if n == "scroll_until":
            extra["found"] = hit is not None
            if hit is not None:
                # 交回 id 与文字：新观察随即会推给模型，下一步可以直接 tap 这个 id。
                extra["matched"] = {"id": hit.id, "text": hit.text}
                if screens == 0:
                    hint = "目标本来就在当前屏幕上，没有滚动。"
            elif never_moved:
                hint = stuck + f"（在找「{target}」）"
            elif reached_end:
                hint = f"滚了 {screens} 屏，滚不动了（到底/到顶），没有出现「{target}」。"
            else:
                hint = (f"滚了 {screens} 屏还没出现「{target}」，"
                        f"到达 {config.SCROLL_MAX_SCREENS} 屏上限 —— **还没到底**。")
        else:
            extra["lines"] = lines
            extra["count"] = len(lines)
            if never_moved:
                hint = stuck + "收到的就是当前这一屏。"
            elif truncated:
                extra["truncated"] = True
                hint = f"文字太多（超过 {config.COLLECT_MAX_CHARS} 字），只收了前 {screens} 屏，后面没滚。"
            elif not reached_end:
                hint = f"到达 {config.SCROLL_MAX_SCREENS} 屏上限，**还没到底**，下面可能还有。"
        res = ToolResult(ok=True, changed=ch.changed, hamming=ch.hamming, text_diff=ch.text_diff,
                         settled=settled, hint=hint, extra=extra)
        # 一屏没滚时不重推同一个观察 —— 消息流里出现重复观察块会让模型以为画面翻了一页。
        return res, (cur if cur is not obs else None)

    @staticmethod
    def _find_text(obs: Observation, text: str):
        """按 OCR 文本行找目标，子串匹配。

        子串而非全等：OCR 常把一行读成「设置 >」「12 条未读 微信」这种带零碎的形态，
        要求全等等于要模型猜 OCR 会怎么断行。
        """
        t = text.strip().lower()
        return next((e for e in obs.elements if t in e.text.lower()), None)

    @staticmethod
    def _absorb(lines: list[str], seen: set[str], obs: Observation, chars: int) -> tuple[int, bool]:
        """把一屏的文本并进汇总：去重，保留首次出现的顺序。

        必须去重：滚动时相邻两屏一定有重叠，不去重的话汇总里全是重复行。
        用集合判重、用列表保序 —— 顺序就是页面上从上到下的阅读顺序，丢了它汇总就没法读。
        """
        for e in obs.elements:
            t = e.text.strip()
            if not t or t in seen:
                continue
            if chars + len(t) > config.COLLECT_MAX_CHARS:
                return chars, True
            seen.add(t)
            lines.append(t)
            chars += len(t)
        return chars, False

    def _identity(self, name: str, new: Observation) -> dict:
        """点完之后问看图的那一方：进的是不是「name」。结果原样写进 extra["identity"]。

        三条路（直达 / Spotlight / 翻主屏）报 ok 之前都过这一道 —— 一个规则一个入口（项目开发约定）。
        verified=True/False 是看过图的结论；None = 没问成（没有看图的一方 / 调用失败 / 答非所问），
        照旧当开成了，但「没核对」留在结果里看得见（§3）。为什么每次都问见 judge.is_target_app。
        """
        chk = judge.is_target_app(new.image, name, self.asker)
        if chk is None:
            return {"verified": None}
        return {"verified": chk.is_app, "confidence": chk.confidence, "seen": chk.seen, "why": chk.why}

    def _open_app(self, name: str, obs: Observation):
        """开 App：先查布局表直达（`_open_app_from_layout`），走不通走 Spotlight（`_open_app_via_spotlight`）。

        直达没成的原因（LAYOUT_MISSES 之一）写进**最终返回的那个结果**的 extra["layout_miss"] ——
        不论后面是 Spotlight 成功、翻主屏成功还是 app_not_found。原来这几种没成全都不留痕，
        评测里「表里没有 / 当前帧找不到标签 / 点完还在主屏」统统显示成 Spotlight，
        和根本没有布局表时一模一样（项目开发约定 失败必须能被看见）。没表不记：没表 = 没尝试。

        每条路点开之后都核对身份（`_identity`）：看图说「不是」就不算开成，换下一条路走；
        一路上开错的每一次都记进最终结果的 extra["wrong_app"]（开错了哪条路、点了什么、看图说是什么）。
        """
        # ⚠ 先查布局表直达：翻页 + 点图标，零打字。打字是这个项目最脆的通道（设计说明 的键盘排查
        #   全在 Spotlight 上撞的；2026-09-08 批跑 39/40 次打字失败），能不打就不打。
        #   走不通（表里没有 / 翻到了当前帧上没这个标签 / 点了还在主屏）退回 Spotlight，
        #   那条路原样不动。
        wrong: list[dict] = []
        direct = self._open_app_from_layout(name, obs, wrong)
        if isinstance(direct, tuple):
            return direct
        res, new = self._open_app_via_spotlight(name, obs, wrong)
        if direct is not None:
            res.extra["layout_miss"] = direct
        if wrong:
            res.extra["wrong_app"] = wrong
        return res, new

    def _open_app_via_spotlight(self, name: str, obs: Observation, wrong: list | None = None):
        """Spotlight 打开 App：输入 → 确认输入落屏 → **点匹配的那一行**，不盲按回车。

        原实现有两个真缺陷（2026-09-07 真机暴露）：
        1. 「输入是否成功」查的是整个画面上有没有出现目标文字。但 Spotlight 一打开，
           即使一个字都没输入，上方也会列出常用 App —— 目标恰好在里面时，检查会
           **因为错误的理由通过**。粘贴输入本身不稳，于是常常是「没输进去但检查过了」。
        2. 盲按回车打开的是当前选中项，不一定是目标。上一次运行就这样打开了错误的 App；
           下一次侥幸对了，只是因为目标恰好排第一。

        现在：输入后只在**搜索框那一行**上找目标文字来确认输入成功；再在结果列表里
        找文字匹配的那一行去点它。两处都失败才报 app_not_found。
        """
        wrong = [] if wrong is None else wrong
        self.dev.key("spotlight")
        frame, _ = settle(self.dev, self._frame_of(obs), IOS_TIMING["key"], ahash)
        before_type = self.per.observe(frame)

        # ⚠ 中文 App 名**不走输入法**，直接打拼音字母。
        #   2026-09-08 实测：Spotlight 自己就按拼音匹配 App 名 ——
        #   干净的搜索框里打 "yimujizhang"，「一木记账」直接出现在「最佳搜索结果」；
        #   而空搜索框里它不在（对照过，基线是干净的）。
        #
        #   为什么不走输入法：「一木记账」这种 App 名根本不在输入法词库里，
        #   打 yimujizhang 给出的候选是「以募集章」「一目几张」，永远选不中。
        #   而 dev.type() 拿到中文会走粘贴通道，粘贴在本环境实测不通（设计说明）。
        #   所以以前 open_app 对中文 App 名是**坏的** —— 之前能打开纯属侥幸：
        #   模型手搓的流程里输入法失败了，反而把拼音字母留在了框里。
        query = spotlight_query(name)
        self.dev.select_all_and_delete()
        self.dev.type(query)
        frame, _ = settle(self.dev, frame, IOS_TIMING["type"], ahash)
        cand = self.per.observe(frame)

        # 落屏校验查的是**打进去的那串**（拼音），匹配结果行查的是 App 名本身 ——
        # Spotlight 用拼音搜，但结果列表里显示的仍然是中文名。
        typed = self._query_landed(before_type, cand, query)
        row = self._match_row(cand, name, exclude_top=not typed)
        if row is None:
            # ⚠ Spotlight 这条路要**打字**。打字整条通道会死（2026-09-08 批跑实测：
            #   39/40 次打字失败，而同一时间点击 6/6、滚动正常）—— 那时候 open_app
            #   就彻底没辙，整个任务卡死在"打不开 App"上。
            #   所以退路必须走**另一条通道**：回主屏、翻页、点图标，全程只用点击和滚动。
            return self._open_app_from_home(name, obs, typed, query, wrong)

        sx, sy = image_to_screen(*row.center, self._frame_of(cand))
        self.dev.tap(sx, sy)
        frame, settled = settle(self.dev, frame, IOS_TIMING["open_app"], ahash)
        new = self.per.observe(frame)
        via = "row"
        if in_spotlight(new):
            # ⚠ 点了「结果行」画面还是 Spotlight = 点中的是标签不是图标。
            #   「最佳搜索结果」是一排图标 + 下面的标签，和主屏一样：OCR 只读得到标签，点标签打不开。
            #   2026-09-10 真机（备忘录）：open_app 点了顶部的「备忘录」标签，报 ok —— 因为 changed
            #   拿的是「按 Spotlight 之前的主屏」做基线，主屏→Spotlight 当然「变了」。模型对着
            #   Spotlight 又点了两下才进去，第一下还是死点。基线要拿 Spotlight 那一帧，
            #   开没开成看的是「还在不在 Spotlight」，不是「画面变没变」。
            iy = icon_y_above_label(row.center[1], row.box)
            sx, sy = image_to_screen(row.center[0], iy, self._frame_of(cand))
            self.dev.tap(sx, sy)
            frame, settled = settle(self.dev, frame, IOS_TIMING["open_app"], ahash)
            new = self.per.observe(frame)
            via = "icon_above"
            if in_spotlight(new):
                return self._open_app_from_home(name, obs, typed, query, wrong)
        # ⚠ 离开了 Spotlight ≠ 进了目标 App。2026-09-10 真机：打字没落屏，搜索框里留着旧词
        #   beiwanglu，结果列表底部的分组标题「设置」被精确匹配中；点它没反应 → 点它上方 →
        #   进了一条备忘录 → 报 ok via=icon_above（runs/example-run）。
        #   身份交给看图的一方；说不是就当没开成，走不打字的那条退路。
        ident = self._identity(name, new)
        if ident["verified"] is False:
            wrong.append({"via": via, "tapped": row.text, "seen": ident["seen"], "why": ident["why"]})
            return self._open_app_from_home(name, obs, typed, query, wrong)
        ch = did_change(cand, new)
        return (ToolResult(ok=True, changed=ch.changed, hamming=ch.hamming, text_diff=ch.text_diff,
                           settled=settled, hint=None if ch.changed else HINTS["open_app"],
                           extra={"typed": typed, "tapped": row.text, "via": via, "screen": [sx, sy],
                                  "identity": ident}),
                new)

    def _open_app_from_layout(self, name: str, obs: Observation, wrong: list | None = None):
        """按布局表直达：回第一页 → 翻到第 k 页 → **在当前帧上找到标签** → 点标签上方的图标 → 核身份。

        任何一步对不上都交给 Spotlight 那条路，绝不按旧行列盲点：布局表是参考，
        当前帧才是真源（设计说明 第 3 条）。点完 in_spotlight / looks_like_home / 还是同一页网格
        任一成立就不算开成 —— 和 Spotlight 那条路一样，开没开成看身份不看画面变没变（2026-09-10）。

        返回三种之一：(ToolResult, 新观察) = 开成了；str = 没开成的原因（LAYOUT_MISSES 之一）；
        None = 没有布局表，根本没尝试。
        """
        if self.layout is None:
            return None
        wrong = [] if wrong is None else wrong
        # ⚠ 2026-09-10 终审（I2）：表里一个 label 是 123，find_app 抛 AttributeError，冒到 _run_once
        #   的 `except Exception` 变成 device_error —— Spotlight 一次都没按，本来能开的 App 开不了。
        #   孪生任何一处坏了都等于没有孪生（设计说明 不变式 5），所以**孪生自己的逻辑**（查表、页码校验、
        #   标签归一化）包起来，坏了记 error 退回 Spotlight。**设备调用（key/scroll/tap/capture/observe）
        #   不包**：设备的异常照旧往外抛，与 Spotlight 那条路一致 —— 镜像断了不是「表没用上」，
        #   吞掉它再去走 Spotlight 只会在一个死通道上再撞一次，还把真正的原因藏起来。
        try:
            hit = self.layout.find_app(name)
            if hit is None:
                return "no_hit"
            if not 1 <= hit.page_order <= config.HOME_PAGES_MAX:
                return "page_out_of_range"      # 表被写坏 / 手改：别翻 49 页再退回来
            want = norm_label(hit.label)
        except Exception:       # noqa: BLE001 —— 孪生坏了 = 没有孪生
            return "error"
        return_to_first_home_page(self.dev, self._frame_of(obs))
        frame = self.dev.capture()
        for _ in range(hit.page_order - 1):
            self.dev.scroll("right", "page")
            frame, _ = settle(self.dev, frame, IOS_TIMING["scroll"], ahash)
            time.sleep(config.AFTER_PAGE_FLIP_S)    # 翻页后 2 秒不可点（config 里那条注释）
            frame = self.dev.capture()
        cur = self.per.observe(frame)
        # 标签相等用和查表同一个 norm_label（去空格、不分大小写）：OCR 把「App Store」读成
        # 「App store」时，查表命中而这里找不到，就白回一趟主屏。
        label = next((e for e in cur.elements if norm_label(e.text) == want), None)
        if label is None:
            return "label_missing"            # 布局过时（用户挪了图标 / 翻错页）：不猜
        iy = icon_y_above_label(label.center[1], label.box)
        try:
            sx, sy = image_to_screen(label.center[0], iy, self._frame_of(cur))
        except OutOfWindow:
            return "out_of_window"
        self.dev.tap(sx, sy)
        frame, settled = settle(self.dev, frame, IOS_TIMING["open_app"], ahash)
        new = self.per.observe(frame)
        if in_spotlight(new) or looks_like_home(new):
            return "still_home"               # 点空了 / 点了标签：没开成，别报假成功
        try:
            same = _still_on_same_home_page(cur, new)
        except Exception:       # noqa: BLE001 —— 孪生的网格推断坏了 = 没有孪生，退回 Spotlight
            return "error"
        if same:
            return "still_home"               # 第三方 App 页上点空了：画面还是这一页（M2）
        ident = self._identity(name, new)
        if ident["verified"] is False:        # 进了别的 App（图标被挪过 / 标签被遮）：不报假成功
            wrong.append({"via": "layout", "tapped": hit.label, "seen": ident["seen"], "why": ident["why"]})
            return "wrong_app"
        ch = did_change(cur, new)
        return (ToolResult(ok=True, changed=ch.changed, hamming=ch.hamming, text_diff=ch.text_diff,
                           settled=settled,
                           extra={"via": "layout", "page": hit.page_order, "row": hit.row,
                                  "col": hit.col, "tapped": hit.label, "screen": [sx, sy],
                                  "identity": ident}),
                new)

    def _open_app_from_home(self, name: str, obs: Observation, typed: bool, query: str,
                            wrong: list | None = None):
        """退路：回主屏幕，一页页翻着找 App 图标。**全程不打字。**

        为什么值得单独做一条：Spotlight 要打字，而打字这条通道会整个死掉
        （2026-09-08 批跑：39/40 次打字失败，同一时间点击和滚动完全正常）。
        退路走另一条通道，才在那种时候还救得回来。

        ⚠ 点图标要用 `icon_above` —— 主屏幕上 OCR 只读得到图标**下面**的标签，
          点标签本身打不开 App（`设计说明`）。
        ⚠ 翻页之后必须硬等 `AFTER_PAGE_FLIP_S`，见那个常量的注释。
        """
        wrong = [] if wrong is None else wrong
        return_to_first_home_page(self.dev, self._frame_of(obs))
        frame = self.dev.capture()
        seen_pages = []
        for page in range(config.HOME_PAGES_MAX):
            cur = self.per.observe(frame)
            hit = next((e for e in cur.elements if name in e.text), None)
            if hit is not None:
                y = icon_y_above_label(hit.center[1], hit.box)
                self.dev.tap(*image_to_screen(hit.center[0], y, self._frame_of(cur)))
                frame, settled = settle(self.dev, frame, IOS_TIMING["open_app"], ahash)
                new = self.per.observe(frame)
                if in_spotlight(new) or looks_like_home(new):
                    # 点了图标还在主屏 / Spotlight：没开成。和 Spotlight 那条路一样，
                    # 开没开成看身份不看变化 —— 报 ok 就是假成功（2026-09-10）。
                    break
                ident = self._identity(name, new)
                if ident["verified"] is False:
                    # 点的是「名字里含 {name}」的那个东西（包含匹配），进的却是别的 App：
                    # 如实说点开了什么 —— 不能再说「翻了几页也没找到」，那是另一种失败。
                    # error 仍是 app_not_found、typed 照带：打字通道失效的阶梯靠它俩触发。
                    wrong.append({"via": "home_icon", "tapped": hit.text, "seen": ident["seen"],
                                  "why": ident["why"]})
                    return ToolResult(
                        ok=False, error="app_not_found",
                        hint=f"主屏第 {page + 1} 页上点了「{hit.text}」，打开的却不是「{name}」"
                             f"（看图说现在是：{ident['seen'] or '别的界面'}）。先看清当前画面再决定下一步。",
                        extra={"typed": typed}), new
                ch = did_change(cur, new)
                return (ToolResult(
                    ok=True, changed=ch.changed, hamming=ch.hamming,
                    text_diff=ch.text_diff, settled=settled,
                    hint=f"Spotlight 那条路没走通（打字没落进去），"
                         f"改在主屏第 {page + 1} 页上点开了「{name}」的图标。",
                    extra={"via": "home_icon", "page": page + 1, "identity": ident}), new)
            seen_pages.append(cur.ahash)
            self.dev.scroll("right", "page")
            frame, _ = settle(self.dev, frame, IOS_TIMING["scroll"], ahash)
            time.sleep(config.AFTER_PAGE_FLIP_S)
            frame = self.dev.capture()
            if self.per.observe(frame).ahash in seen_pages:
                break                          # 翻不动了（到最后一页），别空转
        new = self.per.observe(frame)
        why = ("Spotlight 里没有匹配的结果" if typed
               else f"{query!r} 没能输入到搜索框（打字通道可能失效了）")
        return ToolResult(
            ok=False, error="app_not_found",
            hint=f"打不开「{name}」：{why}；退回主屏幕翻了 {page + 1} 页也没找到这个图标。"
                 f"确认一下 App 名字对不对，或者它是不是在某个文件夹里。",
            extra={"typed": typed}), new

    @staticmethod
    def _query_landed(before: Observation, after: Observation, name: str) -> bool:
        """搜索词有没有真的落进输入框。

        只认「输入之前没有、输入之后才出现」的文本 —— 这正是 next_main 在 Android 上
        学到的落屏校验：画面上本来就有的文字（placeholder、常用 App 列表）不能当成
        输入成功的证据。
        """
        return any(name in e.text for e in after.elements) and \
            not any(name in e.text for e in before.elements)

    @staticmethod
    def _match_row(obs: Observation, name: str, exclude_top: bool):
        """结果列表里文字匹配目标的那一行。

        exclude_top=True 时跳过屏幕上方 25%（搜索框和输入回显所在的区域），
        避免把「自己刚输进去的那串字」误当成一条搜索结果去点。
        """
        # ⚠ 两条都是 2026-09-09 评测集跑出来的：
        #   · 去掉空格再比。OCR 读「Safari 浏览器」常吞掉中间的空格，
        #     `'Safari 浏览器' in 'Safari浏览器'` 是 False —— 真正的 App 行没匹配上，
        #     「提示」区一条含完整字样的长句反而中了，open_app 报假成功。
        #   · 原来 exclude_top 排掉顶部 25%，为的是不把「自己刚输进去的那串字」当结果点。
        #     那是旧 Spotlight 的布局；**iOS 18 的搜索框在底部**（实测 y≈0.92H），
        #     而「最佳搜索结果」的 App 标签恰恰在顶部 y≈0.246H —— 差 6 像素被排掉。
        #     现在：精确等于 App 名的行不看位置（那就是它）；包含匹配才排掉底部
        #     搜索框和输入法候选栏那一带。
        want = name.replace(" ", "")
        exact = [e for e in obs.elements if e.text.replace(" ", "") == want]
        if exact:
            return min(exact, key=lambda e: e.center[1])
        ceiling = obs.height_px * 0.85 if exclude_top else obs.height_px
        loose = [e for e in obs.elements
                 if want in e.text.replace(" ", "") and e.center[1] <= ceiling]
        return min(loose, key=lambda e: e.center[1]) if loose else None

    def _skill_text(self, kind: str, name: str) -> str | None:
        """只读当前索引里可见的条目：draft App、proposed 场景、损坏文件都读不到（spec §4.5）。"""
        from iphone_agent.skills.store import INDEX_TRUST
        if self.catalog is None:
            return None
        if kind == "app":
            a = self.catalog.app(name)
            if a is None or a.status != "verified":
                return None
            return f"【App】{a.display}\n{a.body.strip()}\n{INDEX_TRUST}"
        s = self.catalog.scenarios.get(name)
        if s is None or s.status != "manual":
            return None
        return f"【场景】{s.name}：{s.description}\n{s.body.strip()}\n{INDEX_TRUST}"
