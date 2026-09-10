"""内置动作的内部表示与严格校验（spec §6.1）。名单见 BUILTIN_ACTION_NAMES —— 这里原来写着「9 个动作」，早就不是 9 个了；数字写进文档就会烂，改成指向那个常量。

harness 内部坐标只有图像像素。

坐标路径受 allow_coord_tap 门控：未标定的模型只能按 id 点。"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from iphone_agent import config
from iphone_agent.perceive.elements import ZOOM_MIN_PX

ICON_ABOVE_LABEL_RATIO = 3   # 图标中心 ≈ 标签中心上移 3 倍文字高（2026-09-07 实测）


def icon_y_above_label(center_y, box):
    """标签上方图标的 y 坐标：从标签中心上移 ICON_ABOVE_LABEL_RATIO 倍文字高，下限 0。

    一个规则一个入口（项目开发约定）：主屏幕 / Spotlight「点了标签还在原处」那一步 /
    布局表查表直达，凡是要从标签框推图标位置，都调这里，别再各自重复这条公式。
    `box` 是元素包围盒 `(x1, y1, x2, y2)`。
    """
    label_h = box[3] - box[1]
    return max(0, center_y - ICON_ABOVE_LABEL_RATIO * label_h)


BUILTIN_ACTION_NAMES = ("observe", "zoom", "tap", "scroll", "scroll_until", "collect", "type",
                        "erase", "key", "open_app", "wait", "done", "recall", "recall_runs",
                        "switch_ime", "search_memory", "use_skill", "handover")
ACTION_NAMES = BUILTIN_ACTION_NAMES
# 不计熔断。⚠ 只是「不计熔断」，不是「不消耗步数」—— loop.py 的 steps += 1 不看动作名，
# 这里进不进都照样吃一步预算。两件事必须分开：
#   recall/recall_runs 查的是文字，本来就不改变画面（跟 observe/wait 同类）。留在熔断
#   计数里会撞出一个假信号：RECALL_PER_RUN 和 NO_PROGRESS_WARN 都是 3，模型规规矩矩
#   用满查询预算，第 3 次就被判「连续多步无进展，换一条路」。同屏去重也一样——
#   「这个动作做过了且画面没变」是给屏幕操作设计的规则，对 recall 永远成立。
#   search_memory（计划 B Task 3）是同一类东西——查记忆索引之外的条目，同样不改
#   画面，同样该跟 recall 共用查询预算（loop.py 的 recall_used），不该被当成
#   「原地打转」熔断掉。use_skill（skill 层）读的是知识库的正文，性质一样。
NON_COUNTING = {"observe", "zoom", "wait", "recall", "recall_runs", "search_memory",
                "use_skill", "handover"}   # handover 自己不动手机，画面是人改的
# ⚠ zoom 必须在里面。它只看不动，屏幕本来就不该变 —— 让它去撞「连续无进展」
#   等于惩罚模型「先看清楚再动手」，那正是我们要它养成的习惯。

# 只豁免「连续无进展」这一道，**同屏去重那一道照旧管着**。
# 中文输入是个多步过程：打拼音 → 看候选 → 点候选 → 打下一段，输一个 App 名要六到八步。
# 这些步都在同一片区域里折腾，很容易被判成原地打转，把无进展预算吃光，挤掉任务本身。
# 但 type 不能进 NON_COUNTING —— 那会把同屏去重也关掉，而「往一个没聚焦的输入框里
# 反复打同样的字」正是要靠同屏去重拦的。留着它，那种死循环第二次就被拒。
NO_PROGRESS_EXEMPT = NON_COUNTING | {"type"}

# key(name) 的全部名字 —— 工具 schema（tools.py）和 Device._KEY_COMBOS 都从这里对齐，只此一处。
# 后八个是光标键（硬件键盘的 Cmd+方向键 / 方向键，2026-09-10 真机实测 Cmd+←/→ 在镜像里通）。
# ⚠ 为什么要有它们：光标看不见，点文字时它落在被点的那个字上，不是行末。09-10 真机（备忘录另起一行）
#   模型没有挪光标的手，27 步里 16 步在跟光标搏斗 —— 回车把一行劈成两行、字插在中间、再删。
KEY_NAMES = ("home", "app_switcher", "spotlight", "return",
             "line_start", "line_end", "text_start", "text_end", "left", "right", "up", "down")
CURSOR_KEYS = frozenset(KEY_NAMES[4:])


def screen_neutral(action) -> bool:
    """画面**本来就不该变**的动作：只看不动的那些，和挪光标（光标不可见）。
    三处规矩都从这里判：不计无进展、不做同屏去重、执行层报 changed=None 而不是 False。
    2026-09-08 zoom 没进 NON_COUNTING，「先看清楚再动手」被熔断惩罚 —— 光标键是同一类。"""
    return action.name in NON_COUNTING or (action.name == "key" and action.args.get("name") in CURSOR_KEYS)


class ValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Action:
    name: str
    args: dict
    reason: str
    expect: str | None
    call_id: str
    # 模型对上一步的评价与自己维护的备忘（设计说明）。默认 None：
    # 老模型、老 prompt、测试脚本都不给这两个字段，构造必须照旧能用。
    eval: str | None = None
    memory: str | None = None

    def describe(self) -> str:
        """一句话说这个动作干了什么，给看图复核当上下文（harness/judge.py）。

        ⚠ 只列**动作本身**的参数，不带 reason —— reason 是模型自己写的说辞，
          拿它去问「这个动作生效了吗」，等于让判官先读一遍辩方陈词。
        """
        if not self.args:
            return self.name
        shown = {k: v for k, v in self.args.items() if not k.startswith("_")}
        inner = "，".join(f"{k}={v!r}" for k, v in sorted(shown.items()))
        return f"{self.name}({inner})"


def _num(v, field):
    # bool 是 int 的子类，isinstance(True, (int, float)) 为真，必须显式排除，
    # 否则 [True, False] 会被当成合法包围盒取中点得到 0.5，绕过下面的 bool 拦截。
    if (isinstance(v, list) and len(v) == 2
            and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in v)):
        v = (v[0] + v[1]) / 2  # B129：包围盒取中点
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValidationError("invalid_args", f"{field} 必须是有限数字")
    return v


def _clamp_axis(v: float, size: int, field: str, coord_mode: str) -> int:
    tol = size * config.CLAMP_TOLERANCE
    if v < -tol or v > size + tol:
        # 越界最常见的原因是坐标约定搞错了（给了像素，而这里要 0-1000），
        # 光说「越界」模型只会换个像素值再试一遍 —— 真机上连试五次被熔断。
        hint = ("坐标要用 0-1000 的归一化值，不是像素 —— 看元素列表头部那行。"
                if coord_mode == "norm1000" else "")
        raise ValidationError(
            "out_of_image",
            f"{field}={v} 超出图像 {size} 且超过 {config.CLAMP_TOLERANCE:.0%} 容差。{hint}")
    return int(round(min(max(v, 0), size)))


def _enum(v, allowed, field):
    if v not in allowed:
        raise ValidationError("invalid_args", f"{field} 必须是 {allowed} 之一，收到 {v!r}")
    return v


def validate_action(action: Action, obs, procedures: dict | None = None, *,
                    allow_coord_tap: bool = True) -> Action:
    procedures = procedures or {}
    if action.name in procedures:
        proc = procedures[action.name]
        a = {}
        for k in proc.params:
            v = action.args.get(k)
            if not isinstance(v, str) or not v.strip() or len(v) > 200:
                raise ValidationError("invalid_args", f"剧本 {action.name} 的参数 {k} 必须是 1–200 字的字符串")
            a[k] = v
        extra = set(action.args) - set(proc.params)
        if extra:
            raise ValidationError("invalid_args", f"剧本 {action.name} 没有参数 {sorted(extra)}")
        return replace(action, args=a)
    if action.name not in BUILTIN_ACTION_NAMES:
        raise ValidationError("unknown_tool", f"未知工具 {action.name}")
    a = dict(action.args)
    n = action.name
    if n == "tap":
        has_id = "id" in a
        has_xy = "x" in a and "y" in a
        if has_id == has_xy or ("x" in a) != ("y" in a):
            raise ValidationError("invalid_args", "tap 需要 id 或 (x,y) 二选一")
        target = _enum(a.get("target", "text"),
                       ("text", "icon_above", "row_right", "row_left"), "target")
        a["target"] = target
        if has_id:
            if obs is None or not isinstance(a["id"], int) or isinstance(a["id"], bool):
                raise ValidationError("invalid_args", "id 必须是整数")
            el = obs.element(a["id"])
            if el is None:
                raise ValidationError("stale_element_id", f"id {a['id']} 不在当前 observation #{obs.observation_id}")
            a["x"], a["y"] = el.center
            if target == "icon_above":
                # 图标在标签正上方。偏移取标签文字高度的 3 倍 —— 2026-09-07 真机实测：
                # 主屏幕「设置」标签高 23px、中心 y=796，点 y=796 完全没反应；
                # 点 y=727（上移 3 倍）和 y=750（上移 2 倍）都成功打开。
                # 规则确定且有容差，所以由代码算而不是让模型估坐标
                # —— 坐标永远来自框，模型只负责选目标。
                a["y"] = icon_y_above_label(a["y"], el.box)
            elif target in ("row_right", "row_left"):
                # 列表行的版式很规整：文字在左、chevron/开关/数值在右、图标在最左，
                # 三者在同一个 y 上。所以从行文字能推出同一行上另外两个位置，
                # 而且推出来的是**精确坐标**，不是模型估的（模型估图标 p90 差 117px，
                # 一个图标才 120px 宽 —— 2026-09-08 标定，见 设计说明）。
                #
                # ⚠ row_right 是**开关唯一能点中的地方**：点行文字不会切换开关，
                #   而开关在 OCR 里根本没有元素。实测见 config 里的常量注释。
                ratio = (config.ROW_RIGHT_RATIO if target == "row_right"
                         else config.ROW_LEFT_RATIO)
                a["x"] = int(round(obs.width_px * ratio))
        elif target != "text":
            raise ValidationError(
                "invalid_args",
                f"target={target} 只能配 id 用 —— 它是从那个文字元素的框推出来的坐标")
        else:
            if not allow_coord_tap:
                raise ValidationError(
                    "coord_disabled",
                    "这个模型未标定坐标，坐标点击已关闭，请用元素编号 id（配 target 可点图标/开关/行首）")
            if obs is None:
                raise ValidationError("invalid_args", "尚无观察，不能用坐标")
            x = _num(a["x"], "x"); y = _num(a["y"], "y")
            a["x"] = _clamp_axis(x, obs.width_px, "x", obs.coord_mode)
            a["y"] = _clamp_axis(y, obs.height_px, "y", obs.coord_mode)
    elif n == "scroll":
        # 四个方向都走滚轮。镜像里没有可用的拖拽手势，也没有系统返回，所以没有 swipe。
        _enum(a.get("direction"), ("up", "down", "left", "right"), "direction")
        _enum(a.get("amount", "page"), ("page", "half"), "amount")
        a.setdefault("amount", "page")
    elif n in ("scroll_until", "collect"):
        # direction 的含义与 scroll 逐字相同 —— 同一件事只允许有一个理解，
        # 否则模型会在两个工具之间读到互相矛盾的方向语义。
        # 没有 amount：这两个工具按「屏」计数，一屏就是 scroll 的 page。
        _enum(a.get("direction"), ("up", "down", "left", "right"), "direction")
        if n == "scroll_until":
            # ⚠ 谓词只能是数据。调用方是个发 JSON 的模型，传不了回调函数。
            t = a.get("text")
            if not isinstance(t, str) or not t.strip() or len(t) > config.SCROLL_UNTIL_TEXT_MAX:
                raise ValidationError(
                    "invalid_args",
                    f"text 必须是 1–{config.SCROLL_UNTIL_TEXT_MAX} 字的非空字符串")
    elif n == "type":
        t = a.get("text")
        if not isinstance(t, str) or not t or len(t) > config.MAX_TYPE_TEXT:
            raise ValidationError("invalid_args", f"text 必须是 1–{config.MAX_TYPE_TEXT} 字的字符串")
    elif n == "zoom":
        if obs is None:
            raise ValidationError("invalid_args", "尚无观察，不能放大")
        # 坐标和 tap 走**同一条**换算：一个规则一个入口。elements.py 那段注释记着
        # 「同一段文字里两套坐标约定」害得连拒五次的事故，不能再来一次。
        x1 = _clamp_axis(_num(a.get("x1"), "x1"), obs.width_px, "x1", obs.coord_mode)
        x2 = _clamp_axis(_num(a.get("x2"), "x2"), obs.width_px, "x2", obs.coord_mode)
        y1 = _clamp_axis(_num(a.get("y1"), "y1"), obs.height_px, "y1", obs.coord_mode)
        y2 = _clamp_axis(_num(a.get("y2"), "y2"), obs.height_px, "y2", obs.coord_mode)
        x1, x2 = min(x1, x2), max(x1, x2)      # 角给反了排一下，比拒掉它有用
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 < ZOOM_MIN_PX or y2 - y1 < ZOOM_MIN_PX:
            # 夹大反而危险：模型想看的是它自己框的那块，我们替它扩了就不是同一块了。
            raise ValidationError(
                "invalid_args",
                f"放大区域太小（{x2 - x1}x{y2 - y1} 像素），至少要 "
                f"{ZOOM_MIN_PX}x{ZOOM_MIN_PX}。框大一点，把目标连同周围一起圈进来。")
        a["x1"], a["y1"], a["x2"], a["y2"] = x1, y1, x2, y2
    elif n == "erase":
        # 夹住而不是报错？**不行。** 模型说「退 500 个」时它想的是「清空」，
        # 而这里一夹变成退 100 个，删掉的是它没打算删的正文，且没有回头路。
        # 数量对不上就打回去，让它自己说清楚要删多少。
        c = a.get("count")
        if not isinstance(c, int) or isinstance(c, bool) or not (1 <= c <= config.MAX_ERASE):
            raise ValidationError("invalid_args",
                                  f"count 必须是 1–{config.MAX_ERASE} 的整数")
    elif n == "key":
        _enum(a.get("name"), KEY_NAMES, "name")
    elif n == "open_app":
        t = a.get("name")
        if not isinstance(t, str) or not t.strip() or len(t) > 50:
            raise ValidationError("invalid_args", "name 必须是 1–50 字的字符串")
    elif n == "wait":
        s = _num(a.get("seconds"), "seconds")
        if not (config.WAIT_MIN_S <= s <= config.WAIT_MAX_S):
            raise ValidationError("invalid_args", f"seconds 必须在 {config.WAIT_MIN_S}–{config.WAIT_MAX_S}")
    elif n == "recall":
        if not isinstance(a.get("name"), str) or not a["name"].strip():
            raise ValidationError("invalid_args", "recall 需要一个字符串 name")
    elif n == "search_memory":
        q = a.get("query")
        if not isinstance(q, str) or not (1 <= len(q) <= 100):
            raise ValidationError("invalid_args", "query 必须是 1–100 字的字符串")
    elif n == "recall_runs":
        # 夹住而不是报错：模型给 100 是想多看点，不是犯错。
        v = a.get("limit", 5)
        if isinstance(v, bool) or not isinstance(v, int):
            v = 5
        a["limit"] = min(max(v, 1), config.RECALL_RUNS_MAX)
    elif n == "use_skill":
        _enum(a.get("kind"), ("app", "scenario"), "kind")
        if not isinstance(a.get("name"), str) or not a["name"].strip():
            raise ValidationError("invalid_args", "use_skill 需要一个字符串 name")
    elif n == "handover":
        need = a.get("need")
        if not isinstance(need, str) or not need.strip() or len(need) > 300:
            raise ValidationError("invalid_args", "handover 需要 need：一句话说清楚要人做什么（1–300 字）")
        return replace(action, args={"need": need.strip()})
    elif n == "done":
        _enum(a.get("status"), ("success", "failed"), "status")
        if not isinstance(a.get("result"), str):
            raise ValidationError("invalid_args", "result 必须是字符串")
        # 写入搭 done 的便车：loop.py 对每个通过校验的动作都 steps += 1，
        # 独立的 remember 工具会占掉一步预算，模型就会为了完成任务而不记。
        items = a.get("remember", [])
        if items is None:
            items = []
        if not isinstance(items, list):
            raise ValidationError("invalid_args", "remember 必须是列表")
        if len(items) > config.MEMORY_WRITE_PER_RUN:
            raise ValidationError(
                "invalid_args",
                f"remember 最多 {config.MEMORY_WRITE_PER_RUN} 条，收到 {len(items)}")
        normalized = []
        for it in items:
            if not isinstance(it, dict) or any(
                    not isinstance(it.get(k), str) or not it.get(k)
                    for k in ("name", "description", "content")):
                raise ValidationError(
                    "invalid_args",
                    "remember 的每一条都要有字符串 name、description、content")
            # kind 在这里补默认值而不是留给 store：写盘的地方要显式知道自己在写
            # 哪一类，「缺省即 knowledge」的隐式解析散在两处早晚会分裂。
            # 白名单之外一律拒绝 —— playbook 是「照着做」的步骤，未知的第五个值
            # 该按什么强度对待没人知道，猜一个善意的默认就是在给将来埋雷。
            # ⚠ 这个白名单是两条线合并来的：knowledge / playbook 落记忆库（原来
            #   skill 层管它叫 "memory"，合并后统一用 knowledge 这个名字，两处
            #   缺省值才不会分裂）；app_note / scenario 落 skill 层，而且只是提议，
            #   人批准才生效。
            kind = it.get("kind", "knowledge")
            if kind is None:
                kind = "knowledge"
            _enum(kind, ("knowledge", "playbook", "app_note", "scenario"), "remember[].kind")
            if kind == "app_note" and not (isinstance(it.get("app"), str) and it["app"].strip()):
                raise ValidationError("invalid_args", "kind=app_note 的条目必须给 app")
            if kind == "scenario":
                apps = it.get("apps")
                if not isinstance(apps, list) or not apps or not all(isinstance(x, str) and x for x in apps):
                    raise ValidationError("invalid_args", "kind=scenario 的条目必须给非空的 apps")
            # 复制一份再补 kind：这些 dict 就是模型原样传来的对象，日志里的
            # args_raw 与它们同源，原地改会让「原始参数」记上我们补的默认值。
            normalized.append(it | {"kind": kind})
        a["remember"] = normalized
        # 模型自报「这次哪几条记忆帮上忙了」。上限 10：它是反馈不是清单，
        # 报一屏的名字没有信息量，还会把 run.json 撑大。
        used = a.get("used_memories", [])
        if used is None:
            used = []
        if not isinstance(used, list):
            raise ValidationError("invalid_args", "used_memories 必须是列表")
        if len(used) > config.MEMORY_USED_PER_RUN:
            raise ValidationError(
                "invalid_args",
                f"used_memories 最多 {config.MEMORY_USED_PER_RUN} 条，收到 {len(used)}")
        if any(not isinstance(x, str) or not x.strip() for x in used):
            raise ValidationError("invalid_args", "used_memories 的每一项都要是非空字符串")
        # ⚠ strip 的结果要**存回去**，不能只用来判空：" k \n" 原样存下来，
        #   下游拿它去比对记忆名永远不中，看起来就像模型报了个不存在的名字。
        #   长度上限跟 MEMORY_NAME_RE 的 48 对齐——这里收的是记忆名，不是自由文本；
        #   没有上限就意味着模型能把一整段话塞进 run.json（评审 Minor 7）。
        used = [x.strip() for x in used]
        _NAME_MAX = 48
        if any(len(x) > _NAME_MAX for x in used):
            raise ValidationError(
                "invalid_args", f"used_memories 的每一项最多 {_NAME_MAX} 字符")
        a["used_memories"] = used
    return replace(action, args=a)
