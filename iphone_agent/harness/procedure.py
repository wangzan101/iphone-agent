"""照着剧本走：剧本内部的每个动作都是一等动作 —— 同一套校验、同屏去重、熔断、上限、留档。

为什么走不通就整个交回、不自己重试：一步对不上的原因（弹窗、改版、加载慢）剧本判断不了，
模型看着截图能。所以任何一步失败就停，连同当时的观察一起交回模型。
"""
from __future__ import annotations

import json
import time

from iphone_agent import config
from iphone_agent.harness.actions import Action, ValidationError, validate_action
from iphone_agent.harness.executor import ToolResult
from iphone_agent.harness.runlog import frame_stub, observation_snapshot
from iphone_agent.memory.screenmap import COVERAGE
from iphone_agent.perceive.change import did_change

# 只有这三种计入 fail_streak：它们说的是「剧本和界面对不上」。设备异常、模型错误、窗口移动、
# 熔断、上限说的是别的事，记进这条剧本的账是冤枉它（spec §2.3 迁移表）。
COUNTED_FAILURES = frozenset({"expect_mismatch", "target_not_found", "ambiguous_target"})

_WHY = {
    "expect_mismatch": "后没到该到的屏（期望有『{expected}』）",
    "target_not_found": "时在当前屏找不到目标",
    "ambiguous_target": "时目标命中了多个元素，不敢猜",
    "type_not_verified": "后没在输入区读到打进去的字",
    "repeated_action": "时被同屏去重拦下（同样的动作在这一屏做过且没变化）",
    "no_progress": "时触发了无进展熔断",
    "procedure_limit": "时超过了剧本内部动作上限或任务时限",
    "invalid_step": "时这一步本身校验不过",
}


class _StepFailed(Exception):
    def __init__(self, error: str, detail: str = "", obs=None):
        super().__init__(detail)
        self.error = error
        self.detail = detail
        self.obs = obs


def _fill(s: str | None, args: dict) -> str | None:
    if s is None:
        return None
    for k, v in args.items():
        s = s.replace("{" + k + "}", v)
    return s


def _hits(obs, text: str) -> list:
    t = text.strip().lower()
    return [e for e in obs.elements if t in e.text.lower()]


def read_row(obs, row: str) -> str:
    """标签必须**恰好一个**元素命中；同一行 = y 中心相差 ≤ READ_ROW_TOL_RATIO × 图高；
    只取标签右侧的元素，按 x 排序拼接；标签自身不进结果。"""
    hits = _hits(obs, row)
    if len(hits) != 1:
        raise _StepFailed("ambiguous_target" if hits else "target_not_found",
                          f"read 的标签「{row}」在当前屏命中 {len(hits)} 个元素，必须恰好一个", obs)
    label = hits[0]
    tol = config.READ_ROW_TOL_RATIO * obs.height_px
    same = [e for e in obs.elements
            if e is not label and abs(e.center[1] - label.center[1]) <= tol and e.center[0] > label.center[0]]
    return " ".join(e.text.strip() for e in sorted(same, key=lambda e: e.center[0]))


class ProcedureRunner:
    def __init__(self, executor, guard, log, deadline: float, on_step=None):
        self.ex = executor
        self.guard = guard
        self.log = log
        self.deadline = deadline
        self.on_step = on_step
        self._actions = 0
        self._moved = 0
        self._verdict: str | None = None

    def run(self, proc, args: dict, obs, call_id: str, tool_name: str):
        start = cur = obs
        self._actions = 0
        self._moved = 0
        self._verdict = None
        returns: dict[str, str] = {}
        done = 0
        try:
            for k, step in enumerate(proc.steps, start=1):
                if step.do == "read":
                    # read 不产生动作，但也不能在上限/时限之后还接着走（spec §4.4 的「上限」对整条剧本说话）。
                    self._check_budget(cur)
                    returns[step.as_] = read_row(cur, _fill(step.row, args))
                else:
                    cur = self._do(step, k, args, cur, call_id, tool_name)
                done = k
        except _StepFailed as f:
            step = proc.steps[done]
            target = _fill(step.target or step.row or step.text, args)
            # ⚠ 只 format 模板本身。fallback 里的 f.detail 是执行器的 hint，
            #   里面嵌着模型/剧本给的字符串（`打了拼音 {pinyin!r}…{text!r}`），带个花括号很正常；
            #   拿它去 .format() 会在 except 处理里再抛一次 KeyError，调用方连 ToolResult 都收不到。
            tpl = _WHY.get(f.error)
            why = tpl.format(expected="、".join(step.expect)) if tpl else "时出错：" + f.detail
            hint = (f"剧本 {tool_name} 走到第 {done + 1} 步『{step.do} {target}』{why}。"
                    f"当前屏幕已交给你，从这儿接着手动做；这条剧本会记一次失败。")
            res = ToolResult(ok=False, error=f.error, hint=hint, extra=self._extra({
                "procedure": tool_name, "step": done + 1, "do": step.do, "target": target,
                "expected": list(step.expect), "steps_done": done, "actions": self._actions,
                "moved_screens": self._moved}))
            out = f.obs if f.obs is not None else cur
            return res, (out if out is not start else None)
        ch = did_change(start, cur)
        res = ToolResult(ok=True, changed=ch.changed, hamming=ch.hamming, text_diff=ch.text_diff,
                         extra=self._extra({"procedure": tool_name, "steps_done": done,
                                            "actions": self._actions, "returns": returns}))
        return res, (cur if cur is not start else None)

    def _extra(self, d: dict) -> dict:
        """把熔断的判决原样带给调用方。

        熔断是一次性的：`record_outcome` 只在计数**正好等于**阈值的那一下报 warn/stop，
        之后计数继续往上爬，再也不会报第二次。剧本要是自己把这一枪吃掉（abort 完就不说了），
        主循环的 `verdict == "stop"` 分支就永远等不到 —— 熔断等于被剧本悄悄拆了。
        所以 stop 照旧中止剧本，但判决必须出现在结果里，由调用方决定整轮怎么收场。
        """
        if self._verdict is not None:
            d["guard_verdict"] = self._verdict
        return d

    # ---- 一步 ----
    def _do(self, step, k: int, args: dict, cur, call_id: str, tool_name: str):
        reason = f"剧本 {tool_name} 第 {k} 步"
        cid = f"{call_id}#{k}"
        target = _fill(step.target, args)
        if step.do == "open_app":
            action = Action("open_app", {"name": target}, reason, None, cid)
        elif step.do == "scroll_until":
            action = Action("scroll_until", {"direction": step.direction, "text": target}, reason, None, cid)
        elif step.do == "type":
            action = Action("type", {"text": _fill(step.text, args)}, reason, None, cid)
        elif step.do == "tap":
            hits = _hits(cur, target)
            if not hits and step.find:
                for _ in range(min(step.find["max_screens"], config.SCROLL_MAX_SCREENS)):
                    scroll = Action("scroll", {"direction": step.find["direction"], "amount": "page"}, reason, None, cid)
                    sent = self._actions
                    try:
                        _, cur, rec = self._execute(scroll, cur, k, call_id, tool_name)
                    except _StepFailed:
                        # 已经打出去的滚动要认账：设备层失败也已经动过屏幕，
                        # moved_screens 报 0 会让模型以为剧本压根没滚过。
                        # 上限/去重/校验那几种是**没发出去**就被拦下的，不算。
                        if self._actions > sent:
                            self._moved += 1
                        raise
                    self._moved += 1
                    self._log(rec)
                    hits = _hits(cur, target)
                    if hits:
                        break
            if not hits:
                raise _StepFailed("target_not_found", target, cur)
            if len(hits) > 1:
                raise _StepFailed("ambiguous_target", target, cur)
            action = Action("tap", {"id": hits[0].id, "target": step.how}, reason, None, cid)
        else:
            raise _StepFailed("invalid_step", f"未知 do {step.do}")

        res, new, rec = self._execute(action, cur, k, call_id, tool_name)
        # 各类动作「算成功」的条件（spec §2.3 表）
        if step.do == "type" and res.extra.get("verified") not in ("true", None) and not res.extra.get("picked"):
            rec["expect_ok"] = False
            self._log(rec)
            raise _StepFailed("type_not_verified", "", new)
        if step.do == "scroll_until" and not res.extra.get("found"):
            rec["expect_ok"] = False
            self._log(rec)
            raise _StepFailed("target_not_found", target, new)
        ok = True
        if step.expect:
            ok = len(set(step.expect) & new.text_set) / len(step.expect) >= COVERAGE
        rec["expect"] = list(step.expect)
        rec["expect_ok"] = ok
        self._log(rec)
        if not ok:
            raise _StepFailed("expect_mismatch", "", new)
        return new

    def _check_budget(self, cur) -> None:
        if self._actions >= config.PROC_MAX_ACTIONS:
            raise _StepFailed("procedure_limit", f"内部动作超过 {config.PROC_MAX_ACTIONS} 个", cur)
        if time.time() > self.deadline:
            raise _StepFailed("procedure_limit", "任务总时限已到", cur)

    def _execute(self, action: Action, cur, k: int, call_id: str, tool_name: str):
        """一个内部动作：上限 → 校验 → 同屏去重 → 执行 → 熔断记账 → 子记录。失败抛 _StepFailed。"""
        self._check_budget(cur)
        try:
            action = validate_action(action, cur)
        except ValidationError as e:
            raise _StepFailed("invalid_step", f"{e.code}: {e.message}", cur) from e
        if self.guard.check_repeat(action, cur.state_key, cur.width_px, cur.height_px):
            raise _StepFailed("repeated_action", "", cur)
        res, new = self.ex.run(action, cur)
        self._actions += 1
        if res.ok and not res.changed:
            self.guard.record_executed(action, cur.state_key, cur.width_px, cur.height_px)
        verdict = self.guard.record_result(action, res, new)
        if verdict is not None:
            self._verdict = verdict          # stop 盖过 warn：stop 这一步本来就要中止
        elif self.guard.no_progress == 0:
            self._verdict = None             # 计数被清零 = 真有进展，之前那次 warn 不作数了
        rec = {"parent_call_id": call_id, "procedure": tool_name, "step": k, "ts": time.time(),
               # ⚠ 2026-09-11 final review：漏了 ts，twin/events.py 兜底成 0.0，
               #   剧本碰过的屏全带 1970 年戳（runs 里子记录一直没有 ts 这个键）。
               "observation": observation_snapshot(cur, self.log.save_frame(frame_stub(cur))),
               "action": {"name": action.name, "args": dict(action.args)},
               "result": json.loads(res.to_json())}
        if new is not None:
            rec["after_frame_file"] = self.log.save_frame(frame_stub(new))
        if not res.ok:
            self._log(rec)
            # ⚠ device_error / activate_failed / out_of_window 时 new 是 None，
            #   交回 run() 自己的 cur 就成了**这一步之前**那张屏（find 滚过两屏也白滚）。
            #   hint 说的是「当前屏幕已交给你」，那就得给失败这一刻真正在手上的这张。
            raise _StepFailed(res.error or "device_error", res.hint or "", new if new is not None else cur)
        if verdict == "stop":
            self._log(rec)
            raise _StepFailed("no_progress", "", new)
        return res, (new if new is not None else cur), rec

    def _log(self, rec: dict) -> None:
        self.log.procedure_step(rec)
        if self.on_step:
            try:
                self.on_step(rec)
            except Exception:   # noqa: BLE001 —— 通知失败绝不能影响任务
                pass
