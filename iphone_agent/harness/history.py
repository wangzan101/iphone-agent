"""每一步压成一行，给状态报告用。

⚠ 不压成一个 ok/failed。did_change 是「明显视觉差异」不是「动作生效」
（change.py:1、guard.py:51）：点错入口 changed=true 但错了；recall 成功屏幕不变；
scroll 到底不变但那正是成功。压成一个词等于给模型造伪事实（Codex 评审 §2.2）。
所以四列：执行了没（程序知道）/ 画面变没变（程序知道）/ 达到预期没（模型下一步
自己答，可以 unknown）/ 一句话。让模型自己看列做判断。

失败行保留：擦掉失败就是擦掉证据（Manus）。被拒的动作也进历史，标 R，不占步数。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from iphone_agent import config
from iphone_agent.harness import verdict as verdict_mod

NOTE_MAX = 60
NO_CHANGE_SEMANTICS = {"recall", "recall_runs", "observe", "wait", "done", "handover"}
# ⚠ 2026-09-18（最终评审 M10）：滚着找东西的两个动作，「滚到底了没有这一项」是**正当答案**不是失败。
#   verdict.of() 给它 not_met/text（守卫要靠这个否定判无进展），expected 列照读的话，这一行会以
#   「no(文字)」进【更早】的「没达到预期的」一节 —— 模型读到的是「这一步错了」，而它其实办完了事。
#   这一列对它们保持 unknown、不填，和 by=="pixels" 不填同理（下面那段注释）。
FOUND_OR_NOT = {"scroll_until", "collect"}


@dataclass(frozen=True)
class HistoryRow:
    step: str
    action: str
    executed: str
    changed: str
    expected: str
    note: str


def _action_text(name: str, args: dict, elements_by_id: dict[int, str] | None) -> str:
    a = dict(args)
    if name == "tap":
        if "id" in a:
            label = (elements_by_id or {}).get(a["id"])
            target = f'"{label}"' if label else f"#{a['id']}"
        elif "x" in a and "y" in a:
            target = f"({a['x']},{a['y']})"
        else:
            target = "?"
        t = a.get("target")
        return f"tap {target}" + (f"({t})" if t and t != "text" else "")
    if name in ("scroll", "scroll_until", "collect"):
        s = f"{name} {a.get('direction', '?')}"
        if name == "scroll_until":
            s += f' "{a.get("text", "")}"'
        return s
    if name == "type":
        return f'type "{a.get("text", "")}"'
    if name in ("recall", "open_app"):
        return f"{name} {a.get('name', '')}"
    if name == "key":
        return f"key {a.get('name', '')}"
    if name == "done":
        return f"done {a.get('status', '')}"
    return name


def _first_line(s) -> str:
    s = str(s or "").strip().splitlines()
    return s[0] if s else ""


def _cap(s: str) -> str:
    return s if len(s) <= NOTE_MAX else s[:NOTE_MAX] + "…"


def row_from_record(record: dict, elements_by_id: dict[int, str] | None = None) -> HistoryRow:
    name = record["action"]["name"]
    args = record["action"].get("args", record["action"].get("args_raw", {}))
    result = record.get("result") or {}
    action = _action_text(name, args, elements_by_id)
    rejected = record.get("validation") or record.get("rejected")
    if rejected:
        note = str(rejected)
        eval_note = (record.get("eval") or {}).get("note")
        if eval_note:
            note = f"{note} · {eval_note}"
        return HistoryRow("R", action, "rejected", "n/a", "n/a", _cap(note))
    executed = "yes" if result.get("ok") else "error"
    if name in NO_CHANGE_SEMANTICS or result.get("changed") is None:
        changed = "n/a"
    else:
        tr = record.get("transition") or {}
        if tr.get("changed") is False and tr.get("local_changed"):
            changed = "local"
        else:
            changed = "yes" if result.get("changed") else "no"
    ev = record.get("eval") or {}
    expected = ev.get("expected") or "unknown"
    note = ev.get("note")
    # 模型的自评字段从来没人填（2026-09-09 统计 103 次运行一次没有），这一列永远 unknown。
    # 判官看图判了「变了但没达到预期」时用它补上，并标明是看图判的 —— 模型自己答了以它为准。
    # ⚠ 2026-09-18：这里原来直接读 expect_check / judged 两个旧字段，各读各的含义。Task 3 把
    #   「画面没变的 tap」的 expect_check 改成 {met: None, by: not_asked}（不再拿「生效了吗」的
    #   答案冒充「达到预期了吗」）之后，这条读法看不到判官的结论了，「没变 + 看图说没生效」的 tap
    #   从 no(看图) 静默退成了 unknown。结论其实在 verdict 里（verdict.of() 对新旧留档是同一个
    #   函数，见 verdict.py 模块文档），改读它：旧留档没有 verdict 键，of() 会从旧字段推导，
    #   与这里原来的两段分支逐字节对齐（除一种极少见的旧形状，见下）。
    if expected == "unknown" and name not in FOUND_OR_NOT:
        v = verdict_mod.of(name, args, result)
        if v.state == "met" or v.state == "not_met":
            if v.by == "text":
                label = "(文字)"
            elif v.by in ("vision", "readback", "identity"):
                label = "(看图)"
            else:
                # by == "pixels"（或其它未知来源）：像素判据的成/没成已经在同一行的 changed 列里说过，
                # 不在这一列里再说一遍；而且 changed=no 不等于失败（滚到底就是这样，见 prompt.py 对
                # 这张表的说明）。保持 unknown、不填。
                label = None
            if label:
                expected = ("yes" if v.state == "met" else "no") + label
                if v.state == "not_met":
                    note = note or v.why
    if not note:
        if executed == "error":
            note = result.get("hint") or result.get("error") or ""
        elif name == "recall":
            note = _first_line(result.get("content"))
        elif name == "recall_runs":
            note = f"{len(result.get('runs') or [])} 条"
        elif name == "collect":
            note = f"{result.get('count', 0)} 行，{result.get('screens', 0)} 屏"
        elif result.get("hint"):
            note = result["hint"]
        else:
            note = _first_line((record.get("model") or {}).get("reason"))
            if not note:
                note = "画面无变化" if changed == "no" else ""
    return HistoryRow(str(record.get("step", "?")), action, executed, changed, expected, _cap(_first_line(note)))


def _verb(action: str) -> str:
    return action.split(" ", 1)[0] if action else "?"


def _tap_target(action: str) -> str:
    """从 `tap ...` 动作文本里取出目标。

    去掉 `tap ` 前缀后：以 `"` 开头取到下一个 `"`（元素标签）；以 `(` 开头取到
    对应 `)`（坐标 `(x,y)` 原样保留）；否则取到第一个 `(` 之前（`target=...` 后缀）
    或整段（`#id` 之类）。
    """
    rest = action[len("tap "):]
    if rest.startswith('"'):
        end = rest.find('"', 1)
        return rest[1:end] if end != -1 else rest.lstrip('"')
    if rest.startswith("("):
        end = rest.find(")")
        return rest[:end + 1] if end != -1 else rest
    idx = rest.find("(")
    return rest[:idx] if idx != -1 else rest


def _reject_key(note: str) -> str:
    return note.split(" · ", 1)[0] if " · " in note else note


def _cap_list(items: list[str], list_max: int, unit: str) -> str:
    shown = [_cap(t) for t in items[:list_max]]
    s = ", ".join(shown)
    if len(items) > list_max:
        s += f" …共 {len(items)} {unit}"
    return s


def summarize_omitted(rows: list[HistoryRow], list_max: int = config.EARLIER_LIST_MAX) -> str:
    """把被窗口挤出去的历史行压成一段有界摘要，替代原来那句「[… k 步已省略 …]」。

    ⚠ 为什么必须有这一层：MAX_STEPS 已经是 100，HISTORY_KEEP 是 12。没有它，
    第 90 步的模型看到的是「第 1 步 + 一句省略 + 最近 11 步」，中间 77 步的证据
    全靠它自己曾经塞进 600 字备忘（docs/20 B5+B6）。这里零模型调用，纯机械。
    保留的四样东西按信息价值排：试过但没成的（别再试）、被拒的（别再犯）、
    花步数换来的读数（recall/collect，丢了就得重做）、动作分布（知道自己在哪转过）。
    """
    if not rows:
        return ""
    first, last = rows[0].step, rows[-1].step
    lines = [f"【更早】第 {first}–{last} 步（{len(rows)} 步）"]

    verbs = Counter(_verb(r.action) for r in rows)
    tap_targets = list(dict.fromkeys(
        _tap_target(r.action) for r in rows if r.action.startswith("tap ")))
    tap_targets = [t for t in tap_targets if t]
    parts = []
    for v, n in verbs.most_common():
        if v == "tap" and tap_targets:
            parts.append(f"tap {n}（目标 {_cap_list(tap_targets, list_max, '个')}）")
        else:
            parts.append(f"{v} {n}")
    lines.append("动作分布：" + "、".join(parts))

    def _entries(sel, label):
        items = [f"{r.step} {_cap(r.action)}（{r.note}）" if r.note else f"{r.step} {_cap(r.action)}"
                 for r in rows if sel(r)]
        if not items:
            return
        s = "；".join(items[:list_max])
        if len(items) > list_max:
            s += f"…（共 {len(items)} 条，列前 {list_max}）"
        lines.append(f"{label}：{s}")

    # ⚠ startswith 不是 ==：expected 实际写的是「no(看图)」（row_from_record 按 expect_check 填），裸「no」
    #   只有模型自评时才有。只认 == "no" 时这一节从来没命中过，走错页的步被静默压掉（终审 2026-09-11）。
    _entries(lambda r: r.expected.startswith("no"), "没达到预期的")
    rejected = Counter(_reject_key(r.note) for r in rows if r.executed == "rejected")
    if rejected:
        shown = rejected.most_common(list_max)
        s = "、".join(f"{k} ×{n}" for k, n in shown)
        if len(rejected) > list_max:
            s += f"…（共 {len(rejected)} 种）"
        lines.append(f"被拒 {sum(rejected.values())} 次：" + s)
    _entries(lambda r: r.executed == "yes" and _verb(r.action) in ("recall", "recall_runs", "collect", "scroll_until"), "读到的")
    return "\n".join(lines)


def render_history(rows: list[HistoryRow], keep: int) -> str:
    if not rows:
        return "【到目前为止】\n（还没有动作）"
    lines = ["【到目前为止】", "step | 动作 | executed | changed | expected | 说明"]

    def _line(r):
        return f"{r.step} | {r.action} | {r.executed} | {r.changed} | {r.expected} | {r.note}"

    if len(rows) <= keep:
        lines += [_line(r) for r in rows]
        return "\n".join(lines)
    tail = max(1, keep - 1)
    lines.append(_line(rows[0]))
    lines.append(summarize_omitted(rows[1:-tail]))
    lines += [_line(r) for r in rows[-tail:]]
    return "\n".join(lines)
