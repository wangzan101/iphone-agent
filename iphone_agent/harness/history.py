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

NOTE_MAX = 60
NO_CHANGE_SEMANTICS = {"recall", "recall_runs", "observe", "wait", "done", "handover"}


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
    judged = result.get("judged") or {}
    if expected == "unknown" and result.get("changed") and judged.get("worked") is False:
        expected = "no(看图)"
        note = note or judged.get("why")
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
    全靠它自己曾经塞进 600 字备忘（设计说明 B5+B6）。这里零模型调用，纯机械。
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

    _entries(lambda r: r.expected == "no", "没达到预期的")
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
