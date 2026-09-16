"""孪生的输入：从 steps.jsonl 抽「可重放动作事件」，并推当前 App（spec §1、§2）。

实时（twin/live.py）与重放（twin/record.py）吃同一种事件、走同一个 AppTracker ——
一个规则一个入口（开发原则 §7）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from iphone_agent.memory.screenmap import SYSTEM, UNKNOWN, _icon_tap_label, app_id_for

# 会改变画面、值得记转移的动作
TRANSITION_ACTIONS = frozenset({"tap", "scroll", "scroll_until", "collect", "type", "erase", "key", "open_app"})
# 能当事件（算一次访问）的动作。顶层剧本调用的工具名不在这里 —— 它的内部动作另有子记录（kind=procedure_step）。
EVENT_ACTIONS = TRANSITION_ACTIONS | frozenset({
    "observe", "zoom", "wait", "done", "recall", "recall_runs", "search_memory", "use_skill",
    "switch_ime", "handover"})
# 带这些键的记录不是「执行过的动作」：阶段错误、被拒（校验 / 安全闸 / 多工具调用）。
_SKIP_KEYS = ("phase", "error_phase", "validation", "rejected")


@dataclass(frozen=True)
class Snapshot:
    """一帧观察里孪生用得到的部分：元素行、图像尺寸、帧文件名、视觉标注与当时给过的候选。"""
    elements: tuple[dict, ...]
    width_px: int
    height_px: int
    frame_file: str = ""
    label: dict | None = None                 # observation["screen"] 原样；重放时再过一遍 parse_screen_label
    label_status: str = "off"                 # observation["perception"]["screen"]；老留档没有 → off
    candidates: tuple[dict, ...] = ()

    @classmethod
    def from_observation(cls, obs) -> Snapshot | None:
        els = obs.get("elements") if isinstance(obs, dict) else None
        if not isinstance(els, list):
            return None
        per = obs.get("perception") if isinstance(obs.get("perception"), dict) else {}
        label = obs.get("screen") if isinstance(obs.get("screen"), dict) else None
        cands = obs.get("screen_candidates")
        return cls(tuple(e for e in els if isinstance(e, dict)),
                   int(obs.get("width_px") or 0), int(obs.get("height_px") or 0),
                   str(obs.get("frame_file") or ""), label,
                   str(per.get("screen") or ("ok" if label is not None else "off")),
                   tuple(c for c in cands if isinstance(c, dict)) if isinstance(cands, list) else ())


@dataclass(frozen=True)
class Event:
    """一个执行过的动作：动作前那一帧、动作、结果、动作后那一帧。"""
    event_id: str                 # f"{run_id}:{steps.jsonl 行号}"
    ts: float
    before: Snapshot
    action: dict                  # {"name": str, "args": dict}
    result: dict                  # ToolResult.to_json() 的输出：extra 已摊平在顶层
    after: Snapshot | None
    after_event_id: str | None    # after 那一帧作为事件出现时的 id；after 是新屏时拿它当 first_event

    @property
    def name(self) -> str:
        return str(self.action.get("name") or "")

    @property
    def args(self) -> dict:
        a = self.action.get("args")
        return a if isinstance(a, dict) else {}


def extract_events(run_id: str, lines: list[str]) -> list[Event]:
    """⚠ id 用行号不用 step：被拒动作与下一个动作同号，剧本内部步号是局部的（codex 评审第 6 条）。
    ⚠ after 按帧文件配对，不按「物理下一行」：中间可能夹着出错记录、被拒记录（第 10 条）。
    ⚠ 2026-09-14（spec 按需看图 §8.1）：after 优先用本条记录自己的 after_observation；没有才按帧文件配（老留档）。"""
    raw: list[tuple[int, dict, Snapshot]] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or any(k in rec for k in _SKIP_KEYS):
            continue
        action = rec.get("action")
        if not isinstance(action, dict) or action.get("name") not in EVENT_ACTIONS:
            continue
        snap = Snapshot.from_observation(rec.get("observation"))
        if snap is None:
            continue
        raw.append((lineno, rec, snap))
    # 从后往前扫，记住每个帧文件「下一次作为观察出现」在哪：O(n)，且只会配到后面的事件。
    nxt: dict[str, int] = {}
    after_idx: list[int | None] = [None] * len(raw)
    for i in range(len(raw) - 1, -1, -1):
        after_file = raw[i][1].get("after_frame_file")
        if after_file:
            after_idx[i] = nxt.get(str(after_file))
        frame = (raw[i][1].get("observation") or {}).get("frame_file")
        if frame:
            nxt[str(frame)] = i
    events: list[Event] = []
    for i, (lineno, rec, snap) in enumerate(raw):
        a = rec["action"]
        args = a.get("args") if isinstance(a.get("args"), dict) else a.get("args_raw")
        j = after_idx[i]
        # 本条记录的 after_observation 带着实时孪生用过的标注；最后一个动作之后没有下一条记录，只有它在。
        own = Snapshot.from_observation(rec.get("after_observation"))
        events.append(Event(
            event_id=f"{run_id}:{lineno}", ts=float(rec.get("ts") or 0.0), before=snap,
            action={"name": a["name"], "args": dict(args) if isinstance(args, dict) else {}},
            result=rec["result"] if isinstance(rec.get("result"), dict) else {},
            after=own if own is not None else (raw[j][2] if j is not None else None),
            after_event_id=f"{run_id}:{raw[j][0]}" if j is not None else None))
    return events


def read_run_events(run_dir: Path) -> list[Event]:
    """读不了等于没有事件，不抛（不变式 5）。文件坏编码或目录缺失都返回空。"""
    try:
        lines = (Path(run_dir) / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    return extract_events(Path(run_dir).name, lines)


class AppTracker:
    """当前在哪个 App。⚠ 不是 screenmap.ownership()：那个只看动作参数，会把误开当开对
    （runs/20260910-193640-bfd7）。这里核不了身份就归 unknown —— 宁可不记，不可记错 App。"""

    def __init__(self) -> None:
        self.owner: str = SYSTEM
        self.display: dict[str, str] = {}      # App id → 显示名（open_app 原文 / 图标标签）
        self.entered: str | None = None
        self.entered_label_app: str | None = None

    def advance(self, ev: Event) -> None:
        self.entered = self.entered_label_app = None
        name, args, res = ev.name, ev.args, ev.result
        if name == "open_app":
            if res.get("ok") and res.get("changed"):
                ident = res.get("identity")
                if isinstance(ident, dict) and ident.get("verified") is True:
                    self._enter(str(args.get("name") or ""), ident)
                else:
                    # 没问成（verified=None）或 7322f80 之前的留档（没有 identity）
                    self.owner = UNKNOWN
            else:
                # ⚠ 2026-09-11 final review：失败/未变化的 open_app 曾经「不变」——但每条打开
                #   路线失败前都先回第一张主屏（executor.return_to_first_home_page），
                #   错 App 的路线又在错 App 里返回 ok=False：手机不可能还停在原来那个 App。
                #   ok 但没变化同理不可信。宁可不知道，不可记错（spec §2 同步）。
                self.owner = UNKNOWN
        elif name == "key":
            k = args.get("name")
            if k in ("home", "spotlight"):
                self.owner = SYSTEM
            elif k == "app_switcher":
                self.owner = UNKNOWN
            elif self.owner == SYSTEM and res.get("changed"):
                self.owner = UNKNOWN        # Spotlight 里回车会直接开最佳结果
        elif name == "handover":
            self.owner = UNKNOWN            # 人可能换了 App
        elif name == "tap":
            st = {"observation": {"elements": list(ev.before.elements)}}
            label = _icon_tap_label(st, self.owner, name, args, res)
            if label is not None:
                # ⚠ 2026-09-11（fbc1 用户决定）：点图标进 App 这条路，跟 open_app 用同一条规则——
                #   只信核过身份的入口（CLAUDE.md §7）。runs/20260907-142838-fbc1:2 错归：
                #   模型在主屏第 2 页的前帧上点了「设置」，点击那一刻手机其实已经翻回第 1 页，
                #   同一位置是 Gemini，开的是 Gemini，孪生按前帧标签记成了「设置」。前帧里没有
                #   任何结构信号能发现这件事，只有 executor 核对打开后的身份（result["identity"]）
                #   才能抓到。identity 缺失（老留档 / 没核成）、None、False 都不能进 App，归 unknown。
                ident = res.get("identity")
                if isinstance(ident, dict) and ident.get("verified") is True:
                    self._enter(label, ident)
                else:
                    self.owner = UNKNOWN
            elif self.owner == SYSTEM and res.get("changed"):
                # ⚠ 2026-09-11 闸门 B 错归（runs/20260908-022024-e257）：Spotlight 里点结果文字「记账本」
                #   （target=text）进了 App，这里原先「不变」，状态机仍当 system；之后在 App 里点底部 tab 图标
                #   （icon_above，OCR 读成「G」）被当成主屏点图标，整段记到了 App「g」。
                #   system 不是看出来的，是推出来的：系统界面上画面变了、又不是认得出的主屏点图标
                #   （点 Spotlight 结果 / 小组件 / 图标下的文字），都可能已经进了某个 App → 不知道是哪个。
                self.owner = UNKNOWN

    def mark_unknown(self) -> None:
        self.owner = UNKNOWN
        self.entered = self.entered_label_app = None

    def _enter(self, label: str, ident: dict | None = None) -> None:
        if not label.strip():
            self.owner = UNKNOWN
            return
        app = app_id_for(label)
        self.owner = app
        self.display.setdefault(app, label)
        # 核过身份的进入：下一帧的归属以它为准（spec 2026-09-12 §5.1 第 1 条），并据此学 App 名别名。
        self.entered = app
        la = (ident or {}).get("label_app")
        self.entered_label_app = la if isinstance(la, str) and la.strip() else None

    def take_entry(self) -> tuple[str | None, str | None]:
        e = (self.entered, self.entered_label_app)
        self.entered = self.entered_label_app = None
        return e
