"""孪生记账（spec §4、§5）：内存状态、逐事件记账、唯一写盘入口、重建。

实时（twin/live.py）与重放（record_run / rebuild / simulate）共用 RunReplayer ——
两边结论一致靠的就是只有这一份逻辑（codex 评审第 11 条）。
"""
from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from iphone_agent.memory.screenmap import SYSTEM, UNKNOWN, _run_finished, app_id_for
from iphone_agent.perceive.screen import parse_screen_label
from iphone_agent.twin.events import TRANSITION_ACTIONS, AppTracker, Event, read_run_events
from iphone_agent.twin.identify import Recognition, identify, norm_text, ocr_elements, ocr_texts, plain_app_id
from iphone_agent.twin.schema import RevisionConflict, read_json, write_json
from iphone_agent.twin.screenfile import (
    APP_META_FILE,
    EVIDENCE_MAX,
    AppMeta,
    Screen,
    StaleSchema,
    Transition,
    UnsupportedSchema,
    screen_id,
    utc,
)


@dataclass
class RecordStats:
    """能失败的东西要有统计（开发原则 §3）。写进 run.json["twin"]，成功失败都写。"""
    runs: int = 0
    unfinished: int = 0
    matched: int = 0
    new: int = 0
    skipped: int = 0
    unlabeled: int = 0
    label_conflict: int = 0
    candidate_unresolved: int = 0
    labeled: int = 0
    label_invalid: int = 0
    label_failed: int = 0
    screens_created: int = 0
    screens_confirmed: int = 0
    transitions: int = 0
    skipped_applied: int = 0
    collisions: int = 0
    corrupt: int = 0
    stale_schema: int = 0
    unsupported_schema: int = 0
    write_failed: int = 0
    lock_timeout: int = 0
    orphans: int = 0
    aliases_added: int = 0
    duplicate_names: int = 0
    owner_from_entry: int = 0
    owner_from_label: int = 0
    owner_from_same_as: int = 0
    owner_from_tracker: int = 0
    owner_unknown_app: int = 0
    app_aliases_added: int = 0
    app_alias_conflict: int = 0

    def count(self, state: str) -> None:
        setattr(self, state, getattr(self, state) + 1)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _quarantine(p: Path) -> None:
    """坏文件挪进 screens/.corrupt/，不删、不覆盖：里面可能有人写的东西。"""
    d = p.parent / ".corrupt"
    d.mkdir(exist_ok=True)
    dest, n = d / p.name, 1
    while dest.exists():
        dest, n = d / f"{p.stem}.{n}{p.suffix}", n + 1
    os.replace(p, dest)


class TwinState:
    """一台手机的屏，按 App 懒加载。apps_dir=None 是纯内存（report / bench / 测试）；
    writable=False（实时一侧）时 flush 什么都不做、坏文件也不挪。
    known_apps：主屏布局表里的 App id —— 标注里的 App 名只有「已知」才接受（spec §5.1 第 2 条）。"""

    def __init__(self, apps_dir: Path | None, *, writable: bool = False, stats: RecordStats | None = None,
                 known_apps: frozenset[str] = frozenset()):
        self.apps_dir = Path(apps_dir) if apps_dir is not None else None
        self.writable = writable
        self.stats = stats if stats is not None else RecordStats()
        self.known_apps = frozenset(known_apps)
        self._apps: dict[str, dict[str, Screen]] = {}
        self._meta: dict[str, AppMeta] | None = None
        # 加载时 app.json 是旧版 / 不支持 schema 的 App：它的 app.json 不读、不改、不写（spec §7「也不覆盖」）
        self._meta_blocked: set[str] = set()
        self.dirty: set[tuple[str, str]] = set()
        self.dirty_meta: set[str] = set()
        self._run: str | None = None
        self._fresh: set[str] = set()

    def screens_dir(self, app: str) -> Path:
        assert self.apps_dir is not None
        return self.apps_dir / app / "screens"

    def screens_of(self, app: str) -> list[Screen]:
        if app not in self._apps:
            self._apps[app] = self._load(app)
        return list(self._apps[app].values())

    def _classify_load(self, p: Path, parse):
        """按 schema 版本分类（spec §7）：旧版 / 不支持的跳过计数、不隔离；v2 坏了才隔离。
        返回 (解析结果或 None, 跳过的原因 None / "stale" / "unsupported" / "corrupt")。"""
        data = read_json(p)
        try:
            if data is None:
                raise ValueError("unreadable")
            return parse(data), None
        except StaleSchema:
            self.stats.stale_schema += 1
            return None, "stale"
        except UnsupportedSchema:
            self.stats.unsupported_schema += 1
            return None, "unsupported"
        except ValueError:
            self.stats.corrupt += 1
            if self.writable:
                _quarantine(p)
            return None, "corrupt"

    def _load(self, app: str) -> dict[str, Screen]:
        out: dict[str, Screen] = {}
        if self.apps_dir is None or not self.screens_dir(app).is_dir():
            return out
        for p in sorted(self.screens_dir(app).glob("s_*.json")):
            s, _ = self._classify_load(p, Screen.from_json)
            if s is not None:
                out[s.id] = s
        return out

    def _metas(self) -> dict[str, AppMeta]:
        if self._meta is None:
            self._meta = {}
            if self.apps_dir is not None and self.apps_dir.is_dir():
                for p in sorted(self.apps_dir.glob(f"*/screens/{APP_META_FILE}")):
                    m, why = self._classify_load(p, AppMeta.from_json)
                    if m is not None:
                        self._meta[m.app] = m
                    elif why in ("stale", "unsupported"):
                        # ⚠ 2026-09-12：评审 probe C——she-zhi 的 app.json 是 schema 3，加载时跳过；随后进设置
                        #   note_display 走 _meta_of 新建一份 AppMeta(revision=None)，flush 用 expect_revision=None
                        #   写盘不核对，更新版程序写的文件就被整个换掉了。App id 只能从路径取（内容读不懂）。
                        #   坏文件（corrupt）已隔离走，不在此列：它的位置可以重新长。
                        self._meta_blocked.add(p.parent.parent.name)
        return self._meta

    def _meta_is_blocked(self, app: str) -> bool:
        self._metas()
        return app in self._meta_blocked

    def _meta_of(self, app: str) -> AppMeta:
        ms = self._metas()
        if app not in ms:
            ms[app] = AppMeta(app)
        return ms[app]

    def display(self, app: str) -> str:
        m = self._metas().get(app)
        return m.display if m is not None and m.display else app

    def note_display(self, app: str, name: str) -> None:
        """显示名只在第一次见到时写，之后不变（确定性：重放顺序固定，第一次是谁就是谁）。"""
        if not name or app in (SYSTEM, UNKNOWN) or self._meta_is_blocked(app):
            return
        m = self._meta_of(app)
        if not m.display:
            m.display = name
            self.dirty_meta.add(app)

    def _alias_targets(self, key: str) -> set[str]:
        return {a for a, m in self._metas().items() if any(app_id_for(x) == key for x in m.aliases)}

    def resolve_app(self, label_app: str) -> str | None:
        """标注 App 名 → App id 的唯一入口（spec §5.2）：先 plain_app_id，再查别名表。
        同一个名字被别名指向两个 App → UNKNOWN（不随便挑一个）。"""
        base = plain_app_id(label_app)
        if base is None or base == SYSTEM:
            return base
        targets = self._alias_targets(base)
        # ⚠ 2026-09-12：slug 自己就是已知 App 时，它本身也算一个指向。评审 probe B：进备忘录（核过身份）那一帧
        #   模型把 App 标成「设置」→ 学到 设置→备忘录；只看别名表的话，之后每一帧「设置」都归备忘录，
        #   「通用」「关于本机」全记错 App，而且后来的证据改不回来（learn_alias 在 base == app 时直接返回）。
        #   两边都有证据、程序分不出谁对 → 退成 unknown 并计数（spec §5.2 冲突规则），不挑边、也不自动删别名：
        #   宁可这个名字暂时不认，不可把一个 App 的屏记到另一个 App 名下。
        if targets and self.known_app(base):
            targets = targets | {base}
        if len(targets) > 1:
            self.stats.app_alias_conflict += 1
            return UNKNOWN
        return next(iter(targets)) if targets else base

    def known_app(self, app: str) -> bool:
        if app in (SYSTEM, UNKNOWN):
            return False
        return app in self.known_apps or bool(self.screens_of(app)) or bool(self._metas().get(app, AppMeta(app)).aliases)

    def learn_alias(self, label_app: str, app: str) -> None:
        """App 名别名只在核过身份的那一刻建立（spec §5.2）；程序从不自己猜两个名字是不是同一个 App。"""
        base = plain_app_id(label_app)
        if base in (None, SYSTEM, UNKNOWN) or base == app or app in (SYSTEM, UNKNOWN):
            return
        if self.known_app(base):
            # ⚠ 2026-09-12：一个真 App 自己的名字不能成为另一个 App 的别名（评审 probe B 的劫持）。
            #   进入核过身份，但标注说的是另一个已知 App —— 两边都有证据，这里不替谁拍板，记冲突。
            self.stats.app_alias_conflict += 1
            return
        if self._meta_is_blocked(app):
            return                               # 这个 App 的 app.json 读不懂（spec §7）：不改它
        m = self._meta_of(app)
        name = label_app.strip()
        if name in m.aliases:
            return
        m.aliases.add(name)
        self.dirty_meta.add(app)
        self.stats.app_aliases_added += 1

    def get(self, app: str, sid: str | None) -> Screen | None:
        if sid is None:
            return None
        self.screens_of(app)
        return self._apps[app].get(sid)

    def all_apps(self) -> list[str]:
        apps = {a for a, ss in self._apps.items() if ss}
        if self.apps_dir is not None and self.apps_dir.is_dir():
            apps |= {p.parent.name for p in self.apps_dir.glob("*/screens")}
        return sorted(apps)

    def begin_run(self, run_id: str) -> None:
        self._run, self._fresh = run_id, set()

    def may_update(self, s: Screen) -> bool:
        """这一屏能不能记本 run 的东西。已记过本 run 的屏（上次写到一半崩了）整屏跳过。"""
        if s.id in self._fresh:
            return True
        if self._run in s.applied_runs:
            self.stats.skipped_applied += 1
            return False
        s.applied_runs.add(self._run)
        self._fresh.add(s.id)
        return True

    def create(self, app: str, key: str, name: str, ts: float) -> Screen:
        self.screens_of(app)
        salt = 0
        while True:
            sid = screen_id(app, key, salt)
            cur = self._apps[app].get(sid)
            if cur is None or cur.first_event == key:
                break
            salt += 1
            self.stats.collisions += 1
        if cur is not None:
            return cur                           # 同一帧已经建过（after 帧与下一个事件的 before 是同一帧）
        if any(name in s.names() for s in self._apps[app].values()):
            self.stats.duplicate_names += 1      # 宁可分开，不可合错（spec §4.2）；留给第三期整理
        s = Screen(id=sid, app=app, first_event=key, created=utc(ts), updated=utc(ts), name=name)
        self._apps[app][sid] = s
        # ⚠ 2026-09-12：只有真新建的才标脏。沿用已有的（上面 return cur）不能标：崩溃后重跑时那一屏
        #   已在盘上、本 run 已记过，再标脏会多写一遍、revision 比干净跑多 1（after 帧没有 same_as
        #   就总是 new，每次都会走到这里）——增量与全量重建就对不上了。
        self.dirty.add((app, sid))
        self.stats.screens_created += 1
        return s

    def visit(self, key: str, ts: float, owner: str, rec: Recognition) -> Screen | None:
        if rec.state == "matched":
            s = self.get(owner, rec.screen_id)
        elif rec.state == "new":
            s = self.create(owner, key, rec.name or "", ts)
        else:
            return None
        if s is None or not self.may_update(s):
            return s
        if rec.state == "matched" and s.visits >= 1 and s.status == "provisional":
            s.status = "confirmed"
            self.stats.screens_confirmed += 1
        s.visits += 1
        for a in rec.anchors:
            s.anchors[a] = s.anchors.get(a, 0) + 1
        if rec.alias and rec.alias not in s.aliases:
            s.aliases.add(rec.alias)
            self.stats.aliases_added += 1
        s.updated = utc(ts)
        self.dirty.add((owner, s.id))
        return s

    def add_transition(self, owner: str, s: Screen, ev: Event, target: str | None, sent: bool,
                       effect: str, to: str | None, off_track: bool) -> None:
        if not self.may_update(s):
            return
        key = (ev.name, target or "", sent, effect, to or "")
        t = s.transitions.get(key)
        if t is None:
            t = s.transitions[key] = Transition(ev.name, target, sent, effect, to)
        t.count += 1
        t.off_track += int(off_track)
        t.last = utc(ev.ts)
        t.evidence = (t.evidence + [ev.event_id])[-EVIDENCE_MAX:]
        self.stats.transitions += 1
        self.dirty.add((owner, s.id))

    def flush(self) -> None:
        """把碰过的屏和 App 元数据写盘。每个文件单独原子写（schema.write_json）；写失败只计数。"""
        if self.apps_dir is None or not self.writable:
            self.dirty.clear()
            self.dirty_meta.clear()
            return
        for app, sid in sorted(self.dirty):
            s = self._apps[app][sid]
            try:
                s.revision = write_json(self.screens_dir(app) / f"{sid}.json", s.to_json(), s.revision)
            except (RevisionConflict, OSError):
                self.stats.write_failed += 1
        for app in sorted(self.dirty_meta):
            if app in self._meta_blocked:
                continue                         # spec §7：不支持的 app.json 不覆盖
            m = self._metas()[app]
            try:
                m.revision = write_json(self.screens_dir(app) / APP_META_FILE, m.to_json(), m.revision)
            except (RevisionConflict, OSError):
                self.stats.write_failed += 1
        self.dirty.clear()
        self.dirty_meta.clear()


def target_of(ev: Event) -> str | None:
    """只存语义目标，不存坐标。⚠ type 不记内容（docs/32 §10：不存人打的字）。
    tap 用这一步自己的元素表还原文字；还原不出来、或点的是视觉元素（名字不稳定）→ None。"""
    a = ev.args
    if ev.name == "tap":
        eid = a.get("id")
        if eid is None:
            return None
        hit = next((e for e in ev.before.elements if e.get("id") == eid), None)
        if hit is None or not ocr_elements([hit]):
            return None
        return norm_text(str(hit.get("text") or "")) or None
    if ev.name in ("scroll", "scroll_until", "collect"):
        return str(a.get("direction") or "") or None
    if ev.name in ("key", "open_app"):
        return str(a.get("name") or "") or None
    return None


def off_track_of(result: dict) -> bool:
    """guard 第三态：变了但看图复核说没达到预期。留档里 judged 是摊平的 extra。"""
    j = result.get("judged")
    return isinstance(j, dict) and bool(j.get("on_change")) and j.get("worked") is False


def outcome_of(ev: Event, owner: str, before_id: str, after_rec: Recognition | None,
               after_owner: str) -> tuple[bool, str, str | None]:
    """spec §5，按顺序先中先得。"""
    sent = bool(ev.result.get("ok"))
    if not sent:
        return sent, "unknown", None
    if after_owner != owner:
        return sent, "left_app", None
    if after_rec is None or after_rec.state != "matched":
        return sent, "unknown", None
    if after_rec.screen_id != before_id:
        return sent, "navigated", after_rec.screen_id
    if ev.result.get("changed"):
        return sent, "state_changed", None
    return sent, "none", None


_OWNER_STAT = {"entry": "owner_from_entry", "label": "owner_from_label", "same_as": "owner_from_same_as",
               "tracker": "owner_from_tracker", "unknown_app": "owner_unknown_app"}


def _label_of(snap):
    """快照里的标注再过一遍唯一校验入口：留档可能被手改过，不能直接信。"""
    if snap.label is None:
        return None, snap.label_status
    return parse_screen_label({"screen": snap.label}, len(snap.candidates))


class RunReplayer:
    """一个 run 的逐事件记账：recognize（这一帧归谁、是哪屏）→ apply（记访问、推归属、记转移）。
    实时（twin/live.py）与重放共用这一份逻辑（codex 评审第 11 条）。
    ⚠ 2026-09-12：认屏不再靠几何规则（头部带 / 返回前缀 / 弱屏覆盖率），也不再用「以前最常到哪」
      的先验去消歧义（旧 _expected）——「是哪屏」「归哪个 App」只来自视觉标注，这里只核对和记账
      （spec §4.2、§5.1）。没有标注就不认（§4.3）：不认屏、不建屏、不记转移。"""

    def __init__(self, state: TwinState, run_id: str):
        self.state, self.run_id = state, run_id
        self.tracker = AppTracker()
        self._after_zoom = False
        self._after_memo: tuple[str, str, Recognition] | None = None   # (frame_file, 归属, apply 里那次的认屏)
        state.begin_run(run_id)

    def key_of(self, snap, fallback: str) -> str:
        """屏的首次出处：run:frame_file（spec §4.1，实时与重放同一个值）。老记录没有帧文件名才退回事件 id。"""
        return f"{self.run_id}:{snap.frame_file}" if snap.frame_file else fallback

    def recognize(self, snap) -> tuple[str, Recognition]:
        memo, self._after_memo = self._after_memo, None
        if self._after_zoom:
            # ⚠ zoom 之后的观察是放大的局部（元素只在那一块），当成一屏去认会长出假屏。
            self._after_zoom = False
            self.state.stats.count("skipped")
            return self.tracker.owner, Recognition("skipped")
        if memo is not None and snap.frame_file and memo[0] == snap.frame_file:
            # ⚠ 2026-09-12 终审 FR#1：after 帧在 apply 里已经认过一次（取走了核过身份的进入、计过数），
            #   它又是下一个事件的 before。再认一遍时 take_entry 已空、退到标注：评审 P5 里入口帧「文件夹」
            #   被标成「设置」→ 一帧两屏（备忘录下 0 次访问的幻影 + 设置下的「文件夹」），之后在备忘录里的
            #   点击记成 设置 … left_app（进入核过身份也照样错归）；干净的设置 run 也把 new / labeled /
            #   owner_* 记了两遍。同一帧（同一个 frame_file）只认一次：沿用 apply 那次的结论，不重推归属、
            #   不重新计数。「new」原样沿用：visit 按同一个 run:frame_file 找回 apply 建的那一屏，访问只记一次。
            #   没有帧文件名的老记录认不出「是同一帧」，照旧重认。
            _, owner, rec = memo
            self.tracker.owner = owner
            return owner, rec
        return self.classify(snap)

    def classify(self, snap) -> tuple[str, Recognition]:
        st = self.state.stats
        label, status = _label_of(snap)
        if status == "ok":
            st.labeled += 1
        elif status == "invalid":
            st.label_invalid += 1
        elif status == "failed":
            st.label_failed += 1
        owner, src = self._owner(label, snap.candidates)
        st.count(_OWNER_STAT[src])
        self.tracker.owner = owner            # 动作推断从一个看过图的起点继续（spec §5.1）
        if label is None:
            rec = Recognition("unlabeled")
        else:
            if src == "label":
                self.state.note_display(owner, label.app)
            rec = identify(label, owner, ocr_texts(snap.elements), snap.candidates, self.state)
        st.count(rec.state)
        return owner, rec

    def _owner(self, label, cands) -> tuple[str, str]:
        """spec §5.1，先中先得。"""
        entered, entered_label_app = self.tracker.take_entry()
        if entered is not None:
            shown = self.tracker.display.get(entered, entered)      # open_app 用的名字 / 图标标签
            la = label.app if label is not None else entered_label_app
            seen = self.state.resolve_app(la) if la else None
            differs = seen not in (entered, SYSTEM, UNKNOWN, None)
            # 标注的名字自己的 slug 就是进入的 App（只是别名表把它指到了别处）→ 标注与进入说的是同一个，
            #   不适用下面的规则：否则一条学错的别名（备忘录入口帧标「设置」→ 设置→备忘录，那时设置还不认识）
            #   会借这条规则把之后每一次核过身份的「进设置」都判给备忘录（test_one_mislabelled_entry…）。
            if (differs and plain_app_id(la) != entered
                    and self.state.known_app(seen) and not self.state.known_app(entered)):
                # ⚠ 2026-09-12 终审 FR#3（spec §5.1 第 1 条、§5.2「谁是正名」）：布局表里叫「App Store」，模型
                #   open_app("应用商店") 核过身份、各帧标「App Store」。按「进入的名字为准」会学 App Store →
                #   ying-yong-shang-dian，被 learn_alias「真 App 的名字不能当别名」挡住（记冲突），入口帧归
                #   ying-yong-shang-dian、后面的帧归 app-store，点「游戏」记成 left_app（假的【位置】）。
                #   进入的名字不是已知 App、标注的是 → 以标注的 App 为准，把进入的名字学成它的别名
                #   （别名的 slug 是进入的名字，不是已知 App，learn_alias 不会挡）。两个都是已知 App 且不同
                #   → 仍以进入为准、learn_alias 记冲突（备忘录入口帧标成「设置」不能劫持「设置」）。
                self.state.note_display(seen, la)
                self.state.learn_alias(shown, seen)
                return seen, "entry"
            self.state.note_display(entered, shown)
            if differs:
                self.state.learn_alias(la, entered)
            return entered, "entry"
        if label is not None:
            app = self.state.resolve_app(label.app)
            if app == SYSTEM:
                return SYSTEM, "label"
            if app is not None:
                if app == UNKNOWN or not self.state.known_app(app):
                    return UNKNOWN, "unknown_app"
                return app, "label"
            if label.same_as is not None:
                return str(cands[label.same_as - 1].get("app_id") or UNKNOWN), "same_as"
        return self.tracker.owner, "tracker"

    def apply(self, ev: Event, owner: str, rec: Recognition) -> str | None:
        s = self.state.visit(self.key_of(ev.before, ev.event_id), ev.ts, owner, rec)
        self.tracker.advance(ev)
        self._after_zoom = ev.name == "zoom"
        if s is None or ev.name not in TRANSITION_ACTIONS:
            return s.id if s is not None else None
        target = target_of(ev)
        if ev.name == "tap" and target is None:
            return s.id
        after_rec, after_owner = None, self.tracker.owner
        if ev.after is not None:
            after_owner, after_rec = self.classify(ev.after)
            if ev.after.frame_file:
                self._after_memo = (ev.after.frame_file, after_owner, after_rec)   # 见 recognize：同一帧只认一次
            if after_rec.state == "new":
                # 新屏以「它自己那一帧」为首次出处（run:frame_file）：下一个事件认它时是同一个 id。
                ns = self.state.create(after_owner, self.key_of(ev.after, ev.after_event_id or ev.event_id),
                                       after_rec.name or "", ev.ts)
                after_rec = Recognition("matched", ns.id, ns.name)
        sent, effect, to = outcome_of(ev, owner, s.id, after_rec, after_owner)
        self.state.add_transition(owner, s, ev, target, sent, effect, to, off_track_of(ev.result))
        return s.id


def _replay(state: TwinState, run_dir: Path, sink: dict | None = None) -> None:
    r = RunReplayer(state, Path(run_dir).name)
    last: Event | None = None
    for ev in read_run_events(run_dir):
        owner, rec = r.recognize(ev.before)
        sid = r.apply(ev, owner, rec)
        if sink is not None:
            sink[ev.event_id] = (owner, rec.state, sid)
        last = ev
    if last is not None and last.after is not None:
        # ⚠ 2026-09-15（spec 按需看图 §8.1）：实时在 push_obs 里对推给模型的每一帧都 recognize 一次，包括最后一个
        #   动作的 after 帧；重放只在 apply 里认 after 帧，而 apply 在归属未知（visit 返回 None）或按坐标点击
        #   （target 为 None）时提前返回 —— 平时下一个事件的 before 会补认这一帧，最后一个动作之后没有下一个
        #   事件，这一帧就只有实时认过：max_steps / 停止 / 设备出错 / 无进展结束的 run，live 与 record 的
        #   labeled 等计数差一。这里照实时的 push_obs 补认一次，同样走 recognize：apply 已认过的同一帧沿用 memo
        #   不重计，zoom 之后照旧 skipped。不 visit、不建屏。after 为 None 的末事件（done 等）实时也不再认，不补。
        r.recognize(last.after)
    state.stats.runs += 1


def replay_run(state: TwinState, run_dir: Path) -> None:
    _replay(state, Path(run_dir))


def _finish_ts(run_dir: Path) -> float:
    """结束时间 = steps.jsonl 最后一条的 ts。增量记账按任务结束的顺序发生，重建按它排序才能对上。
    ⚠ 编码坏也等于「没有」（不变式 5，同 events.read_run_events）：破损的留档不该让 simulate 炸。"""
    try:
        lines = (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return 0.0
    for line in reversed(lines):
        try:
            ts = json.loads(line).get("ts")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(ts, (int, float)):
            return float(ts)
    return 0.0


def finished_runs(runs_root: Path) -> list[Path]:
    root = Path(runs_root)
    if not root.is_dir():
        return []
    runs = [d for d in root.iterdir() if d.is_dir() and (d / "steps.jsonl").exists() and _run_finished(d)]
    return sorted(runs, key=lambda d: (_finish_ts(d), d.name))


@dataclass
class Simulation:
    state: TwinState
    stats: RecordStats
    by_event: dict[str, tuple[str, str, str | None]]     # event_id → (归属, 认屏态, 落到的屏 id)


def known_apps(ws) -> frozenset[str]:
    """主屏布局表里的 App id（spec §5.1 第 2 条的「已知」）。布局表空 / 坏 = 没有。
    增量记账在记账那一刻读布局表，重建在重建那一刻读：两者只在中间 layout.json 改过时才可能不同；
    布局不变时两次重建照旧逐字节相同。"""
    from iphone_agent.twin.layout import Layout
    lay = Layout.load_or_none(ws.twin_device / "layout.json")
    if lay is None:
        return frozenset()
    return frozenset(c.app for p in lay.pages for c in p.cells if c.app)


def simulate(ws) -> Simulation:
    """纯内存把全部留档过一遍：report 和 bench 用，不写盘。"""
    stats = RecordStats()
    state = TwinState(None, stats=stats, known_apps=known_apps(ws))
    by_event: dict[str, tuple[str, str, str | None]] = {}
    for run_dir in finished_runs(ws.runs):
        _replay(state, run_dir, by_event)
    return Simulation(state, stats, by_event)


LOCK_WAIT_S = 10.0       # 等锁上限：任务收尾不能被另一个进程的记账无限拖住
# ⚠ 2026-09-12：schema 2 升级后必须重建一次（历史留档没有标注 → App 层孪生从空开始，spec §7）。
REBUILT_MARKER = ".twin-rebuilt-v2"


class LockTimeout(Exception):
    """等孪生写锁超时。"""


def _kdir(ws) -> Path:
    return ws.dot / "knowledge"


@contextmanager
def twin_lock(ws, wait_s: float = LOCK_WAIT_S):
    """进程间写锁（fcntl.flock）。⚠ rebuild 持一次锁、内部调 _record_run_locked，
    不能重入 record_run —— 同一进程两次 open 也会互相挡住（codex 评审第 14 条）。"""
    path = _kdir(ws) / ".twin.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a")                                   # noqa: SIM115 —— 锁跟着句柄走，finally 里关
    deadline = time.monotonic() + wait_s
    try:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise LockTimeout(str(path)) from None
                time.sleep(0.05)
        yield
    finally:
        fh.close()                                         # 关句柄即释放锁


def _pending_path(ws) -> Path:
    return _kdir(ws) / "twin-pending.txt"


def _add_pending(ws, run_id: str) -> None:
    p = _pending_path(ws)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(run_id + "\n")


def _read_pending(ws) -> list[str]:
    """只读，不删——删要等这条真记完（见 record_run）：否则中途炸了就永远丢了这个 run（评审第 1 条）。"""
    p = _pending_path(ws)
    try:
        ids = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, UnicodeDecodeError):
        return []
    return list(dict.fromkeys(ids))


def _write_pending(ws, ids: list[str]) -> None:
    """待办文件整份重写成 ids；ids 为空就删掉文件（没有半个空文件的意义）。"""
    p = _pending_path(ws)
    if not ids:
        p.unlink(missing_ok=True)
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(rid + "\n" for rid in ids), encoding="utf-8")


def _record_run_locked(ws, run_dir: Path, stats: RecordStats, state: TwinState | None = None) -> None:
    if not _run_finished(run_dir):
        stats.unfinished += 1
        return
    state = state if state is not None else TwinState(ws.twin_apps, writable=True, stats=stats, known_apps=known_apps(ws))
    _replay(state, run_dir)
    state.flush()


def record_run(ws, run_dir: Path, *, lock_wait_s: float = LOCK_WAIT_S) -> RecordStats:
    """唯一写盘入口：按这个 run 的留档重放一遍写进孪生。任务结束后由 loop 调。
    等锁超时不阻塞收尾：run 进待办，下一次拿到锁时先补上。
    ⚠ 待办逐条出队：每条记完才从待办文件里划掉；中途某条炸了，后面没处理到的
    ——包括正在炸的这条——原样留在待办文件里，下次拿到锁还会补记（不能先删文件再记，那样炸一半就永远丢了）。"""
    stats = RecordStats()
    run_dir = Path(run_dir)
    try:
        with twin_lock(ws, lock_wait_s):
            pending = _read_pending(ws)
            for i, rid in enumerate(pending):
                _record_run_locked(ws, ws.runs / rid, stats)
                _write_pending(ws, pending[i + 1:])
            _record_run_locked(ws, run_dir, stats)
    except LockTimeout:
        _add_pending(ws, run_dir.name)
        stats.lock_timeout += 1
    return stats


def _collect_humans(apps_dir: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if not apps_dir.is_dir():
        return out
    for p in sorted(apps_dir.glob("*/screens/s_*.json")):
        d = read_json(p)
        if d and isinstance(d.get("human"), dict) and d["human"]:
            out[(p.parent.parent.name, p.stem)] = d["human"]
    return out


def _restore_humans(apps_tmp: Path, humans: dict, orphans_path: Path, stats: RecordStats) -> None:
    orphans: dict[str, dict] = {}
    for (app, sid), h in sorted(humans.items()):
        p = apps_tmp / app / "screens" / f"{sid}.json"
        d = read_json(p)
        if d is None:
            orphans[f"{app}/{sid}"] = h
            continue
        d["human"] = h
        write_json(p, d, d.get("revision"))
    if orphans:
        prev = read_json(orphans_path) or {}
        prev.pop("revision", None)
        write_json(orphans_path, {**prev, **orphans}, None)
        stats.orphans += len(orphans)


def _rename(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def _swap_in(apps_tmp: Path, apps_dir: Path) -> None:
    """两阶段换入 screens/，任一步失败整体回滚：中途失败旧孪生必须原样（spec 不变式，codex 评审第 1 条）——
    按 App 挨个改名不是原子的，第 N 个 App 炸时前 N-1 个已经换了，光靠 try/except 兜不住半途状态。
    阶段一：把每个 App 现在的 live screens/ 挪到 .screens-old（先留一份，不删）；
    阶段二：把每个 App 新的 screens/ 挪进 live。两阶段全成功才继续；任何一步炸，
    按相反顺序把已经换进 live 的新目录扔掉、把已经挪开的 .screens-old 挪回 live，再把异常抛出去。"""
    apps: set[str] = set()
    if apps_tmp.is_dir():
        apps |= {p.name for p in apps_tmp.iterdir() if p.is_dir()}
    if apps_dir.is_dir():
        apps |= {p.parent.name for p in apps_dir.glob("*/screens")}
    apps = sorted(apps)
    moved_old: list[str] = []
    moved_new: list[str] = []
    try:
        for app in apps:
            live, old = apps_dir / app / "screens", apps_dir / app / ".screens-old"
            shutil.rmtree(old, ignore_errors=True)
            if live.exists():
                _rename(live, old)
            moved_old.append(app)
        for app in apps:
            new, live = apps_tmp / app / "screens", apps_dir / app / "screens"
            if new.exists():
                live.parent.mkdir(parents=True, exist_ok=True)
                _rename(new, live)
            moved_new.append(app)
    except Exception:
        for app in reversed(moved_new):
            live = apps_dir / app / "screens"
            if live.exists():
                shutil.rmtree(live, ignore_errors=True)
        for app in reversed(moved_old):
            live, old = apps_dir / app / "screens", apps_dir / app / ".screens-old"
            if old.exists():
                shutil.rmtree(live, ignore_errors=True)
                _rename(old, live)
        raise
    # ⚠ 2026-09-12（终审，暂缓）：live screens/ 里不支持 schema 的 s_*.json / app.json（更新版程序写的）
    #   在这里随 .screens-old 一起删掉 —— 只有旧版程序重建新版孪生（降级）才走得到，但违背 spec §7「不覆盖」。
    #   下次升 schema 之前，要像下面的 .corrupt 一样把它们带过换入。
    for app in apps:
        live, old = apps_dir / app / "screens", apps_dir / app / ".screens-old"
        if (old / ".corrupt").is_dir():
            live.mkdir(parents=True, exist_ok=True)
            if not (live / ".corrupt").exists():
                _rename(old / ".corrupt", live / ".corrupt")
        shutil.rmtree(old, ignore_errors=True)


def rebuild(ws, *, lock_wait_s: float = LOCK_WAIT_S) -> RecordStats:
    """从全部已完成的留档重建。写到临时目录，全部成功才换入；中途失败旧孪生原样（codex 第 1 条）。"""
    stats = RecordStats()
    with twin_lock(ws, lock_wait_s):
        humans = _collect_humans(ws.twin_apps)
        tmp = _kdir(ws) / ".twin-rebuild"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            state = TwinState(tmp / "apps", writable=True, stats=stats, known_apps=known_apps(ws))
            for run_dir in finished_runs(ws.runs):
                _replay(state, run_dir)
                state.flush()                               # 每个 run 写一次：revision 与增量记账一致
            _restore_humans(tmp / "apps", humans, _kdir(ws) / "twin-orphans.json", stats)
            _swap_in(tmp / "apps", ws.twin_apps)
            (_kdir(ws) / REBUILT_MARKER).touch()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return stats


def needs_initial_rebuild(ws) -> bool:
    """第一次用孪生而已有跑完的 run：该从留档长一次。重建过就不再自动重建（之后靠增量）。"""
    if (_kdir(ws) / REBUILT_MARKER).exists():
        return False
    runs = ws.runs
    return runs.is_dir() and any(d.is_dir() and _run_finished(d) for d in runs.iterdir())
