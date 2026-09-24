"""孪生屏文件（spec §4.2）：数据模型、稳定序列化、确定性 id。只管格式，不管什么时候写。

⚠ 重建两遍要逐字节相同（docs/32 §8）：id 不能随机、时间取事件 ts、所有列表都有固定排序。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

SCHEMA = 2
APP_META_FILE = "app.json"
EVIDENCE_MAX = 5


def utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def screen_id(app: str, first_event: str, salt: int = 0) -> str:
    """64 bit：v1 的 24 bit 长期会撞（codex 评审第 6 条）。撞了由调用方加 salt 重算。"""
    key = f"{app}\x00{first_event}" + (f"\x00{salt}" if salt else "")
    return "s_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class StaleSchema(ValueError):
    """旧版本（schema 1）的文件：不是坏文件，跳过计数，不隔离（spec §7）。"""


class UnsupportedSchema(ValueError):
    """缺 schema 或比这版程序新：可能是更新的程序写的，跳过计数，绝不隔离也不覆盖（spec §7）。"""


def check_schema(d: dict) -> None:
    s = d.get("schema")
    if type(s) is int and s == SCHEMA:
        return
    if type(s) is int and s == 1:
        raise StaleSchema("schema=1")
    raise UnsupportedSchema(f"schema={s!r}")


def _rev(d: dict) -> int | None:
    rev = d.get("revision")
    return rev if isinstance(rev, int) and not isinstance(rev, bool) else None


@dataclass
class Transition:
    action: str
    target: str | None
    sent: bool
    effect: str                     # navigated | state_changed | none | left_app | unknown
    to: str | None
    count: int = 0
    off_track: int = 0              # guard 第三态（变了但没达到预期）的次数
    last: str = ""
    evidence: list[str] = field(default_factory=list)

    def key(self) -> tuple:
        return (self.action, self.target or "", self.sent, self.effect, self.to or "")

    @property
    def unstable(self) -> bool:
        return self.off_track * 2 > self.count

    def to_json(self) -> dict:
        return {"action": self.action, "target": self.target,
                "outcome": {"sent": self.sent, "effect": self.effect, "to": self.to},
                "count": self.count, "off_track": self.off_track, "last": self.last,
                "evidence": list(self.evidence)}

    @classmethod
    def from_json(cls, d: dict) -> Transition:
        o = d["outcome"]
        return cls(str(d["action"]), d.get("target"), bool(o["sent"]), str(o["effect"]), o.get("to"),
                   int(d["count"]), int(d.get("off_track", 0)), str(d.get("last", "")),
                   [str(x) for x in d.get("evidence", [])])


@dataclass
class Screen:
    """⚠ 2026-09-12：身份从 title/back/tabs/weak/texts（几何规则的产物）换成视觉标注的
    name / aliases / anchors（spec 2026-09-12 §4.2）。skeleton / bottom_seen 只服务旧认屏规则，一并删掉。"""
    id: str
    app: str
    first_event: str                              # 首次出处 run:frame_file（没有帧文件的老记录退回事件 id）
    created: str
    updated: str
    name: str
    aliases: set[str] = field(default_factory=set)
    anchors: dict[str, int] = field(default_factory=dict)   # 落地锚点 → 落地次数
    # ⚠ 2026-09-22（spec 打开App落点 §3.3）：按票存、不存「最新一次答的是什么」——模型偶尔会把一屏
    #   答错层级（比如把 top 看成 inner），单帧误判不该直接翻转一屏的身份；多帧一起投票才稳。
    levels: dict[str, int] = field(default_factory=dict)    # 标注说的层级 → 票数
    transitions: dict[tuple, Transition] = field(default_factory=dict)
    visits: int = 0
    status: str = "provisional"
    applied_runs: set[str] = field(default_factory=set)
    human: dict = field(default_factory=dict)
    revision: int | None = None

    def names(self) -> set[str]:
        return {self.name} | self.aliases

    @property
    def level(self) -> str | None:
        """票数最多的层级；平票或没票 → None（宁可不说，不说错）。
        ⚠ 2026-09-22（spec 打开App落点 §3.3、开发原则 §2）：取多数而不是最新一次，理由同 anchors ——
          代理指标（这里是「模型标注」）平时对、关键时刻错，一次误判不该盖掉之前多次一致的答案；
          票数打平说明证据本身就不足以下结论，返回 None 比随便挑一个更安全（比如 goto 拿它判「是不是首页」）。"""
        if not self.levels:
            return None
        top = max(self.levels.values())
        best = [k for k, v in self.levels.items() if v == top]
        return best[0] if len(best) == 1 else None

    def to_json(self) -> dict:
        ident = {"name": self.name, "aliases": sorted(self.aliases), "anchors": dict(sorted(self.anchors.items()))}
        # ⚠ 2026-09-22：levels 为空才不写这个键（不写 "levels": {}）——这是本任务的硬约束：老屏文件
        #   （schema 2、level 加入之前写的）和这次全量重建的结果必须逐字节相同，写一个空 dict 会破坏它。
        if self.levels:                     # 空的不写：老屏文件与重建结果逐字节不变
            ident["levels"] = dict(sorted(self.levels.items()))
        return {
            "schema": SCHEMA, "id": self.id, "app": self.app, "created": self.created, "updated": self.updated,
            "identity": ident,
            "transitions": [t.to_json() for _, t in sorted(self.transitions.items())],
            "visits": self.visits, "status": self.status, "first_event": self.first_event,
            "applied_runs": sorted(self.applied_runs), "human": self.human,
        }

    @classmethod
    def from_json(cls, d: dict) -> Screen:
        """schema 不对抛 StaleSchema / UnsupportedSchema；v2 坏格式抛普通 ValueError（调用方隔离）。"""
        check_schema(d)
        try:
            ident = d["identity"]
            name = ident["name"]
            if not isinstance(name, str) or not name:
                raise ValueError("identity.name")
            s = cls(id=str(d["id"]), app=str(d["app"]), first_event=str(d["first_event"]),
                    created=str(d["created"]), updated=str(d["updated"]), name=name,
                    aliases={str(x) for x in ident.get("aliases") or []},
                    anchors={str(k): int(v) for k, v in (ident.get("anchors") or {}).items()},
                    levels={str(k): int(v) for k, v in (ident.get("levels") or {}).items()},
                    visits=int(d["visits"]), status=str(d["status"]),
                    applied_runs={str(x) for x in d.get("applied_runs", [])},
                    human=dict(d.get("human") or {}), revision=_rev(d))
            for t in d.get("transitions", []):
                tr = Transition.from_json(t)
                s.transitions[tr.key()] = tr
            return s
        except (KeyError, TypeError, AttributeError) as e:
            raise ValueError(f"{type(e).__name__}: {e}") from e


@dataclass
class AppMeta:
    """App 层元数据：显示名、App 名别名（spec §5.2）。放在 screens/ 里，随 rebuild 的两阶段换入一起替换。"""
    app: str
    display: str = ""
    aliases: set[str] = field(default_factory=set)
    revision: int | None = None

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "app": self.app, "display": self.display, "aliases": sorted(self.aliases)}

    @classmethod
    def from_json(cls, d: dict) -> AppMeta:
        check_schema(d)
        try:
            return cls(app=str(d["app"]), display=str(d.get("display") or ""),
                       aliases={str(x) for x in d.get("aliases") or []}, revision=_rev(d))
        except (KeyError, TypeError, AttributeError) as e:
            raise ValueError(f"{type(e).__name__}: {e}") from e
