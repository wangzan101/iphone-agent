"""孪生的两道合入闸门（spec §8.1）。零模型，只读留档与标注。

A 错合：标「different」的一对被认成同一屏 —— 必须为 0（误认比认不出更糟：错的【位置】会把模型带偏）。
B 错归：事件被归到**错误的具体 App** —— 必须为 0。归成 system / unknown 允许（那一帧不记，安全）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchResult:
    pairs: int = 0
    owners: int = 0
    wrong_merge: list[tuple[str, str]] = field(default_factory=list)
    wrong_split: list[tuple[str, str]] = field(default_factory=list)
    wrong_owner: list[tuple[str, str, str]] = field(default_factory=list)
    unknown_owner: int = 0
    unresolved: list[str] = field(default_factory=list)
    # ⚠ 2026-09-11 final review：闸门原先只看 wrong_merge / wrong_owner，没跑完（unresolved）
    #   和「全归 system/unknown」（零覆盖）都能悄悄混进「通过」——闸门形同虚设。
    #   concrete_owner：owner 事件里归到具体 App（非 system/unknown）的个数；
    #   resolved_different：「different」标注的两个事件都落到了具体屏 id 的对数。
    concrete_owner: int = 0
    resolved_different: int = 0

    @property
    def passed(self) -> bool:
        return (not self.wrong_merge and not self.wrong_owner and not self.unresolved
                and self.concrete_owner > 0 and self.resolved_different > 0)


def load_labels(d: Path) -> tuple[list[dict], list[dict]]:
    def read(name: str) -> list[dict]:
        p = Path(d) / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    return read("pairs.json"), read("owners.json")


def run_bench(sim, pairs: list[dict], owners: list[dict]) -> BenchResult:
    r = BenchResult(pairs=len(pairs), owners=len(owners))
    for p in pairs:
        a, b = sim.by_event.get(p["a"]), sim.by_event.get(p["b"])
        missing = [k for k, v in ((p["a"], a), (p["b"], b)) if v is None]
        if missing:
            r.unresolved.extend(missing)
            continue
        same = a[2] is not None and a[2] == b[2]
        if p["label"] == "different" and same:
            r.wrong_merge.append((p["a"], p["b"]))
        elif p["label"] == "same" and not same:
            r.wrong_split.append((p["a"], p["b"]))
        if p["label"] == "different" and a[2] is not None and b[2] is not None:
            r.resolved_different += 1
    for o in owners:
        got = sim.by_event.get(o["event"])
        if got is None:
            r.unresolved.append(o["event"])
            continue
        owner = got[0]
        if owner == "unknown":
            r.unknown_owner += 1
        else:
            if owner != o["app"] and owner != "system":
                r.wrong_owner.append((o["event"], owner, o["app"]))
            if owner != "system":
                r.concrete_owner += 1
    return r


def format_result(r: BenchResult) -> str:
    lines = [f"闸门 A 错合：{len(r.wrong_merge)} / {r.pairs} 对（必须 0）；错分 {len(r.wrong_split)}（只报告）",
             f"闸门 B 错归：{len(r.wrong_owner)} / {r.owners} 个事件（必须 0）；归 unknown {r.unknown_owner}",
             f"覆盖：归到具体 App {r.concrete_owner} 个事件；「不同」且两边都落到具体屏 {r.resolved_different} 对"]
    lines += [f"  错合 {a} ≈ {b}" for a, b in r.wrong_merge]
    lines += [f"  错归 {e}：认成 {got}，应为 {want}" for e, got, want in r.wrong_owner]
    lines += [f"  错分 {a} / {b}" for a, b in r.wrong_split]
    if r.unresolved:
        lines.append(f"  找不到的事件（留档没有或没跑完）：{', '.join(sorted(set(r.unresolved)))}")
    if not r.passed and not r.wrong_merge and not r.wrong_owner:
        if r.unresolved:
            lines.append("覆盖为 0：闸门没有被真正检验（有事件没跑完/找不到）")
        elif r.concrete_owner == 0:
            lines.append("覆盖为 0：闸门没有被真正检验（没有一个事件归到具体 App）")
        elif r.resolved_different == 0:
            lines.append("覆盖为 0：闸门没有被真正检验（没有一对「不同」标注两边都落到具体屏）")
    lines.append("通过" if r.passed else "未通过")
    return "\n".join(lines)
