"""把个人层里 2026-09-17 之前那套拼音 App id 迁到规范 id（`iphone skill migrate-ids`）。

为什么这件事要有一个命令、而不是「加载时自动改名」：改名要搬目录、要合并两边的
访问计数和转移、要改屏 id —— 这是一次有后果的写操作，得先让人看见它要干什么
（CLAUDE.md §3、§6）。所以：

- **默认只打印**，`--apply` 才动手；
- 动手之前把 `knowledge/apps/` 和 `skills/` 整棵拷一份到 `backup-migrate-ids-<时间戳>/`；
- 目的地已经有内容就**拒绝**，除非明确说了 `--merge`；合并是把两边的
  visits / applied_runs / anchors / transitions 加起来，不是拿一边盖另一边。

⚠ 屏 id 是 `sha256(app + first_event)` 算出来的（`twin/screenfile.screen_id`），App id 一改，
  同一屏的 id 就变了。所以迁移必须**重算 id、改文件名、并把所有转移的目的地 `to` 一起改掉** ——
  只改目录名的话，下一次 `iphone twin rebuild` 会按新 id 重新建一遍屏，旧文件全成孤儿，
  而且没有一行日志会说这件事（同一类静默失败）。`to` 的重写要扫**所有** App 的屏，
  不只是改了名的那几个：第三方 App 的屏也可能有一条指向 `she-zhi` 某一屏的转移。

⚠ 剧本工具名 `app__proc` 不落盘：它由 `skills/tools.tool_name(p.app, p.name)` 在运行时算，
  而 `p.app` 来自 SKILL.md frontmatter 的 `apps[0]`。所以这里只改 `apps:` 那一行，工具名自动跟着变。
  `runs/` 里的旧记录一个字不改（它们是历史，不是状态），读的时候由解析器认出旧 id。
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from iphone_agent.appids import resolve_app_id
from iphone_agent.skills import frontmatter
from iphone_agent.twin.screenfile import EVIDENCE_MAX, screen_id

BACKUP_PREFIX = "backup-migrate-ids-"


class MigrateRefused(RuntimeError):
    """目的地已经有内容，而没人说可以合并。"""


@dataclass
class Move:
    old_id: str
    new_id: str
    src: Path
    dst: Path
    screens: int = 0
    merging: list[str] = field(default_factory=list)      # 两边都有、要合起来的屏 id（新 id）


@dataclass
class SkillEdit:
    path: Path
    old_apps: list[str]
    new_apps: list[str]


@dataclass
class Plan:
    apps_dir: Path
    skills_dir: Path
    moves: list[Move] = field(default_factory=list)
    skills: list[SkillEdit] = field(default_factory=list)
    needs_merge: list[str] = field(default_factory=list)   # 目的地已有内容的新 id


@dataclass
class Report:
    backup: Path
    moved: int
    merged_screens: int
    skills: int


def _read(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return d if isinstance(d, dict) else None


def _screen_files(app_dir: Path) -> list[Path]:
    return sorted((app_dir / "screens").glob("s_*.json"))


def _has_content(app_dir: Path) -> bool:
    return bool(_screen_files(app_dir)) or (app_dir / "APP.md").exists()


def plan(apps_dir: Path, skills_dir: Path) -> Plan:
    """看一遍要动什么。**不写任何东西。**"""
    apps_dir, skills_dir = Path(apps_dir), Path(skills_dir)
    p = Plan(apps_dir, skills_dir)
    renames: dict[str, str] = {}
    try:
        entries = sorted(d for d in apps_dir.iterdir() if d.is_dir())
    except OSError:
        entries = []
    for d in entries:
        new_id = resolve_app_id(d.name)
        if new_id == d.name:
            continue
        renames[d.name] = new_id
        dst = apps_dir / new_id
        mv = Move(d.name, new_id, d, dst, screens=len(_screen_files(d)))
        if dst.is_dir() and _has_content(dst):
            p.needs_merge.append(new_id)
            here = {f.stem for f in _screen_files(dst)}
            for f in _screen_files(d):
                s = _read(f) or {}
                sid = screen_id(new_id, str(s.get("first_event") or f.stem))
                if sid in here:
                    mv.merging.append(sid)
        p.moves.append(mv)

    try:
        skill_dirs = sorted(x for x in skills_dir.iterdir() if x.is_dir() and not x.name.startswith(("_", ".")))
    except OSError:
        skill_dirs = []
    for d in skill_dirs:
        f = d / "SKILL.md"
        try:
            meta, _ = frontmatter.parse(f.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, frontmatter.FrontmatterError):
            continue
        old = [str(x) for x in (meta.get("apps") or []) if isinstance(x, str)]
        new = [renames.get(x, x) for x in old]
        if new != old:
            p.skills.append(SkillEdit(f, old, new))
    return p


def format_plan(p: Plan) -> str:
    if not p.moves and not p.skills:
        return "没有要迁的东西：个人层里的 App id 已经都是规范 id。"
    lines = []
    for mv in p.moves:
        how = "合并进" if mv.merging or mv.new_id in p.needs_merge else "改名为"
        lines.append(f"  {mv.old_id} {how} {mv.new_id}（{mv.screens} 个屏文件）")
        if mv.merging:
            lines.append(f"    两边都有、要合起来的屏：{len(mv.merging)} 个"
                         f"（visits / applied_runs / anchors / 转移相加，不覆盖）")
    for s in p.skills:
        lines.append(f"  {s.path.parent.name}/SKILL.md 的 apps：{s.old_apps} → {s.new_apps}"
                     f"（剧本工具名 app__proc 跟着变）")
    head = [f"要迁 {len(p.moves)} 个 App、{len(p.skills)} 个技能："]
    if p.needs_merge:
        head.append(f"⚠ 这些目的地已经有内容：{'、'.join(p.needs_merge)} —— 要合并请加 --merge。")
    head.append("⚠ 屏 id 会按新 App id 重算，转移的目的地一起改；runs/ 一个字不动。")
    return "\n".join([*head, *lines, "这是预演，什么都没写。真做请加 --apply。"])


# ---- 合并 ----

def _merge_transitions(dst: list[dict], src: list[dict]) -> list[dict]:
    def key(t):
        o = t.get("outcome") or {}
        return (t.get("action"), t.get("target") or "", bool(o.get("sent")),
                str(o.get("effect")), o.get("to") or "")
    out = {key(t): json.loads(json.dumps(t)) for t in dst}
    for t in src:
        k = key(t)
        if k not in out:
            out[k] = json.loads(json.dumps(t))
            continue
        cur = out[k]
        cur["count"] = int(cur.get("count", 0)) + int(t.get("count", 0))
        cur["off_track"] = int(cur.get("off_track", 0)) + int(t.get("off_track", 0))
        cur["last"] = max(str(cur.get("last", "")), str(t.get("last", "")))
        ev = list(dict.fromkeys([*(cur.get("evidence") or []), *(t.get("evidence") or [])]))
        cur["evidence"] = ev[:EVIDENCE_MAX]
    return [out[k] for k in sorted(out)]


def _merge_screen(dst: dict, src: dict) -> dict:
    """两边都有的同一屏。**加起来**，不覆盖 —— 覆盖等于把一边的历史悄悄扔掉。"""
    out = json.loads(json.dumps(dst))
    out["visits"] = int(dst.get("visits", 0)) + int(src.get("visits", 0))
    out["applied_runs"] = sorted({*(dst.get("applied_runs") or []), *(src.get("applied_runs") or [])})
    di, si = dst.get("identity") or {}, src.get("identity") or {}
    anchors = dict(di.get("anchors") or {})
    for k, v in (si.get("anchors") or {}).items():
        anchors[k] = anchors.get(k, 0) + int(v)
    names = {*(di.get("aliases") or []), *(si.get("aliases") or []), str(si.get("name") or "")}
    names.discard(str(di.get("name") or ""))
    names.discard("")
    out["identity"] = {"name": di.get("name") or si.get("name"),
                       "aliases": sorted(names), "anchors": dict(sorted(anchors.items()))}
    out["transitions"] = _merge_transitions(dst.get("transitions") or [], src.get("transitions") or [])
    out["created"] = min(str(dst.get("created") or ""), str(src.get("created") or ""))
    out["updated"] = max(str(dst.get("updated") or ""), str(src.get("updated") or ""))
    if "confirmed" in (dst.get("status"), src.get("status")):
        out["status"] = "confirmed"
    # 人写的东西以目的地为准，源那边缺的补上：谁都不删
    out["human"] = {**(src.get("human") or {}), **(dst.get("human") or {})}
    return out


def _merge_meta(dst: dict, src: dict, old_id: str, new_id: str) -> dict:
    aliases = {*(dst.get("aliases") or []), *(src.get("aliases") or []), old_id}
    display = dst.get("display") or src.get("display") or ""
    return {"schema": dst.get("schema") or src.get("schema") or 2, "app": new_id,
            "display": display, "aliases": sorted(a for a in aliases if a)}


# ---- 真做 ----

def apply(p: Plan, *, merge: bool = False) -> Report:
    if p.needs_merge and not merge:
        raise MigrateRefused(
            f"目的地已经有内容：{'、'.join(p.needs_merge)}。想把两边合起来请加 --merge；"
            f"不想合就先自己把那几个目录挪开。")
    root = p.apps_dir.parent.parent            # .iphone/
    backup = root / (BACKUP_PREFIX + datetime.now().strftime("%Y%m%d-%H%M%S"))
    backup.mkdir(parents=True, exist_ok=False)
    if p.apps_dir.is_dir():
        shutil.copytree(p.apps_dir, backup / "apps")
    if p.skills_dir.is_dir():
        shutil.copytree(p.skills_dir, backup / "skills")

    # 1. 先把「旧屏 id → 新屏 id」的全表算出来：转移的 to 可能跨 App，必须一次算齐，
    #    而且要在**合并之前**就换成新 id —— 否则同一条转移会以旧 to 和新 to 两副面孔
    #    并排躺在合并结果里，count 也就永远合不到一起。
    id_map: dict[str, str] = {}
    staged: dict[str, list[tuple[str, dict]]] = {}          # 新 App id → [(新屏 id, 内容)]
    for mv in p.moves:
        out = []
        for f in _screen_files(mv.src):
            d = _read(f)
            if d is None:
                continue                                    # 坏文件留在备份里，不往新目录搬
            key = str(d.get("first_event") or f.stem)
            salt = 0
            sid = screen_id(mv.new_id, key, salt)
            while True:
                there = _read(mv.dst / "screens" / f"{sid}.json")
                # 目的地已有同一屏（同 first_event）→ 要合并的那种「撞」，不加盐；
                # 真撞上别人（不同 first_event 同 hash）才加盐，规则同 TwinState.create。
                if there is None or str(there.get("first_event") or "") == key:
                    break
                salt += 1
                sid = screen_id(mv.new_id, key, salt)
            d["app"], d["id"] = mv.new_id, sid
            id_map[str(f.stem)] = sid
            out.append((sid, d))
        staged[mv.new_id] = out

    # 2. 转移的目的地：盘上的（所有 App，不只是改了名的那几个）和刚搬进来的，都换成新屏 id
    _rewrite_transition_targets(p.apps_dir, id_map)
    for out in staged.values():
        for _, d in out:
            _remap(d, id_map)

    merged = 0
    for mv in p.moves:
        (mv.dst / "screens").mkdir(parents=True, exist_ok=True)
        for sid, d in staged[mv.new_id]:
            target = mv.dst / "screens" / f"{sid}.json"
            old = _read(target) if target.exists() else None
            if old is not None:
                d = _merge_screen(old, d)
                merged += 1
            target.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        # app.json
        src_meta = _read(mv.src / "screens" / "app.json") or {}
        dst_meta = _read(mv.dst / "screens" / "app.json") or {}
        if src_meta or dst_meta:
            (mv.dst / "screens" / "app.json").write_text(
                json.dumps(_merge_meta(dst_meta, src_meta, mv.old_id, mv.new_id),
                           ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        _move_side_files(mv)
        shutil.rmtree(mv.src, ignore_errors=True)

    # 3. 技能的 apps:（剧本工具名 app__proc 由它算出来）
    for s in p.skills:
        text = s.path.read_text(encoding="utf-8")
        meta, body = frontmatter.parse(text)
        meta["apps"] = s.new_apps
        s.path.write_text(frontmatter.dump(meta, body), encoding="utf-8")

    return Report(backup=backup, moved=len(p.moves), merged_screens=merged, skills=len(p.skills))


def _move_side_files(mv: Move) -> None:
    """APP.md / map.json / NOTES-proposed.md 这些跟着走；已有的不覆盖（人写的东西不删）。"""
    for name in ("APP.md", "map.json", "NOTES-proposed.md"):
        src, dst = mv.src / name, mv.dst / name
        if not src.exists():
            continue
        if dst.exists():
            continue
        if name == "APP.md":
            dst.write_text(_rewrite_app_md(src.read_text(encoding="utf-8"), mv.old_id, mv.new_id),
                           encoding="utf-8")
        elif name == "map.json":
            d = _read(src)
            if d is None:
                continue
            d["app"] = mv.new_id
            dst.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        else:
            shutil.copy2(src, dst)


def _rewrite_app_md(text: str, old_id: str, new_id: str) -> str:
    """name: 跟目录名走（加载时目录名是权威，得让这句话继续为真）；旧 id 进 aliases。"""
    try:
        meta, body = frontmatter.parse(text)
    except frontmatter.FrontmatterError:
        return text
    meta["name"] = new_id
    aliases = [str(x) for x in (meta.get("aliases") or []) if isinstance(x, str)]
    if old_id not in aliases:
        aliases.append(old_id)
    meta["aliases"] = aliases
    return frontmatter.dump(meta, body)


def _remap(d: dict, id_map: dict[str, str]) -> bool:
    changed = False
    for t in d.get("transitions") or []:
        o = t.get("outcome") or {}
        to = o.get("to")
        if isinstance(to, str) and id_map.get(to, to) != to:
            o["to"] = id_map[to]
            changed = True
    return changed


def _rewrite_transition_targets(apps_dir: Path, id_map: dict[str, str]) -> None:
    if not id_map:
        return
    for f in sorted(apps_dir.glob("*/screens/s_*.json")):
        d = _read(f)
        if d is not None and _remap(d, id_map):
            f.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
