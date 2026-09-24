"""屏幕上的 App 名 → App id。**全项目只有这一个入口**（CLAUDE.md §7 一个规则一个入口）。

## 为什么要有这个文件

2026-09-17 之前，App id 直接是显示名的拼音（`slugify`）：中文机上 设置 → `she-zhi`，
健康 → `jian-kang`。但同一台手机换成英文，这两个 App 在屏幕上叫 Settings / Health，
id 就变成了 `settings` / `health` —— **仓库里那份共享知识只对中文机有效**。
对一个「装上就能用」的开源产品来说这是事实错误，不是小瑕疵：英文机用户的知识目录
永远是空的，第一天什么都借不到。

规矩定成这样：

1. **苹果系统 App 的 id 是英文 slug**（`CANONICAL_APP_IDS`）。这是一份人定的表，
   跟手机语言无关 —— 它就是这些 App 在全世界的同一个名字。
2. **第三方 App 不发明规范 id**，照旧走 `slugify(显示名)`。我们不认识它们，
   替它们编一个英文名只会编错（「记账本」不叫 jizhang notebook）。
3. **别名写在各自的 `knowledge/apps/<id>/APP.md` 的 frontmatter `aliases:` 里**，
   一个 App 的所有叫法（中文显示名、英文显示名、它曾经用过的拼音 id）都在那一处。
   要支持一种新语言，就往那个 App 的 `aliases:` 里加一行它在那种语言下的显示名 ——
   **不改代码**，也不必改这个文件。（仓库里只填了写得出证据的那几种：简中、英文、
   以及 2026-09-17 之前的旧拼音 id。没人能核实的语言不猜。）

## 解析顺序

已知 id 命中 → 别名索引 → `slugify` 兜底 → slug 再查一次索引（繁体、异体字走这条：
「設置」不是别名，但拼音化之后是 `she-zhi`，而那是 settings 的旧 id）。

⚠ 别名索引**不许自己拍板**：同一个名字被两个 App 认领时抛 `AppIdConflict`，不挑一个
（CLAUDE.md §3 失败必须能被看见）。挑错了的后果是一整个 App 的屏被记到另一个 App 名下，
而且没有任何一行日志会说这件事发生过 —— `twin/record.resolve_app` 为同一个坑退成 unknown 计数过一次。

⚠ 旧 id 永远留在别名里。用户不迁移也不能坏：盘上写着 `she-zhi` 的老数据，读的时候
照样解析回 `settings`（迁移命令见 `iphone_agent/appmigrate.py`）。
"""
from __future__ import annotations

import re as _re
from collections.abc import Iterable, Mapping
from pathlib import Path

from pypinyin import lazy_pinyin

from iphone_agent import config
from iphone_agent.skills import frontmatter

# 苹果系统 App 的规范 id。**只列我们真的有知识、也真的认得出的那些**：
# 往里加一个 id 之前，先有一份 knowledge/apps/<id>/APP.md 和它的别名，否则加了也没人用。
CANONICAL_APP_IDS = frozenset({
    "settings", "health", "calendar", "clock", "photos", "weather",
    "notes", "reminders", "maps", "files", "calculator",
    "app-store", "shortcuts",
})


class AppIdError(ValueError):
    pass


class AppIdConflict(AppIdError):
    """同一个名字被两个 App 认领。程序不挑边（CLAUDE.md §2/§3）。"""


def slugify(text: str, max_len: int = 48) -> str:
    """任意文字 → 合法 id 片段：中文转拼音，只留 [a-z0-9-]，合并连字符。空则 'x'。

    ⚠ 这是**兜底**，不是 App id 的定义。要 App id 请用 `resolve_app_id` ——
    直接调它就绕过了别名，中文机和英文机又会分叉成两套 id。
    （skills/model.auto_name 拿它生成技能名，那是另一回事，不涉及别名。）
    """
    parts = lazy_pinyin(text) if not text.isascii() else [text]
    s = "-".join(parts).lower()
    s = _re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    s = _re.sub(r"-{2,}", "-", s)[:max_len].strip("-")
    if not s or not s[0].isalnum():
        s = "x" + s
    return s


def normalize(name: str) -> str:
    """比别名时的唯一归一化：去首尾空白、内部连续空白压成一个、大小写不敏感。"""
    return " ".join((name or "").split()).casefold()


class AliasIndex:
    """别名 → App id。由知识目录（或任意 id → 别名表）建出来，程序不内置第二份。"""

    def __init__(self, ids: frozenset[str], alias_to_id: Mapping[str, str],
                 conflicts: Mapping[str, tuple[str, ...]]):
        self.ids = ids
        self.alias_to_id = dict(alias_to_id)
        self.conflicts = dict(conflicts)

    @classmethod
    def build(cls, apps: Mapping[str, Iterable[str]]) -> AliasIndex:
        """apps: App id → 它的别名。规范 id 无条件在册，哪怕这台机器上还没有它的知识文件。"""
        claims: dict[str, list[str]] = {}
        for app_id in sorted(apps):
            for alias in apps[app_id] or ():
                key = normalize(str(alias))
                if not key or key == normalize(app_id):
                    continue
                owners = claims.setdefault(key, [])
                if app_id not in owners:
                    owners.append(app_id)

        # ⚠ 旧 id 的目录还在盘上时（用户没迁移），它自己也是一个「已知 id」——若不折叠，
        #   `she-zhi` 就会解析成 `she-zhi`，而它的显示名「设置」又会同时被 settings 和
        #   she-zhi 认领，整个「设置」这个名字当场变成冲突、谁都开不了。所以：
        #   **非规范 id 被某个 App 认领为别名 = 它是那个 App 的旧 id**，折叠过去。
        #   规范 id 永不折叠 —— 一个真 App 的 id 不能被别人的别名劫持
        #   （twin/record.py 那条 probe B 的同一条规矩）。
        fold: dict[str, str] = {}
        for x in sorted(set(apps) | CANONICAL_APP_IDS):
            if x in CANONICAL_APP_IDS:
                continue
            owners = [o for o in claims.get(normalize(x), []) if o != x]
            if len(owners) == 1:
                fold[x] = owners[0]

        def final(a: str) -> str:
            seen = set()
            while a in fold and a not in seen:
                seen.add(a)
                a = fold[a]
            return a

        ids = frozenset(x for x in (set(apps) | CANONICAL_APP_IDS) if x not in fold)
        known = {normalize(i) for i in ids}
        alias_to_id, conflicts = {}, {}
        for key, owners in claims.items():
            if key in known:
                continue
            eff = sorted({final(o) for o in owners})
            if len(eff) > 1:
                conflicts[key] = tuple(eff)
            else:
                alias_to_id[key] = eff[0]
        return cls(ids, alias_to_id, conflicts)

    @classmethod
    def from_catalog(cls, catalog) -> AliasIndex:
        """`skills.store.Catalog` → 索引。App 自己的 display / open 也算别名：
        人手写 APP.md 时常常只填 display，忘了 aliases，那份显示名照样该认得出来。

        ⚠ 随包走的那份共享知识**永远一起算**：catalog 只反映这个工作区看得见的文件，
        而规范 id 和旧 id 的对照表是产品的一部分，不该因为某个工作区还没有知识目录就消失。"""
        apps = _apps_from_roots([_SHARED_APPS])
        for app_id, profile in getattr(catalog, "apps", {}).items():
            names = [*getattr(profile, "aliases", ()), getattr(profile, "display", ""),
                     getattr(profile, "open", "")]
            apps.setdefault(app_id, [])
            apps[app_id] += [n for n in names if n]
        return cls.build(apps)

    def resolve(self, display_or_id: str) -> str:
        raw = (display_or_id or "").strip()
        key = normalize(raw)
        if key in self.ids:
            return key
        hit = self._alias(key)
        if hit is not None:
            return hit
        s = slugify(raw)
        if s in self.ids:
            return s
        return self._alias(s) or s

    def _alias(self, key: str) -> str | None:
        if key in self.conflicts:
            owners = self.conflicts[key]
            raise AppIdConflict(
                f"别名 {key!r} 同时指向 {', '.join(owners)}：改掉其中一份 APP.md 的 aliases，"
                f"程序不替你挑一个")
        return self.alias_to_id.get(key)


# ---- 默认索引：从盘上的知识目录读 ----
#
# 共享层（仓库里那份 knowledge/）按**包的位置**找，不按 cwd：cwd 在测试里、在别人的
# 工作目录里都可能不是仓库根，而共享知识是随源码一起走的东西，跟着包走才稳。
# 个人层按 cwd（config.KNOWLEDGE_DIR）找 —— 它本来就是「这个工作区的」。
_SHARED_APPS = Path(__file__).resolve().parent.parent / "knowledge" / "apps"

# 缓存的键是两个目录里每份 APP.md 的 (路径, mtime)：加一个 App、改一行 aliases 都会变。
# 不做时间兜底的缓存 —— 改了文件却不生效，就是 CLAUDE.md §3 说的那种看不见的失败。
_cache: tuple[tuple, AliasIndex] | None = None


def _fingerprint(roots: list[Path]) -> tuple:
    out = []
    for root in roots:
        try:
            entries = sorted(root.iterdir())
        except OSError:
            out.append((str(root), None))
            continue
        for d in entries:
            f = d / "APP.md"
            try:
                out.append((str(f), f.stat().st_mtime_ns))
            except OSError:
                continue
    return tuple(out)


def _aliases_of(path: Path) -> list[str]:
    """只读 frontmatter 的 aliases / display / open。坏文件不抛：解析 App id 这条路
    挡在每一次 open_app 前面，一份写坏的 APP.md 不该让整台手机停摆（它坏在哪由
    `skills.store.load` 报 broken）。"""
    try:
        meta, _ = frontmatter.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, frontmatter.FrontmatterError):
        return []
    names = meta.get("aliases")
    out = [str(x) for x in names] if isinstance(names, list) else []
    for k in ("display", "open"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v)
    return out


def _apps_from_roots(roots: list[Path]) -> dict[str, list[str]]:
    apps: dict[str, list[str]] = {}
    for root in roots:
        try:
            entries = sorted(root.iterdir())
        except OSError:
            continue
        for d in entries:
            if not (d / "APP.md").is_file():
                continue
            apps.setdefault(d.name, [])
            apps[d.name] += _aliases_of(d / "APP.md")
    return apps


def default_index() -> AliasIndex:
    roots = [_SHARED_APPS, config.KNOWLEDGE_DIR / "apps"]
    key = _fingerprint(roots)
    global _cache
    if _cache is not None and _cache[0] == key:
        return _cache[1]
    idx = AliasIndex.build(_apps_from_roots(roots))
    _cache = (key, idx)
    return idx


def resolve_app_id(display_or_id: str, catalog=None) -> str:
    """显示名或旧 id → 规范 App id。给了 catalog 就用它的别名，否则读盘上的知识目录。"""
    idx = AliasIndex.from_catalog(catalog) if catalog is not None else default_index()
    return idx.resolve(display_or_id)


def app_id_for(open_name: str, catalog=None) -> str:
    """`resolve_app_id` 的旧名。open_app 的参数、主屏图标标签、标注里的 App 名都走它。"""
    return resolve_app_id(open_name, catalog)
