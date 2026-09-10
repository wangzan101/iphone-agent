"""记忆的落盘与校验。

⚠ 校验一律**拒绝**不合规输入，从不规整。规整会让两个不同的名字撞成同一个文件、
静默覆盖别人的记忆；拒绝了模型自己会重起一个（spec §3.2）。
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from iphone_agent import config
from iphone_agent.memory.similarity import score as _bigram_score


class MemoryRejected(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# 敏感内容拦截：验证码/手机号/密码这类东西一旦写进记忆文件，就会在每次 recall 时
# 原样回喂进模型上下文——记忆不是聊天记录，没有「过期」这个概念，一旦写进去
# 就会被反复召回，比一次性泄露更糟。
# ⚠ 拒绝信息里不给"换个说法"之类的提示：验证码/密码这类内容换个说法还是同一件事，
#   给这种提示等于是在教对方怎么绕过拦截。
_SENSITIVE = [
    re.compile(r"验证码.{0,10}?\d{4,8}(?!\s*年)|\d{4,8}(?!\s*年).{0,10}?验证码"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(密码|password|passcode|口令)\s*[:：=]\s*\S+", re.I),
]


@dataclass(frozen=True)
class MemoryEntry:
    name: str
    description: str
    source: str
    source_outcome: str
    created: str
    # 以下四个是可选字段，旧文件里没有——缺省值就是「知识类、从没被用过」。
    # kind 区分「陈述性知识」和「操作步骤」，将来 recap/search 可能按 kind 分开展示；
    # used_success/used_failed/last_used 是使用反馈，为后续按「有没有用」筛选记忆铺路
    # （本任务只落盘这几个字段，不消费它们——消费逻辑是别的任务）。
    kind: str = "knowledge"
    used_success: int = 0
    used_failed: int = 0
    last_used: str = ""


class MemoryStore:
    def __init__(self, root: Path | None = None):
        # root 为 None 时落到默认工作区（env IPHONE_WORKSPACE，否则 cwd）——
        # 位置和原来的 config.MEMORY_DIR 一样，只是现在能整块换掉（设计说明 D3）。
        if root is None:
            from iphone_agent.workspace import Workspace
            root = Workspace.default().memory_dir
        self.root = Path(root)
        self.trash = self.root / "trash"

    def validate(self, name: str, description: str, content: str) -> None:
        if not isinstance(name, str) or not config.MEMORY_NAME_RE.match(name):
            raise MemoryRejected(
                "invalid_name",
                f"name 必须匹配 {config.MEMORY_NAME_RE.pattern}（小写字母数字和连字符，"
                f"1–48 字符，字母数字开头），收到 {name!r}")
        if name in config.MEMORY_RESERVED_NAMES:
            raise MemoryRejected("invalid_name",
                                 f"{name!r} 是保留名，换一个")

        if not isinstance(description, str) or not description.strip():
            raise MemoryRejected("invalid_description", "description 不能为空")
        if len(description) > config.MEMORY_DESC_MAX:
            raise MemoryRejected(
                "invalid_description",
                f"description 最长 {config.MEMORY_DESC_MAX} 字符，收到 {len(description)}")
        # 换行能在 frontmatter 里伪造 source / created 字段；
        # 以 - 开头能在索引里伪造出一条不存在的记忆；--- 能提前闭合 frontmatter。
        if any(c in description for c in "\n\r"):
            raise MemoryRejected("invalid_description", "description 必须是一行，不能含换行")
        if "---" in description:
            raise MemoryRejected("invalid_description", "description 不能含 ---")
        if description.lstrip()[:1] in ("-", ":"):
            raise MemoryRejected("invalid_description", "description 不能以 - 或 : 开头")

        if not isinstance(content, str):
            raise MemoryRejected("content_too_long", "content 必须是字符串")
        if len(content) > config.MEMORY_BODY_MAX:
            raise MemoryRejected(
                "content_too_long",
                f"正文最长 {config.MEMORY_BODY_MAX} 字符（再长会被消息滑窗截掉，"
                f"recall 就读不全了），收到 {len(content)}")

        # content 和 description 都要查——description 也会被索引展示、被反复召回，
        # 不能因为它短就当作安全区。
        for pat in _SENSITIVE:
            if pat.search(content) or pat.search(description):
                raise MemoryRejected(
                    "sensitive_content", "记忆里不能有验证码、手机号、密码这类内容")

    # 名字里绝不该出现的东西。出现了就说明对方是在**拼路径**，不是在起名字。
    # 反斜杠也在内：POSIX 上 `..\private` 只是个奇怪的文件名（containment 拦不住它），
    # 但它换到 Windows 就是穿越，而且没有任何正当理由出现在一条记忆的名字里。
    _NAME_FORBIDDEN = ("/", "\\", "\0")

    def _path(self, name: str) -> Path:
        """名字 → 文件路径。**这里必须挡住越权**。

        ⚠ 写入路径靠 validate() 挡住了非法名字，但读取路径（recall）以前直接拼，
        于是 recall("../private") 能读到记忆目录之外的 .md，内容进模型上下文，
        而模型有 type 工具能把它打进任意 App（2026-09-07 最终 review 实测复现）。
        同一件事两处用不同判据 —— 这是本分支第三次踩这个形状。

        ⚠ 判据是 containment，**不是 MEMORY_NAME_RE**。index() 刻意支持人手改文件名
        （改成 Foo.md 之后索引显示 Foo、read("Foo") 要能读到），套正则会拒绝它，
        那就是又造一次「索引和读取判据分裂」。只挡越权，不挡手改。
        """
        if not isinstance(name, str) or not name:
            raise MemoryRejected("invalid_name", f"name 必须是非空字符串，收到 {name!r}")
        if name in (".", "..") or any(c in name for c in self._NAME_FORBIDDEN):
            raise MemoryRejected(
                "invalid_name",
                f"name 是一个名字、不是路径：不能含 / \\ 或 ..，收到 {name!r}")
        path = self.root / f"{name}.md"
        # 真正的判据：解析后的父目录必须就是记忆目录本身。
        # 解析父目录而不是文件本身 —— 文件可能是人手工做的软链，那不是越权；
        # 而 root 可能是相对路径，两边都 resolve() 才比得对。
        try:
            escaped = path.parent.resolve() != self.root.resolve()
        except (OSError, ValueError, RuntimeError):
            escaped = True
        if escaped:
            raise MemoryRejected("invalid_name", f"{name!r} 会落到记忆目录之外")
        return path

    def _write_tmp(self, path: Path, data: bytes) -> str:
        """把 data 写进 path 同目录下的一个临时文件，返回临时文件路径。
        不做 replace —— 调用方决定什么时候、要不要把它变成正式文件。

        收字节而不是字符串：归档要按原样保住可能不是 UTF-8 的旧内容，
        编码这件事因此必须由调用方决定（正文写入自己 encode，归档不 encode）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return tmp

    def _atomic_write(self, path: Path, data: bytes) -> None:
        """临时文件 + os.replace。直接写会在崩溃时留下半个文件，
        而半个 frontmatter 会让整条记忆变成损坏项。"""
        tmp = self._write_tmp(path, data)
        try:
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _archive_to_trash(self, path: Path, why: str) -> None:
        """把 path 现有内容归档进 trash，但不动 path 本身。
        单独拆出来是因为 write() 覆盖时不能在这里顺手 unlink 原文件——
        谁来删、什么时候删，必须由调用方根据后续步骤是否成功来决定
        （见 write() 里的注释）。"""
        # 时间戳是为了反复「建了删、删了建」时旧版本不互相覆盖。
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        dest = self.trash / f"{path.stem}-{stamp}.md"
        # ⚠ 按字节复制，不 read_text。归档的目的是「原样保住旧内容」，
        #   而旧内容可能是人手工编辑时存成 GBK 之类的非 UTF-8 字节 ——
        #   解码会抛异常，让整条覆盖路径失败（read() 已经不抛了，归档再抛就等于
        #   把「不崩溃」这件事只做了一半）。归档不需要理解内容，只需要保住它：
        #   一个字节都不改，只在末尾追加一句理由。
        raw = path.read_bytes() + f"\n\n<!-- 移入 trash 的理由：{why} -->\n".encode()
        self._atomic_write(dest, raw)

    def write(self, name: str, description: str, content: str,
              source: str, source_outcome: str, kind: str = "knowledge") -> Path:
        # 校验必须在任何落盘动作之前 —— 否则一个非法名字已经建过目录了。
        self.validate(name, description, content)
        path = self._path(name)
        # 新写入的记忆总是显式落 kind——不留给「缺省即 knowledge」的隐式解析去猜，
        # 手改文件时也能一眼看见它是什么类型。used_success/used_failed/last_used
        # 不在这里落盘：一条刚写的记忆从没被用过，缺省值（parse 时补）已经是对的，
        # 写三行全零/全空只是让文件变长，没有信息量。
        body = (f"---\nname: {name}\ndescription: {description}\n"
                f"source: {source}\nsource_outcome: {source_outcome}\n"
                f"created: {date.today().isoformat()}\nkind: {kind}\n---\n\n{content}\n")
        # 覆盖操作本身必须是原子的，不能拆成「先删旧文件、再写新文件」两个独立步骤：
        # 旧实现是 move_to_trash（归档旧内容 + unlink 正式路径）在前、写新内容在后，
        # 一旦写新内容失败，正式路径已经被删了，记忆就凭空消失（read() 返回 None），
        # trash 里的旧版本也不会被当成「当前」记忆——这是 review 抓到的真实 bug。
        #
        # 现在的顺序：先把新内容稳妥地写进临时文件（不动旧文件）；再把旧内容
        # 归档进 trash（同样不动旧文件，只是多存一份副本）；最后用一次 os.replace
        # 把临时文件原子地换成正式文件——这一步是唯一真正「动」旧文件的地方，
        # 而 os.replace 本身是原子的，失败时旧文件必然还在、内容必然完整。
        tmp = self._write_tmp(path, body.encode("utf-8"))
        try:
            if path.exists():
                # 覆盖也进 trash：覆盖同样会毁掉旧内容，没理由只对删除保历史。
                self._archive_to_trash(path, "被同名记忆覆盖")
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    def read(self, name: str) -> str | None:
        try:
            path = self._path(name)
        except MemoryRejected:
            # 越权名字对模型来说就是「没有这条」：返回 None，不抛。
            # 一来 read() 现有契约就是「读不出来返回 None」（CLI 的 memory show
            # 和 commit_memories 都指着它不抛）；二来「没有这条」不泄露
            # 记忆目录外面那个路径到底存不存在。
            return None
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # ⚠ 与 index() 里同一条判断：读不出来就是「这个文件用不了」，
            #   返回 None 而不是抛 —— 抛会让 CLI 的 memory show 直接崩，
            #   也会让 commit_memories 在「要写的名字恰好是那个坏文件」时整批失败。
            return None
        parts = text.split("---\n", 2)
        return parts[2].strip() if len(parts) == 3 else text.strip()

    def move_to_trash(self, name: str, why: str) -> bool:
        try:
            path = self._path(name)
        except MemoryRejected:
            # 与 read() 同一条判断：越权名字就是「没有这条」，返回 False。
            # 尤其不能让 iphone memory rm ../private 把记忆目录外面的文件挪走。
            return False
        if not path.exists():
            return False
        self._archive_to_trash(path, why)
        path.unlink()
        return True

    _FIELDS = ("name", "description", "source", "source_outcome", "created")

    @staticmethod
    def _parse_frontmatter(text: str) -> dict | None:
        """按行解析，不引入 YAML 依赖。任何一步不合预期就返回 None（视为损坏）。"""
        if not text.startswith("---\n"):
            return None
        end = text.find("\n---\n", 4)
        if end == -1:
            return None
        meta: dict[str, str] = {}
        for line in text[4:end].splitlines():
            key, sep, value = line.partition(":")
            if not sep:
                return None
            meta[key.strip()] = value.strip()
        return meta if all(k in meta for k in MemoryStore._FIELDS) else None

    def index(self) -> tuple[list[MemoryEntry], int]:
        """现算索引。返回 (条目, 损坏文件数)。

        不落盘是刻意的：只要存了第二份数据，就有它和文件不一致的可能，
        而「写文件 + 重建索引」中间崩溃必然造成漂移。没有第二份数据就没有这个问题。
        """
        if not self.root.exists():
            return [], 0
        entries, broken = [], 0
        for path in sorted(self.root.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                # ⚠ 读不出来和 frontmatter 格式错是同一类「这个文件坏了」，走同一条路径。
                #    这个项目刻意让人手改记忆文件，而内容是中文 —— 编辑器存成 GBK
                #    就会产生非 UTF-8 字节。不在这里接住的话，坏文件会让 count() 抛异常，
                #    进而让**之后所有** done(remember=...) 静默失败，直到有人找到它。
                broken += 1
                continue
            meta = self._parse_frontmatter(text)
            if meta is None:
                broken += 1
                continue
            # 可选字段缺省时按 MemoryEntry 的默认值来（旧文件没有这几行，
            # 必须照常解析——这是本任务的硬约束）；出现了但转不成 int
            # 就是手改坏了，和五个必填字段解析失败同一个下场：算损坏、跳过。
            try:
                used_success = int(meta["used_success"]) if "used_success" in meta else 0
                used_failed = int(meta["used_failed"]) if "used_failed" in meta else 0
            except ValueError:
                broken += 1
                continue
            # name 以文件名为准，不用 frontmatter 里那个。两者只在手工编辑时会分裂，
            # 而这个项目选文件格式就是为了让人手改 —— 重命名文件是最自然的动作之一。
            # 若以 frontmatter 为准，索引会展示一个 read() 解析不了的名字；
            # 若把不符当成损坏，一次改名就会让整条记忆从索引里消失。
            # frontmatter 里的 name 因此只是给人看的冗余标注。
            entries.append(MemoryEntry(**{
                **{k: meta[k] for k in self._FIELDS},
                "name": path.stem,
                "kind": meta.get("kind", "knowledge"),
                "used_success": used_success,
                "used_failed": used_failed,
                "last_used": meta.get("last_used", ""),
            }))
        return entries, broken

    def count(self) -> int:
        return len(self.index()[0])

    def update_usage(self, name: str, outcome: str, run_id: str) -> bool:
        """记一次使用反馈：把 used_success/used_failed/last_used 三行改掉，正文和
        其余 frontmatter 字段一个字节都不碰。

        ⚠ 不走 read() + write()：write() 会把整条记忆当成"新内容"重新格式化、
        还会把旧版本挪进 trash——一次使用反馈不该在 trash 里留一条几乎相同的历史，
        更不该有内容被静默重排的风险（比如 description 里巧合含有正文特征）。
        这里直接在原文本里定位 frontmatter 区块、替换/追加三行、其余原样保留。
        """
        try:
            path = self._path(name)
        except MemoryRejected:
            return False
        if not path.exists():
            return False
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        if not text.startswith("---\n"):
            return False
        end = text.find("\n---\n", 4)
        if end == -1:
            return False
        header = text[4:end]
        # 从 end 开始（含开头的 "\n"）到文件结尾一个字符都不动——这就是"正文"，
        # 以及闭合 frontmatter 的那个 "---\n" 本身。
        rest = text[end:]

        meta: dict[str, str] = {}
        kept_lines: list[str] = []
        for line in header.split("\n"):
            key, sep, value = line.partition(":")
            if not sep:
                return False  # 格式已经坏了，别在坏文件上动手
            k = key.strip()
            meta[k] = value.strip()
            if k not in ("used_success", "used_failed", "last_used"):
                kept_lines.append(line)
        if not all(k in meta for k in self._FIELDS):
            return False  # 必填字段都不全，这是个损坏文件，不修

        try:
            used_success = int(meta.get("used_success", "0") or "0")
            used_failed = int(meta.get("used_failed", "0") or "0")
        except ValueError:
            # 和 index() 同一条判断标准：这两个字段转不成 int 就是文件损坏，
            # 不在这里"自愈"覆盖旧计数——直接拒绝这次反馈，一个字节都不写。
            return False

        if outcome == "success":
            used_success += 1
        elif outcome == "failed":
            used_failed += 1
        else:
            raise ValueError(f"outcome 必须是 'success' 或 'failed'，收到 {outcome!r}")

        kept_lines.append(f"used_success: {used_success}")
        kept_lines.append(f"used_failed: {used_failed}")
        kept_lines.append(f"last_used: {run_id}")
        new_text = "---\n" + "\n".join(kept_lines) + rest
        self._atomic_write(path, new_text.encode("utf-8"))
        return True

    def search(self, query: str, k: int) -> list[tuple[MemoryEntry, float]]:
        """按 name+description 的 bigram 相似度打分，降序，只留 score > 0 的前 k 条。

        只搜 name+description、不搜正文：description 本来就是"一句话摘要"，
        搜它等价于搜一份浓缩过的索引；正文可能到 2000 字符，参与打分只会
        让长记忆的分数被稀释或者反过来靠字多取胜，两种偏差都不是想要的。
        """
        entries, _broken = self.index()
        scored = [(e, _bigram_score(query, f"{e.name} {e.description}"))
                  for e in entries]
        scored = [(e, s) for e, s in scored if s > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
