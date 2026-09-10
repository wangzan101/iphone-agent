"""孪生文件的读写底座：随机 id、原子替换、revision 检查（设计说明、§7 不变式 3）。

为什么 id 是随机串不是名字的拼音（codex 评审第 5 点）：多音字、同名「详情」页会撞；
合并后总要淘汰一个 id，「永不变」要靠重定向兑现 —— 名字只能是属性。
为什么带 revision：任务后的后台整理和下一个任务、页面上的人工改名会同时写同一个文件，
拿着旧快照整文件写回会丢掉别人刚写的东西。写之前核对磁盘上的 revision，不对就拒绝。
"""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path

_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


class RevisionConflict(Exception):
    """磁盘上的 revision 和调用方以为的不一样：有人先写了。调用方重读再合，别覆盖。"""


def new_id(prefix: str) -> str:
    return prefix + "_" + "".join(secrets.choice(_ALPHABET) for _ in range(6))


def read_json(path: Path) -> dict | None:
    """不存在 / 坏 JSON / 不是对象 → None。读失败等于「没有孪生」，绝不抛（不变式 5）。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, data: dict, expect_revision: int | None) -> int:
    """原子写：临时文件 + os.replace。`expect_revision` 与磁盘上的不等就拒绝。返回新 revision。"""
    path = Path(path)
    current = read_json(path)
    on_disk = current.get("revision") if current else None
    # 磁盘上的值是不可信输入（可能被手改坏、被旧版本写坏）：不是 int 就当「没有 revision」。
    # bool 是 int 的子类，要单独排掉，否则 True/False 会被当成合法 revision。
    # 不净化的话 (on_disk or 0) + 1 对着字符串之类的东西做算术直接抛 TypeError——
    # 这个模块的调用方（Layout.save）只接 RevisionConflict/OSError，接不住 TypeError；
    # 净化成「没有 revision」等于让坏文件被下一次写整个覆盖掉，而不是从此永远写不进去。
    if isinstance(on_disk, bool) or not isinstance(on_disk, int):
        on_disk = None
    if expect_revision is not None and on_disk != expect_revision:
        raise RevisionConflict(f"{path.name}: 磁盘 revision={on_disk}，调用方以为是 {expect_revision}")
    new_rev = (on_disk or 0) + 1
    body = dict(data)
    body["revision"] = new_rev
    # 先序列化再建临时文件（学 model/configwrite.py）：数据有问题时 dumps 就抛，
    # 根本不会有一个要靠 except 分支去清理的 .tmp。
    text = json.dumps(body, ensure_ascii=False, indent=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return new_rev
