"""SKILL.md 式 frontmatter 的通用词法：`---` 包围、每行 `key: value`。

不引 YAML：标准库没有，而这个项目的规矩是零新依赖。所以数组和对象必须写成
JSON 字面量（`apps: ["alipay", "yimujizhang"]`），以 `[` 或 `{` 开头的值走 json.loads；
其余值原样字符串（去首尾空白）。

⚠ 不能直接复用 MemoryStore._parse_frontmatter：它硬要求记忆专属的五个字段。
这里只做词法，字段校验由每种文件自己的 schema 做（skills/model.py）。
"""
from __future__ import annotations

import json


class FrontmatterError(ValueError):
    pass


def parse(text: str) -> tuple[dict, str]:
    """→ (meta, body)。body 是闭合 `---` 之后的全部文本（去掉紧跟的空行）。"""
    if not text.startswith("---\n"):
        raise FrontmatterError("文件必须以 --- 开头")
    end = text.find("\n---\n", 3)
    if end == -1:
        # 允许文件在闭合 --- 处结束（没有正文）
        if text.endswith("\n---"):
            end = len(text) - 4
        else:
            raise FrontmatterError("frontmatter 没有闭合的 ---")
    meta: dict = {}
    # Guard against negative slice when end < 4 (empty meta section)
    meta_text = text[4:end] if end >= 4 else ""
    for lineno, line in enumerate(meta_text.splitlines(), start=2):
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        key = key.strip()
        if not sep or not key or " " in key:
            raise FrontmatterError(f"第 {lineno} 行不是 key: value：{line!r}")
        if key in meta:
            raise FrontmatterError(f"字段 {key} 重复")
        value = value.strip()
        if value[:1] in ("[", "{"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as e:
                raise FrontmatterError(f"字段 {key} 的值不是合法 JSON：{e}") from e
        meta[key] = value
    body = text[end + 5:]
    return meta, body.lstrip("\n")


def dump(meta: dict, body: str) -> str:
    """把 meta 写成 frontmatter。**值里不许有换行**，见下面的注释：这是安全边界。"""
    lines = ["---"]
    for k, v in meta.items():
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)     # json.dumps 自己会转义换行，安全
        elif v is None:
            v = ""
        else:
            _check_value(k, str(v))
        lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n" + body.rstrip("\n") + "\n"


def _check_value(key: str, value: str) -> None:
    """dump 的输入里有模型可控的字符串（场景 description、自动建 App 的 display / open）。
    值里一个换行就能提前闭合 frontmatter，把后面几行伪造成 status: manual / verified ——
    人批准这道闸门就废了。与 memory/store.py 里 description 的那道防线同源、同理由。
    这里只拒绝，不清洗：调用方比这里更清楚该降级成什么（extract 取首行，loop 直接拒）。"""
    if any(c in value for c in "\n\r"):
        raise FrontmatterError(f"字段 {key} 的值不能含换行：{value!r}")
    if value.strip().startswith("---"):
        raise FrontmatterError(f"字段 {key} 的值不能以 --- 开头：{value!r}")
