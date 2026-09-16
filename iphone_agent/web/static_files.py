"""发静态资源。

为什么不用 `http.server.SimpleHTTPRequestHandler`：它按**进程的当前目录**发文件。
这个服务能操作你的真手机，而 `iphone serve` 是在用户的项目目录里跑的 ——
把 cwd 整个暴露出去不可接受。这里只发 `web/static/` 里的东西，别的一律 404。

⚠ 路径穿越必须自己挡。`resolve()` 之后检查它是不是真的落在 static 目录下面，
不能只看字符串前缀 —— 符号链接会绕过前缀检查。

**没有构建步骤**：这些文件原样发出去，改完刷新就看得见（docs/22 D3）。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent / "static"

# 白名单，不是黑名单。没列出来的后缀一律不发 —— 万一哪天有人往这个目录里
# 放了个 .py 或 .toml，也不会被当成静态资源发出去。
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".woff2": "font/woff2",
    ".ico": "image/x-icon",
}


def resolve_asset(url_path: str, root: Path = ROOT) -> tuple[Path, str] | None:
    """把 URL 路径映射成一个真实文件。不合法或不存在都返回 None（调用方 404）。"""
    rel = url_path.lstrip("/")
    if not rel or rel.endswith("/"):
        rel = (rel + "index.html") if rel else "index.html"
    # 有 NUL 的路径在某些系统调用上会被截断，直接拒。
    if "\x00" in rel:
        return None
    root = root.resolve()
    try:
        p = (root / rel).resolve()
    except OSError:
        return None
    # ⚠ 必须是 resolve 之后再判包含关系：符号链接能让字符串前缀检查通过而实际指到外面。
    if p != root and root not in p.parents:
        return None
    ctype = TYPES.get(p.suffix.lower())
    if ctype is None or not p.is_file():
        return None
    return p, ctype


def read_asset(url_path: str, root: Path = ROOT) -> tuple[bytes, str] | None:
    hit = resolve_asset(url_path, root)
    if hit is None:
        return None
    p, ctype = hit
    try:
        return p.read_bytes(), ctype
    except OSError:
        return None
