"""静态资源。

这个服务能操作你的真手机，而它是在用户的项目目录里跑起来的 ——
所以「只发 static 目录里的东西」是一条安全边界，不是整洁问题。
"""
import pytest

from iphone_agent.web import static_files as sf


@pytest.fixture
def root(tmp_path):
    (tmp_path / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    (tmp_path / "app.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.js").write_text("//", encoding="utf-8")
    (tmp_path / "secret.py").write_text("SECRET = 1", encoding="utf-8")
    outside = tmp_path.parent / "outside.html"
    outside.write_text("<p>不该发出去</p>", encoding="utf-8")
    return tmp_path


def test_root_serves_index(root):
    body, ctype = sf.read_asset("/", root)
    assert b"hi" in body and ctype.startswith("text/html")


def test_normal_files(root):
    assert sf.read_asset("/app.css", root)[1].startswith("text/css")
    assert sf.read_asset("/sub/a.js", root)[1].startswith("text/javascript")


@pytest.mark.parametrize("path", [
    "/../outside.html",
    "/sub/../../outside.html",
    "/%2e%2e/outside.html",          # 已解码的形式由调用方给，这里再挡一次原样的
    "/....//outside.html",
])
def test_path_traversal_is_refused(root, path):
    assert sf.read_asset(path, root) is None


def test_symlink_escaping_the_root_is_refused(root):
    """字符串前缀检查会被符号链接绕过 —— 所以必须 resolve 之后再判包含关系。"""
    link = root / "escape.html"
    link.symlink_to(root.parent / "outside.html")
    assert sf.read_asset("/escape.html", root) is None


def test_unlisted_suffixes_are_never_served(root):
    """白名单不是黑名单：万一有人往这个目录里放了 .py 或 .toml，也不该被发出去。"""
    assert sf.read_asset("/secret.py", root) is None


def test_missing_file_is_none(root):
    assert sf.read_asset("/nope.js", root) is None


def test_nul_byte_is_refused(root):
    assert sf.read_asset("/app.css\x00.png", root) is None


def test_directory_itself_is_not_served(root):
    assert sf.read_asset("/sub", root) is None


# ── 真实资源确实在包里 ──────────────────────────────────────────────────

def test_the_shipped_app_is_present_and_self_contained():
    """pip 装完 iphone serve 就该能打开 —— 资源必须真的在包里，且不依赖外网。"""
    body, ctype = sf.read_asset("/")
    assert ctype.startswith("text/html")
    html = body.decode()
    assert "app.css" in html and "app.js" in html
    # 外部 CDN 一律不行：这是个离线的本机工具，断网也得能用。
    for bad in ("https://", "//cdn", "unpkg", "jsdelivr", "googleapis"):
        assert bad not in html, f"页面里不该出现外部资源：{bad}"


def test_shipped_css_and_js_load():
    assert sf.read_asset("/app.css")
    assert sf.read_asset("/app.js")
