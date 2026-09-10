"""往 `.iphone/config.toml` 里写 —— 设置页存密钥、换模型走这里。

为什么要有它：到现在为止这个文件**只能手写**。命令行用户会 `vim` 一下再
`chmod 600`；界面用户不会，也不该会。设置页要能替他把这两件事一起做对。

三条硬约束，每条都有来历：

1. **权限位必须是 0600，而且是创建时就是**。先建 0644 再 chmod 的话，中间那一瞬
   密钥是全机可读的。所以用 `os.open(..., 0o600)` 直接建，不靠事后补救。
   读那一侧（`read_config_file`）本来就会拒绝权限太松的文件 —— 写这一侧不能自己
   造出一个会被自己拒绝的文件。
2. **原子替换**。写到同目录的临时文件再 `os.replace`。中途断电/崩溃不会留下一个
   写了一半的配置 —— 那会让下次启动读出 TOMLDecodeError，而用户完全不知道发生了什么。
3. **异常里一个字都不能带密钥**。这个项目的密钥泄过一次（见 ResolvedModel 的注释），
   写入路径比读取路径更容易犯这个错，因为手上就攥着明文。

⚠ **注释会丢**。这里是「读成 dict → 合并 → 整个重写」，TOML 的注释和排版留不住。
标准库只有 tomllib（读），没有写；为一个设置页引入 tomlkit 不值得（设计说明 D3 的
同一条理由）。所以文件顶部会写一行说明，告诉手改过的人这件事。
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from iphone_agent.model.config import CONFIG_FILE, read_config_file
from iphone_agent.model.errors import ConfigError

HEADER = (
    "# iPhone Agent 的配置。设置页会重写这个文件 —— **注释和排版不会保留**。\n"
    "# 权限固定为 600（只有你能读），里面有 API 密钥。\n"
)

_BARE_KEY_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _key(k: str) -> str:
    """TOML 键。模型 spec 带冒号和点（alibaba:qwen3.7-plus），必须加引号。"""
    if k and all(c in _BARE_KEY_OK for c in k):
        return k
    return _string(k)


def _string(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _value(v) -> str:
    # bool 必须排在 int 前面：Python 里 True 是 int 的实例，顺序反了会写出 1 而不是 true。
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return _string(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            raise ConfigError(f"写不进 TOML 的浮点值：{v!r}")
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    raise ConfigError(f"写不进 TOML 的值类型：{type(v).__name__}")


def dumps(data: dict) -> str:
    """够用的 TOML 子集：标量、数组、嵌套表。这个配置文件的 schema 就这些。"""
    if not isinstance(data, dict):
        raise ConfigError("配置必须是键值对")
    lines: list[str] = []

    def emit(table: dict, path: list[str]) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        subs = {k: v for k, v in table.items() if isinstance(v, dict)}
        # 只有自己有标量、或者压根没有子表时才写表头。
        # 不然 [providers] 会作为一个空表头出现在 [providers.alibaba] 前面 ——
        # 合法但难看，手改过这个文件的人会以为哪里出错了。
        if path and (scalars or not subs):
            if lines:
                lines.append("")
            lines.append("[" + ".".join(_key(p) for p in path) + "]")
        for k, v in scalars.items():
            lines.append(f"{_key(k)} = {_value(v)}")
        for k, v in subs.items():
            emit(v, [*path, k])

    emit(data, [])
    return HEADER + "\n" + "\n".join(lines) + "\n"


def write_config_file(data: dict, path: Path = CONFIG_FILE) -> None:
    """整个重写。0600 创建 + 原子替换 —— 理由见模块 docstring。"""
    text = dumps(data)                       # 先序列化：内容不合法就别去动磁盘上的文件
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 目录也收紧：.iphone/ 底下不止这一个文件，运行记录里有截图。
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass                                 # 目录本来就是别人的/只读的，不该因此写不成配置
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".config-", suffix=".toml")
    try:
        os.fchmod(fd, 0o600)                 # 替换过去之前就定死权限，不留全机可读的窗口
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # 失败就把临时文件收干净。⚠ 这里绝不能把异常内容再包一层往外抛 ——
        # 手上攥着明文密钥，包装信息里很容易带出去。
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update_config(path: Path = CONFIG_FILE, **changes) -> dict:
    """读 → 合并 → 写。返回合并后的配置（**不含密钥的调用方自己判断**，这里原样返回）。

    值为 None 表示删掉这个键 —— 设置页要能把密钥清空，退回用环境变量。
    """
    path = Path(path)
    data = read_config_file(path)            # 权限太松会在这里就拒绝，不会被我们悄悄覆盖掉
    for k, v in changes.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    write_config_file(data, path)
    return data


def set_model(spec: str, path: Path = CONFIG_FILE) -> dict:
    """设置页换模型。只写顶层 `model`，不碰别的。"""
    from iphone_agent.model import registry
    registry.parse_spec(spec)                # 先校验：别把一个跑不起来的 spec 写进文件
    return update_config(path, model=spec)


# 自定义服务商的名字。它会变成 TOML 的表名、也会变成 spec 的前半段（myvllm:xxx），
# 所以不能带冒号、不能带空格。
PROVIDER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,31}$")


def set_provider(name: str, *, base_url: str | None = None, api_key: str | None = None,
                 path: Path = CONFIG_FILE) -> dict:
    """写 `[providers.<name>]`。**None = 不动这个字段，"" = 删掉它。**

    自定义端点走的就是这里 —— 后端本来就支持（`_provider_overlay` 会给未知
    provider 现造一个 profile，还会自动推导环境变量名 `<NAME>_API_KEY`），
    只是一直没有写入的口子，用户只能回去 vim。

    ⚠ resolve() 要求不在内置表里的 provider **必须有 base_url**，否则报错。
      这里不强制（用户可能分两步填），但界面要在保存前挡住。
    ⚠ 任何报错都不许带 key 的内容。
    """
    name = (name or "").strip().lower()
    if not PROVIDER_NAME_RE.match(name):
        raise ConfigError("服务商名字只能用小写字母、数字和 _ - .，不能带冒号或空格",
                          code="config.providerName", params={})
    if base_url is not None:
        base_url = base_url.strip()
        if base_url:
            if not base_url.startswith(("http://", "https://")):
                raise ConfigError("接口地址要以 http:// 或 https:// 开头", code="config.baseScheme", params={})
            if not base_url.isascii() or any(c.isspace() for c in base_url):
                raise ConfigError("接口地址里有空格或非 ASCII 字符 —— 多半是复制时带进来的",
                                  code="config.baseCharacters", params={})
            if len(base_url) > 500:
                raise ConfigError("接口地址太长了", code="config.baseLength", params={})
    if api_key is not None:
        api_key = api_key.strip()
        if api_key and not api_key.isascii():
            # 和 resolve() 里那道检查对齐。报错里只说「含非 ASCII」，不回显任何字符。
            raise ConfigError("密钥含非 ASCII 字符 —— 多半是把占位符原样复制了。请填真实密钥。",
                              code="config.keyCharacters", params={})

    data = read_config_file(path)
    provs = dict(data.get("providers") or {})
    entry = dict(provs.get(name) or {})
    for field, value in (("base_url", base_url), ("api_key", api_key)):
        if value is None:
            continue                              # 不动
        if value:
            entry[field] = value
        else:
            entry.pop(field, None)                # 空串 = 删掉
    if entry:
        provs[name] = entry
    else:
        provs.pop(name, None)                     # 空表别留在文件里，读起来像配过其实没有
    if provs:
        data["providers"] = provs
    else:
        data.pop("providers", None)
    write_config_file(data, path)
    return data


def set_api_key(provider: str, key: str | None, path: Path = CONFIG_FILE) -> dict:
    """设置页存密钥。`key=None` 或空串 = 删掉它，退回环境变量那条路。

    ⚠ 这个函数的任何报错都不许出现 key 的内容。
    """
    if not (provider or "").strip():
        raise ConfigError("没说是哪个 provider 的密钥")
    return set_provider(provider, api_key=key or "", path=path)
