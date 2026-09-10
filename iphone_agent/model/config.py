"""配置层唯一的出口：resolve(spec, conf) -> ResolvedModel。

桥、Session、doctor、serve 只消费 ResolvedModel。注册表只负责纯 profile 查询。
优先级：环境变量 > config.toml > 内置表（脚本和 CI 要能临时覆盖，而不用去改文件）。
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

from iphone_agent.model import registry
from iphone_agent.model.errors import ConfigError
from iphone_agent.model.profile import COORD_MODES, ModelProfile, ProviderProfile

CONFIG_FILE = Path(".iphone/config.toml")

# 旧格式：顶层这几个键等价于 [providers.alibaba] 的同名键。不打弃用警告，本期不逼用户改文件。
_LEGACY_TOP_KEYS = ("api_key", "base_url")
_PROVIDER_KEYS = frozenset({"api_key", "base_url", "omit_params", "default_headers", "extra_body"})
_MODEL_KEYS = frozenset({"coord_mode", "allow_coord_tap", "supports_tools", "max_tokens",
                         "extra_body", "calibrated"})


@dataclass(frozen=True)
class ResolvedModel:
    provider: ProviderProfile        # 已合并 TOML 覆盖
    model: ModelProfile              # 已合并 TOML 覆盖
    # repr=False：密钥不进 repr。堆栈里的局部变量、调试时的 print(resolved) 都会展开
    # dataclass 的 repr，一展开密钥就落进日志和聊天记录里 —— 这个项目就这么泄过一次。
    api_key: str = field(repr=False)
    api_key_source: str              # "env:DASHSCOPE_API_KEY" | "toml:[providers.alibaba]" | "none"
    spec: str                        # "alibaba:qwen3.7-plus"，留档与打印用
    notices: tuple[str, ...] = ()    # 启动时要打给用户看的一行行提示


def read_config_file(path: Path = CONFIG_FILE) -> dict:
    """`.iphone/config.toml` —— 让密钥不必每次写在命令行上。

    ⚠ 这个文件存在的**唯一理由**就是密钥不该出现在命令行里：命令行会进 shell 历史、
    进 ps 输出、进日志、进聊天记录。这个项目的密钥就是这么泄出去过一次的。

    所以 group / other 权限位必须全为 0，否则拒绝使用它 —— 一个 644 的密钥文件
    比环境变量还糟，它是持久的。0600、0400 都过。
    用 TOML 不用 YAML：tomllib 是标准库，零新依赖。
    """
    if not path.exists():
        return {}
    if path.stat().st_mode & 0o077:
        raise ConfigError(
            f"{path} 的权限太松（组或其他用户可读）—— 里面是密钥。先 chmod 600 {path} 再跑。",
            code="config.permissions", params={"path": str(path)})
    import tomllib
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"{path} 读不出来：{type(e).__name__}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path} 顶层必须是键值对")
    return data


def _check_keys(table: dict, allowed: frozenset, where: str) -> None:
    bad = sorted(set(table) - allowed)
    if bad:
        raise ConfigError(f"{where} 里有不认识的键：{bad}。可用的是 {sorted(allowed)}")


def _sole_key(tables: dict, keys: list[str], fallback: str, render) -> str:
    """同一个目标写了两张表时当场喊停。

    合并两张表没有唯一解（谁覆盖谁？），静默挑一张就等于把用户另一半配置扔了 ——
    而「配置被悄悄丢掉」正是这个模块要消灭的那类故障。空表不算数：它没内容可丢。
    """
    hit = [k for k in keys if tables.get(k)]
    if len(hit) > 1:
        names = "、".join(render(k) for k in hit)
        raise ConfigError(f"config.toml 里 {names} 指的是同一个东西，只会有一张生效。"
                          f"把它们合并成一张再跑。")
    return hit[0] if hit else fallback


def _provider_overlay(conf: dict, name: str) -> tuple[ProviderProfile, str | None]:
    """返回 (合并后的 provider, toml 里的 api_key 或 None)。

    overlay 是一次性的局部对象：内置表只读，写回去会让 web 的长驻 Session 和
    同进程里下一次 resolve 读到上一次的配置。
    """
    tables = conf.get("providers") or {}
    if not isinstance(tables, dict):
        raise ConfigError("config.toml 的 providers 必须是表")
    builtin = registry.find_provider(name)
    # 别名和正名在配置里等价：写 [providers.kimi] 的人不该被静默忽略。name 已是正名。
    keys = [name, *(builtin.aliases if builtin is not None else ())]
    used = _sole_key(tables, keys, name, lambda k: f"[providers.{k}]")
    table = dict(tables.get(used) or {})
    if name == registry.DEFAULT_PROVIDER:            # 旧格式的顶层键并进来，已有的不覆盖
        for k in _LEGACY_TOP_KEYS:
            if k in conf and k not in table:
                table[k] = conf[k]
    _check_keys(table, _PROVIDER_KEYS, f"[providers.{used}]")

    toml_key = table.pop("api_key", None)
    try:
        if builtin is None:
            if not table.get("base_url"):
                raise ConfigError(f"provider {name!r} 不在内置表里（{registry.known_providers()}），"
                                  f"自定义端点必须在 [providers.{name}] 里给 base_url")
            env_name = name.upper().replace("-", "_") + "_API_KEY"
            p = ProviderProfile(name=name, env_vars=(env_name,), **table)
        else:
            p = dataclasses.replace(builtin, **table) if table else builtin
    except (TypeError, ValueError) as e:          # default_headers = "abc" 这种，构造时才炸
        raise ConfigError(f"[providers.{used}]：{type(e).__name__}: {e}") from e
    if p.name == registry.DEFAULT_PROVIDER and os.environ.get("DASHSCOPE_BASE_URL", "").strip():
        p = dataclasses.replace(p, base_url=os.environ["DASHSCOPE_BASE_URL"].strip())
    return p, (str(toml_key).strip() if toml_key else None)


def _canon_spec(key: str) -> str | None:
    """把 [models.*] 的表名归一成正名 spec；不是合法 spec 就返回 None（当它不存在）。"""
    try:
        prov, mid = registry.parse_spec(key)
    except ConfigError:
        return None
    builtin = registry.find_provider(prov)
    return f"{builtin.name if builtin is not None else prov}:{mid}"


def _model_overlay(conf: dict, provider: ProviderProfile, model_id: str) -> tuple[ModelProfile, list[str]]:
    spec = f"{provider.name}:{model_id}"
    tables = conf.get("models") or {}
    if not isinstance(tables, dict):
        raise ConfigError("config.toml 的 models 必须是表")
    # 表名同样按别名归一：写 [models."dashscope:qwen-vl-max"] 的人不该被静默忽略。
    keys = [spec, *(k for k in tables if k != spec and _canon_spec(k) == spec)]
    used = _sole_key(tables, keys, spec, lambda k: f'[models."{k}"]')
    table = dict(tables.get(used) or {})
    _check_keys(table, _MODEL_KEYS, f'[models."{used}"]')
    builtin = registry.find_model(spec)
    notices: list[str] = []
    try:
        if builtin is None:
            m = ModelProfile(id=model_id, provider=provider.name, **table)
        else:
            m = dataclasses.replace(builtin, **table) if table else builtin
    except (TypeError, ValueError) as e:          # coord_mode 非法（ValueError）、extra_body = 5（TypeError）
        raise ConfigError(f'[models."{used}"]：{type(e).__name__}: {e}') from e
    legacy = os.environ.get("IPHONE_USE_COORD_MODE", "").strip()
    if legacy:
        if legacy not in COORD_MODES:
            raise ConfigError(f"IPHONE_USE_COORD_MODE={legacy!r} 不是 {COORD_MODES} 之一")
        # 设了这个变量就是用户亲口声明了坐标约定，坐标点击一并打开：这是它当初的生产语义，
        # 静默忽略等于悄悄改掉别人已经在用的行为。既然打开了，就不能再说「未标定、已关闭」。
        m = dataclasses.replace(m, coord_mode=legacy, allow_coord_tap=True)
        notices.append(f'IPHONE_USE_COORD_MODE 已改为按模型配置，请迁到 [models."{spec}"] coord_mode。'
                       f"本次按 {legacy} 处理并打开坐标点击。")
    elif not m.allow_coord_tap:
        notices.append(
            f"{spec} 未标定坐标，坐标点击已关闭，只按元素编号点。"
            f'跑 iphone qualify，或在 {CONFIG_FILE} 的 [models."{spec}"] 里手动打开。')
    if not m.supports_tools:
        raise ConfigError(f"{spec} 声明不支持工具调用，本项目的循环没法跑；换模型")
    return m, notices


def _api_key(provider: ProviderProfile, toml_key: str | None) -> tuple[str, str]:
    for env in provider.env_vars:
        v = os.environ.get(env, "").strip()
        if v:
            return v, f"env:{env}"
    if toml_key:
        return toml_key, f"toml:[providers.{provider.name}]"
    raise ConfigError(
        f"没有 {provider.name} 的 API key：设环境变量 {' 或 '.join(provider.env_vars)}，"
        f'或者在 {CONFIG_FILE} 的 [providers.{provider.name}] 里写 api_key = "..."（记得 chmod 600）',
        code="config.missingKey", params={"provider": provider.name, "env": " / ".join(provider.env_vars)})


def resolve(spec: str | None = None, conf: dict | None = None, *, need_key: bool = True) -> ResolvedModel:
    """spec 为 None 时依次取 IPHONE_USE_MODEL、conf["model"]、内置默认。
    need_key=False：不查 key（screen / tap / doctor 这些不碰模型的命令用）。"""
    if conf is None:
        conf = read_config_file()
    # 顶层也要查错字：写成 mdoel = "deepseek:..." 的人不该悄悄拿到默认模型。
    # [providers.*] 和 [models.*] 早就有这道检查，只有顶层漏了。
    if conf:
        _check_keys(conf, frozenset({"model", "providers", "models", *_LEGACY_TOP_KEYS}),
                    "config.toml 顶层")
    raw = (spec or os.environ.get("IPHONE_USE_MODEL", "").strip()
           or str(conf.get("model") or "").strip() or registry.DEFAULT_MODEL_SPEC)
    prov_name, model_id = registry.parse_spec(raw)
    builtin = registry.find_provider(prov_name)
    declared = conf.get("providers") or {}
    if builtin is not None:
        # 别名先归一成正名，否则写 dashscope:xxx 的人会找不到自己的 [providers.alibaba]。
        prov_name = builtin.name
    elif not (isinstance(declared, dict) and prov_name in declared):
        raise ConfigError(f"不认识的 provider {prov_name!r}。已知：{registry.known_providers()}；"
                          f"自定义端点在 {CONFIG_FILE} 的 [providers.{prov_name}] 里声明 base_url")
    provider, toml_key = _provider_overlay(conf, prov_name)
    model, notices = _model_overlay(conf, provider, model_id)
    if need_key:
        key, source = _api_key(provider, toml_key)
        # 非 ASCII 的密钥装不进 HTTP 的 Authorization 头，发请求时才会以
        # 「'ascii' codec can't encode characters」这种看不出所以然的方式炸。
        # 真踩过：把文档里的占位符「你的key」原样 export 了进来。
        if not key.isascii():
            bad = "".join(ch for ch in key if not ch.isascii())[:8]
            raise ConfigError(f"{source} 含非 ASCII 字符（{bad!r}）—— 多半是把占位符原样复制了。请填真实密钥。",
                              code="config.keyCharacters", params={})
    else:
        key, source = "", "none"
    return ResolvedModel(provider=provider, model=model, api_key=key, api_key_source=source,
                         spec=f"{provider.name}:{model_id}", notices=tuple(notices))
