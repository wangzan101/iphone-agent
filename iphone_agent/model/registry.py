"""内置只读表。

用户在 config.toml 里声明的 provider / model **不进这里** —— resolve() 每次建一个局部 overlay，
查完即弃。理由：web 的 Session 长驻、测试在同一进程反复 resolve，写全局会让上一次的
配置残留到下一次。
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from iphone_agent.model.errors import ConfigError
from iphone_agent.model.profile import ModelProfile, ProviderProfile
from iphone_agent.model.providers import ALL as _BUILTIN_MODULES

DEFAULT_PROVIDER = "alibaba"
DEFAULT_MODEL_SPEC = "alibaba:qwen3.7-plus"


def _build() -> tuple[Mapping[str, ProviderProfile], Mapping[str, str], Mapping[str, ModelProfile]]:
    providers: dict[str, ProviderProfile] = {}
    aliases: dict[str, str] = {}
    models: dict[str, ModelProfile] = {}
    for mod in _BUILTIN_MODULES:
        p: ProviderProfile = mod.PROVIDER
        if p.name in providers or p.name in aliases:
            raise RuntimeError(f"内置 provider 重名：{p.name}")
        providers[p.name] = p
        for a in p.aliases:
            if a in providers or a in aliases:
                raise RuntimeError(f"内置 provider 别名撞车：{a}")
            aliases[a] = p.name
        for m in getattr(mod, "MODELS", ()):
            if m.provider != p.name:
                raise RuntimeError(f"{mod.__name__} 里的模型 {m.id} 挂错了 provider：{m.provider}")
            models[m.spec] = m
    return MappingProxyType(providers), MappingProxyType(aliases), MappingProxyType(models)


PROVIDERS, ALIASES, MODELS = _build()


def parse_spec(spec: str) -> tuple[str, str]:
    """`provider:model`。含冒号只切第一个（ollama 的模型名自己带冒号）；不含冒号补默认 provider。
    这条规则对 IPHONE_USE_MODEL、--model、TOML 顶层 model 三处一视同仁。"""
    s = (spec or "").strip()
    if not s:
        raise ConfigError("模型标识为空", code="config.emptyModel", params={})
    if ":" in s:
        prov, mid = s.split(":", 1)
    else:
        prov, mid = DEFAULT_PROVIDER, s
    prov, mid = prov.strip(), mid.strip()
    if not prov or not mid:
        raise ConfigError(f"模型标识 {spec!r} 格式应为 provider:model", code="config.modelFormat", params={})
    return prov, mid


def find_provider(name: str) -> ProviderProfile | None:
    n = name.strip()
    if n in PROVIDERS:
        return PROVIDERS[n]
    if n in ALIASES:
        return PROVIDERS[ALIASES[n]]
    return None


def find_model(spec: str) -> ModelProfile | None:
    prov, mid = parse_spec(spec)
    p = find_provider(prov)
    if p is None:
        return None
    return MODELS.get(f"{p.name}:{mid}")


def known_providers() -> list[str]:
    return sorted(PROVIDERS)
