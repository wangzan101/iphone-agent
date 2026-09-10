"""两张表：ProviderProfile（一个端点的脾气）、ModelProfile（一个模型的脾气）。

照 Hermes Agent 的 ProviderProfile：**声明式**，只描述行为，不拥有 client 构造、不发请求。
桥（transports/）读这两张表拼请求，而不是接收二十个布尔开关。

两张表而不是一张：一个 provider 下挂多个模型，而 coord_mode 是模型的属性不是端点的。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

COORD_MODES = ("norm1000", "pixel")
API_MODES = ("chat_completions",)      # 子项目 2 加 anthropic_messages


def _frozen_map(m: Mapping | None) -> Mapping:
    # frozen=True 冻不住 dict：注册表共享实例，调用方原地改一下就污染所有请求。
    return MappingProxyType(dict(m or {}))


@dataclass(frozen=True)
class ProviderProfile:
    name: str                                   # "alibaba"
    aliases: tuple[str, ...] = ()               # ("dashscope",)
    api_mode: str = "chat_completions"
    base_url: str = ""
    env_vars: tuple[str, ...] = ()              # 取 key 的环境变量，按优先级
    default_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)   # 端点级专有参数
    # 不少 OpenAI 兼容端点**收到不认识的顶层字段直接 400** 而不是忽略（Hermes 的
    # supports_prompt_cache_key 就是为这个设的）。parallel_tool_calls 最常见。
    # 桥发请求前按这个集合把顶层参数过滤掉。
    omit_params: frozenset[str] = frozenset()

    def __post_init__(self):
        if self.api_mode not in API_MODES:
            raise ValueError(f"api_mode 必须是 {API_MODES} 之一，收到 {self.api_mode!r}")
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "env_vars", tuple(self.env_vars))
        object.__setattr__(self, "omit_params", frozenset(self.omit_params))
        object.__setattr__(self, "default_headers", _frozen_map(self.default_headers))
        object.__setattr__(self, "extra_body", _frozen_map(self.extra_body))


@dataclass(frozen=True)
class ModelProfile:
    id: str                                     # "qwen3.7-plus"，原样发给 API
    provider: str                               # "alibaba"
    coord_mode: str = "norm1000"                # norm1000 | pixel
    # ⚠ 默认关。坐标约定猜错是 20 倍误差（2026-09-08 标定：norm1000 中位 8.4px，
    #   pixel 185px），让它盲指不如不指。id 路径（tap(id)/icon_above/row_right/row_left）
    #   全部从 OCR 框推坐标，不依赖模型的坐标约定，照样能跑。
    #   只有内置已标定的模型显式 True。用户只写 max_tokens 的部分声明也不会把它打开。
    allow_coord_tap: bool = False
    supports_tools: bool = True                 # False 本期拒绝启动：主循环离不开工具调用
    max_tokens: int = 1024
    extra_body: Mapping[str, Any] = field(default_factory=dict)   # 模型级，覆盖 provider 级
    calibrated: bool = False                    # coord_mode 是标定出来的还是猜的；doctor 显示用

    def __post_init__(self):
        if self.coord_mode not in COORD_MODES:
            raise ValueError(f"coord_mode 必须是 {COORD_MODES} 之一，收到 {self.coord_mode!r}")
        object.__setattr__(self, "extra_body", _frozen_map(self.extra_body))

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.id}"
