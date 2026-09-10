"""测试共用：不读环境、不读文件，直接从内置表拼一个 ResolvedModel。"""
from __future__ import annotations

import dataclasses

from iphone_agent.model import registry
from iphone_agent.model.config import ResolvedModel
from iphone_agent.model.profile import ModelProfile


def fake_resolved(spec: str = "alibaba:qwen3.7-plus", **model_overrides) -> ResolvedModel:
    prov, mid = registry.parse_spec(spec)
    provider = registry.find_provider(prov)
    assert provider is not None, f"测试只用内置 provider，{prov} 不在表里"
    model = registry.find_model(spec) or ModelProfile(id=mid, provider=provider.name)
    if model_overrides:
        model = dataclasses.replace(model, **model_overrides)
    return ResolvedModel(provider=provider, model=model, api_key="k", api_key_source="test",
                         spec=f"{provider.name}:{mid}")
