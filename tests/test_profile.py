import dataclasses
from types import MappingProxyType

import pytest

from iphone_agent.model.profile import ModelProfile, ProviderProfile


def test_provider_dict_fields_are_frozen_copies():
    """frozen=True 冻不住 dict：注册表共享实例，调用方原地改一下就污染所有请求。"""
    src = {"enable_thinking": False}
    p = ProviderProfile(name="x", extra_body=src, default_headers={"A": "b"})
    src["enable_thinking"] = True                      # 改原 dict 不影响 profile
    assert p.extra_body == {"enable_thinking": False}
    assert isinstance(p.extra_body, MappingProxyType)
    with pytest.raises(TypeError):
        p.extra_body["k"] = 1                          # type: ignore[index]
    with pytest.raises(TypeError):
        p.default_headers["A"] = "c"                   # type: ignore[index]


def test_provider_sequence_fields_are_normalised():
    p = ProviderProfile(name="x", aliases=["a", "b"], env_vars=["K"], omit_params=["parallel_tool_calls"])
    assert p.aliases == ("a", "b") and p.env_vars == ("K",)
    assert p.omit_params == frozenset({"parallel_tool_calls"})


def test_provider_rejects_unknown_api_mode():
    with pytest.raises(ValueError) as ei:
        ProviderProfile(name="x", api_mode="grpc")
    assert "api_mode" in str(ei.value)


def test_model_allow_coord_tap_defaults_to_false():
    """⚠ 安全承诺：没标定过的模型不许盲指坐标。默认值就是 False，不靠任何分支。"""
    m = ModelProfile(id="m", provider="p")
    assert m.allow_coord_tap is False and m.calibrated is False
    assert m.coord_mode == "norm1000" and m.supports_tools is True and m.max_tokens == 1024


def test_model_rejects_bad_coord_mode():
    with pytest.raises(ValueError) as ei:
        ModelProfile(id="m", provider="p", coord_mode="percent")
    assert "coord_mode" in str(ei.value) and "norm1000" in str(ei.value)


def test_model_spec_and_frozen_extra_body():
    m = ModelProfile(id="qwen3.7-plus", provider="alibaba", extra_body={"enable_thinking": False})
    assert m.spec == "alibaba:qwen3.7-plus"
    assert isinstance(m.extra_body, MappingProxyType)


def test_replace_keeps_invariants():
    """overlay 用 dataclasses.replace 叠字段，__post_init__ 必须再跑一次。"""
    m = ModelProfile(id="m", provider="p")
    m2 = dataclasses.replace(m, extra_body={"a": 1}, coord_mode="pixel")
    assert isinstance(m2.extra_body, MappingProxyType) and m2.coord_mode == "pixel"
    with pytest.raises(ValueError):
        dataclasses.replace(m, coord_mode="nope")
