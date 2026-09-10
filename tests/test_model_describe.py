"""模型信息的结构化出口。

两条底线：
1. **一个字都不能带密钥** —— 这份数据要走 HTTP 到网页上去。
2. `calibrated` 必须出现在目录里 —— 内置表是不对称的，只有一个模型标定过坐标，
   界面得能把这件事标出来，否则用户随手换一个然后觉得「它变笨了」。
"""
import json

import pytest

from iphone_agent.model import registry
from iphone_agent.model.config import resolve
from iphone_agent.model.describe import catalog, describe, providers

SECRET = "sk-do-not-leak-0123456789"


@pytest.fixture
def resolved(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("IPHONE_USE_MODEL", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", SECRET)
    return resolve()


def test_describe_never_leaks_the_key(resolved):
    d = describe(resolved)
    assert SECRET not in json.dumps(d, ensure_ascii=False)
    assert d["has_key"] is True
    assert d["api_key_source"].startswith("env:")     # 只说来源


def test_describe_carries_the_coordinate_facts(resolved):
    d = describe(resolved)
    assert d["spec"] == registry.DEFAULT_MODEL_SPEC
    assert d["calibrated"] is True                    # 默认模型是唯一标定过的那个
    assert d["coord_mode"] in ("norm1000", "pixel")
    assert isinstance(d["allow_coord_tap"], bool)


def test_catalog_lists_every_builtin_model():
    c = catalog()
    assert {x["spec"] for x in c} == set(registry.MODELS)
    json.dumps(c)                                     # 要走 HTTP


def test_catalog_marks_exactly_one_recommended():
    rec = [x for x in catalog() if x["recommended"]]
    assert [x["spec"] for x in rec] == [registry.DEFAULT_MODEL_SPEC]


def test_the_asymmetry_is_between_the_builtin_model_and_anything_you_type(monkeypatch, tmp_path):
    """内置表的真实形状：**7 个 provider，只有 1 个标定过的模型**。

    别的模型 id 是自由输入 —— resolve 会照收，但按「未标定、坐标点击关闭」处理。
    所以设置页不能是一个固定的模型下拉：provider 是列表，model id 是输入框，
    而那个唯一标定过的 spec 要作为推荐项摆在最前面。
    """
    assert [x["spec"] for x in catalog() if x["calibrated"]] == [registry.DEFAULT_MODEL_SPEC]
    assert len(providers()) > len(catalog()), "provider 比内置模型多 —— 这正是自由输入的理由"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    d = describe(resolve("deepseek:deepseek-v3"))
    assert d["spec"] == "deepseek:deepseek-v3"
    assert d["calibrated"] is False
    assert d["allow_coord_tap"] is False, "没标定过就不该给坐标工具，误差 185px 起步"
    assert any("未标定坐标" in n for n in d["notices"]), "界面要能把这句话原样显示出来"


def test_catalog_says_where_the_key_comes_from():
    """设置页要能告诉用户「这个模型的 key 去哪拿」。"""
    for x in catalog():
        assert isinstance(x["env_vars"], list)


def test_providers_listing():
    p = providers()
    assert {x["name"] for x in p} == set(registry.PROVIDERS)
    json.dumps(p)
