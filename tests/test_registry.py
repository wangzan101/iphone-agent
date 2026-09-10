import pytest

from iphone_agent.model import registry
from iphone_agent.model.config import ResolvedModel, read_config_file, resolve
from iphone_agent.model.errors import ConfigError


def test_parse_spec_with_colon():
    assert registry.parse_spec("deepseek:deepseek-v3") == ("deepseek", "deepseek-v3")


def test_parse_spec_bare_name_means_alibaba():
    """向后兼容：IPHONE_USE_MODEL=qwen3.7-plus 一直是这么用的。"""
    assert registry.parse_spec("qwen3.7-plus") == ("alibaba", "qwen3.7-plus")


def test_parse_spec_model_id_may_contain_colon():
    """只切第一个冒号：ollama 的模型名自己带冒号。"""
    assert registry.parse_spec("ollama:qwen2.5:7b") == ("ollama", "qwen2.5:7b")


@pytest.mark.parametrize("bad", ["", "  ", ":x", "x:", ":"])
def test_parse_spec_rejects_empty_parts(bad):
    with pytest.raises(ConfigError):
        registry.parse_spec(bad)


def test_builtin_alibaba_by_name_and_alias():
    p = registry.find_provider("alibaba")
    assert p is not None and p.env_vars == ("DASHSCOPE_API_KEY",)
    assert p.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert registry.find_provider("dashscope") is p


def test_unknown_provider_is_none_and_known_list_is_sorted():
    assert registry.find_provider("nope") is None
    names = registry.known_providers()
    assert names == sorted(names) and "alibaba" in names and "openrouter" in names


def test_builtin_qwen_model_is_calibrated_and_allows_coords():
    m = registry.find_model("alibaba:qwen3.7-plus")
    assert m is not None
    assert m.coord_mode == "norm1000" and m.allow_coord_tap is True and m.calibrated is True
    assert dict(m.extra_body) == {"enable_thinking": False}
    assert registry.find_model("alibaba:qwen-vl-max") is None


def test_builtin_tables_are_read_only():
    with pytest.raises(TypeError):
        registry.PROVIDERS["evil"] = None      # type: ignore[index]
    with pytest.raises(TypeError):
        registry.MODELS["evil"] = None         # type: ignore[index]


# ---- resolve()：配置层唯一出口 ----

ALL_ENV = ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "IPHONE_USE_MODEL", "IPHONE_USE_COORD_MODE",
           "DEEPSEEK_API_KEY", "OPENAI_API_KEY")


def _clear_env(monkeypatch):
    for k in ALL_ENV:
        monkeypatch.delenv(k, raising=False)


def _write(tmp_path, body, mode=0o600):
    d = tmp_path / ".iphone"
    d.mkdir(exist_ok=True)
    f = d / "config.toml"
    f.write_text(body, encoding="utf-8")
    f.chmod(mode)
    return f


def test_resolve_default_is_builtin_qwen_from_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-env")
    r = resolve(None, conf={})
    assert isinstance(r, ResolvedModel)
    assert r.spec == "alibaba:qwen3.7-plus" and r.api_key == "sk-env"
    assert r.api_key_source == "env:DASHSCOPE_API_KEY"
    assert r.model.allow_coord_tap is True and r.notices == ()


def test_env_model_spec_bare_name_and_provider_colon(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setenv("IPHONE_USE_MODEL", "qwen-vl-max")
    assert resolve(None, conf={}).spec == "alibaba:qwen-vl-max"
    monkeypatch.setenv("IPHONE_USE_MODEL", "dashscope:qwen-vl-max")   # 别名也归一成正名
    assert resolve(None, conf={}).spec == "alibaba:qwen-vl-max"


def test_unknown_model_defaults_to_no_coords_with_notice(monkeypatch):
    """§1.4：没在表里的模型默认关坐标，并告诉用户怎么打开。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    r = resolve("alibaba:qwen-vl-max", conf={})
    assert r.model.allow_coord_tap is False and r.model.calibrated is False
    assert r.model.extra_body == {}                       # 不继承 qwen3.7-plus 的 enable_thinking
    assert any("未标定" in n and "qualify" in n and "alibaba:qwen-vl-max" in n for n in r.notices)


def test_partial_toml_override_does_not_open_coords(monkeypatch):
    """Codex 审出的坑：用户只写 max_tokens，dataclass 默认值必须仍然是 False。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    conf = {"models": {"alibaba:qwen-vl-max": {"max_tokens": 2048}}}
    r = resolve("alibaba:qwen-vl-max", conf=conf)
    assert r.model.max_tokens == 2048 and r.model.allow_coord_tap is False


def test_toml_model_override_and_custom_provider(monkeypatch):
    _clear_env(monkeypatch)
    conf = {
        "providers": {"my-vllm": {"base_url": "http://10.0.0.5:8000/v1", "api_key": "x",
                                  "omit_params": ["parallel_tool_calls"]}},
        "models": {"my-vllm:qwen2.5-vl-72b": {"coord_mode": "pixel", "allow_coord_tap": True}},
    }
    r = resolve("my-vllm:qwen2.5-vl-72b", conf=conf)
    assert r.provider.base_url == "http://10.0.0.5:8000/v1"
    assert r.provider.omit_params == frozenset({"parallel_tool_calls"})
    assert r.provider.env_vars == ("MY_VLLM_API_KEY",)
    assert r.api_key == "x" and r.api_key_source == "toml:[providers.my-vllm]"
    assert r.model.coord_mode == "pixel" and r.model.allow_coord_tap is True
    assert r.notices == ()


def test_custom_provider_env_key_wins_over_toml(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MY_VLLM_API_KEY", "from-env")
    conf = {"providers": {"my-vllm": {"base_url": "http://h/v1", "api_key": "from-toml"}}}
    r = resolve("my-vllm:m", conf=conf)
    assert r.api_key == "from-env" and r.api_key_source == "env:MY_VLLM_API_KEY"
    monkeypatch.delenv("MY_VLLM_API_KEY")


def test_custom_provider_requires_base_url(monkeypatch):
    _clear_env(monkeypatch)
    with pytest.raises(ConfigError) as ei:
        resolve("my-vllm:m", conf={"providers": {"my-vllm": {"api_key": "x"}}})
    assert "base_url" in str(ei.value) and "my-vllm" in str(ei.value)


def test_old_toml_format_still_works(monkeypatch):
    """顶层 api_key / base_url / model 三个键 = 旧格式，原样能用，不打警告。"""
    _clear_env(monkeypatch)
    conf = {"api_key": "sk-old", "base_url": "https://old/v1", "model": "qwen3.7-plus"}
    r = resolve(None, conf=conf)
    assert r.spec == "alibaba:qwen3.7-plus" and r.api_key == "sk-old"
    assert r.api_key_source == "toml:[providers.alibaba]" and r.provider.base_url == "https://old/v1"
    assert r.notices == ()


def test_provider_alias_and_canonical_name_are_interchangeable(monkeypatch):
    """别名不是二等公民：按别名点名的模型要认正名下的表，按别名写的表也要认。"""
    _clear_env(monkeypatch)
    conf = {"api_key": "sk-old", "model": "dashscope:qwen3.7-plus"}   # 旧格式顶层键 + 别名 spec
    assert resolve(None, conf=conf).api_key == "sk-old"
    conf = {"providers": {"dashscope": {"base_url": "https://alias/v1", "api_key": "sk-a"}}}
    r = resolve("alibaba:qwen3.7-plus", conf=conf)
    assert r.provider.base_url == "https://alias/v1" and r.api_key == "sk-a"
    assert r.api_key_source == "toml:[providers.alibaba]"


def test_model_table_key_may_use_provider_alias(monkeypatch):
    """[models."dashscope:x"] 与 [models."alibaba:x"] 等价：写别名的表不该被静默丢掉。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    conf = {"models": {"dashscope:qwen-vl-max": {"allow_coord_tap": True, "coord_mode": "pixel"}}}
    r = resolve("dashscope:qwen-vl-max", conf=conf)
    assert r.spec == "alibaba:qwen-vl-max"
    assert r.model.coord_mode == "pixel" and r.model.allow_coord_tap is True
    assert r.notices == ()                       # 配了就不该再收到「未标定」的提示
    # 反向：spec 写正名、表写别名，同样认。
    assert resolve("alibaba:qwen-vl-max", conf=conf).model.allow_coord_tap is True


def test_model_table_key_alias_still_requires_explicit_allow(monkeypatch):
    """别名表生效不等于坐标默认打开：没写 allow_coord_tap 就仍然是关的。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    conf = {"models": {"dashscope:qwen-vl-max": {"max_tokens": 2048}}}
    r = resolve("alibaba:qwen-vl-max", conf=conf)
    assert r.model.max_tokens == 2048 and r.model.allow_coord_tap is False
    assert any("未标定" in n for n in r.notices)


def test_model_table_key_alias_typo_is_reported(monkeypatch):
    """别名表也要过键名检查：以前它被忽略，里面的错字也跟着看不见。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    with pytest.raises(ConfigError) as ei:
        resolve("alibaba:qwen-vl-max", conf={"models": {"dashscope:qwen-vl-max": {"allow_cord_tap": True}}})
    assert "allow_cord_tap" in str(ei.value)


def test_two_provider_tables_for_one_provider_is_an_error(monkeypatch):
    """正名和别名各写一张表 —— 合并没有唯一解，静默挑一张就丢了另一半配置。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    conf = {"providers": {"alibaba": {"base_url": "https://a/v1"}, "dashscope": {"extra_body": {"x": 1}}}}
    with pytest.raises(ConfigError) as ei:
        resolve("alibaba:qwen3.7-plus", conf=conf)
    msg = str(ei.value)
    assert "[providers.alibaba]" in msg and "[providers.dashscope]" in msg


def test_two_model_tables_for_one_spec_is_an_error(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    conf = {"models": {"alibaba:qwen-vl-max": {"max_tokens": 2048},
                       "dashscope:qwen-vl-max": {"coord_mode": "pixel"}}}
    with pytest.raises(ConfigError) as ei:
        resolve("alibaba:qwen-vl-max", conf=conf)
    msg = str(ei.value)
    assert '[models."alibaba:qwen-vl-max"]' in msg and '[models."dashscope:qwen-vl-max"]' in msg


def test_bad_provider_value_type_is_a_config_error(monkeypatch):
    """default_headers = "abc" 该是一句人话，不是 dict("abc") 的 traceback。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    with pytest.raises(ConfigError) as ei:
        resolve("alibaba:qwen3.7-plus", conf={"providers": {"alibaba": {"default_headers": "abc"}}})
    assert "[providers.alibaba]" in str(ei.value)


def test_env_model_spec_beats_toml_model(monkeypatch):
    """优先级：环境变量 > config.toml > 内置。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("IPHONE_USE_MODEL", "deepseek:deepseek-v3")
    assert resolve(None, conf={"model": "alibaba:qwen3.7-plus"}).spec == "deepseek:deepseek-v3"


def test_toml_model_key_with_colon_is_not_double_prefixed(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    assert resolve(None, conf={"model": "deepseek:deepseek-v3"}).spec == "deepseek:deepseek-v3"


def test_dashscope_base_url_env_overrides_alibaba(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://proxy/v1")
    assert resolve("alibaba:qwen3.7-plus", conf={}).provider.base_url == "https://proxy/v1"


def test_legacy_coord_mode_env_is_top_priority_and_opens_coords(monkeypatch):
    """IPHONE_USE_COORD_MODE 曾直接控制生产行为，不能静默忽略。设了就当用户为它负责。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setenv("IPHONE_USE_COORD_MODE", "pixel")
    r = resolve("alibaba:qwen-vl-max", conf={})
    assert r.model.coord_mode == "pixel" and r.model.allow_coord_tap is True
    assert any("IPHONE_USE_COORD_MODE" in n and "coord_mode" in n for n in r.notices)
    monkeypatch.setenv("IPHONE_USE_COORD_MODE", "percent")
    with pytest.raises(ConfigError):
        resolve("alibaba:qwen-vl-max", conf={})


def test_unknown_provider_lists_known(monkeypatch):
    _clear_env(monkeypatch)
    with pytest.raises(ConfigError) as ei:
        resolve("nope:m", conf={})
    assert "nope" in str(ei.value) and "alibaba" in str(ei.value) and "openrouter" in str(ei.value)


def test_missing_key_names_env_and_toml_path(monkeypatch):
    _clear_env(monkeypatch)
    with pytest.raises(ConfigError) as ei:
        resolve("deepseek:deepseek-v3", conf={})
    msg = str(ei.value)
    assert "DEEPSEEK_API_KEY" in msg and "[providers.deepseek]" in msg and "config.toml" in msg


def test_need_key_false_skips_key(monkeypatch):
    """screen / tap / doctor 这些不碰模型的命令也要能拿到 coord_mode，不该被「没 key」拦住。"""
    _clear_env(monkeypatch)
    r = resolve("deepseek:deepseek-v3", conf={}, need_key=False)
    assert r.api_key == "" and r.api_key_source == "none" and r.model.coord_mode == "norm1000"


def test_non_ascii_key_rejected(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "你的key")
    with pytest.raises(ConfigError) as ei:
        resolve(None, conf={})
    assert "非 ASCII" in str(ei.value) and "占位符" in str(ei.value)


def test_supports_tools_false_refuses(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    conf = {"models": {"alibaba:x": {"supports_tools": False}}}
    with pytest.raises(ConfigError) as ei:
        resolve("alibaba:x", conf=conf)
    assert "工具调用" in str(ei.value)


def test_bad_coord_mode_in_toml_rejected_at_resolve(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    with pytest.raises(ConfigError) as ei:
        resolve("alibaba:x", conf={"models": {"alibaba:x": {"coord_mode": "percent"}}})
    assert "coord_mode" in str(ei.value)


def test_unknown_toml_key_rejected(monkeypatch):
    """打错字不该静默：allow_cord_tap 不是字段。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    with pytest.raises(ConfigError) as ei:
        resolve("alibaba:x", conf={"models": {"alibaba:x": {"allow_cord_tap": True}}})
    assert "allow_cord_tap" in str(ei.value)


def test_overlay_does_not_leak_between_resolves(monkeypatch):
    """内置表只读：上一次的 conf 不能残留到下一次。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    resolve("alibaba:qwen-vl-max", conf={"models": {"alibaba:qwen-vl-max": {"allow_coord_tap": True}}})
    assert resolve("alibaba:qwen-vl-max", conf={}).model.allow_coord_tap is False
    assert "alibaba:qwen-vl-max" not in registry.MODELS


# ---- 文件读取：权限与格式 ----

def test_read_config_file_accepts_owner_only_modes(tmp_path, monkeypatch):
    """规则是禁止 group/other 位，不是只认 0600：0400 也是只有本人可读。"""
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, 'api_key = "sk-x"\n', mode=0o400)
    assert read_config_file()["api_key"] == "sk-x"


def test_read_config_file_refuses_group_readable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, 'api_key = "sk-x"\n', mode=0o644)
    with pytest.raises(ConfigError) as ei:
        read_config_file()
    assert "chmod 600" in str(ei.value)


def test_read_config_file_missing_is_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert read_config_file() == {}


def test_read_config_file_broken_toml_names_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "这不是 toml ][\n")
    with pytest.raises(ConfigError) as ei:
        read_config_file()
    assert "config.toml" in str(ei.value)


def test_resolve_reads_file_when_conf_not_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _clear_env(monkeypatch)
    _write(tmp_path, 'model = "deepseek:deepseek-v3"\n[providers.deepseek]\napi_key = "sk-f"\n')
    r = resolve()
    assert r.spec == "deepseek:deepseek-v3" and r.api_key == "sk-f"


# ---- 顶层键与类型：错字和坏类型都不许悄悄溜过去 ----

def test_unknown_top_level_key_rejected(monkeypatch):
    """顶层也要查错字：写成 mdoel 的人以前会静默拿到默认模型，配置整个被丢掉。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    with pytest.raises(ConfigError) as ei:
        resolve(None, conf={"mdoel": "deepseek:deepseek-v3"})
    assert "mdoel" in str(ei.value) and "顶层" in str(ei.value)


def test_empty_conf_and_all_legal_top_keys_pass(monkeypatch):
    """空配置不该被这道检查绊住；合法的顶层键一个都不能误伤（含旧格式的两个）。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    assert resolve(None, conf={}).spec == registry.DEFAULT_MODEL_SPEC
    conf = {"model": "alibaba:qwen3.7-plus", "api_key": "sk", "base_url": "https://x/v1",
            "providers": {}, "models": {}}
    assert resolve(None, conf=conf).spec == "alibaba:qwen3.7-plus"


def test_bad_model_value_type_is_a_config_error(monkeypatch):
    """extra_body = 5：以前 dataclasses.replace 抛的 TypeError 直接糊成一坨堆栈给用户。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    with pytest.raises(ConfigError) as ei:
        resolve("alibaba:x", conf={"models": {"alibaba:x": {"extra_body": 5}}})
    assert "TypeError" in str(ei.value) and 'models."alibaba:x"' in str(ei.value)


def test_api_key_not_in_repr(monkeypatch):
    """repr 一展开就落进日志和聊天记录 —— 这个项目就这么泄过一次。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-secret-value")
    r = resolve(None, conf={})
    assert "sk-secret-value" not in repr(r) and r.api_key == "sk-secret-value"
