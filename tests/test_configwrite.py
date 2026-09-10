"""往 config.toml 里写。

这一层的价值全在**安全**上，所以测试也压在那儿：
- 写出来的文件必须是 0600，而且中途不能有更松的窗口；
- 写出来的东西必须能被 read_config_file 读回来（不能造出一个自己会拒绝的文件）；
- 报错里一个字都不能带密钥。
"""
import os
import stat

import pytest

from iphone_agent.model.config import read_config_file, resolve
from iphone_agent.model.configwrite import (
    dumps,
    set_api_key,
    set_model,
    update_config,
    write_config_file,
)
from iphone_agent.model.errors import ConfigError

SECRET = "sk-never-appear-in-an-error-9876543210"


@pytest.fixture
def cfg(tmp_path):
    return tmp_path / ".iphone" / "config.toml"


# ── 权限 ────────────────────────────────────────────────────────────────

def test_written_file_is_0600(cfg):
    write_config_file({"model": "alibaba:qwen3.7-plus"}, cfg)
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600


def test_parent_directory_is_locked_down_too(cfg):
    write_config_file({"model": "x:y"}, cfg)
    assert stat.S_IMODE(cfg.parent.stat().st_mode) == 0o700


def test_round_trips_through_the_reader(cfg):
    """写出来的东西必须读得回来 —— 读那一侧会拒绝权限太松的文件，不能自相矛盾。"""
    data = {"model": "deepseek:deepseek-v3",
            "providers": {"deepseek": {"api_key": SECRET, "base_url": "https://x/v1"}},
            "models": {"deepseek:deepseek-v3": {"coord_mode": "pixel", "allow_coord_tap": True}}}
    write_config_file(data, cfg)
    assert read_config_file(cfg) == data


def test_overwriting_keeps_permissions_tight(cfg):
    write_config_file({"model": "a:b"}, cfg)
    write_config_file({"model": "c:d"}, cfg)
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600
    assert read_config_file(cfg)["model"] == "c:d"


def test_refuses_to_silently_overwrite_a_world_readable_file(cfg):
    """已经存在一个 644 的密钥文件时，读那一侧会报错 —— 写这一侧不能绕过它。"""
    cfg.parent.mkdir(parents=True)
    cfg.write_text('model = "a:b"\n', encoding="utf-8")
    cfg.chmod(0o644)
    with pytest.raises(ConfigError, match="chmod 600"):
        update_config(cfg, model="c:d")


def test_no_partial_file_when_serialization_fails(cfg):
    """内容不合法就别去动磁盘上那份 —— 半个配置比没有配置更难查。"""
    write_config_file({"model": "good:one"}, cfg)
    with pytest.raises(ConfigError):
        write_config_file({"bad": object()}, cfg)
    assert read_config_file(cfg)["model"] == "good:one"
    leftovers = [p for p in cfg.parent.iterdir() if p.name.startswith(".config-")]
    assert not leftovers, "临时文件没收干净"


# ── 密钥不外泄 ──────────────────────────────────────────────────────────

def test_non_ascii_key_is_rejected_without_echoing_it(cfg):
    with pytest.raises(ConfigError) as ei:
        set_api_key("alibaba", "你的key", cfg)
    assert "你的key" not in str(ei.value)
    assert "非 ASCII" in str(ei.value)


def test_clearing_the_key_falls_back_to_env(cfg, monkeypatch):
    set_api_key("alibaba", SECRET, cfg)
    assert read_config_file(cfg)["providers"]["alibaba"]["api_key"] == SECRET
    set_api_key("alibaba", None, cfg)
    assert "providers" not in read_config_file(cfg), "空表不该留在文件里"


def test_key_written_here_is_what_resolve_picks_up(cfg, monkeypatch, tmp_path):
    """端到端：设置页存的密钥，下一次 resolve 要真的用得上。"""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("IPHONE_USE_MODEL", raising=False)
    set_api_key("alibaba", SECRET, cfg)
    r = resolve(conf=read_config_file(cfg))
    assert r.api_key == SECRET
    assert r.api_key_source == "toml:[providers.alibaba]"


def test_env_still_wins_over_the_file(cfg, monkeypatch):
    """环境变量优先级更高这条不能因为有了设置页就变 —— 脚本和 CI 靠它临时覆盖。"""
    set_api_key("alibaba", SECRET, cfg)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "from-env")
    r = resolve(conf=read_config_file(cfg))
    assert r.api_key == "from-env"


# ── 换模型 ──────────────────────────────────────────────────────────────

def test_set_model_validates_before_writing(cfg):
    with pytest.raises(ConfigError):
        set_model("", cfg)
    assert not cfg.exists(), "校验没过就不该建出文件来"


def test_set_model_only_touches_the_model_key(cfg):
    set_api_key("alibaba", SECRET, cfg)
    set_model("deepseek:deepseek-v3", cfg)
    d = read_config_file(cfg)
    assert d["model"] == "deepseek:deepseek-v3"
    assert d["providers"]["alibaba"]["api_key"] == SECRET, "换模型不该把密钥冲掉"


# ── 序列化 ──────────────────────────────────────────────────────────────

def test_model_specs_get_quoted_because_they_contain_colons():
    out = dumps({"models": {"alibaba:qwen3.7-plus": {"coord_mode": "pixel"}}})
    assert '[models."alibaba:qwen3.7-plus"]' in out


def test_bool_is_not_written_as_int():
    """Python 里 True 是 int 的实例 —— 判断顺序反了就会写出 1，TOML 那边类型就变了。"""
    out = dumps({"models": {"a:b": {"allow_coord_tap": True, "max_tokens": 1}}})
    assert "allow_coord_tap = true" in out
    assert "max_tokens = 1" in out


def test_strings_with_quotes_and_backslashes_survive(cfg):
    weird = 'he said "hi"\\n\tand\ttabs'
    write_config_file({"providers": {"p": {"base_url": weird}}}, cfg)
    assert read_config_file(cfg)["providers"]["p"]["base_url"] == weird


def test_header_tells_hand_editors_that_comments_are_lost():
    assert "注释" in dumps({"model": "a:b"})


def test_dumps_rejects_unserializable_values():
    with pytest.raises(ConfigError):
        dumps({"x": {1, 2}})
    with pytest.raises(ConfigError):
        dumps({"x": float("inf")})


def test_update_config_can_delete_a_key(cfg):
    update_config(cfg, model="a:b")
    update_config(cfg, model=None)
    assert "model" not in read_config_file(cfg)


def test_writing_into_a_missing_directory_creates_it(cfg):
    assert not cfg.parent.exists()
    write_config_file({"model": "a:b"}, cfg)
    assert cfg.exists() and os.access(cfg, os.R_OK)
