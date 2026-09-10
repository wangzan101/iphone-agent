from iphone_agent import config


def test_constants_exist():
    # 保险丝只断言存在且合理，不钉死数值——它们不是设计假设，按任务可覆盖（config.py 注释）。
    assert config.MAX_STEPS >= 30
    assert config.TASK_TIMEOUT_S >= 600
    assert 0 < config.CLAMP_TOLERANCE < 0.1


def test_config_error_alias_still_importable():
    """外部脚本可能还 from iphone_agent.config import ConfigError。"""
    from iphone_agent.model.errors import ConfigError
    assert config.ConfigError is ConfigError
