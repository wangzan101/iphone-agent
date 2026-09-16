"""光标键：一个入口（actions.KEY_NAMES），三处规矩（不计熔断、不去重、changed=None）。

2026-09-10 真机（备忘录另起一行输中文，runs/20260910-094921-d53a）：模型没有挪光标的手，
27 步里 16 步在跟光标搏斗 —— tap 到「明天」上光标落在字中间，回车把一行劈成两行，
type 把字插在中间，再 erase 收拾，三轮。Cmd+←/→ 在镜像里实测通（Spotlight 框得 xabcy）。
"""
from iphone_agent.harness.actions import CURSOR_KEYS, KEY_NAMES, Action, screen_neutral, validate_action
from iphone_agent.harness.guard import ActionGuard


def act(kind, **args):
    return Action(kind, args, "r", None, "c")


def test_key_names_are_the_single_source_for_schema_and_device():
    from iphone_agent.driver.device import Device
    from iphone_agent.harness.tools import tool_defs
    assert set(Device._KEY_COMBOS) == set(KEY_NAMES), "Device 的组合键表和 KEY_NAMES 不一致"
    key_tool = next(t for t in tool_defs() if t["function"]["name"] == "key")
    assert key_tool["function"]["parameters"]["properties"]["name"]["enum"] == list(KEY_NAMES)


def test_cursor_keys_are_cmd_arrows_and_plain_arrows():
    from iphone_agent.driver.device import Device
    from iphone_agent.driver.injector import KEY
    combos = Device._KEY_COMBOS
    assert combos["line_end"] == (KEY["right"], [KEY["cmd"]])
    assert combos["line_start"] == (KEY["left"], [KEY["cmd"]])
    assert combos["text_end"] == (KEY["down"], [KEY["cmd"]])
    assert combos["text_start"] == (KEY["up"], [KEY["cmd"]])
    assert combos["right"] == (KEY["right"], []) and combos["down"] == (KEY["down"], [])


def test_cursor_keys_validate_and_system_keys_still_do():
    for k in KEY_NAMES:
        validate_action(act("key", name=k), None)
    assert all(screen_neutral(act("key", name=k)) for k in CURSOR_KEYS)
    assert not any(screen_neutral(act("key", name=k)) for k in ("home", "spotlight", "return"))


def test_cursor_keys_neither_burn_progress_nor_get_deduped():
    """按 Cmd+→ 画面本来就不变。判无进展会让模型以为没按中；同屏去重会拦第二次 key(right)。"""
    g = ActionGuard()
    g.record_screen(1)
    for _ in range(8):
        assert g.record_outcome(act("key", name="line_end"), changed=False, new_screen_hash=1) is None
    assert g.no_progress == 0
    a = act("key", name="right")
    g.record_executed(a, 1, 800, 1700)
    assert not g.check_repeat(a, 1, 800, 1700), "同屏第二次 key(right) 不能被当成重复动作拒掉"
    h = act("key", name="home")
    g.record_executed(h, 1, 800, 1700)
    assert g.check_repeat(h, 1, 800, 1700), "系统键的去重规矩不变"


def test_executor_reports_cursor_keys_as_screen_neutral(fake_env):
    from iphone_agent.harness.executor import Executor
    dev, per, _ = fake_env([["你好", "明天"]] * 8)
    obs = per.observe(dev.capture())
    res, new = Executor(dev, per).run(act("key", name="line_end"), obs)
    assert res.ok and res.changed is None, res.to_json()
    assert ("key", "line_end") in dev.calls
    assert new is not None and "未检测到变化" not in (res.hint or "")
