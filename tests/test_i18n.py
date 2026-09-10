"""Language boundaries: UI metadata must not change phone data or approval semantics."""
import json
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from iphone_agent import health
from iphone_agent.harness import safety
from iphone_agent.harness.executor import in_spotlight, looks_like_home
from iphone_agent.model.config import resolve
from iphone_agent.model.errors import ConfigError
from iphone_agent.skills.model import scenario_risk
from iphone_agent.twin.layout import _is_label
from iphone_agent.web import api
from iphone_agent.web.server import Chat


def observation(*labels):
    return SimpleNamespace(elements=[SimpleNamespace(text=text) for text in labels])


@pytest.mark.parametrize("labels", [
    ("最佳搜索结果", "Siri 建议"),
    ("Top Hit", "Siri Suggestions"),
    ("TOP HITS", "SEARCH IN APPS"),
])
def test_spotlight_markers_cover_chinese_and_english(labels):
    assert in_spotlight(observation(*labels))


@pytest.mark.parametrize("labels", [
    ("设置", "备忘录", "时钟", "Safari", "App Store"),
    ("Settings", "Notes", "Clock", "Safari", "App Store"),
])
def test_home_markers_cover_chinese_and_english(labels):
    assert looks_like_home(observation(*labels))


def test_one_mixed_label_does_not_count_as_several_home_icons():
    assert not looks_like_home(observation("设置 Settings Notes Clock"))
    assert not looks_like_home(observation("Settings", "Settings storage", "Settings account"))
    assert not looks_like_home(observation("设置", "Settings", "Safari"))


def test_unrelated_text_does_not_count_as_home_or_spotlight():
    obs = observation("Welcome", "Account", "Help")
    assert not looks_like_home(obs)
    assert not in_spotlight(obs)


@pytest.mark.parametrize("label", ["搜索", "Search", "SEARCH", " search "])
def test_search_control_is_not_an_app_label(label):
    assert not _is_label(label)


@pytest.mark.parametrize("body,expected", [
    ("发送消息", "write"), ("SEND a message", "write"),
    ("保存记录", "write"), ("Save a note", "write"),
    ("删除记录", "irreversible"), ("Delete a record", "irreversible"),
    ("付款", "irreversible"), ("Make a payment", "irreversible"),
    ("Read posture advice", "read"), ("Read repayment history", "read"),
])
def test_skill_risk_supports_both_languages_without_substring_false_positives(body, expected):
    assert scenario_risk("read", [], body) == expected


def test_risk_never_downgrades_an_existing_declaration():
    assert scenario_risk("irreversible", [], "Read a note") == "irreversible"


def test_config_error_keeps_legacy_text_and_structured_message():
    e = ConfigError("原始错误", code="config.emptyModel", params={})
    assert str(e) == "原始错误"
    assert e.as_message() == {"code": "config.emptyModel", "params": {}}


def test_missing_key_has_actionable_message_without_credentials(monkeypatch):
    for name in ("DASHSCOPE_API_KEY", "IPHONE_USE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigError) as caught:
        resolve("alibaba:qwen3.7-plus", conf={})
    assert caught.value.as_message()["code"] == "config.missingKey"
    assert "api_key" not in caught.value.as_message()["params"]


@pytest.mark.parametrize("error,code", [
    ("401 unauthorized", "model.auth"), ("403", "model.forbidden"),
    ("404", "model.endpoint"), ("429", "model.rateLimit"),
    ("timed out", "model.timeout"), ("SSL certificate", "model.tls"),
    ("unknown transport failure", "model.error"),
])
def test_model_error_classification_has_stable_codes(error, code):
    assert api._error_kind(RuntimeError(error)) == code


def test_permission_checks_supply_ui_metadata(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "ApplicationServices", SimpleNamespace(AXIsProcessTrusted=lambda: False))
    check = health.check_accessibility()
    assert check.messages["title"]["code"] == "check.perm.accessibility.title"
    assert check.messages["fix"]["code"] == "check.accessibility.fix"
    assert check.title == "辅助功能权限"  # CLI unchanged.
    json.dumps(check.messages)


def test_confirm_metadata_fans_out_without_changing_approval():
    chat = Chat()
    one, two = chat.subscribe(), chat.subscribe()
    request = safety.ConfirmationRequest("发送 <script>", "用户的原始理由")
    result = []
    worker = threading.Thread(target=lambda: result.append(chat.confirm(request)), daemon=True)
    worker.start()
    try:
        a, b = one.get(timeout=2), two.get(timeout=2)
        assert a == b
        assert a["message"]["params"] == {"target": "发送 <script>", "reason": "用户的原始理由"}
        assert chat.pending_confirm_message == a["message"]
        assert chat.confirm_timeout_remaining > 0
        assert chat.answer(False)
        worker.join(timeout=2)
        assert result == [False]
        assert chat.pending_confirm is None and chat.pending_confirm_message is None
    finally:
        chat.answer(False)
        worker.join(timeout=2)


def test_ui_unit_suite():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the dependency-free UI message tests")
    result = subprocess.run([node, "--test", str(Path(__file__).with_name("web_i18n.test.cjs"))],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
