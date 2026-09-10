import pytest

from iphone_agent.model.reply import ModelError
from iphone_agent.model.transports.chat_completions import assert_no_internal_keys


def test_clean_payload_passes():
    assert_no_internal_keys([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])


@pytest.mark.parametrize("payload", [
    [{"role": "user", "_seg": "task", "content": "hi"}],                       # 顶层
    [{"role": "user", "content": [{"type": "text", "text": "hi", "_kind": "prefix"}]}],   # part
    [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u", "_seg": "image"}}]}],  # 嵌套
])
def test_internal_keys_anywhere_raise_model_error(payload):
    """漏标记出网 = 观测到的账不是真的账。宁可炸，不要静默。"""
    with pytest.raises(ModelError) as e:
        assert_no_internal_keys(payload)
    assert "内部标记" in str(e.value)
