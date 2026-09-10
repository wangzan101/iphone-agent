"""judge.is_target_app：开 App 之后问看图的那一方「进的是不是这个 App」。

2026-09-10 真机：open_app("设置") 点进了一条备忘录，只因为「离开了 Spotlight」就报 ok。
「离开 Spotlight / 离开主屏」只说明有东西被打开了，说明不了打开的是谁 —— 身份交给会看图的一方。
"""
from PIL import Image

from iphone_agent.harness import judge

IMG = Image.new("RGB", (10, 20), "white")


class _Asker:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def ask_json(self, prompt, images, max_tokens=None):
        self.prompts.append(prompt)
        return self.reply


def test_no_asker_means_not_asked():
    assert judge.is_target_app(IMG, "设置", None) is None


def test_a_reply_without_a_verdict_is_not_a_no():
    """答非所问 / 没回 = 没问成（None），绝不能当成「不是」——
    否则模型抽风一次，就把一个真开成了的 App 判成开错、白走一趟退路。"""
    assert judge.is_target_app(IMG, "设置", _Asker({"worked": True})) is None
    assert judge.is_target_app(IMG, "设置", _Asker(None)) is None
    assert judge.is_target_app(IMG, "设置", _Asker({"is_app": "maybe"})) is None


def test_the_verdict_and_what_it_saw_are_kept():
    a = _Asker({"is_app": False, "confidence": 0.9, "seen": "备忘录里的一条笔记", "why": "顶部是返回和分享"})
    chk = judge.is_target_app(IMG, "设置", a)
    assert chk.is_app is False and chk.seen == "备忘录里的一条笔记" and chk.confidence == 0.9
    assert "要打开的 App 是「设置」" in a.prompts[0]


def test_string_booleans_are_read_as_booleans():
    """bool("false") 是 True —— 模型把布尔值写成字符串时不能被读反。"""
    assert judge.is_target_app(IMG, "设置", _Asker({"is_app": "false"})).is_app is False
    assert judge.is_target_app(IMG, "设置", _Asker({"is_app": "true"})).is_app is True


def test_confidence_is_clamped_and_bad_values_do_not_crash():
    assert judge.is_target_app(IMG, "设置", _Asker({"is_app": True, "confidence": "high"})).confidence == 0.0
    assert judge.is_target_app(IMG, "设置", _Asker({"is_app": True, "confidence": 7})).confidence == 1.0
