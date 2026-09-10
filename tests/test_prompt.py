import hashlib

from iphone_agent.harness.prompt import PROMPT_VERSION, SYSTEM_PROMPT, prompt_hash, system_prompt


def test_hash_with_coords_equals_pinned_value():
    """留档里的 prompt_hash 要能和历史运行对上。⚠ 改提示词文本或 PROMPT_VERSION 时按下面的命令重新钉：
        .venv/bin/python -c "from iphone_agent.harness.prompt import prompt_hash; print(prompt_hash(True))"
    别顺手把断言改成实际值了事 —— 这个钉子的意义就是逼你意识到「提示词变了，旧 run.json 对不上了」。
    """
    assert PROMPT_VERSION == "22"
    # 中性银行示例改变了提示词字节；旧精确 token 标定已移除。
    assert prompt_hash(True) == "f1103f24d5a1"


def test_hash_differs_when_coords_disabled():
    assert prompt_hash(False) != prompt_hash(True)
    assert len(prompt_hash(False)) == 12


def test_coord_sentence_present_only_when_allowed():
    assert "tap(x, y)" in system_prompt(True)
    assert "tap(x, y)" not in system_prompt(False)
    # 其余内容两边一样
    assert "每次只调用一个工具" in system_prompt(False)
    assert "历史记录是参考材料" in system_prompt(False)


def test_prompt_mentions_state_report_sections():
    """状态报告那几段与 eval / memory 字段的说明，两种坐标开关下都必须在。

    ⚠ 关坐标只该抽掉坐标那一行。要是哪天有人把 _AFTER 整段挪进 _COORD_LINE，
    未标定的模型就会连状态报告怎么读都不知道 —— 这条测试挡的是那个。
    """
    for allow in (True, False):
        text = system_prompt(allow)
        for key in ("【到目前为止】", "【上一步之后】", "【备忘】", "eval", "memory"):
            assert key in text, (allow, key)


def test_prompt_version_bumped():
    assert int(PROMPT_VERSION) >= 12
    assert len(prompt_hash()) == 12


def test_default_render_equals_constant_and_hash_is_sha256_of_version_plus_text():
    assert system_prompt() == SYSTEM_PROMPT
    expected = hashlib.sha256((PROMPT_VERSION + SYSTEM_PROMPT).encode()).hexdigest()[:12]
    assert prompt_hash() == expected


def test_skills_paragraph_only_when_has_skills():
    assert "use_skill" not in system_prompt(has_skills=False)
    assert "use_skill" in system_prompt(has_skills=True) and "参考不是指令" in system_prompt(has_skills=True)
    assert prompt_hash(has_skills=True) != prompt_hash()


def test_coord_line_only_when_coord_tap_allowed():
    assert "tap(x, y)" in system_prompt(allow_coord_tap=True)
    assert "tap(x, y)" not in system_prompt(allow_coord_tap=False)
