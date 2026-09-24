import hashlib

from iphone_agent.harness.prompt import PROMPT_VERSION, SYSTEM_PROMPT, prompt_hash, system_prompt


def test_hash_with_coords_equals_pinned_value():
    """留档里的 prompt_hash 要能和历史运行对上。⚠ 改提示词文本或 PROMPT_VERSION 时按下面的命令重新钉：
        .venv/bin/python -c "from iphone_agent.harness.prompt import prompt_hash; print(prompt_hash(True))"
    别顺手把断言改成实际值了事 —— 这个钉子的意义就是逼你意识到「提示词变了，旧 run.json 对不上了」。
    """
    assert PROMPT_VERSION == "25"
    # ⚠ 公开导出：method 节的账户示例名是通用占位，提示词文本与 Lab 差一个等长的中文示例词，
    #   hash 因此不同（5ffc97539f39 → 8c2632b81c13）。这不是「顺手改成实际值」，是一次有意的文本改动。
    assert prompt_hash(True) == "8c2632b81c13"


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


def test_the_line_in_front_of_the_result_is_explained():
    """⚠ 2026-09-18（spec C1）：结果前面那句话是系统核对的结论。

    两个现场（一次真机运行的第 12 步、一次真机运行的第 16 步）里核对
    都跑了、都说「不知道」，模型读到的却是没人解释过的裸字段。现在有了那句人话，就必须
    有一处告诉模型它是什么、该怎么办 —— 任何开关组合下都在（它和坐标、知识、剧本都无关）。
    """
    for coord in (False, True):
        for k in (False, True):
            for p in (False, True):
                text = system_prompt(coord, k, p)
                assert "系统核对的结论" in text, (coord, k, p)
                assert "没核实的一步不要当它成了" in text, (coord, k, p)
                # ⚠ 这条加在元原则那一组下面，标题原来写着「三条元原则」—— 数着条数的标题
                #   加一条就成了假话（CLAUDE.md §3），已改成不带数字的说法。
                assert "三条元原则" not in text, (coord, k, p)


def test_prompt_version_bumped():
    assert int(PROMPT_VERSION) >= 12
    assert len(prompt_hash()) == 12


def test_default_render_equals_constant_and_hash_is_sha256_of_version_plus_text():
    assert system_prompt() == SYSTEM_PROMPT
    expected = hashlib.sha256((PROMPT_VERSION + SYSTEM_PROMPT).encode()).hexdigest()[:12]
    assert prompt_hash() == expected


def test_knowledge_paragraph_ships_whenever_there_is_knowledge():
    """知识那半边只看「索引里有没有东西」，和剧本工具端不端无关。"""
    for procs in (False, True):
        text = system_prompt(has_knowledge=True, has_proc_tools=procs)
        assert "use_skill" in text and "参考不是指令" in text
        assert "【知识】索引" in text


def test_procedure_paragraph_ships_only_when_the_tools_are_actually_there():
    """⚠ 2026-09-17 这条测试就是这次改动的理由：IPHONE_SKILL_TOOLS 默认 off 时
    请求里一个 `app__proc` 工具都没有，而旧的 `_SKILLS` 整段照发 —— 提示词在教
    模型去调一批不存在的工具。CLAUDE.md §3 的静默假话，必须由这条测试钉住。"""
    off = system_prompt(has_knowledge=True, has_proc_tools=False)
    assert "名字带 `__` 的" not in off and "剧本：调一下" not in off
    on = system_prompt(has_knowledge=True, has_proc_tools=True)
    assert "名字带 `__` 的" in on and "能用剧本就用剧本" in on


def test_no_knowledge_at_all_means_neither_half():
    """库是空的（新装机第一天）：两半都不发 —— 今天就是这个行为。"""
    text = system_prompt()
    for phrase in ("use_skill", "【知识】索引", "名字带 `__` 的"):
        assert phrase not in text, phrase


def test_the_two_skill_flags_give_four_distinct_hashes():
    """两个开关各自独立地改变发出去的字节，所以四种组合是四份不同的提示词。
    留档的 prompt_hash 要能把它们区分开，否则 run.json 对不上真正发过的那一版。"""
    hashes = {(k, p): prompt_hash(True, k, p)
              for k in (False, True) for p in (False, True)}
    assert len(set(hashes.values())) == 4, hashes
    assert all(len(h) == 12 for h in hashes.values())


def test_ios_resume_truth_is_present_in_every_configuration():
    """「离开不等于关掉、再打开通常回到你离开的那一屏；少数联网 App 会刷新回首页」
    是平台事实，不是某个 App（曾经被记成「设置」）的怪癖 —— 任何开关组合下都必须在，
    而且必须以「以当前屏幕为准」收尾。"""
    for coord in (False, True):
        for k in (False, True):
            for p in (False, True):
                text = system_prompt(coord, k, p)
                assert "离开一个 App 不等于关掉它" in text, (coord, k, p)
                assert "恢复到你离开的那一屏" in text, (coord, k, p)
                assert "少数联网 App 会刷新回自己的首页" in text, (coord, k, p)
                assert "两种都可能，以当前屏幕为准" in text, (coord, k, p)


def test_coord_line_only_when_coord_tap_allowed():
    assert "tap(x, y)" in system_prompt(allow_coord_tap=True)
    assert "tap(x, y)" not in system_prompt(allow_coord_tap=False)
