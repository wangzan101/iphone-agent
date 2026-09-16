"""系统提示词的七节，每节一条测试钉它的**意图**。

不钉短语 —— 2026-09-09 改版本 20 时三条钉着旧措辞的测试红了，意图明明没变。
钉意图的办法：每节该回答一个问题，测试就问那个问题；措辞怎么改都行，答案变了才红。
"""
from iphone_agent.harness.prompt import SECTIONS, section, system_prompt


def test_seven_sections_in_this_order():
    """顺序本身是设计：最要紧的（身份、元原则）在开头，不可逆和信任边界在结尾。"""
    assert SECTIONS == ("who", "world", "ios", "mirror", "hands", "method", "see")
    text = system_prompt()
    pos = [text.find(section(n)[:40]) for n in SECTIONS]
    assert pos == sorted(pos) and -1 not in pos


def test_who_says_what_counts_as_success():
    """身份那一节要说清：假成功比失败糟，不可逆的事只能一次做对。"""
    w = section("who")
    assert "done(success)" in w and "done(failed)" in w
    assert "假成功" in w or "必须是真的" in w
    assert "不可逆" in w


def test_world_says_kind_is_a_guess_and_lists_what_a_tap_can_do():
    """世界模型：感知是猜测；点了之后的几种结果，每种都要说**怎么从工具返回认出来**。"""
    w = section("world")
    assert "猜测" in w and "不是事实" in w
    for outcome in ("没反应", "跳到新页面", "原地切换", "弹出", "键盘", "跳出", "不可逆"):
        assert outcome in w, outcome
    assert "appeared" in w and "local_mad" in w
    # 元原则：同位置不同态、相似不同功能、做对的标志
    assert "同一个位置" in w and "长得一样" in w and "做对的标志" in w


def test_ios_conventions_hedge_and_defer_to_the_screen():
    """iOS 惯例那一节写的是惯例不是规则 —— 必须留余地，并说清以当前屏幕为准。"""
    i = section("ios")
    assert "通常" in i or "多半" in i
    assert "以当前屏幕为准" in i
    # 对点击任务最要紧的几条
    assert "›" in i and "开关" in i and "恢复上次" in i


def test_mirror_section_corrects_ios_common_sense():
    """镜像差异那一节：模型的 iOS 常识里在这里不成立的，每条要给替代做法。"""
    m = section("mirror")
    assert "没有手势" in m or "做不到" in m
    assert "编辑" in m, "左滑删除做不到，得说用「编辑」"
    assert "open_app" in m, "主屏翻不了页，得说用 open_app"
    assert "相机" in m and "麦克风" in m


def test_hands_cover_every_tool_the_model_can_call():
    """手那一节：17 个工具，模型能调的每一个都得在提示词里露过面。
    2026-09-09 zoom/erase/key(return) 加了一天提示词里一个字没提，模型就不用。"""
    from iphone_agent.harness.actions import BUILTIN_ACTION_NAMES

    h = section("hands")
    for tool in BUILTIN_ACTION_NAMES:
        if tool in ("observe", "switch_ime", "wait", "use_skill"):
            continue        # 自动做 / 罕用 / 自解释 / 只在有剧本时才注入（_SKILLS）
        assert tool in h, f"工具 {tool} 在「你的手」里没提到"


def test_hands_keep_tab_bars_out_of_icon_above():
    """纯图标（tab 栏）不能用 icon_above —— 真机踩过三次。"""
    h = section("hands")
    icon_above_para = next(p for p in h.split("· ") if 'target="icon_above"' in p)
    assert "tab 栏" not in icon_above_para
    assert "绝对不要" in h and "乱码" in h


def test_method_says_verify_each_step_and_when_to_stop():
    m = section("method")
    assert "先认位置" in m or "先看清" in m
    assert "appeared" in m, "每步做完要看 appeared，这是「做对的标志」在方法层的落实"
    assert "done(failed)" in m, "得说清什么时候放弃"


def test_see_explains_report_sections_and_trust_boundary():
    s = section("see")
    for key in ("【到目前为止】", "【上一步之后】", "【备忘】", "eval", "memory"):
        assert key in s, key
    assert "不能授权" in s and "以当前屏幕为准" in s


def test_coord_line_only_lives_in_hands_and_world():
    """2026-09-14：world 一节也提坐标这条路（zoom → 坐标 → observe，spec §11），
    这是设计变化，不是旧假设的例外 —— 别的节仍然不该提它。"""
    assert "tap(x, y)" in section("hands", allow_coord_tap=True)
    assert "tap(x, y)" not in section("hands", allow_coord_tap=False)
    assert "tap(x, y)" in section("world", allow_coord_tap=True)
    assert "tap(x, y)" not in section("world", allow_coord_tap=False)
    for n in SECTIONS:
        if n not in ("hands", "world"):
            assert "tap(x, y)" not in section(n)


def test_no_length_cap_but_no_app_specific_knowledge():
    """2026-09-10 用户决定：**不设字数上限**，字数限制是错误的设计 —— 为了凑上限删过有用的句子。
    去掉长度断言后，这条测试只守它原本想守的那件事：提示词里不放 App 特定知识（那是 skills 的事）。"""
    text = system_prompt(True, True)
    for app_name in ("记账本", "微信", "小红书", "备忘录", "淘宝"):
        assert app_name not in text, f"提示词里出现了具体 App「{app_name}」，该下沉到 skills/app_note"


def test_start_point_is_uncertain_and_the_way_back_to_root_is_taught():
    """起点不确定：App 从上次离开的那页恢复，open_app 成功不等于在首页。
    三节各管一段：ios 节教「回根的通用路径」，world 节把它列为 open_app 的一种结果，
    method 节把「先认位置」变成第一步。2026-09-09：记账本停在记账页，下一次任务在那页上迷路。"""
    i = section("ios")
    assert "恢复" in i and "不退出" in i
    for way in ("<", "tab", "取消", "键盘"):
        assert way in i, f"回根路径缺了 {way}"
    assert "open_app" in section("world") and "回根" in section("world")
    m = section("method")
    assert "根页面" in m and "返回箭头" in m


def test_method_says_read_current_value_before_writing():
    """写前先读：恢复出来的页面上，金额/输入框/开关/选中项可能已经有值。
    2026-09-09 真机：记账页留着金额 5，模型 reason 写「当前金额应为 0.00」，点 1 成了 51。"""
    m = section("method")
    assert "状态" in m and "先读" in m
    for thing in ("金额", "输入框", "开关", "选中"):
        assert thing in m, thing


def test_hands_teach_that_the_cursor_is_invisible_and_how_to_move_it():
    """光标看不见、点文字落在那个字上、行末用 key(line_end)。09-10 备忘录 27 步里 16 步在搏斗光标。"""
    h = section("hands")
    assert "光标" in h and "line_end" in h and "不是行末" in h
    assert "光标" in section("ios")


def test_world_says_the_list_is_ocr_only_by_default_and_names_the_ways_out():
    """spec 2026-09-14 §4.2：如实说元素表缺什么；zoom 放在坐标前面（§11），observe 是慢的全屏。"""
    w = section("world", True)
    assert "默认只有 OCR" in w and "以截图为准" in w
    assert w.index("zoom 那一块") < w.index("tap(x, y)") < w.index("observe 看全屏")
    w0 = section("world", False)
    assert "tap(x, y)" not in w0 and "zoom 那一块" in w0 and "observe 看全屏" in w0


def test_hands_say_observe_is_the_slow_full_screen_look():
    h = section("hands")
    assert "observe 看全屏" in h and "旧编号作废" in h
