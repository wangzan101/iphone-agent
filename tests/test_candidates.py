"""输入法候选栏的识别。

样本全部来自 2026-09-08 的真机探针（截图 runs/ime-q*.png），原样回放 ——
下面每一组 `(文字, 中心点)` 都是当时 OCR 真的吐出来的东西，包括读错的。
图像 624x1388。
"""
from types import SimpleNamespace

from iphone_agent.harness.executor import Executor

H = 1388


def obs(pairs):
    els = [SimpleNamespace(id=i, text=t, center=c) for i, (t, c) in enumerate(pairs)]
    return SimpleNamespace(elements=els, height_px=H, width_px=624)


def cands(pairs):
    return [e.text for e in Executor._candidates(Executor.__new__(Executor), obs(pairs))]


# 打完 shezhi。注意 '10000' 和 '2026/8/25' —— 它们是 Spotlight 的搜索结果，
# 以数字开头又在屏幕下部，老规则会把它们当候选。
Q1 = [('信息', (59, 1057)), ('在App中搜索 a', (501, 1058)),
      ('10000', (190, 1121)), ('2026/8/25', (530, 1125)),
      ('【中国电信】尊敬的客户，截至08月24日23', (359, 1156)),
      ('时50分-你江购套终木日国内诵用浴量20⋯', (366, 1185)),
      ('1设置', (168, 1218)), ('2 摄制', (303, 1217)), ('3 摄制组', (441, 1218)),
      ('Q shezhi', (128, 1283))]

# 紧接着打 weixin
Q2 = [('信息', (59, 1006)), ('在App中搜索', (487, 1007)),
      ('10000', (190, 1071)), ('2026/8/25', (530, 1072)),
      ('【中国电信】 尊敬的客户，截至08月24日23', (359, 1103)),
      ('时59分，您订购套餐本月国内通用流量30…', (367, 1137)),
      ('1微信', (196, 1218)), ('2 威信', (335, 1217)), ('3 违心', (471, 1218)),
      ('设置weixin', (177, 1284))]

# 打 yi。**这一组是老规则彻底失灵的那一组**：OCR 没读到搜索框，
# 候选栏自己成了最底行；而 '1—' 是 OCR 把「一」读成了破折号（截图证实）。
Q3_YI = [('240813群分享《“会说话”和⋯', (334, 980)), ('pdf', (83, 1013)),
         ('382 KB•PDF文稿•2024/8/13', (312, 1015)), ('Q', (559, 1013)),
         ('16:07', (179, 1044)), ('OverloadYield', (236, 1124)),
         ('97字节•JavaScript•2025/8/9', (317, 1156)), ('Q', (556, 1156)),
         ('1—', (153, 1216)), ('2', (229, 1218)), ('3以', (358, 1218)),
         ('4', (444, 1218)), ('V', (541, 1218))]

# 打 mu
Q3_MU = [('显示更多结果', (311, 1160)),
         ('1 幕', (170, 1217)), ('2一亩', (307, 1218)), ('3异母', (446, 1218)),
         ('在Amm', (68, 1244)), ('Q', (77, 1280)), ('yimu', (134, 1283))]


def test_finds_the_real_bar_among_spotlight_results():
    assert cands(Q1) == ['1设置', '2 摄制', '3 摄制组']


def test_number_prefixed_search_results_are_not_candidates():
    """'10000' 和 '2026/8/25' 同处一行、编号看着像 1 和 2 —— 但剥掉编号是
    '0000' 和 '026/8/25'，一个汉字都没有。老规则栽在这里。"""
    just_that_row = [('10000', (190, 1121)), ('2026/8/25', (530, 1125))]
    assert cands(just_that_row) == []


def test_second_word_after_committing_the_first():
    assert cands(Q2) == ['1微信', '2 威信', '3 违心']


def test_works_when_ocr_missed_the_input_box():
    """老规则排除「最底下那一行」（以为那是输入框）。这一组里 OCR 根本没读到
    输入框，候选栏自己成了最底行 —— 老规则把真候选全排除，返回
    ['97字节•JavaScript•2025/8/9']。"""
    got = cands(Q3_YI)
    assert got == ['1—', '2', '3以', '4'], got
    assert 'V' not in got, "展开箭头没有编号，不是候选"


def test_single_char_bar_still_found_even_though_ocr_misread_it():
    """OCR 把「一」读成了破折号。候选栏该照样认出来 ——
    读错的是**内容**，不该连**这是不是候选栏**都判错。"""
    assert cands(Q3_YI)[0] == '1—'


def test_third_word_bar():
    assert cands(Q3_MU) == ['1 幕', '2一亩', '3异母']


def test_no_elements_no_candidates():
    assert cands([]) == []


def test_a_lone_numbered_row_is_not_a_bar():
    """候选栏至少两项。单独一个 '1条新消息' 不是候选栏。"""
    assert cands([('1条新消息', (200, 1250))]) == []


# ---- 2026-09-09 真机：App 内容框（备忘录）里打 'nihao' ----
# 这一组是老规则的死穴。候选栏 OCR 读得好好的，但 y=318/1388=0.229，
# 被 CAND_MIN_Y_RATIO=0.70 整条筛掉 → 判定「不在中文模式」→ toggle_ime 把好好的
# 中文模式切走 → 报 ime_not_chinese。App 内容框 8 次尝试 8 次全挂，就挂在这里。
# 数据是探针原样打印的，包括读错的（'14:114'、'3你还' 少了空格）。
NOTES_NIHAO = [('14:114', (300, 118)), ('74', (560, 117)),
               ('＜ 返回', (60, 182)), ('◎', (300, 183)), ('完成', (570, 183)),
               ('nihao', (100, 257)),
               ('1 你好', (110, 318)), ('2', (230, 318)), ('3你还', (330, 319))]


def test_candidate_bar_in_app_body_not_stuck_to_keyboard():
    """候选栏贴的是**光标**，不是键盘。这条带在屏幕上部 23%。"""
    assert cands(NOTES_NIHAO) == ['1 你好', '2', '3你还']


def test_anchor_picks_the_bar_next_to_the_pinyin_not_the_lowest():
    """两条带都合格时，选离拼音最近的那条，不是最低的那条。

    构造：备忘录的候选栏在上（318），下面再摆一条同样合格的假带（1218）。
    没有锚点时取最低（老行为）；给了 'nihao' 就该选上面那条。
    """
    decoy = [('1 别的', (110, 1218)), ('2 其他', (230, 1218)), ('3 另一个', (330, 1218))]
    pairs = NOTES_NIHAO + decoy
    els = [SimpleNamespace(id=i, text=t, center=c) for i, (t, c) in enumerate(pairs)]
    o = SimpleNamespace(elements=els, height_px=H, width_px=624)
    ex = Executor.__new__(Executor)
    assert [e.text for e in Executor._candidates(ex, o)] == ['1 别的', '2 其他', '3 另一个']
    assert [e.text for e in Executor._candidates(ex, o, 'nihao')] == ['1 你好', '2', '3你还']


def test_spotlight_anchor_is_below_the_bar():
    """Spotlight 里拼音在候选栏**下面**（'Q shezhi'@1283 vs 候选栏@1218）——
    方向和备忘录正好相反，所以按绝对距离挑，不看方向。"""
    assert cands(Q1) == ['1设置', '2 摄制', '3 摄制组']
    els = [SimpleNamespace(id=i, text=t, center=c) for i, (t, c) in enumerate(Q1)]
    o = SimpleNamespace(elements=els, height_px=H, width_px=624)
    got = [e.text for e in Executor._candidates(Executor.__new__(Executor), o, 'shezhi')]
    assert got == ['1设置', '2 摄制', '3 摄制组'], got
