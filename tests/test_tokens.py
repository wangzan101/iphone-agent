import pytest

from iphone_agent.harness import tokens


def test_calibration_id_is_set_after_the_2026_09_09_run():
    """2026-09-09 真机标定完成，见 docs/superpowers/acceptance/
    2026-09-09-token标定.md。CALIBRATION_ID 不再是「未标定」的占位值 ——
    这条测试替换了原来的 test_calibration_id_starts_uninitialized（该测试
    钉的是「标定之前不许冒充」，现在已经真标定过，钉字面值 "uninitialized"
    只会制造无意义的红）。这里改钉「非占位值」这条更弱、但对下一轮标定
    仍然成立的性质，具体字符串在 EXACT_SEGMENT_TOKENS 那组测试里核对。"""
    assert tokens.CALIBRATION_ID != "uninitialized"
    assert tokens.CALIBRATION_ID == "2026-09-09-qwen3.7-plus"


def test_empty_text_is_zero():
    assert tokens.estimate_text("") == 0
    assert tokens.count_chars("") == dict.fromkeys(tokens.BUCKETS, 0)


def test_char_buckets():
    c = tokens.count_chars("你好abc 12，🙂\n")
    assert c["han"] == 2            # 你 好
    assert c["cjk_punct"] == 1      # ，
    assert c["ascii_alpha"] == 3    # a b c
    assert c["ascii_digit"] == 2    # 1 2
    assert c["ascii_sym"] == 2      # 空格 换行
    assert c["astral"] == 1         # 🙂
    assert c["other"] == 0


def test_fullwidth_punctuation_is_not_in_the_han_bucket():
    """本次改动的核心。老实现把全角标点和汉字塞进同一个桶，注释里写的理由是
    「分词行为接近」—— 2026-09-09 真机实测把它证伪了：汉字 0.334 token/字符，
    中文标点 0.933，差 2.79 倍。同桶意味着一段满是标点的文本会被低估近 3 倍。
    """
    assert tokens.bucket_of("，") == "cjk_punct"
    assert tokens.bucket_of("。") == "cjk_punct"
    assert tokens.bucket_of("【") == "cjk_punct"
    assert tokens.bucket_of("国") == "han"
    assert tokens.TOKENS_PER_CHAR["cjk_punct"] > tokens.TOKENS_PER_CHAR["han"]


def test_general_punctuation_block_is_its_own_bucket_not_folded_into_cjk_punct():
    """— (U+2014) … (U+2026) “ ” 在通用标点区（0x2010-0x2027），单独一个桶。

    这条测试**推翻了它的上一版**。上一版断言这些字符落 cjk_punct，理由是老的
    标点语料 87% 全角 + 13% 这一段、混起来仍是线性 —— 但任何固定配比都线性，
    那条语料根本证伪不了「两者不同价」，上一版自己也把这条不确定性写在注释里，
    并留下判据：**再单发一条纯语料，斜率差超过 15% 就该拆桶。**

    2026-09-09 单发的纯 west_punct 语料量出 1.166，cjk_punct 0.933，差 25% ——
    超阈值，按它自己的判据拆开。
    """
    for ch in "—…“”‘’":
        assert tokens.bucket_of(ch) == "west_punct", ch
    # 全角区不受影响，仍归 cjk_punct。
    for ch in "，。！？【】":
        assert tokens.bucket_of(ch) == "cjk_punct", ch
    d = tokens.TOKENS_PER_CHAR
    assert abs(d["west_punct"] - d["cjk_punct"]) / d["cjk_punct"] > 0.15, \
        "两个桶的系数差不到 15% 的话，当初就没有拆桶的依据 —— 该合回去"


def test_ascii_digits_have_their_own_bucket():
    """真实元素行里数字占 45%（每行 `[id] 文字 (cx,cy) 0.99`），而老探针连一条
    纯数字语料都没有 —— 数字的系数是从「重复英文串」里外推出来的。分桶是让它
    有机会被单独标定的前提。
    """
    assert tokens.bucket_of("7") == "ascii_digit"
    assert tokens.bucket_of("a") == "ascii_alpha"
    assert tokens.bucket_of("[") == "ascii_sym"


def test_newline_counts_as_ascii_not_other():
    """元素列表每行一个换行。老实现只认 0x20-0x7E，换行落进 other 桶按
    「其余 BMP」的系数记 —— 一整类高频字符记错了桶。"""
    assert tokens.bucket_of("\n") == "ascii_sym"
    assert tokens.bucket_of("\t") == "ascii_sym"


def test_every_bucket_has_a_non_negative_coefficient():
    """单调性（下面那条测试）靠系数非负；桶名对不上会让某个桶静默按 0 记。"""
    assert set(tokens.TOKENS_PER_CHAR) == set(tokens.BUCKETS)
    assert all(v >= 0 for v in tokens.TOKENS_PER_CHAR.values())


def test_every_bucket_coefficient_is_the_measured_pure_corpus_slope():
    """八个桶的系数必须是 2026-09-09 P1 纯语料实测出来的斜率，不是待标定初值。

    出处：docs/superpowers/acceptance/2026-09-09-token标定/
          probe-text-20260909-152108.json（第二轮，八条纯语料齐全）
          + probe-text-20260909-143809.json（第一轮，前四个桶）。
    算法：过原点最小二乘 Σ(x·y)/Σ(x²)，x = meta.chars（200/500/1000/2000），
          y = prompt_tokens − baseline_tokens（= 59）。

    为什么要有这条：ascii_alpha / ascii_digit / ascii_sym / other 这四个桶
    的纯语料 2026-09-09 已经跑过了，代码里却一直留着待标定初值
    （0.25 / 0.50 / 0.35 / 0.50），注释还写着「等真跑」—— 而**当时全量 910
    条测试没有一条拦得住这件事**，也没有一条钉住那四个错值。填错一个系数
    只会让第 3 层兜底静默偏，不会有任何红灯。这条测试就是那盏灯。

    ⚠ 下一轮重标时：改 CALIBRATION_ID 的同时改这里，并把新的原始数据文件名
      写进 docstring。不许只改 tokens.py 不改这里。
    """
    assert tokens.TOKENS_PER_CHAR == {
        "han": 0.334,          # 0.3343
        "cjk_punct": 0.933,    # 0.9329
        "west_punct": 1.166,   # 1.1659
        "astral": 2.555,       # 2.5548（emoji 语料含 1/9 的 U+FE0F，见 tokens.py）
        "ascii_alpha": 0.302,  # 0.3023
        "ascii_digit": 1.001,  # 1.0007  ← 旧初值 0.50，差整整一倍
        "ascii_sym": 0.625,    # 0.6250  ← 旧初值 0.35
        "other": 0.667,        # 0.6668  ← 旧初值 0.50
    }
    # 那四个待标定初值一个都不许再出现。
    assert tokens.TOKENS_PER_CHAR["ascii_alpha"] != 0.25
    assert tokens.TOKENS_PER_CHAR["ascii_digit"] != 0.50
    assert tokens.TOKENS_PER_CHAR["ascii_sym"] != 0.35
    assert tokens.TOKENS_PER_CHAR["other"] != 0.50


def test_pure_ascii_is_cheaper_per_char_than_pure_cjk():
    assert tokens.estimate_text("你" * 100) > tokens.estimate_text("a" * 100)


@pytest.mark.parametrize("s", ["", "a", "你好", "abc 你好 🙂", "x" * 5000, "你" * 5000])
def test_monotonic_in_length(s):
    """更长的文本估出的 token 不减 —— 这条一破，所有占比结论都不可信。"""
    assert tokens.estimate_text(s + "更多文字 and more") >= tokens.estimate_text(s)


def test_returns_non_negative_int():
    for s in ["", "a", "你", "🙂", "\n\n\n"]:
        v = tokens.estimate_text(s)
        assert isinstance(v, int) and v >= 0


@pytest.mark.parametrize("s", ["", "a你🙂 ，", "x" * 100, "。" * 50, "\t\n\r"])
def test_buckets_are_exhaustive_and_disjoint(s):
    c = tokens.count_chars(s)
    assert set(c) == set(tokens.BUCKETS)
    assert sum(c.values()) == len(s)


def test_unknown_model_returns_none_not_a_default():
    """套默认系数会产出一个看起来正常的错数，比没有数更糟（spec §2.6）。"""
    assert tokens.estimate_image(624, 1388, model_id="no-such-model") is None


def test_image_rules_contains_the_official_qwen3_7_plus_rule():
    """T0-3 的结论：qwen3.7-plus 的图片规则来自阿里云百炼「视觉理解」官方文档
    （查证日期 2026-09-02，含官方 smart_resize 参考代码）。文档把 qwen3.7-plus
    归入「Qwen3.8 / Qwen3.7 / Qwen3.6 / Qwen3.5 / Qwen3-VL 系列」这一组。

    （这条测试替换了原来的 test_image_rules_is_empty_until_t0_3_passes。）
    """
    assert set(tokens.IMAGE_RULES) == {"qwen3.7-plus"}
    r = tokens.IMAGE_RULES["qwen3.7-plus"]
    assert r.pixels_per_token == 1024          # 32*32，不是 QVQ/Qwen2.5-VL 的 28*28
    assert r.constant_tokens == 2              # <vision_bos> + <vision_eos>
    assert r.size_multiple == 32
    assert r.min_pixels == 4096                # 4 * 32 * 32，官方代码写死
    assert r.max_pixels == 2_621_440           # 默认值；我们不传 vl_high_resolution_images
    # 这一代 factor 一身兼二职（取整倍数 == 每 token 边长），两个字段必须自洽。
    assert r.pixels_per_token == r.size_multiple ** 2


def test_image_rules_and_text_calibration_are_tracked_separately(monkeypatch):
    """图片规则和文本标定是两本账，**互不依赖** —— 这条测试要真的测到这个意图。

    原来这条测试用 `CALIBRATION_ID == "uninitialized"` 证明「填了图片规则
    ≠ 模块已标定」。2026-09-09 文本系数真标定完之后，它被替换成了两条互不
    相关的常量断言（`IMAGE_RULES` 有一条 + `CALIBRATION_ID` 是某个字面值），
    那两条各自已经被别的测试钉着，这里等于什么都没测 —— 名字说的「分开记账」
    这件事，反而没有一条断言碰到。

    现在改成直接构造「两本账各自出错」的两种情形，各验一次：

      1. 文本标定退回未标定（CALIBRATION_ID 变回占位值）—— 图片估算的结果
         **必须一个 token 都不变**。图片走的是查文档得来的 smart_resize 规则，
         跟哪天对哪个模型标过文本系数无关。
      2. 文本系数被整体清零（第 3 层全塌）—— 图片估算仍然**必须**给出 862。
         这条排除的是「图片路径偷偷复用了 TOKENS_PER_CHAR」这种耦合。

    反过来的方向由 budget 的分层记账保证：图片段落在 image_rule 层，
    永远不进 exact / segment_ratio / char_class 任何一层（见
    tests/test_budget.py 对 est_by_layer 的断言）。
    """
    before = tokens.estimate_image(624, 1388, model_id="qwen3.7-plus")
    assert before == 862

    monkeypatch.setattr(tokens, "CALIBRATION_ID", "uninitialized")
    assert tokens.estimate_image(624, 1388, model_id="qwen3.7-plus") == before

    monkeypatch.setattr(
        tokens, "TOKENS_PER_CHAR", dict.fromkeys(tokens.BUCKETS, 0.0))
    assert tokens.estimate_text("任何文本 whatever 123") == 0   # 第 3 层确实塌了
    assert tokens.estimate_image(624, 1388, model_id="qwen3.7-plus") == before


def test_every_production_segment_name_is_accounted_for():
    """生产里发得出去的每个 `_seg` 名字，必须落在 SEGMENT_TOKENS_PER_CHAR
    或 SEGMENTS_PENDING_CALIBRATION 里 —— 二选一，不许都不在。

    为什么必须有这条：estimate_segment 查不到段名就**静默**降级到第 3 层
    字符类兜底。也就是说，`seg="memory_index"` 打错成 `seg="memory_idnex"`，
    行为上只是那一段的估算精度悄悄退化，测试全绿、budget 里也只显示成一个
    看着正常的 char_class。这份工作的核心主张是「层要诚实」：一个段走的是
    哪一层必须是有人**决定**过的，不能是打错字的副产品。

    实现上扫源码而不是跑主循环：段名是散在 messages.py / loop.py / budget.py
    里的字面量，跑一遍主循环只能覆盖这次走到的分支（比如 scenario / app_note /
    obs_route 都要特定输入才出现），扫源码才覆盖得全。代价是正则依赖写法 ——
    所以下面同时断言「扫出来的条数不少于 24」，防止正则哪天失配、扫出空集
    而测试照样绿。

    2026-09-09 这条测试写出来时立刻抓到一个：`repair_prompt`（loop.py 在模型
    只回文字时补发的那句）两个表都不在，一直在静默走兜底。已补进
    SEGMENTS_PENDING_CALIBRATION。

    2026-09-09 补扫 budget.py：`assistant_tool_args`（budget.py 里合成的
    工具调用参数段）和 `tools_schema`（合成的工具表段）都不是 `"_seg": "X"` /
    `seg="X"` 字面量，也不是 `(obs_X, …)` / `(state_X, …)` 元组，原来的两条
    正则扫不到——`_est("assistant_tool_args", ...)` 打错字也会静默走兜底，
    全量测试照样绿（reviewer 变异验证过）。加一条正则专门扫这两个调用点。
    """
    import pathlib
    import re

    root = pathlib.Path(tokens.__file__).parent
    # `"_seg": "X"` / `seg="X"`（MessageLog 的入口）
    kw = re.compile(r'(?:"_seg"\s*:\s*|\bseg\s*=\s*)"([a-z_][a-z_0-9]*)"')
    # `("state_X", …)` / `("obs_X", …)`（build_state_parts / loop 的 obs_parts）
    tup = re.compile(r'\(\s*"((?:obs|state)_[a-z_0-9]+)"\s*,')
    # `_est("X", …)` / `estimate_segment("X", …)`（budget.py 里合成的段名）
    call = re.compile(r'(?:\b_est|\bestimate_segment)\(\s*"([a-z_][a-z_0-9]*)"')

    found: set[str] = set()
    for name in ("messages.py", "loop.py", "budget.py"):
        src = (root / name).read_text(encoding="utf-8")
        found |= set(kw.findall(src))
        found |= set(tup.findall(src))
        found |= set(call.findall(src))

    # "image" 不是文本段：budget 走 estimate_image + LAYER_IMAGE_RULE，
    # 根本不查这两个表。"unknown" 是 `_seg` 缺失/不认识时的兜底桶本身
    # （budget.py `seg or "unknown"`），故意不进两张表、故意走字符类兜底——
    # 不是遗漏。这两个是唯一的豁免。
    found -= {"image", "unknown"}

    assert len(found) >= 24, (
        f"只扫到 {len(found)} 个段名（{sorted(found)}）——正则大概率失配了，"
        "而不是生产里真的只剩这么几段。先修正则。")

    known = set(tokens.SEGMENT_TOKENS_PER_CHAR) | set(tokens.SEGMENTS_PENDING_CALIBRATION)
    missing = sorted(found - known)
    assert not missing, (
        f"这些段名生产里发得出去，但 SEGMENT_TOKENS_PER_CHAR 和 "
        f"SEGMENTS_PENDING_CALIBRATION 都不认识：{missing}。"
        "它们会静默走第 3 层 char_class 兜底。要么是打错字（改回去），"
        "要么是新段（量得到就填 SEGMENT_TOKENS_PER_CHAR，量不到就登记进 "
        "SEGMENTS_PENDING_CALIBRATION）——不许两个表都不写。")


def test_real_device_screenshot_624x1388_is_862_tokens():
    """真机截图尺寸（624x1388）按官方 smart_resize 手算：

        round(624/32) = round(19.5) = 20 -> w_bar = 640
        round(1388/32) = round(43.375) = 43 -> h_bar = 1376
        640*1376 = 880640，介于 min_pixels(4096) 和 max_pixels(2621440) 之间，不缩放
        880640 // 1024 = 860；860 + 2 = 862

    ⚠ 旧实现（ceil）会得到 640x1408 -> 882，差 2.3%：不炸、数量级也对，
      但就是错的。这条测试钉死这个差异。
    """
    r = tokens.IMAGE_RULES["qwen3.7-plus"]
    assert tokens.apply_image_rule(624, 1388, r) == 862
    assert tokens.estimate_image(624, 1388, model_id="qwen3.7-plus") == 862


def test_known_model_applies_the_rule_including_the_constant_term():
    """用一条构造出来的规则验算法本身，不依赖任何真实模型的数值。

    64x64，size_multiple=32：round(64/32)=2 -> 64x64，面积 4096，
    4096 // 1024 = 4；4 + 2 = 6。
    """
    rule = tokens.ImageRule(pixels_per_token=32 * 32, constant_tokens=2,
                            size_multiple=32, min_pixels=None, max_pixels=None)
    assert tokens.apply_image_rule(64, 64, rule) == 6


def test_rule_rounds_dimensions_to_nearest_multiple_not_up():
    """正常范围内用 round（就近）而不是 ceil —— 特意挑一个两者分叉的尺寸。

    100x100：100/32 = 3.125，余数 4 < 16，所以 round 向下。
        round(3.125) = 3 -> 96；ceil(3.125) = 4 -> 128
        官方：96*96 = 9216，9216 // 1024 = 9
        旧实现（ceil）：128*128 = 16384 // 1024 = 16
    """
    rule = tokens.ImageRule(pixels_per_token=32 * 32, constant_tokens=0,
                            size_multiple=32, min_pixels=None, max_pixels=None)
    assert tokens.apply_image_rule(100, 100, rule) == 9


def test_rule_scales_down_with_floor_and_from_the_original_side_lengths():
    """超过 max_pixels：缩小分支用 floor，且基准是**原始**边长除以 beta。

    200x100，size_multiple=32，max_pixels=4096：
        round(200/32) = round(6.25) = 6 -> 192
        round(100/32) = round(3.125) = 3 -> 96
        192*96 = 18432 > 4096，进缩小分支
        beta = sqrt(200*100 / 4096) = sqrt(4.8828125) = 2.209709...
        w: 200 / beta = 90.5097 -> floor(90.5097/32) = floor(2.8284) = 2 -> 64
        h: 100 / beta = 45.2548 -> floor(45.2548/32) = floor(1.4142) = 1 -> 32
        64*32 = 2048，2048 // 1024 = 2
    （误用 round 会得到 96x32 -> 3；从取整后的 192/96 出发也会分叉。）
    """
    rule = tokens.ImageRule(pixels_per_token=32 * 32, constant_tokens=0,
                            size_multiple=32, min_pixels=None, max_pixels=4096)
    assert tokens.apply_image_rule(200, 100, rule) == 2


def test_rule_scales_up_with_ceil_when_below_min_pixels():
    """低于 min_pixels：放大分支用 ceil —— 挑一个 ceil 和 round 分叉的尺寸。

    100x30，size_multiple=32，min_pixels=4096：
        round(100/32) = 3 -> 96；round(30/32) = round(0.9375) = 1 -> 32
        96*32 = 3072 < 4096，进放大分支
        beta = sqrt(4096 / (100*30)) = sqrt(1.3653333) = 1.1684748...
        w: 100 * beta = 116.8475 -> ceil(116.8475/32) = ceil(3.6515) = 4 -> 128
        h: 30 * beta = 35.0542 -> ceil(35.0542/32) = ceil(1.0954) = 2 -> 64
        128*64 = 8192，8192 // 1024 = 8
    （误用 round：h 会取 32，结果是 4，正好差一倍。）
    """
    rule = tokens.ImageRule(pixels_per_token=32 * 32, constant_tokens=0,
                            size_multiple=32, min_pixels=4096, max_pixels=None)
    assert tokens.apply_image_rule(100, 30, rule) == 8


def test_over_max_is_judged_on_the_rounded_area_not_the_raw_area():
    """本次改动的核心分叉点：**取整后**面积越界、但原始面积没越界。

    112x112，size_multiple=32，max_pixels=13000：
        原始面积 112*112 = 12544 <= 13000 —— 旧实现到这里就认为「没越界」，
        直接 ceil 到 128x128 = 16384 // 1024 = 16 token。
        官方：round(112/32) = round(3.5) = 4 -> 128（banker's rounding，3.5 -> 4）
              128*128 = 16384 > 13000，越界，进缩小分支
              beta = sqrt(12544 / 13000) = sqrt(0.9649231) = 0.9823050...
              w = h: 112 / beta = 114.0175 -> floor(114.0175/32) = floor(3.5630) = 3 -> 96
              96*96 = 9216，9216 // 1024 = 9
    9 vs 16：新旧算法在这里差了近一倍，且旧实现不会报任何错。
    """
    rule = tokens.ImageRule(pixels_per_token=32 * 32, constant_tokens=0,
                            size_multiple=32, min_pixels=None, max_pixels=13000)
    assert tokens.apply_image_rule(112, 112, rule) == 9


# --- review fix #1: 非正宽高必须返回 None，不能返回负数或崩在除零上 ---

def test_negative_dimensions_do_not_produce_a_negative_token_count():
    """负数尺寸（上游 bug）曾经会直接算出 -12288 这种看起来正常的错数。"""
    rule = tokens.ImageRule(pixels_per_token=32 * 32, constant_tokens=0,
                            size_multiple=32, min_pixels=None, max_pixels=None)
    assert tokens.apply_image_rule(-100, 100, rule) is None
    assert tokens.estimate_image(-100, 100, model_id="no-such-model") is None


def test_zero_dimensions_with_min_pixels_do_not_raise():
    """0x0（截图失败/黑屏）曾经会在 min_pixels 缩放那步除零崩溃。"""
    rule = tokens.ImageRule(pixels_per_token=32 * 32, constant_tokens=0,
                            size_multiple=32, min_pixels=32 * 32, max_pixels=None)
    assert tokens.apply_image_rule(0, 0, rule) is None


# --- review fix #2: ImageRule 必须校验 min_pixels <= max_pixels ---

def test_image_rule_rejects_min_greater_than_max():
    """min/max 填反（配置手误）必须在构造时就炸，而不是静默产出错数。"""
    with pytest.raises(ValueError):
        tokens.ImageRule(pixels_per_token=32 * 32, constant_tokens=0,
                          size_multiple=32, min_pixels=100 * 100, max_pixels=10 * 10)


# --- review fix #3: 取整后小幅超出 min/max_pixels 是刻意行为，不是 bug ---

def test_rounding_can_push_area_above_min_pixels_by_design():
    """放大分支用 ceil，取整后的面积会明显超过 min_pixels ——这是官方算法
    只缩放一次、不迭代的刻意结果，不是需要「修掉」的 bug。见 fix #3。

    100x30、size_multiple=32、min_pixels=4096：放大后取整到 128x64 = 8192，
    是 min_pixels 的 2 倍。
    """
    rule = tokens.ImageRule(pixels_per_token=1, constant_tokens=0,
                            size_multiple=32, min_pixels=4096, max_pixels=None)
    got = tokens.apply_image_rule(100, 30, rule)
    assert got == 128 * 64
    assert got > 4096 * 1.9


# ============================ 分层估算 ============================

def test_exact_constants_are_keyed_by_hashes_the_code_actually_produces():
    """常量表的键必须是**当前代码算出来的** hash，不能是人手编的字符串。

    这条测试就是那个「别自己编一个键」的执行版：system 的键由
    prompt.prompt_hash 现算，tools_schema 的键由 budget 里那套序列化现算。
    提示词、PROMPT_VERSION、工具表任何一处一改，这条测试就红 —— 那正是提醒
    「实测值失效了，要么重标，要么把这条常量删掉」。
    """
    import hashlib
    import json

    from iphone_agent.harness.prompt import prompt_hash
    from iphone_agent.harness.tools import tool_defs

    sys_key = prompt_hash(allow_coord_tap=True, has_skills=True)
    assert ("system", sys_key) in tokens.EXACT_SEGMENT_TOKENS, (
        "system 的实测常量对不上当前提示词的 hash。要么提示词改了（重跑 "
        "--probe segments 重标），要么这个键是手编的。")

    blob = json.dumps(tool_defs(True), sort_keys=True, ensure_ascii=False)
    tools_key = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    assert ("tools_schema", tools_key) in tokens.EXACT_SEGMENT_TOKENS


def test_exact_layer_wins_and_reports_itself():
    (seg, key), val = next(iter(tokens.EXACT_SEGMENT_TOKENS.items()))
    est, layer = tokens.estimate_segment(seg, "任意文本", key=key)
    assert (est, layer) == (val, tokens.LAYER_EXACT)


def test_wrong_key_falls_through_instead_of_using_a_stale_constant():
    """hash 对不上就是「这段内容已经不是当初量的那一段」——必须降级，
    绝不能拿旧常量顶替（那是本模块最怕的「看起来正常的错数」）。"""
    est, layer = tokens.estimate_segment("system", "新提示词", key="deadbeefdead")
    assert layer == tokens.LAYER_SEGMENT_RATIO
    assert est == round(len("新提示词") * tokens.SEGMENT_TOKENS_PER_CHAR["system"])


def test_segment_ratio_layer_for_elements():
    text = "[1] 设置 (170,405) 0.99\n" * 20
    est, layer = tokens.estimate_segment("state_elements", text)
    assert layer == tokens.LAYER_SEGMENT_RATIO
    # 2026-09-09 真机标定：state_elements 独立测出 0.884（之前是和 obs_elements
    # 共用的占位值 0.868），见 docs/superpowers/acceptance/2026-09-09-token标定.md。
    assert est == round(len(text) * 0.884)
    # 元素列表是字符类模型低估最狠的一段（实测 0.884 / 兜底误差 52.6%）：
    # 分层的收益就体现在这里，比兜底大。
    assert est > tokens.estimate_text(text)


def test_uncalibrated_segments_fall_back_to_char_class_and_say_so():
    """没标定的段走第 3 层，而且必须**如实报出来**是兜底 —— 这个标签就是
    「这个数字别拿去卡阈值」的唯一提示。"""
    for seg in sorted(tokens.SEGMENTS_PENDING_CALIBRATION):
        est, layer = tokens.estimate_segment(seg, "测试 test 123")
        assert layer == tokens.LAYER_CHAR_CLASS, seg
        assert est == tokens.estimate_text("测试 test 123")


def test_pending_and_measured_segment_lists_do_not_overlap():
    """一个段要么有实测比值、要么在待标定名单里，不能同时在两边 ——
    同时在两边意味着名单没跟着标定结果更新，读的人会以为它还没标。"""
    assert not (tokens.SEGMENTS_PENDING_CALIBRATION
                & set(tokens.SEGMENT_TOKENS_PER_CHAR))


def test_estimate_segment_never_raises_on_bad_input():
    """观测设施绝不顶掉任务。"""
    for seg, text, key in [(None, "abc", None), ("system", None, None),
                           (123, "abc", "x"), ("system", 3.5, object())]:
        est, layer = tokens.estimate_segment(seg, text, key=key)  # type: ignore[arg-type]
        assert isinstance(est, int) and est >= 0
        assert isinstance(layer, str)
