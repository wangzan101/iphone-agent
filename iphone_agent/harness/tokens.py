"""文本与图片的 token 估算。纯函数，无 IO。

估算分三层，由 estimate_segment() 按段选层（层名会一路带到 budget 的输出里，
所以看账的人能知道每一段的数字是「测出来的」还是「猜出来的」）：

    第 1 层 exact          已测常量表 EXACT_SEGMENT_TOKENS。按「段名 + 内容 hash」
                           查表。system / tools_schema 这两段在一次任务里逐字节
                           恒定，标一次就该直接用，根本不该估。
    第 2 层 segment_ratio  段内风格稳定的，按段给一个 token/字符比
                           （SEGMENT_TOKENS_PER_CHAR）。
    第 3 层 char_class     下面那套 8 桶字符类估算，兜底。

⚠ 为什么不是「把字符桶做细一点」（2026-09-09 实测结论，见 TOKENS_PER_CHAR
  下方的失败记录）：真实语料上 tools 桶构成与 mixed 桶构成相近，误差方向却
  相反。数字出处是 probe-text-20260909-152108.json 的 real_eval.rows[]
  （signed 值 = (predicted_content_tokens − measured_content_tokens)
  / measured_content_tokens，即该行 rel_error 带上方向）：

      语料                预测      实测    方向     偏差
      real_tools_json     4974.6    4409    高估    +12.8%
      mixed                196.9     224    低估    −12.1%
      real_system_prompt  1517.2    1920    低估    −21.0%
      real_elements       2464.7    2877    低估    −14.3%

  tools 与 mixed 的字符桶构成相近（都以 alpha + sym + digit 为主），一个高估
  一个低估 —— 同样的字符配比推不出同样的 token 数。驱动量是**词表命中密度**
  （`"description"` 重复 15 次压得极好，`[12] (170,405) 0.99` 压不动），
  不是字符类。加桶治不了这个，所以不加桶。

  ⚠ 更早的版本在这里写过「tools 高估 14%、mixed 低估 19%」—— 那两个数在
    real_eval 的任何算法下都出不来（是另一套基线下的临时值，不该进代码）。
    方向性结论不变，数字以上表为准；改数字的同时把出处写死在这里，别再出现
    无出处的百分比。

⚠ 2026-09-09 用 scripts/calibrate_tokens.py 跑了一轮受控探针，结果写进
  docs/superpowers/acceptance/2026-09-09-token标定.md，并把
  EXACT_SEGMENT_TOKENS / SEGMENT_TOKENS_PER_CHAR / CALIBRATION_ID 三处
  手动填回了这里 —— budget 的 calibration 字段现在报
  "2026-09-09-qwen3.7-plus"，不再是 "uninitialized"。

  但这不等于**全模块**都标定过了：
    - 第 1/2 层（EXACT_SEGMENT_TOKENS / SEGMENT_TOKENS_PER_CHAR）覆盖的段是
      有实测支撑的。
    - 第 3 层 TOKENS_PER_CHAR（下方的 8 桶字符类表）八个桶**都有纯语料实测
      斜率**（第二批四个是 2026-09-09 补的，见该表内的分组注释），但它在真实
      语料上已经被证明不能简单线性叠加（见上面那条 ⚠：−21% ~ +12.8%）——
      它只作兜底，不作为「本模块已标定」的依据。「每个桶的单价都测过」和
      「按桶线性叠加能预测真实文本」是两件事，前者成立不蕴含后者。
      calibration 字段报出的 id 说明的是「哪天、对哪个模型标的」，
      不代表每一段都已精确；具体到某一段该信到什么程度，看 budget 输出里
      那段的 `layer` 字段（exact > segment_ratio > char_class）。见 spec §7 第 7 条。
"""
from __future__ import annotations

import dataclasses
import math

# 标定产物的标识，与验收文档 docs/superpowers/acceptance/2026-09-09-token标定.md
# 同名。2026-09-09 用 scripts/calibrate_tokens.py --probe segments（外加
# gate/preflight/probe-text/probe-tools/probe-cache/probe-image 六个探针）
# 在 alibaba:qwen3.7-plus 上跑出了下面这套系数。
#
# ⚠ 这个 id 只能说明「哪天、对着哪个模型别名标的」，**不能**绑到具体模型快照：
#   preflight 测出来 raw_usage 里的 model_version 返回的就是别名 "qwen3.7-plus"
#   本身，不是一个能唯一定位服务端具体版本的值（见验收文档「结论」一节的
#   WARN）。如果服务端在别名不变的情况下换了实现，我们发现不了，这套系数
#   会带着旧版本的误差继续被当真值用。
#
# 三个会让这套系数失效的事件（任一发生就该重标）：
#   1. 提示词或 PROMPT_VERSION 改了 —— EXACT_SEGMENT_TOKENS 里 system 那条
#      按 prompt_hash 查表，hash 一变查不到，自动降级到第 2 层，不会拿旧值顶替。
#   2. 工具表改了 —— tools_schema 那条按 schema_hash 查表，同上自动降级。
#   3. "qwen3.7-plus" 这个别名在服务端换了实现 —— **没有测试钉着这条**，
#      只能定期重标或靠人工发现行为漂移。
#   前两条各有一条测试钉着（tests/test_tokens.py 的
#   test_exact_constants_are_keyed_by_hashes_the_code_actually_produces），
#   hash 一变测试就红，红了就是「该重标」的信号。
CALIBRATION_ID = "2026-09-09-qwen3.7-plus"

# ============================ 字符分桶 ============================
# 2026-09-09 用 scripts/calibrate_tokens.py --probe text 在 qwen3.7-plus 上跑的
# **第一轮**（出处 probe-text-20260909-143809.json，records[]）。单语料斜率
# （每种语料 200/500/1000/2000 字各一点，扣掉 0 字基线；每种语料**单独看**
# 都是完美线性，斜率跨 200→2000 字稳到 3~4 位有效数字）：
#
#     语料          每字符 token   该语料落在哪个桶
#     汉字            0.334        han（100%）
#     全角+中文标点   0.933        cjk_punct（100%，见下面的区间说明）
#     重复英文词      0.185        ascii_alpha + ascii_sym 的混合（空格占 ~15%）
#     emoji           2.555        astral 8/9 + other 1/9（🛠️ 尾部的 U+FE0F）
#     真实混排        0.700        han/digit/alpha/sym/other 全都有
#
# 旧的三桶模型（cjk / ascii / other）在这组数据上 rms 残差 191 token、最大 487，
# 根本不成立。两条根因，各对应下面一处改动：
#
#  1) 全角标点原来和汉字同桶，注释里写着「分词行为接近」—— 实测差 2.8 倍
#     （0.933 vs 0.334）。那条假设是错的，现在 cjk_punct 独立成桶。
#
#  2) 更要命的一条：合成语料标出来的系数套不到真实文本上。0.185 是**重复英文串**
#     压出来的；真实混排语料里 ASCII 部分反推是 ~0.82 token/字符（(28 - 10*0.334)/30），
#     差 4 倍多。数字/括号这类「每个几乎自成一个 token」的东西，和重复的英文单词
#     完全不是一个分词效率。所以 ASCII 拆成 alpha / digit / sym 三桶，
#     **并且**探针必须拿真实语料做检验（scripts/calibrate_tokens.py 的 REAL_CORPORA）。
#
# ⚠ 拆桶只是让模型有机会成立，不等于它成立。判据不是拟合残差小，是
#   **真实语料上的相对误差小** —— 见 calibrate_tokens.probe_text 的 verdict 一节。

#: 桶名 → 每字符 token 数。所有系数非负（estimate_text 的单调性靠这条）。
#:
#: ⚠ 填值规则（本模块的自律）：**只有当某条语料几乎 100% 落在这个桶里时，
#:   才把那条语料的斜率写成这个桶的初值** —— 那种情况下斜率可以直接读出来，
#:   不用解方程，也就没有「几个系数互相污染」的问题。做不到的桶保持待标定初值，
#:   并在下面注明「凭什么知道它是错的」。
#:
#: ⚠ 这条自律的另一半，2026-09-09 之后才补上：**探针跑完必须把值填回来。**
#:   后四个桶的纯语料当天就跑了、验收文档也列了实测值，代码却一直留着待标定
#:   初值，注释还写着「没有纯语料、等真跑」—— 文档和代码互相矛盾了整整一轮，
#:   而全量 910 条测试一条都拦不住。现在
#:   tests/test_tokens.py::test_every_bucket_coefficient_is_the_measured_pure_corpus_slope
#:   钉着这八个数：跑了新探针不填回来，或者填错了，那条测试会红。
#:
#: ⚠ 八个桶现在**全部**有纯语料实测值（2026-09-09 的 P1 第二轮补齐了原先缺的四个），
#:   但这**不等于**「字符类模型成立」：单语料斜率只说明「这一类字符单独成篇时是这个价」，
#:   不说明「按字符类线性叠加」在真实文本上成立 —— 真实语料上它仍然错 12%~21%
#:   （见模块顶部那条 ⚠ 和 real_eval 的四行数据）。这一层是兜底，不是模型。
TOKENS_PER_CHAR: dict[str, float] = {
    # ── 第一批实测（P1 第一轮就有纯语料的四个）─────────────────
    # 2026-09-09 实测：汉字语料（常用双字词为主）0.3340。
    # ⚠ 这条语料是「国家标准化管理委员会……」这类高频官话词，压缩得比一般中文好。
    #   真实屏幕上的短标签（"关于本机"、"通用"）未必这么便宜 —— 真实语料检验就是干这个的。
    "han": 0.334,
    # 2026-09-09 实测：中文标点语料（全角区 0x3000/0xFF00）0.9330，是汉字的 2.79 倍。
    "cjk_punct": 0.933,
    # 2026-09-09 实测：通用标点区（0x2010-0x2027，即 — … “ ” ‘ ’ 这些）1.166。
    # ⚠ 这个桶是**上一轮假设被证伪之后拆出来的**：上一轮把 0x2010-0x2027 并进
    #   cjk_punct，理由是「老的标点语料 87% 全角 + 13% 这一段、混起来线性」，
    #   并且实施者自己写下判据「两条纯语料斜率差 >15% 就该单独立桶」。
    #   同一天 west_punct 纯语料量出 1.166 vs cjk_punct 0.933 —— 差 25%，
    #   超过它自己设的阈值。按它自己的判据拆开，不是我另立标准。
    "west_punct": 1.166,
    # 2026-09-09 实测：emoji 语料 2.5550。⚠ 那条语料 9 个码点里有 8 个星平面
    #   + 1 个 U+FE0F（落 other 桶），所以真正的 astral 单价在 2.555~2.874 之间，
    #   取决于 VS-16 值多少钱。要收敛它得给 other 桶一条纯语料（P1 已加 other_bmp）。
    "astral": 2.555,

    # ── 第二批实测（P1 第二轮为这四个桶补的纯语料）──────────────
    # 出处：runs 与 docs/superpowers/acceptance/2026-09-09-token标定/
    #       probe-text-20260909-152108.json，字段 records[]（label 形如
    #       "ascii_alpha×2000"，meta.group == "pure"）。
    # 算法：斜率 = 过原点最小二乘 Σ(x·y)/Σ(x²)，x = meta.chars（200/500/1000/2000），
    #       y = prompt_tokens − baseline_tokens（该文件 baseline_tokens = 59）。
    # 线性度：每桶 4 个点，逐点比值（y/x）如下 —— 跨 200→2000 字稳到 3~4 位有效数字。
    #
    #     桶            200      500     1000     2000    过原点斜率   旧初值   旧值偏差
    #     ascii_alpha  0.305    0.300    0.302    0.3025    0.3023      0.25     −17%
    #     ascii_digit  1.005    1.002    1.001    1.0005    1.0007      0.50     −50%
    #     ascii_sym    0.625    0.626    0.625    0.6250    0.6250      0.35     −44%
    #     other        0.670    0.666    0.666    0.6670    0.6668      0.50     −25%
    #
    # ⚠ 旧的四个值（0.25 / 0.50 / 0.35 / 0.50）是**待标定初值**，2026-09-09 的
    #   探针跑完后一直没填回来，代码注释还写着「等真跑」—— 这次补上。数字最刺眼的是
    #   ascii_digit：真实值正好是初值的两倍，而元素列表（坐标 + 置信度）几乎全是数字，
    #   这一段正是兜底误差最大的地方。
    # ⚠ 各语料的构成见 scripts/calibrate_tokens.py 的 PURE_ALPHABETS：
    #   ascii_digit 刻意不用 0123456789 这种递增串（递增串在 BPE 里压得异常好）；
    #   ascii_sym 含空格与换行（元素行的骨架）；other 用的是 other_bmp
    #   （希腊 + 西里尔 + 假名）—— 它们各自 100% 落在本桶里，斜率可以直接读，
    #   不用解方程。
    "ascii_alpha": 0.302,
    "ascii_digit": 1.001,
    "ascii_sym": 0.625,
    "other": 0.667,
}

# ── 桶的判据（互斥且穷尽，按下面的顺序判） ────────────────────────

# 星平面：码点 > 0xFFFF。emoji 的主体在这里（U+1F300~U+1FAFF），也包括
# 扩展 B 之后的生僻汉字 —— 它们和 emoji 一样是 4 字节 UTF-8、词表命中率低，
# 归一起是有理由的；真出现大量生僻字再拆。
_ASTRAL_MIN = 0x10000

# ASCII：可打印区 + 三个真会出现在正文里的控制符（\t \n \r）。
# ⚠ 旧实现只认 0x20-0x7E，换行落进 other 桶按 0.5 记 —— 元素列表每行一个换行，
#   这是一整类被记错桶的高频字符。ASCII 三桶共用同一条 BPE 预分词规则，
#   把换行放进 ascii_sym 比放进「其余 BMP」更贴近实际。
_ASCII_EXTRA = frozenset("\t\n\r")

# CJK 表意文字本体。不含标点。
_HAN_RANGES = (
    (0x3400, 0x4DBF),   # 扩展 A
    (0x4E00, 0x9FFF),   # 基本区
    (0xF900, 0xFAFF),   # 兼容表意
)

# 中文排版用的全角标点。两段：
#   0x3000-0x303F  CJK 符号与标点（、。〈〉《》【】…）
#   0xFF00-0xFFEF  半角/全角形式（，！？；：（））
#     ⚠ 这一段里还有全角字母 Ａ-Ｚ 和全角数字 ０-９。它们理论上更像 alpha/digit
#       而不是标点，但真实屏幕上极罕见，且 2026-09-09 的标点语料里一个都没有 ——
#       没有证据就不另立桶，留在这里并记下这条已知的不精确。
_CJK_PUNCT_RANGES = (
    (0x3000, 0x303F),
    (0xFF00, 0xFFEF),
)

# 通用标点里中文排版实际在用的那一段（— … “ ” ‘ ’ ・）。
# ⚠ 上一轮把它并进 cjk_punct，理由是老语料 87%/13% 的混合线性 —— 但**任何**
#   固定配比都是线性的，那条语料证伪不了「两者不同价」。2026-09-09 单发的
#   纯 west_punct 语料量出 1.166，cjk_punct 0.933，差 25%，超过上一轮自己
#   写下的 15% 拆桶阈值。据此独立成桶。
_WEST_PUNCT_RANGES = (
    (0x2010, 0x2027),
)

#: 桶名的规范顺序。count_chars 的返回值一定含且只含这些键。
BUCKETS: tuple[str, ...] = ("han", "cjk_punct", "west_punct", "ascii_alpha",
                            "ascii_digit", "ascii_sym", "astral", "other")


def _in(cp: int, ranges) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def bucket_of(ch: str) -> str:
    """单个字符落哪个桶。判定顺序即优先级，各桶互斥。"""
    cp = ord(ch)
    if cp >= _ASTRAL_MIN:
        return "astral"
    if 0x20 <= cp <= 0x7E or ch in _ASCII_EXTRA:
        if ch.isascii() and ch.isalpha():
            return "ascii_alpha"
        if "0" <= ch <= "9":
            return "ascii_digit"
        return "ascii_sym"
    if _in(cp, _HAN_RANGES):
        return "han"
    if _in(cp, _CJK_PUNCT_RANGES):
        return "cjk_punct"
    if _in(cp, _WEST_PUNCT_RANGES):
        return "west_punct"
    return "other"


def count_chars(s: str) -> dict[str, int]:
    """按字符类分桶。各桶**互斥且穷尽**：所有桶计数之和 == len(s)。

    ⚠ 这条不变式被 tests/test_tokens.py 钉着。加桶、改判据都必须保住它 ——
      它一破，「某段占多少」就不再是把各桶加起来能得到的东西。
    """
    out = dict.fromkeys(BUCKETS, 0)
    for ch in s:
        out[bucket_of(ch)] += 1
    return out


def estimate_text(s: str) -> int:
    """字符类加权求和。

    ⚠ 系数非负 + round 单调，所以 estimate_text 对文本追加是单调不减的。
      这条性质被 tests/test_tokens.py 钉着：它一破，所有占比结论都不可信。

    ⚠ 缺系数的桶按 0 记而不是抛：这是观测设施，坏输入降级不砸任务。
      正常情况下 TOKENS_PER_CHAR 覆盖全部 BUCKETS，走不到那条路。
    """
    c = count_chars(s)
    return int(round(sum(n * TOKENS_PER_CHAR.get(b, 0.0) for b, n in c.items())))


# ============================ 分层估算 ============================
# 层名。这三个字符串会原样出现在 budget 的每段输出里（`layer` 字段）和 run 级
# 汇总里（`token_provenance`），是「这个数字有多可信」的唯一来源，别随手改名。
LAYER_EXACT = "exact"                   # 查已测常量表，逐 token 精确
LAYER_SEGMENT_RATIO = "segment_ratio"   # 这一段实测过的 token/字符比
LAYER_CHAR_CLASS = "char_class"         # 8 桶字符类兜底，真实语料上误差 12%~21%

#: 第 1 层：已测常量表。键是 (段名, 内容 hash)，值是**实测** token 数。
#:
#: 为什么这一层存在：system 和 tools_schema 在一次任务里是确定性的 —— 同一个
#: PROMPT_VERSION、同一套工具集，每次请求逐字节相同。确定性的东西不该估。
#: 这两段合计 1920 + 5300 = **7220** token，是冻结前缀的大头；把它们做成精确的，
#: 整本账的可信度就从「±20% 的估计」变成「大头精确 + 小头估计」。
#:
#: 这个「大头」到底占多少，随每次调用里可变段（元素列表长度、图片尺寸、有没有
#: 记忆注入）变化，不是一个常数。2026-09-09 的端到端实测（出处
#: endtoend-20260909-160000.json 的 cases[].budget.est_by_layer，
#: measured_share = (exact + image_rule) / est_total）：
#:
#:     场景                    est_total  exact  image_rule  measured_share  实测   误差
#:     稀疏屏 12 元素             7929     6549      862         0.935       7947  0.23%
#:     首次单样本 60 元素         9063     6549      862         0.818       9061  0.02%
#:     大图 1086x2415 + 80 元素  11206     6549     2552         0.812      11233  0.24%
#:     密集屏 200 元素           12440     6549      862         0.596      12483  0.34%
#:     超密集 400 元素           17354     6549      862         0.427      17341  0.07%
#:
#: 也就是说：屏幕越密，measured_share 越低（分子恒定、分母随元素列表涨），
#: 这五档从 0.94 一路掉到 0.43。**别把 measured_share 当成一个「约 0.9」的常数**
#: —— 要看某次真实运行的值，读 budget.summarize() 的输出。整本账的准头反倒
#: 与它无关：五档的 abs_pct_error 都在 0.02%~0.34%，因为第 2 层的段比值在
#: 元素列表这种「段内风格稳定」的文本上本来就很准。
#:
#: 键怎么来（**不许手编，必须由代码算**）：
#:   system        harness.prompt.hash_of(渲染后的系统提示词)
#:                 = sha256(PROMPT_VERSION + text)[:12]，与 prompt_hash() 同源，
#:                 提示词或 PROMPT_VERSION 一改它就变、这条常量自动失效落到第 2 层。
#:   tools_schema  budget.account 里已经在算的 schema_hash
#:                 = sha256(json.dumps(tools, sort_keys=True, ensure_ascii=False))
#:
#: 值的来源（2026-09-09 实测，qwen3.7-plus，
#: docs/superpowers/acceptance/2026-09-09-token标定/probe-segments-20260909-154117.json，见
#: scripts/calibrate_tokens.py --probe segments）：
#:   system 1920 —— system_prompt(allow_coord_tap=True, has_skills=True)，3486 字，
#:          实测 0.551 token/字符（字符类模型兜底预测的误差 31.1%，见验收文档）。
#:   tools_schema 5300 —— tool_defs(True) 共 17 个工具，**走 `tools=` 参数发**的实测值。
#:          出处 docs/superpowers/acceptance/2026-09-09-token标定/
#:          probe-tools-reconstant-20260909-merge.json（tools=[] 的 13 与
#:          tools=tool_defs(True) 的 5313 之差）。
#:
#:          ⚠ 这条常量重标过一次：main 合入后工具表从 15 个（4629 token）涨到 17 个
#:          （新增 zoom、erase，5300 token），schema_hash 随之改变，旧的
#:          ("tools_schema", "f2130f12…8831df"): 4629 这条 hash 对不上新工具表、
#:          自动查不到，测试红了两条——安全网设计上就该这样。详见验收文档
#:          「合并触发的重标」一节。15→17 个工具净花了 671 token（4629→5300），
#:          这个对照本身有信息量，所以在这里记一笔，而不是把旧条目留着当"两套并存"。
#:
#:          ⚠⚠ tools= 参数发的实测值**不等于**把同一份 schema 当文本 json.dumps 再估的量
#:          （服务端把工具表渲染成自己的模板，和我们序列化出的 JSON 不是同一段字节）。
#:          15 个工具时两条路差 220 token / 4.8%（4629 vs 4409，textual json 值见
#:          probe-segments-20260909-154117.json）。**17 个工具下这个百分比没有重测**——
#:          别直接拿 4.8% 套到新工具表上，也别拿新的 5300 直接减一个没测过的文本估值。
#:          **这里必须用 tools= 参数量出来的值（5300）**——budget 记的是这次请求真正
#:          花掉的 token，我们的调用方式就是传 tools=。
#:
#: ⚠ 只有 (True, True) 这一组的 system 有实测值。别的开关组合（无坐标 / 无剧本）
#:   hash 不同、查不到，会**自动**落到第 2 层，不会拿这个值顶替。这是设计，不是遗漏。
EXACT_SEGMENT_TOKENS: dict[tuple[str, str], int] = {
    # 2026-09-09 晚：提示词版本 21（七节：你是谁 / 世界怎么运作 / iOS 惯例 / 镜像差别 /
    # 你的手 / 怎么做事 / 每轮看到什么），calibrate_tokens.py --probe segments 重标。4463 字 → 2485 token。
    # 2026-09-09 深夜：加「起点不确定 / 回根路径」三处（ios、world、method），重标。4735 字 → 2645 token。
    # 2026-09-10：method 节补「写前先读当前值」，重标。4804 字 → 2690 token。
    # 2026-09-10 上午：版本 22，手那一节加「交给人」（handover + 安全闸的存在），重标。4898 字 → 2747 token。
    #   同一次工具表 17→18（新增 handover，expect 描述改成「换页动作请填」），5296 → 6087。
    # 2026-09-10 下午（main）：hands/ios 加光标键（key 的 enum 从 4 个到 12 个，工具表一起变），重标。4967 字 → 2763 token，工具表 5384。
    # 2026-09-10 合并两路（handover + 光标键）后重标，并去掉字数上限、恢复为凑上限删过的句子：
    #   5157 字 → 2873 token；工具表 18 个（含 handover，key 的 enum 12 个）6175（走 tools= 参数）。
    #   出处 docs/superpowers/acceptance/2026-09-09-token标定/probe-segments-20260910-150711.json。
    # 2026-09-11：tap 的 expect 改必填、描述改成「」写法（docs/superpowers/specs/2026-09-11-点击预期核对-design.md），
    #   只重标工具表：6226（走 tools= 参数）。出处 docs/superpowers/acceptance/2026-09-11-token标定/probe-segments-20260911-172845.json。
    # 2026-09-11 终审：tap 的 expect 描述换掉「飞行模式」示例、加「别把你要点的那几个字放进「」」，只重标工具表：
    #   6226 → 6265（走 tools= 参数）。出处 docs/superpowers/acceptance/2026-09-11-token标定/probe-segments-20260911-182254.json。
    # 2026-09-1x：按需看图 —— 提示词版本 23（感知一节改为「默认只有 OCR」）、observe 描述改为看全屏，重标。
    #   system 4967→5446 字（2763→3051 token）；工具表仍 18 个（observe 描述改字），6265→6308（走 tools= 参数）。
    #   出处 docs/superpowers/acceptance/2026-09-15-token标定/probe-segments-20260915-115841.json。
    # ⚠ 2026-09-16 公开导出：method 节里的账户示例名改成了通用占位（公开仓库不带个人内容），
    #   prompt_hash 因此从 e9b0ed49927c 变成 42680c245e3a。**3051 这个值没有重测**：改动只是把
    #   一个 4 字的中文示例词换成另一个 4 字的中文示例词，system 段字数不变（仍是 5446 字），
    #   所以沿用了上一次实测值。要拿它当精确值用之前，重跑 --probe segments 重标一次。
    ("system", "42680c245e3a"): 3051,
    ("tools_schema",
     "e13f9bd621eb74014f56a6421041af54766fd9d2ccfd6b15c209b03ed376bcc9"): 6308,
}

#: 第 2 层：每段的 token/字符比。只填**这一段自己量过**（样本 ≥200 字）的值。
#:
#: 2026-09-09 实测，docs/superpowers/acceptance/2026-09-09-token标定/probe-segments-20260909-154117.json
#: （下表「兜底误差」是第 3 层字符类模型在**这一段**上的相对误差 —— 它是这一层
#: 相对兜底的收益，不是这一层自己的误差）：
#:
#:     段                    token/字符   样本字符   兜底误差
#:     state_elements        0.884        3254       52.6%
#:     obs_elements          0.885        3247       52.7%
#:     state_last_result     0.564        4009       38.4%
#:     tool_result           0.564        4001       38.5%
#:     assistant_tool_args   0.556        619        38.4%
#:     state_transition      0.652        316        43.7%
#:     memory_index          0.608        398        30.6%
#:     recent_runs           0.608        398        30.6%
#:     state_history         0.503        1026       30.4%
#:     system                0.551        3486       31.1%
#:     tools_schema          0.391        11262      16.2%
#:
#: 三条必须记住的限制：
#:
#:  1. **memory_index 和 recent_runs 是合量测的**，不是分别测的 —— 探针发的
#:     是 memory.injected 这一整段（留档里它本来就是两段 join 后的结果），
#:     没法在真实语料里把它俩拆开。两段填同一个值（0.608）是**近似**，
#:     不是各自的实测；哪天两段能单独取样了，这条近似要拆开重标。
#:  2. system（0.551）和 tools_schema（0.391）只作**第 1 层查不到时的后备**
#:     （正常路径走 EXACT_SEGMENT_TOKENS 的精确常量）。tools_schema 这个比值是
#:     2026-09-09 量在 15 个工具的 **JSON 文本**上的（11262 字 -> 4409 token），
#:     套到 chars=len(blob) 上得到的是「当文本发」的量，当时比 `tools=` 参数的
#:     真实开销（4629）低约 4.8%。
#:     ⚠ 这个比值本身没有跟着 2026-09-09 合并后的重标（15→17 个工具）重测——
#:     17 个工具下 tools= 的真实开销已经是 5300（见 EXACT_SEGMENT_TOKENS 上面
#:     的注），但没有配套的「17 个工具当文本发」的量，所以这里的 4.8% 和 0.391
#:     都还是 15 个工具时代的数，别直接套到新工具表的偏差估计上。
#:     查不到常量时这是我们能给的最好的数，但它系统性偏低，别拿它去卡阈值。
#:  3. **没有填** task（89 字）、assistant_text（73 字）、state_step（8 字）——
#:     样本太短，±1 token 的量化误差就能把比值挪好几个百分点（state_step
#:     净 9 token / 8 字符，比值 1.125，量一次和量两次就能差出 10%+）。
#:     这三段继续走第 3 层兜底，在 SEGMENTS_PENDING_CALIBRATION 里单独分组注明。
#: state_elements / obs_elements 元素列表 digit/sym 占比高、词表命中差，
#:   兜底误差 52%+ 是本表里最大的，分层的收益在这一段最明显。
SEGMENT_TOKENS_PER_CHAR: dict[str, float] = {
    "state_elements": 0.884,
    "obs_elements": 0.885,
    "state_last_result": 0.564,
    "tool_result": 0.564,
    "assistant_tool_args": 0.556,
    "state_transition": 0.652,
    "memory_index": 0.608,
    "recent_runs": 0.608,
    "state_history": 0.503,
    "system": 0.551,
    "tools_schema": 0.391,
}

#: 已知存在、但**还没有实测比值**的段。列出来是为了让「没标定」这件事显形：
#: 它们现在一律走第 3 层字符类兜底，budget 里会标成 char_class。
#:
#: ⚠ 不要给这里的段瞎填数字。scripts/calibrate_tokens.py --probe segments 会
#:   逐段发真实语料把它们量出来；量到了再往 SEGMENT_TOKENS_PER_CHAR 里填，
#:   量不到的（造不出真实语料的）就继续留在这里。
#:
#: 2026-09-09 这轮标定把能测的都挪进了 SEGMENT_TOKENS_PER_CHAR，剩下两类
#: 原因不同，分开列，别混在一起看：
SEGMENTS_PENDING_CALIBRATION: frozenset[str] = frozenset({
    # ── 完全没有语料（10 个）：探针里从没造过这些段的真实文本，无从测起 ──
    "chat_history", "skill_index", "scenario", "app_note",
    "state_memo", "state_report",
    "obs_location", "obs_route", "image_placeholder_text",
    # 2026-09-09 加「暂停 → 补一句话 → 继续」：window 视图里是一条普通用户消息（user_note），
    # state 视图里是状态报告的一段（state_user_note）。人打的短句，量不出可信比值。
    "user_note", "state_user_note",
    # repair_prompt 是 loop.py 在「模型只回文字、没调工具」时补发的一句固定
    # 短串（16 字）。2026-09-09 补进来是因为
    # tests/test_tokens.py::test_every_production_segment_name_is_accounted_for
    # 扫出它两个表都不在 —— 也就是它一直在静默走第 3 层兜底而没人知道。
    # 和 task/state_step 同理：太短，量它没意义，但**必须显形**。
    "repair_prompt",
    # ── 有样本，但样本太短（3 个）：量过，量出来的比值不可信 ──────────
    #   task 89 字、assistant_text 73 字、state_step 8 字（净 9 token，
    #   比值 1.125）——这么短的文本，±1 token 的量化误差就能把比值挪好几个
    #   百分点，2026-09-09 没有据此填 SEGMENT_TOKENS_PER_CHAR。
    "task", "assistant_text", "state_step",
})


def estimate_segment(seg: str, text: str, *, key: str | None = None) -> tuple[int, str]:
    """按段估 token，返回 (token 数, 用了哪一层)。

    查表顺序就是分层顺序：exact -> segment_ratio -> char_class。

    ⚠ 观测设施绝不顶掉任务：key 给了但查不到、段名不认识、text 不是字符串 ——
      一律**降级到下一层**，不抛、不报错。最坏情况就是退回原来那套字符类估算，
      也就是这次改动之前的行为。

    ⚠ 这个函数**不保证**对文本追加单调不减（第 1 层是常量、第 2 层换比值时
      会跳变）。单调性是 estimate_text 的性质，被 tests/test_tokens.py 钉着；
      分层这一层的语义是「哪个数字最接近真值」，不是「构造一个单调泛函」。
    """
    if not isinstance(text, str):
        text = ""
    if isinstance(seg, str):
        if key is not None:
            v = EXACT_SEGMENT_TOKENS.get((seg, key))
            if v is not None:
                return int(v), LAYER_EXACT
        ratio = SEGMENT_TOKENS_PER_CHAR.get(seg)
        if ratio is not None:
            return int(round(len(text) * ratio)), LAYER_SEGMENT_RATIO
    return estimate_text(text), LAYER_CHAR_CLASS


@dataclasses.dataclass(frozen=True)
class ImageRule:
    """某个模型的图片 token 规则。字段对应服务端的处理顺序。

    ⚠ 这些不是拟合参数，是查官方文档抄下来的规则（spec §2.6、T0-3）。
      标定脚本用若干分辨率**验证**它，不用来反推它。

    ⚠ 验证结论有边界：2026-09-09 的实测只覆盖了「正常范围」和「超过 max_pixels」
      两条分支，**低于 min_pixels 的放大分支实测与公式不符**。详见 apply_image_rule。
    """
    pixels_per_token: int        # 每个视觉 token 覆盖多少像素
    constant_tokens: int         # 固定项（有的模型每张图额外加几个 token）
    size_multiple: int           # 服务端把边长取整到它的倍数（官方 smart_resize 的 factor）
    min_pixels: int | None       # 取整后面积小于它，按原始边长等比放大
                                 # ⚠ 这条分支 2026-09-09 实测**不成立**（小图像有 ~72
                                 #   token 的地板），详见 apply_image_rule 的说明。
    max_pixels: int | None       # 取整后面积大于它，按原始边长等比缩小
                                 # ✅ 2026-09-09 实测吻合（1535x3415 -> 2477，逐 token 命中）

    # ⚠ 官方 smart_resize 里 factor 一身兼二职：既是取整倍数，也是每 token 的
    #   边长（token = 面积 / (factor*factor)）。我们拆成 pixels_per_token 和
    #   size_multiple 两个独立字段，是为了让取整粒度和 token 粒度不同的模型也能
    #   表达。对 qwen3.7-plus 这一代（factor=32），两者满足
    #   pixels_per_token == size_multiple ** 2，即 1024 == 32**2。

    def __post_init__(self) -> None:
        # 标定任务填规则时手抖填反 min/max，会产出一个静默的错数（先缩到
        # max、再放大到 min，结果比 max_pixels 还大好几倍且不报错）——
        # 这正是本模块反复强调要避免的：宁可炸，不可悄悄错。见 fix #2。
        if (self.min_pixels is not None and self.max_pixels is not None
                and self.min_pixels > self.max_pixels):
            raise ValueError(
                f"ImageRule 配置写反了：min_pixels={self.min_pixels} > "
                f"max_pixels={self.max_pixels}，min 应当 <= max")


# 规则来源：阿里云百炼「视觉理解」官方文档（查证日期 2026-09-02），含官方
# smart_resize 参考代码。文档把 qwen3.7-plus 点名归入「Qwen3.8、Qwen3.7、
# Qwen3.6、Qwen3.5、Qwen3-VL 系列」这一组，下面五个参数是这一组共用的。
#
# ⚠ 这些是**查文档查来的规则**，不是标定出来的系数。图片规则对不对，和文本系数
#   有没有标定过是两件独立的事：CALIBRATION_ID 只反映文本那一次标定
#   （2026-09-09 之前它是 "uninitialized"，现在是 "2026-09-09-qwen3.7-plus"），
#   图片规则的正确性由 gate 的 T0-3 和 --probe image 单独验证，不依赖它。
IMAGE_RULES: dict[str, ImageRule] = {
    "qwen3.7-plus": ImageRule(
        # 32×32：官方 smart_resize 的 factor=32，token = 面积 /(factor*factor)。
        # ⚠ 28×28 是 QVQ / Qwen2.5-VL 那一代的 factor，别搞混（同一篇文档里
        #   两组模型分列两张表）。
        pixels_per_token=1024,
        # <vision_bos> + <vision_eos>，官方公式里的 "+ 2"（文档 2026-09-02）。
        constant_tokens=2,
        size_multiple=32,
        # 4 * factor * factor = 4096。官方参考代码里写死，不随请求参数变。
        min_pixels=4096,
        # 默认 max_pixels（文档 2026-09-02）。
        # ⚠ 不是 16_777_216（= 16384 * 32 * 32）——那是 vl_high_resolution_images
        #   = true 时服务端强制换上的值。我们的请求不传这个参数：
        #   model/providers/alibaba.py 的 extra_body 只有 enable_thinking，
        #   transports/chat_completions.py 的 request_kwargs 也不加它。
        #   哪天开始传了，这里要同步换成 16_777_216。
        max_pixels=2_621_440,
    ),
}


def apply_image_rule(width_px: int, height_px: int, rule: ImageRule) -> int | None:
    """忠实实现阿里云官方的 smart_resize（文档 2026-09-02 的参考代码）。

    顺序是：**先**把边长「就近」取整到 size_multiple 的倍数，**再**用取整后
    的面积判是否越界；越界了才按**原始边长**等比缩放并重新取整。

    ⚠ 三处极易写错、且写错了不会炸只会给出一个小幅偏离的错数：
      1. 正常范围内是 round（就近），不是 ceil。624x1388 用 ceil 会得到
         640x1408 → 882 token，官方是 640x1376 → 862 token，差 2.3%。
      2. 判越界用的是**取整后**的面积（h_bar*w_bar），不是原始面积。所以存在
         「原始面积没超 max、取整后超了」的图，这种图官方会真的缩小。
      3. 越界后两个分支的取整函数不同：缩小用 floor、放大用 ceil；且基准都是
         原始边长除/乘 beta，不是已经取整过的边长。
      这三条各有一条测试钉着，见 tests/test_tokens.py。

    ⚠ 取整之后面积仍可能小幅超出 min/max_pixels——这是官方算法本身的性质
      （它只缩放一次、不迭代），不是需要「修掉」的 bug。见 fix #3。

    ⚠⚠ **放大分支（面积 < min_pixels）没有通过实测验证，而且实测与官方公式对不上。**
      2026-09-09 --probe image 在 qwen3.7-plus 上的八档（实测 = 有图请求减去
      除图片外逐字节相同的无图请求）：

          尺寸          实测    本函数预测    结论
          32x48          72         8        不符
          32x67          74         8        不符
          43x95          74         8        不符
          61x135         74        10        不符
          624x1388      862       862        吻合（真机截图尺寸）
          768x1707     1274      1274        吻合
          1086x2415    2552      2552        吻合
          1535x3415    2477      2477        吻合（走 max_pixels 的 floor 缩小分支）

      四个大尺寸**逐 token 精确命中**，含跨过 max_pixels 阈值的那一档 ——
      正常分支和缩小分支是验过的。四个极小尺寸全部不符：实测像是有一个
      ~72 token 的地板（面积翻 4 倍只从 72 涨到 74），而官方 min_pixels=4096
      的等比放大公式给出的是 8~10。也就是说服务端在小图上做的事不是
      「按 4096 像素等比放大再算面积」，具体是什么我们不知道。

      为什么不动它：
        1. 我们只发全屏截图（最小 624x1388 = 866,112 像素，是 min_pixels 的 211 倍），
           这条分支在生产路径上**永远走不到**。
        2. 只有四个点、且四个点几乎都压在同一个地板上，不足以定出一条规则。
           照着这四个点凑参数，就是拿没验证的东西冒充验证过的 —— 本模块整篇
           都在反对这件事。
        3. 官方 smart_resize 的放大分支照抄得没错（tests/test_tokens.py 里
           test_rule_scales_up_with_ceil_when_below_min_pixels 钉着算法本身）。
           错的是「服务端真按这个公式算小图」这个假设，不是这几行代码。

      为什么也没让它返回 None（评审时权衡过）：
        a. None 在 budget 里的语义是「不认识的模型 / 拿不到分辨率」，也就是
           「不知道」。小图不是不知道 —— 我们知道真值大概是 72~74，只是不知道
           规则。把两件事塞进同一个 unknown 段，会让 unknown 段失去诊断价值。
        b. 这条分支在生产路径上走不到，改行为收益为零；而 estimate_image 是
           已验证函数，给它加第二套语义（大图给数、小图给 None）是拿确定的东西
           换不确定的东西。
        c. 真要改也只是一行（放大分支 return None）+ 改两条测试；哪天真开始发
           小图，先跑 --probe image 把那条地板量清楚，再决定是给规则还是给 None。

    宽/高非正（<=0）返回 None 而不是负数或抛除零异常：截图失败/黑屏会给
    上游传 0，bug 会传负数，这条路径是真实存在的（下游 budget.account()
    读图片 part 的 `_px` 键，缺省值就是 (0, 0)）。见 fix #1，呼应
    「不认识就 None，不猜」的原则——负数 token 或崩溃都比 None 更糟。
    """
    if width_px <= 0 or height_px <= 0:
        return None
    w, h = float(width_px), float(height_px)
    m = rule.size_multiple
    # round_by_factor：官方用的是 Python 内建 round（banker's rounding），
    # 这里照抄，不换成 floor(x+0.5)——半数边界上两者会分叉。
    w_bar, h_bar = round(w / m) * m, round(h / m) * m
    if rule.max_pixels is not None and w_bar * h_bar > rule.max_pixels:
        beta = math.sqrt((w * h) / rule.max_pixels)
        w_bar = math.floor(w / beta / m) * m
        h_bar = math.floor(h / beta / m) * m
    elif rule.min_pixels is not None and w_bar * h_bar < rule.min_pixels:
        beta = math.sqrt(rule.min_pixels / (w * h))
        w_bar = math.ceil(w * beta / m) * m
        h_bar = math.ceil(h * beta / m) * m
    return int(w_bar * h_bar) // rule.pixels_per_token + rule.constant_tokens


def estimate_image(width_px: int, height_px: int, *, model_id: str) -> int | None:
    """不认识的模型、或非正的宽高，返回 None，由 budget 记进 unknown 段。"""
    if width_px <= 0 or height_px <= 0:
        return None
    rule = IMAGE_RULES.get(model_id)
    return None if rule is None else apply_image_rule(width_px, height_px, rule)
