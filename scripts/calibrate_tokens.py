"""token 估算系数的离线标定。发真实请求，花真钱，人手动跑。

⚠ 不在 CI 跑、不在任务里跑。结果手动填进 harness/tokens.py 的常量，
  并把 CALIBRATION_ID 改成验收文档的名字。

  三条硬门（T0）不过就不许写系数 —— 它们是整个估算方案的隐含假设，
  写成可执行的门就是为了不让它们变成静默假设（spec §6.1）。


════════════════ 给用户的操作手册（按顺序跑） ════════════════

前置：`export DASHSCOPE_API_KEY=...`（或写进 .iphone/config.toml 并 chmod 600）。
每一步都可以先加 `--dry-run` 看清将要发多少请求、多少 token，**dry-run 一个请求都不发**。
除 `--dry-run` 外，每个探针发请求前都会打印成本预估并要求敲 y 确认（`--yes` 跳过）。

  0)  .venv/bin/python scripts/calibrate_tokens.py --check-drift runs
      零成本，不发请求。先看看现有 runs 里的自然差分误差有多大、样本够不够。
      pairs 少于 ~30 时中位数没有意义，先去多跑几个真机任务。

  1)  先查官方文档，把目标模型的图片规则填进 tokens.IMAGE_RULES
      （IMAGE_RULES 现在是空字典，所以 T0-3 必然 FAIL —— 那是设计本意，不是 bug）。
      DashScope 的 Qwen-VL 系文档给的是「每 token 覆盖多少像素、边长取整到多少的倍数、
      像素数上下限、每张图的固定加项」这四项，一一对应 ImageRule 的四个字段。

  2)  .venv/bin/python scripts/calibrate_tokens.py --preflight
      8 次请求，输入合计 ~1 万 token（其中 8500 是全量工具表，发了两遍；另加一条 max_tokens=64 对照）。
      验证 usage 字段路径、tools 是否计入
      prompt_tokens、工具顺序是否影响计数、max_tokens=1 是否被接受且不改 prompt 计数、
      同分辨率不同内容/不同 PNG 编码是否等价、model_version 是否足以绑定这次标定。

  3)  .venv/bin/python scripts/calibrate_tokens.py --gate
      4 次请求（IMAGE_RULES 填了之后 8 次），输入合计 ~9000 token
      （T0-2 的长前缀本身 ~4200 token，发两遍）。
      全过打印 GATE PASS。任一不过 → **不许写系数**，先按打印出来的提示处理。

  4)  .venv/bin/python scripts/calibrate_tokens.py --probe cache
      24 次请求（12 个前缀长度 × 2 遍），输入合计 ~3 万 token —— 前缀越长越贵，
      因为它要把前缀从明显低于 1024 token 扫到明显高于。产出「cached_tokens 从哪个
      前缀长度开始非零」，回答 docs/19 §3.6 的立项问题。

  5)  .venv/bin/python scripts/calibrate_tokens.py --probe text
      37 次请求（9 条纯语料 × 4 个长度 + 1 条基线 + 4 条真实语料），
      输入合计 ~3 万 token（按当前未标定的初值估，真实值会更高 —— 先 --dry-run 看）。
      产出 tokens.TOKENS_PER_CHAR 的每个桶。

      ⚠ 分两段看输出，**验收判据是第二段**：
        P1-1 纯语料拟合  → 每个桶的系数 + 该桶纯语料上的斜率与残差。
        P1-2 真实语料检验 → 真机 OCR 元素列表 / 真 system prompt / 真工具表 JSON /
                            真记忆索引上的「预测 vs 实测」和相对误差。
        P1-3 一句结论    → 真实语料上的相对误差 <= 10% 才算成功、才许填系数。

      ⚠ 老标准（「拟合残差小就算成功」）是错的，2026-09-09 栽在这上面：三桶模型
        在每条合成语料上单独看都完美线性（斜率稳到 3-4 位有效数字），套到真实
        混排文本上却差 2.6 倍。合成语料标出来的系数套不到真实文本上。
        真实语料上误差还是大时，**不许填数字**，按 P1-3 打印的两条岔路走。

      ⚠ 在 git worktree 里跑时 runs/ 是空的，真实记忆语料会缺席。想要它就
        `export IPHONE_CALIB_RUNS=/path/to/主checkout/runs`。

  6)  .venv/bin/python scripts/calibrate_tokens.py --probe wrap
      9 次请求，输入合计 ~2700 token —— 全脚本最便宜的一步。产出 budget.OVERHEAD_PER_MESSAGE 与
      budget.OVERHEAD_PER_SEGMENT_JOIN，以及**端点收不收哪几种消息形状**。
      每条请求独立记账：被 400 拒掉的那条只丢自己的数，不再一条抛、全组死。

  7)  .venv/bin/python scripts/calibrate_tokens.py --probe tools
      N+2 次请求（N 是工具条数，默认 15），输入合计 ~4 万 token —— 逐个累加工具意味着
      工具表被反复重发，这是仅次于 cache 的第二贵。产出 tools_schema 的
      总量校验，以及「工具顺序是否影响计数」。

  8)  .venv/bin/python scripts/calibrate_tokens.py --probe image
      2×梯子档数 次请求（默认 8~14 次），文本部分只有几百 token，成本几乎全在图片上。**验证** IMAGE_RULES，
      不拟合它。分辨率梯子由 IMAGE_RULES 里的 min_pixels/max_pixels 生成，保证跨过阈值。

  9)  .venv/bin/python scripts/calibrate_tokens.py --probe segments
      2 + 段数 次请求（当前 ~15 次），输入合计 ~2 万 token。**按段**标定：
      产出 tokens.EXACT_SEGMENT_TOKENS（system / tools_schema 的精确常量，
      连代码现算的 hash 键一起打印，直接抄）和 tokens.SEGMENT_TOKENS_PER_CHAR
      （每段的 token/字符），以及一张「哪些段没有真实语料、为什么」的清单。
      ⚠ 在 worktree 里跑要先 export IPHONE_CALIB_RUNS=<主 checkout>/runs，
        否则大半个段都会报「无语料」。--dry-run 会先把这张清单打给你看。

  10) 填回代码：
      - tokens.py: EXACT_SEGMENT_TOKENS / SEGMENT_TOKENS_PER_CHAR ← 第 9 步
                   （分层估算的第 1、2 层，收益最大的一块）
      - tokens.py: TOKENS_PER_CHAR 的每个桶 ← 第 5 步（**且第 5 步的 P1-3 说「可以填」**）；
                   IMAGE_RULES ← 第 1 步（第 8 步验证过；⚠ 放大分支实测不成立，见
                   tokens.apply_image_rule 的说明，别拿小尺寸的不吻合去改规则）
      - budget.py: OVERHEAD_PER_MESSAGE / OVERHEAD_PER_SEGMENT_JOIN ← 第 6 步
      - tokens.py: CALIBRATION_ID ← "2026-09-09-<gate 打印出来的 model_version>"
      每个探针的原始数据都写在 --out 目录（默认 runs/calibrate-tokens/）里，验收文档引它。

  10) 替换测试：把 tests/test_tokens.py 的 `test_image_rules_is_empty_until_t0_3_passes`
      **替换**（不是删除）成断言具体规则内容的测试，commit 里写明依据的官方文档版本。
      再按 spec §4.4 写 docs/superpowers/acceptance/2026-09-09-token标定.md：机型、模型 id、
      返回的 model_version、日期、每个系数的值与残差、样本数、探针原始数据路径。

  11) .venv/bin/python -m pytest -q  → 全绿再提交。

⚠ 「max_tokens=1 所以成本可忽略」是错的：探针的输入本身就是大段真实文本，
  那才是主要成本。所以每个探针都在发请求前把输入 token 打出来。
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import math
import os
import pathlib
import random
import statistics
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from iphone_agent.harness import budget as budget_mod
from iphone_agent.harness import tokens
from iphone_agent.harness.tools import tool_defs

# ============================ 探针语料 ============================
# 语料分两类，用途**完全不同**，绝不能混着用：
#
#   纯语料（PURE_ALPHABETS）——「拟合」用。每一条几乎 100% 落进一个桶，所以
#     该桶的系数可以从这条语料的斜率**直接读出来**，不用解方程，也就没有
#     「几个系数互相污染、残差看着还行但每个都错」的问题。
#
#   真实语料（build_real_corpora）——「检验」用。它们是模型真正每一步看到的东西
#     （真机 OCR 元素列表、真 system prompt、真工具表 JSON、真记忆索引）。
#     **验收判据只看它们**：拟合残差小说明不了什么，真实语料上的相对误差小才算数。
#
# ⚠ 为什么必须这么分（2026-09-09 的教训）：老探针只有合成语料，标出来
#   TOKENS_PER_ASCII_CHAR = 0.185 —— 那是**重复英文串**压出来的数。同一天的
#   mixed 语料（长得像真实元素行）实测 0.700 token/字符，用那套系数预测只有
#   0.268，差 2.6 倍。反推出来真实文本里 ASCII 部分是 ~0.82 token/字符。
#   原因是 `[12]` `(170,405)` `0.99` 这类数字/括号每个几乎自成一个 token，
#   而重复的英文单词压缩得极好 —— **分词效率取决于实际词表命中，不是字符类**。
#   合成语料标出来的系数套不到真实文本上，这是整套方法当时最大的漏洞。

PURE_ALPHABETS = {
    # 汉字本体：常用汉字，不含任何标点、不含 ASCII。100% han 桶。
    "han": "国家标准化管理委员会发布实施细则通知关于进一步加强城市建设管理工作的意见",
    # 中文标点：全角标点 + CJK 标点。100% cjk_punct 桶（0x3000-0x303F / 0xFF00-0xFFEF）。
    # 2026-09-09 实测 0.933 token/字符，是汉字的 2.79 倍 —— 这一条就是把
    # 全角标点从 cjk 桶里拆出来的直接证据。
    "cjk_punct": "，。！？；：、（）《》【】〈〉",
    # 通用标点里中文实际在用的那一段（0x2010-0x2027）。tokens.py 把它并进了
    # cjk_punct 桶，但那是**假设**：老的标点语料是 87% 全角 + 13% 这一段的固定
    # 混合，而任何固定配比都是线性的，光看那条语料**证伪不了**两者不同价。
    # 这条纯语料就是来证伪它的：如果斜率和 cjk_punct 明显不同，就得再拆一个桶。
    "west_punct": "—…“”‘’–―‖†‡•",
    # 纯 ASCII 字母，**不含空格** —— 老语料里字母和空格混在一起，两个桶解不开。
    "ascii_alpha": "thequickbrownfoxjumpsoverthelazydogwhileparsingtokens",
    # 纯 ASCII 数字。怀疑它是真实混排贵出来的主因（元素行里全是坐标和置信度），
    # 老探针里一条纯数字语料都没有。
    # ⚠ 数字串刻意不用 0123456789 这种规整递增：递增串在 BPE 里压得异常好，
    #   量到的会是「最省的数字」而不是「一般的数字」—— 那正是重复英文串犯过的错。
    "ascii_digit": "8137502946271840653917248605392718460537",
    # 纯 ASCII 符号：空格、换行、括号、标点。元素行的骨架（`[` `]` `(` `,` `)` 空格 换行）
    # 全在这里。老探针同样没有。
    "ascii_sym": " []() ,.;:/-_=+*&^%$#@!?~|<>\"' \n",
    # 纯 emoji：主体是星平面码点。
    # ⚠ 9 个码点里有 1 个 U+FE0F（BMP，落 other 桶）—— 所以这条语料不是 100% 纯，
    #   astral 的系数要靠它和 other_bmp 两条一起解。
    "emoji": "😀🎉🚀🌟🔥🧭📱🛠️",
    # 其余 BMP：希腊字母 + 西里尔 + 假名。它本身在我们的真实文本里几乎不出现，
    # 但 other 桶必须有一条自己的语料，否则 emoji 语料里那个 U+FE0F 无处安放，
    # astral 的系数就会被它污染。
    "other_bmp": "αβγδεζηθικλμνжзийклмнопカタカナひらがなアイウエオ",
    # 受控混排：人手编的、长得像真实元素行的一段。它**不参与拟合**，和真实语料
    # 一起进检验组 —— 2026-09-09 就是这一条把三桶模型证伪的（实测 0.700 / 预测 0.268）。
    "mixed": "元素 [12] 设置 (170,405) 0.99 通用 About 关于本机\n",
}

#: 参与拟合的语料。mixed 明确排除在外：它是检验组的一员。
FIT_KINDS = tuple(k for k in PURE_ALPHABETS if k != "mixed")

# 每一类的长度梯子（字符数）。0 是基线：它把 system/tools/包装这些恒定量整个消掉。
LENGTHS = (0, 200, 500, 1000, 2000)

#: 真实语料上可接受的相对误差。超过它就**不许把系数填进 tokens.py**。
#:
#: 为什么是 10%：budget 这一层的用途是「哪一段吃掉了多少上下文」和「离窗口上限
#: 还有多远」。段占比在 10% 的估算误差下不改变排序结论，用来定阈值时留一档余量
#: 也还够。再松就没意义了 —— 20% 的误差下 "state_elements 占 30%" 和 "占 36%"
#: 分不开，这个数就不能拿来做决策。
REAL_CORPUS_MAX_REL_ERROR = 0.10

# 真实语料的来源。⚠ 这些路径指向仓库根，脚本从任何目录跑都要能找到。
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCREENMAP_PATH = _REPO_ROOT / ".iphone" / "screenmap.json"
# ⚠ 在 git worktree 里跑时 runs/ 是空的（历史 run 都在主 checkout）。
#   IPHONE_CALIB_RUNS 指过去就能取到真实记忆语料；取不到就少一条语料，不报错。
RUNS_ROOT = pathlib.Path(os.environ.get("IPHONE_CALIB_RUNS") or (_REPO_ROOT / "runs"))


def _real_elements_text(limit_lines: int = 120) -> str | None:
    """从 .iphone/screenmap.json 拼一段**真机元素列表**。

    这是模型每一步真正看到的最大一段文本，也是最难估的一段（中文标签 + 方括号
    编号 + 坐标数字 + 置信度小数混在一起）。

    ⚠ 为什么不从历史 runs 里直接读：`steps.jsonl` 里**没有** elements_text ——
      `_` 开头的进程内材料不落盘。screenmap 是唯一一条能拿到真机 OCR 文字的路。

    ⚠ 文字是真的（4382 条真机 OCR），**坐标和置信度是合成的**：screenmap 只存
      文字，不存框。合成值用确定性伪随机铺在 0-1000（norm1000 就是这个量程），
      位数分布和真实坐标一致 —— 对分词来说要紧的正是「三位数、逗号、两位小数」
      这个形状，不是具体数值。这一点必须写在验收文档里，别让人以为整行都是实测。

    行格式抄自 perceive/elements.py 的 build_observation：
        [{id}] {text} ({cx},{cy}) {conf:.2f}
    头部那行也照抄（norm1000 分支），它每步都在。
    """
    try:
        data = json.loads(SCREENMAP_PATH.read_text(encoding="utf-8"))
        nodes = data.get("nodes") or []
    except Exception as e:                      # noqa: BLE001 —— 语料取不到就降级，不砸整个探针
        print(f"  ⚠ 读不到 {SCREENMAP_PATH}（{type(e).__name__}），跳过 real_elements 语料")
        return None
    # ⚠ 全池确定性抽样，不是「取前 N 条」：前 N 条来自同一两屏，一屏的文字风格
    #   高度相关（全是设置项、或全是单词卡），量到的是那一屏的分词效率而不是
    #   「真实屏幕」的。也不按屏幕轮转 —— 那会让每屏的第一条（多半是状态栏
    #   "1,119"、"12:21" 这种纯数字）占满整段语料，同样不具代表性。
    pool = [t for n in nodes if isinstance(n, dict)
            for t in (n.get("representative") or [])
            if isinstance(t, str) and t.strip()]
    if not pool:
        return None
    texts = random.Random(20260909).sample(pool, min(limit_lines, len(pool)))
    # 确定性伪随机：同一份 screenmap 每次跑出同一段语料，两次标定可比。
    rnd = random.Random(20260909)
    lines = [f"observation #7  图像 624x1388 像素；"
             f"**下面每个元素的坐标、以及你给 tap 的 x,y，统一用 0-1000 的归一化值**"
             f"（x 按宽、y 按高各自折算）。"]
    for i, t in enumerate(texts, start=1):
        lines.append(f"[{i}] {t} ({rnd.randrange(1000)},{rnd.randrange(1000)}) "
                     f"{rnd.uniform(0.60, 1.0):.2f}")
    return "\n".join(lines)


def _real_memory_text() -> str | None:
    """历史 run.json 的 memory.injected —— 真实的记忆索引 / 最近运行段。

    ⚠ 大多数 run 没有这个字段（只有开了记忆层的那些有），所以要挑最长的一条，
      挑不到就返回 None 让这一条语料整个缺席，不要拿空串顶上去。
    """
    best: tuple[int, str] | None = None
    try:
        paths = sorted(RUNS_ROOT.glob("*/run.json"))
    except Exception:                            # noqa: BLE001
        return None
    for path in paths:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                        # noqa: BLE001 —— 单个 run 坏了就跳过
            continue
        mem = d.get("memory") if isinstance(d, dict) else None
        inj = mem.get("injected") if isinstance(mem, dict) else None
        if isinstance(inj, str) and inj.strip() and (best is None or len(inj) > best[0]):
            best = (len(inj), inj)
    if best is None:
        print(f"  ⚠ {RUNS_ROOT} 下没有带 memory.injected 的 run，跳过 real_memory 语料")
    return best[1] if best else None


def build_real_corpora() -> dict[str, str]:
    """检验用的真实语料。取不到的那条**缺席**，不拿合成品顶替。

    每一条都是主循环真的会发出去的字节：
      real_system_prompt  ← harness.prompt.system_prompt()（一字不改）
      real_tools_json     ← harness.tools.tool_defs() 的 JSON，序列化方式与
                            budget.account 里 tools_schema 那段**完全一致**
                            （sort_keys=True, ensure_ascii=False）—— 两边不一致
                            的话，这里验过的系数到了 budget 里仍然是错的
      real_elements       ← 真机 OCR 文字拼成的元素列表（见 _real_elements_text）
      real_memory         ← 历史 run.json 的 memory.injected
      mixed               ← 人手编的元素行样本，留作最便宜的回归点
    """
    from iphone_agent.harness.prompt import system_prompt
    out: dict[str, str] = {}
    out["real_system_prompt"] = system_prompt(allow_coord_tap=True, has_skills=True)
    out["real_tools_json"] = json.dumps(tool_defs(True), sort_keys=True, ensure_ascii=False)
    el = _real_elements_text()
    if el:
        out["real_elements"] = el
    mem = _real_memory_text()
    if mem:
        out["real_memory"] = mem
    out["mixed"] = PURE_ALPHABETS["mixed"] * 8       # 单条 40 字太短，重复到 320 字
    return out


# ============================ 按段标定的语料（P4）============================
# 和上面「拟合 / 检验」两组语料的用途都不同：这一组是**按段**的，每一条都要
# 尽量就是 loop.py 装配时那一段的真实形状（前缀、标题、截断额度全都照搬）。
#
# ⚠ 铁律：造不出真实样本的段就**明确报「无语料，未标定」**，不拿别的段的文本
#   顶替，也不合成一段「看起来差不多」的。合成语料标出来的系数套不到真实文本上
#   ——2026-09-09 三桶模型就是这么翻的车（重复英文串 0.185 vs 真实 ASCII 0.82）。

#: 段的短样本告警阈值（字符）。比这还短的段，±1 token 的量化误差就能把比值
#: 挪动好几个百分点 —— 量到了也只能当参考，不该直接填进 SEGMENT_TOKENS_PER_CHAR。
SEGMENT_MIN_CHARS_FOR_RATIO = 200


def _iter_step_records(limit_runs: int = 40):
    """真机 steps.jsonl 里的逐步记录，新的在前。取不到就是空 —— 不合成。"""
    try:
        files = sorted(RUNS_ROOT.glob("*/steps.jsonl"), reverse=True)[:limit_runs]
    except Exception:                            # noqa: BLE001
        return
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:                        # noqa: BLE001 —— 单个 run 坏了就跳过
            continue
        for ln in lines:
            try:
                r = json.loads(ln)
            except Exception:                    # noqa: BLE001
                continue
            if isinstance(r, dict) and isinstance(r.get("action"), dict):
                yield r


def _seg_state_history(records: list[dict]) -> str | None:
    """【到目前为止】那一段。真记录 -> row_from_record -> render_history，
    和 loop.py:512 走的是同一条路，形状逐字符一致。"""
    from iphone_agent import config
    from iphone_agent.harness.history import render_history, row_from_record
    rows = []
    for r in records[:config.HISTORY_KEEP * 2]:
        try:
            rows.append(row_from_record(r, r.get("_elements_by_id")))
        except Exception:                        # noqa: BLE001 —— 单条不成形就跳过
            continue
    if not rows:
        return None
    # step 号在留档里是真号，render_history 直接用；这里补一个连续序号，
    # 免得抽样出来的行全叫同一个 step。
    import dataclasses as _dc
    rows = [_dc.replace(r, step=str(i + 1)) for i, r in enumerate(rows)]
    return render_history(rows, config.HISTORY_KEEP)


def _seg_last_result(records: list[dict]) -> str | None:
    """【上一步结果】。内容是 ToolResult.to_json() 按 LATEST_TOOL_RESULT_MAX_CHARS
    截断后的串（loop._bound_result）。这里直接拿留档里的 result 重建同一段字节。"""
    from iphone_agent import config
    best = ""
    for r in records:
        res = r.get("result")
        if isinstance(res, dict):
            s = json.dumps(res, ensure_ascii=False)
            if len(s) > len(best):
                best = s
    if not best:
        return None
    # ⚠ 真实额度是 LATEST_TOOL_RESULT_MAX_CHARS（20000），但**比值和长度无关**
    #   （P1 里每条纯语料跨 200→2000 字斜率都稳到 3~4 位有效数字），
    #   而按 20000 字发两遍要多花两万 token。这里砍到 4000 字：字节仍是真的
    #   （同一份 result 的前缀），只是短一截。要量「20000 字这一档会不会非线性」
    #   得另开一条长度梯子，那是别的问题。
    cap = min(4000, config.LATEST_TOOL_RESULT_MAX_CHARS)
    return best[:cap] + "…" if len(best) > cap else best


def _seg_transition(records: list[dict]) -> str | None:
    """【上一步之后】。留档里的 transition 字典能原样重建 Transition 并 render。"""
    from iphone_agent.perceive.transition import Transition, render
    best: str | None = None
    for r in records:
        t = r.get("transition")
        if not isinstance(t, dict):
            continue
        try:
            tr = Transition(changed=bool(t["changed"]), local_changed=t.get("local_changed"),
                            added=tuple(t.get("added") or ()), removed=tuple(t.get("removed") or ()),
                            added_total=int(t.get("added_total") or 0),
                            removed_total=int(t.get("removed_total") or 0))
            s = render(tr)
        except Exception:                        # noqa: BLE001
            continue
        if best is None or len(s) > len(best):
            best = s
    return best


def _seg_from_model_field(records: list[dict], field: str) -> str | None:
    """model.reason / model.memory 这类模型自己写的字段里最长的一条。"""
    best: str | None = None
    for r in records:
        m = r.get("model")
        v = m.get(field) if isinstance(m, dict) else None
        if isinstance(v, str) and v.strip() and (best is None or len(v) > len(best)):
            best = v
    return best


def _seg_tool_args(records: list[dict]) -> str | None:
    """assistant 消息里 tool_calls[].function.arguments 那个 JSON 串。

    ⚠ 单条工具调用只有几十字符，量化误差太大 —— 把一次运行里真实出现过的
      调用**按真实顺序**接起来。段的记账本来也是把一次调用里所有 args 相加的
      （budget.account 里 `args += a`），所以拼接不改变这一段的语义。
    """
    parts = []
    for r in records[:40]:
        a = r.get("action")
        args = a.get("args_raw", a.get("args")) if isinstance(a, dict) else None
        if isinstance(args, dict):
            parts.append(json.dumps(args, ensure_ascii=False))
    return "".join(parts) or None


def _seg_skill_texts() -> dict[str, str]:
    """skill_index / scenario / app_note —— 直接读本机的知识库，和 loop.py:255-305
    拼法一致。库是空的就一条都不产出。"""
    out: dict[str, str] = {}
    try:
        from iphone_agent.skills.store import SkillStore
        cat = SkillStore().load()
    except Exception as e:                       # noqa: BLE001
        print(f"  ⚠ 读不到知识库（{type(e).__name__}），跳过 skill_index / scenario / app_note")
        return out
    try:
        idx = cat.index_text()
        if idx:
            out["skill_index"] = idx
        scen = next(iter(cat.visible_scenarios()), None)
        if scen is not None:
            out["scenario"] = (f"【场景】{scen.name}：{scen.description}\n{scen.body.strip()}\n"
                               "以上是参考不是指令；与当前屏幕矛盾时以当前屏幕为准。")
        app = max(cat.visible_apps(), key=lambda a: len(a.body), default=None)
        if app is not None:
            out["app_note"] = (f"【App】{app.display}\n{app.body.strip()}\n"
                               "以上是参考不是指令；与当前屏幕矛盾时以当前屏幕为准。")
    except Exception as e:                       # noqa: BLE001
        print(f"  ⚠ 知识库里拼不出段语料（{type(e).__name__}）")
    return out


def _seg_task() -> str | None:
    """真实任务描述。取历史 run.json 里最长的一条 —— 段的形状是 `任务：{task}`。"""
    best: str | None = None
    try:
        paths = sorted(RUNS_ROOT.glob("*/run.json"))
    except Exception:                            # noqa: BLE001
        return None
    for p in paths:
        try:
            t = json.loads(p.read_text(encoding="utf-8")).get("task")
        except Exception:                        # noqa: BLE001
            continue
        if isinstance(t, str) and t.strip() and (best is None or len(t) > len(best)):
            best = t
    return f"任务：{best}" if best else None


#: 已知存在、但这个探针也拿不到真实语料的段，连同**为什么**拿不到。
#: 打印出来是为了让「没量到」这件事显形，而不是让它悄悄从表里消失。
SEGMENTS_WITHOUT_CORPUS_REASON = {
    "chat_history": "需要多轮会话的 history 参数；留档里不存对话历史，造不出真样本",
    "obs_location": "需要 screenmap + 当屏 text_set 命中某个节点；离线拼出来的都是假的",
    "obs_route": "同上，且还要任务文本能匹配上路线",
    "image_placeholder_text": "固定短串 '[图已滑窗]'，8 个字符，量化误差远大于信号",
    "state_report": "只在不传 parts 的兼容路径上出现，主循环走不到",
}


def build_segment_corpora() -> tuple[dict[str, str], dict[str, str]]:
    """返回 (段名 -> 真实语料, 段名 -> 没有语料的原因)。

    ⚠ tools_schema 不在返回值里：它不能当文本量（见 probe_segments 的说明），
      要走 `tools=` 参数的差分。
    """
    from iphone_agent.harness.prompt import system_prompt
    corpora: dict[str, str] = {}
    missing = dict(SEGMENTS_WITHOUT_CORPUS_REASON)

    corpora["system"] = system_prompt(allow_coord_tap=True, has_skills=True)

    el = _real_elements_text()
    if el:
        # 两个入口同一种内容：window 模式叫 obs_elements，state 模式在前面多一个
        # 【当前屏幕】标题（build_state_parts）。分别量，别假定两者比值相同。
        corpora["obs_elements"] = el
        corpora["state_elements"] = "【当前屏幕】\n" + el
    else:
        missing["obs_elements"] = missing["state_elements"] = \
            f"读不到 {SCREENMAP_PATH}"

    mem = _real_memory_text()
    if mem:
        # ⚠ run.json 的 memory.injected 是 memory_index 和 recent_runs **拼在一起**
        #   的（loop.py:317 那个 join）。留档里分不开，所以这里只能给一个合量。
        #   要分开量，得先让 run.json 分别留档两段 —— 那是另一个任务。
        corpora["memory_index+recent_runs(合量)"] = mem
    else:
        missing["memory_index"] = missing["recent_runs"] = \
            f"{RUNS_ROOT} 下没有带 memory.injected 的 run"

    records = list(_iter_step_records())
    if records:
        h = _seg_state_history(records)
        if h:
            corpora["state_history"] = h
        lr = _seg_last_result(records)
        if lr:
            corpora["state_last_result"] = "【上一步结果】\n" + lr
            # window 模式下同一份内容作为 tool 消息发出，没有标题。
            corpora["tool_result"] = lr
        tr = _seg_transition(records)
        if tr:
            corpora["state_transition"] = tr
        memo = _seg_from_model_field(records, "memory")
        if memo:
            corpora["state_memo"] = "【备忘】\n" + memo
        else:
            missing["state_memo"] = "留档里的 model.memory 全是 null（模型没写过备忘）"
        reason = _seg_from_model_field(records, "reason")
        if reason:
            corpora["assistant_text"] = reason
        args = _seg_tool_args(records)
        if args:
            corpora["assistant_tool_args"] = args
    else:
        for s in ("state_history", "state_last_result", "tool_result", "state_transition",
                  "state_memo", "assistant_text", "assistant_tool_args"):
            missing[s] = f"{RUNS_ROOT} 下没有可用的 steps.jsonl（worktree 里要设 IPHONE_CALIB_RUNS）"

    corpora.update(_seg_skill_texts())
    for s in ("skill_index", "scenario", "app_note"):
        if s not in corpora:
            missing[s] = "本机知识库里没有这一类条目"

    t = _seg_task()
    if t:
        corpora["task"] = t
    else:
        missing["task"] = f"{RUNS_ROOT} 下没有带 task 的 run.json"

    corpora["state_step"] = "第 7/30 步"
    # 有语料的段不该再出现在 missing 里（两边都在会让人以为它没量到）
    for k in corpora:
        missing.pop(k, None)
    return corpora, missing


# 缓存探针的前缀长度梯子（CJK 字符数）。按 0.7 token/汉字的初值粗算，
# 500 字 ≈ 350 token（明显低于 1024），4000 字 ≈ 2800 token（明显高于）。
# ⚠ 用的是**待标定的初值**来选梯子，所以梯子必须跨得足够宽，宽到初值哪怕错一倍
#   也仍然跨过 1024 —— 这就是这里从 300 一直排到 4000 的原因。
CACHE_PREFIX_CHARS = (300, 500, 700, 900, 1100, 1300, 1600, 2000, 2500, 3000, 3500, 4000)


# ============================ 请求计划 ============================

@dataclasses.dataclass(frozen=True)
class Request:
    """一条待发请求。构造出来先进计划，确认成本之后才发。

    messages 已经是**出网形状**（不带任何内部 `_` 键）：本脚本不走 MessageLog，
    它要的正是「除待测变量外逐字节相同」，自己拼比借用视图函数更可控。
    """
    label: str
    messages: list[dict]
    tools: list[dict] | None            # None = 用 profile 默认工具表；[] = 明确不带工具
    image_px: tuple[int, int] = (0, 0)  # 只用于成本预估；没有图就是 (0, 0)
    meta: dict = dataclasses.field(default_factory=dict)
    max_tokens: int | None = None       # None = 用 runner.max_tokens；填了就覆盖 —— 只用于
                                         # 「同 messages、只变 max_tokens」的 T0 前置对照组


def _text_tokens_of(messages: list[dict]) -> int:
    """计划里所有文本的估算量。用的是**待标定的初值**，只用来估成本，不用来下结论。"""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += tokens.estimate_text(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    total += tokens.estimate_text(p["text"])
        for tc in m.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments")
            if isinstance(args, str):
                total += tokens.estimate_text(args)
    return total


def estimate_plan(plan: list[Request], model_id: str, max_tokens: int) -> dict:
    """成本预估。图片那一项在 IMAGE_RULES 填之前是 None —— 报 None 而不是 0：
    「不知道」和「零」是不同的事实，把不知道写成 0 就是在瞒报成本。"""
    text = 0
    tools_est = 0
    img: int | None = 0
    for r in plan:
        text += _text_tokens_of(r.messages)
        if r.tools:
            tools_est += tokens.estimate_text(
                json.dumps(r.tools, sort_keys=True, ensure_ascii=False))
        if r.image_px != (0, 0):
            one = tokens.estimate_image(*r.image_px, model_id=model_id)
            img = None if (one is None or img is None) else img + one
    return {"requests": len(plan), "text_tokens": text, "tools_tokens": tools_est,
            "image_tokens": img, "output_tokens": len(plan) * max_tokens}


def _fmt_estimate(est: dict) -> str:
    img = "未知（IMAGE_RULES 还没填，这部分成本估不出来）" if est["image_tokens"] is None \
        else f"{est['image_tokens']}"
    return (f"  请求数        : {est['requests']}\n"
            f"  输入文本 token: {est['text_tokens']}\n"
            f"  工具表 token  : {est['tools_tokens']}\n"
            f"  图片 token    : {img}\n"
            f"  输出 token    : {est['output_tokens']}（max_tokens 上限 × 请求数）\n"
            f"  ⚠ 主要成本在输入，不在输出 —— 别因为 max_tokens=1 就当它免费。")


# ============================ 执行器 ============================

class Runner:
    """发请求 + 成本护栏 + 原始数据留档。dry-run 下 send() 一定不会被调到。"""

    def __init__(self, transport, *, dry_run: bool, yes: bool, out_dir: pathlib.Path,
                 max_tokens: int = 1):
        self.transport = transport
        self.dry_run = dry_run
        self.yes = yes
        self.out_dir = out_dir
        self.max_tokens = max_tokens
        self.model_id = getattr(transport, "resolved", None) and transport.resolved.model.id or "?"
        self.records: list[dict] = []
        self.model_versions: set[str] = set()

    def confirm(self, name: str, plan: list[Request]) -> bool:
        """打印成本并要确认。返回 False 表示不要继续（dry-run 或用户拒绝）。"""
        print(f"\n=== {name} ===")
        print(_fmt_estimate(estimate_plan(plan, self.model_id, self.max_tokens)))
        for r in plan:
            print(f"    - {r.label}")
        if self.dry_run:
            print("  [dry-run] 不发任何请求。")
            return False
        if self.yes:
            return True
        try:
            ans = input("继续？[y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans != "y":
            print("  已放弃。")
            return False
        return True

    def send(self, req: Request) -> dict:
        """发一条，返回 usage 摘要。dry-run 下这个方法根本不该被走到。"""
        if self.dry_run:
            raise RuntimeError("dry-run 下不许发请求 —— 调用方漏了 confirm() 的返回值判断")
        mt = req.max_tokens if req.max_tokens is not None else self.max_tokens
        reply = self.transport.decide(req.messages, req.image_px or (0, 0),
                                      tools=req.tools, max_tokens=mt)
        u = reply.usage if isinstance(reply.usage, dict) else {}
        cached = u.get("cached_tokens")
        if cached is None and isinstance(u.get("prompt_tokens_details"), dict):
            cached = u["prompt_tokens_details"].get("cached_tokens")
        rec = {"label": req.label, "prompt_tokens": u.get("prompt_tokens"),
               "completion_tokens": u.get("completion_tokens"),
               # ⚠ cached 保持 None/0 的区分：0 是「确实没命中」，None 是「这个端点没报」。
               #   把 None 折成 0 会让「不支持隐式缓存」看起来像「一直没命中」。
               "cached_tokens": cached,
               "model_version": reply.model_version, "latency_ms": reply.latency_ms,
               "raw_usage": u, "meta": dict(req.meta)}
        if isinstance(reply.model_version, str) and reply.model_version:
            self.model_versions.add(reply.model_version)
        self.records.append(rec)
        return rec

    def prompt_tokens(self, req: Request) -> int | None:
        r = self.send(req)
        v = r["prompt_tokens"]
        return int(v) if isinstance(v, int) else None

    def try_send(self, req: Request) -> dict:
        """发一条，端点拒绝也不炸整套探针 —— 把拒绝理由记进记录并继续。

        ⚠ 只在**故意要试探端点收不收某种形状**的探针里用（P2 消息包装）。
          其它探针里一条发不出去就意味着这一组差分作废，那种情况该炸就炸，
          不能把 None 混进差分里算出一堆看着正常的错数。

        ⚠ 2026-09-09 踩到的：P2 里有一种消息形状被阿里云拒了
          （InvalidParameter: The provided messages input is invalid.
            The error info is [Can only get item pairs from a mapping.]），
          第一条就抛，整个探针一个数都没产出、连原始数据都没落盘。
          现在每条独立记账，跑完还能告诉你**是哪一种形状**不合法。
        """
        try:
            return self.send(req)
        except Exception as e:                    # noqa: BLE001 —— 这里就是要把拒绝当数据收下来
            rec = {"label": req.label, "prompt_tokens": None, "completion_tokens": None,
                   "cached_tokens": None, "model_version": None, "latency_ms": None,
                   "raw_usage": {}, "meta": dict(req.meta),
                   "error": f"{type(e).__name__}: {e}"}
            self.records.append(rec)
            print(f"  ✗ {req.label}：端点拒绝 —— {type(e).__name__}: {str(e)[:300]}")
            return rec

    def dump(self, name: str, extra: dict) -> pathlib.Path | None:
        """原始数据留档。验收文档要引这个路径（spec §4.4）。"""
        if self.dry_run or not self.records:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        p = self.out_dir / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
        p.write_text(json.dumps(
            {"probe": name, "model_id": self.model_id,
             "model_versions": sorted(self.model_versions),
             "calibration_id_at_run": tokens.CALIBRATION_ID,
             "records": self.records, **extra},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  原始数据: {p}")
        return p


# ============================ 消息构造 ============================
# 本脚本所有差分的通用规矩：**同一组内的两条请求，除待测变量外逐字节相同**。
# 实现手段就是所有请求都从下面这几个函数出，变量只从参数进。

_SYSTEM = "你在操作一台 iPhone。只用工具回复。"


def _probe_messages(payload: str, *, nonce: str = "") -> list[dict]:
    """最朴素的一条 user 文本消息。nonce 放在最前面用来破缓存。

    ⚠ 整套探针**共用同一个 nonce**，不是每个相邻请求换一个字符：换字符会在
      「待测变量」之外引入第二个变化量，差分就不再只反映待测变量（Codex 阻断 #4）。
      同一个 nonce 意味着同一套探针内部会互相命中前缀缓存 —— 这不要紧，因为
      T0-2 正是在验「缓存命中不改变 prompt_tokens」。T0-2 不过时，本脚本所有
      差分结论一并作废。
    """
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": (nonce + payload) if nonce else payload}]


def _repeat_to(alphabet: str, n: int) -> str:
    """把字母表重复/截断到恰好 n 个字符。截断可能切断一个 emoji 的 ZWJ 序列，
    但 count_chars 是按 code point 数的，切断后仍然是自洽的 —— 拟合用的
    自变量和实际发出去的字节始终是同一段文本。"""
    if n <= 0:
        return ""
    reps = math.ceil(n / len(alphabet))
    return (alphabet * reps)[:n]


def _png_b64(w: int, h: int, *, noisy: bool = False) -> str:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), "white")
    if noisy:
        # 故意让 PNG 压不动：base64 长度会差一两个数量级。用来验证
        # 「同一分辨率、不同编码长度 → 图片 token 相同」这条前置断言。
        d = ImageDraw.Draw(img)
        for y in range(0, h, 3):
            for x in range(0, w, 7):
                d.point((x, y), fill=((x * 37) % 256, (y * 91) % 256, ((x + y) * 13) % 256))
    else:
        ImageDraw.Draw(img).rectangle([0, 0, max(1, w // 2), max(1, h // 2)], fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _image_messages(text: str, b64: str | None) -> list[dict]:
    """带图/不带图的两条消息，**除 image part 外逐字节相同**。"""
    content: list[dict] = [{"type": "text", "text": text}]
    if b64 is not None:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": content}]


def assert_only_one_difference(a: Request, b: Request, *, field: str) -> None:
    """把「除待测变量外逐字节相同」变成可执行的断言（spec §6.2 最后一条）。

    field 是允许不同的那一项：messages / tools。其余项必须逐字节相同 ——
    包括 tools、image 有无、以及 transport 拼出来的两级 extra_body 和
    omit_params（那些由同一个 transport 拼，只要 tools/max_tokens 一致就一致）。
    """
    def blob(r: Request, key: str):
        return json.dumps(getattr(r, key), sort_keys=True, ensure_ascii=False, default=str)
    for key in ("messages", "tools"):
        same = blob(a, key) == blob(b, key)
        if key == field and same:
            raise AssertionError(f"{a.label} 与 {b.label} 的 {key} 完全相同 —— 待测变量没变")
        if key != field and not same:
            raise AssertionError(
                f"{a.label} 与 {b.label} 除 {field} 外还有第二处不同：{key}。"
                f"差分组内出现第二个变化量，这一组的结论作废。")


# ============================ 纯计算：拟合 ============================

def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """高斯消元（部分主元）。奇异返回 None —— 不返回一组「看起来正常的」错数。"""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit_char_coefficients(samples: list[tuple[str, int]],
                          buckets: tuple[str, ...] = tokens.BUCKETS) -> dict:
    """按字符类做最小二乘：tokens ≈ Σ coeff[bucket] * count[bucket] + intercept。

    桶名由 tokens.BUCKETS 决定，**不写死** —— 分桶方案改了这里跟着改，
    不需要动这个函数。

    ⚠ 只喂**纯语料**。真实语料要留作检验：拿它一起拟合等于把判据和被判者混在
      一起，残差自然好看，但那时候「误差小」不再说明任何事情。

    ⚠ **必须报残差**，但更要紧的是：残差小**不**说明模型成立。2026-09-09 的
      教训是三桶模型在每条合成语料上单独看都是完美线性（斜率稳到 3-4 位有效
      数字），套到真实文本上却差 2.6 倍。判据在 evaluate_on_real_corpora 那边。

    intercept 吸收 system/tools/消息包装这些恒定量；它**不是**要填回代码的东西，
    OVERHEAD_PER_MESSAGE 由 P2 单独测。

    ⚠ 某个桶在所有样本里都是 0（没给它纯语料）时，正规方程奇异 —— 这里不返回
      「看起来正常的」错数，而是把那些桶的名字点出来，告诉调用方缺哪条语料。
    """
    rows: list[list[float]] = []
    ys: list[float] = []
    for text, measured in samples:
        if not isinstance(measured, int) or isinstance(measured, bool):
            continue
        c = tokens.count_chars(text)
        rows.append([float(c.get(b, 0)) for b in buckets] + [1.0])
        ys.append(float(measured))
    n = len(rows)
    k = len(buckets) + 1
    if n < k:
        return {"n": n, "buckets": list(buckets),
                "note": f"样本不足 {k} 条，{k} 个未知数解不出来"}
    empty = [b for i, b in enumerate(buckets) if all(r[i] == 0.0 for r in rows)]
    if empty:
        return {"n": n, "buckets": list(buckets), "empty_buckets": empty,
                "note": f"这些桶在所有样本里计数都是 0：{empty} —— "
                        f"给每个桶补一条几乎 100% 落它的纯语料，否则它的系数无从谈起"}
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    atb = [sum(rows[t][i] * ys[t] for t in range(n)) for i in range(k)]
    sol = _solve(ata, atb)
    if sol is None:
        return {"n": n, "buckets": list(buckets),
                "note": "正规方程奇异 —— 多半是某两个桶在所有语料里配比恒定"
                        "（自变量线性相关）。给它们各补一条纯语料。"}
    resid = [ys[t] - sum(rows[t][i] * sol[i] for i in range(k)) for t in range(n)]
    return {"n": n, "buckets": list(buckets),
            "coefficients": {b: sol[i] for i, b in enumerate(buckets)},
            "intercept": sol[-1],
            "residuals": resid,
            "max_abs_residual": max(abs(r) for r in resid),
            "rms_residual": math.sqrt(sum(r * r for r in resid) / n)}


def predict_tokens(text: str, fit: dict) -> float | None:
    """用拟合结果预测一段文本的**内容** token（不含 intercept）。

    不含 intercept 是刻意的：检验组要回答的是「这段文本本身估得准不准」，
    把恒定的 system/包装开销算进去会把相对误差稀释掉，看着比实际好。
    """
    coeff = fit.get("coefficients")
    if not isinstance(coeff, dict):
        return None
    c = tokens.count_chars(text)
    return sum(c.get(b, 0) * v for b, v in coeff.items())


def evaluate_on_real_corpora(fit: dict, samples: list[tuple[str, str, int]],
                             baseline_tokens: int | None) -> dict:
    """**验收判据在这里**：拟合出来的系数在真实语料上准不准。

    samples 是 [(名字, 文本, 实测 prompt_tokens), …]；baseline_tokens 是同一次
    探针里 0 字请求的 prompt_tokens（system + 包装的恒定量）。
    实测内容 token = prompt_tokens - baseline，和 predict_tokens 对齐。

    ⚠ baseline 拿不到就整个降级成「算不了」，**不要**拿 fit 的 intercept 顶替：
      intercept 是拟合出来的，用它当基线等于让被检验的模型自己定义判据。
    """
    rows = []
    for name, text, measured in samples:
        pred = predict_tokens(text, fit)
        content = (measured - baseline_tokens) if (_is_int(measured)
                                                   and _is_int(baseline_tokens)) else None
        rel = (abs(pred - content) / content
               if pred is not None and _is_int(content) and content > 0 else None)
        rows.append({"corpus": name, "chars": len(text), "prompt_tokens": measured,
                     "measured_content_tokens": content, "predicted_content_tokens": pred,
                     "rel_error": rel,
                     "measured_tokens_per_char": (content / len(text)
                                                  if _is_int(content) and text else None),
                     "buckets": tokens.count_chars(text)})
    usable = [r["rel_error"] for r in rows if r["rel_error"] is not None]
    return {"rows": rows,
            "max_rel_error": max(usable) if usable else None,
            "threshold": REAL_CORPUS_MAX_REL_ERROR,
            # ⚠ 一条都算不出来时 verdict 是 None（「不知道」），不是 False，
            #   更不是 True。把「没数据」折成任何一个结论都是在编。
            "passes": (max(usable) <= REAL_CORPUS_MAX_REL_ERROR) if usable else None}


# ============================ 纯计算：C2 自然差分 ============================

def check_drift_from_budgets(runs: list[list[dict]]) -> dict:
    """整个可变尾部的持续 sanity check。纯函数，不读盘（所以单测不用造 runs/ 目录）。

    ⚠ 只比「整个可变尾部的变化量」，**不声称验证各段绝对系数**（Codex 阻断 #2）：
      各段在同一条 content 里用 "\\n\\n" 拼接，tokenizer 在边界会 merge；一次调用里
      多段同时变；恒定的 system / tools / 固定前缀在差分里整个消失，根本不在被
      验证之列。单段系数只能来自 §4.1 的受控探针。

    ⚠ 用 call_index 而不是 steps 做索引：一次有效 step 可能对应多次模型调用。
    ⚠ prefix_drift 为真的调用整段跳过：那些调用的前缀变了，差分没有意义。
    ⚠ 只取 context_mode == "state"：window 模式的尾部构成是另一回事，混在一起
      得到的中位数谁也代表不了。
    """
    pairs: list[tuple[int, int]] = []
    calls_seen = 0
    for run in runs if isinstance(runs, list) else []:
        budgets = _usable_budgets(run)
        calls_seen += len(budgets)
        for prev, cur in zip(budgets, budgets[1:]):
            if prev.get("prefix_drift") or cur.get("prefix_drift"):
                continue
            if not _is_int(prev.get("actual_prompt_tokens")) or not _is_int(cur.get("actual_prompt_tokens")):
                continue
            if not _is_int(prev.get("est_total")) or not _is_int(cur.get("est_total")):
                continue
            pairs.append((cur["actual_prompt_tokens"] - prev["actual_prompt_tokens"],
                          cur["est_total"] - prev["est_total"]))
    if not pairs:
        # ⚠ pairs 一定要出现在返回值里，包括 0 的情况：调用方对「没数据」和
        #   「误差很小」的处置完全不同。
        return {"pairs": 0, "calls_considered": calls_seen,
                "note": "没有可用的相邻调用对（可能全是 window 模式或缺 usage）"}
    errs = [abs(e - a) / max(1, abs(a)) for a, e in pairs]
    # ⚠ p90 的索引用 int()（截断）在 n=2 时算出 int(0.9*1)=0 —— 取到的是
    #   **最小**误差，会打印出「p90 < median」这种自相矛盾的数字（reviewer 实测
    #   pairs=2 时出过 median 0.0310 / p90 0.0204）。中位数在样本少时「没意义但
    #   仍报出来讓人判断」是本模块的既定原则（pairs 本身就是那个信号），但 p90
    #   不一样：索引法本身错了，报出来的不是「没意义的正确值」而是「错误值」，
    #   两者不能同等对待。这里改成向上取整索引（budget.summarize 的 _pct 用的是
    #   round-最近，同样不会出现 p90 < median；这里选 ceil 是因为它对任意 n 都
    #   保证单调不减、不依赖四舍五入的边界行为，更适合小样本场景下的保守估计）。
    p90_idx = min(len(errs) - 1, math.ceil(0.9 * (len(errs) - 1)))
    return {"pairs": len(pairs), "calls_considered": calls_seen,
            "abs_rel_error_median": statistics.median(errs),
            "abs_rel_error_p90": sorted(errs)[p90_idx]}


def _is_int(v) -> bool:
    """bool 是 int 的子类，但 True 不是一个 token 数。"""
    return isinstance(v, int) and not isinstance(v, bool)


def _usable_budgets(run: list[dict]) -> list[dict]:
    """一个 run 里按 call_index 排好序的 state 模式 budget。

    形状不对的（不是 dict、budget_error 占位符、缺 call_index）整条丢掉：这是
    观测设施，读到坏数据要少算一对，不要抛出去。
    """
    out = [b for b in (run if isinstance(run, list) else [])
           if isinstance(b, dict) and "budget_error" not in b
           and b.get("context_mode") == "state" and _is_int(b.get("call_index"))]
    return sorted(out, key=lambda b: b["call_index"])


def _budgets_of(run_dir: pathlib.Path) -> list[dict]:
    """从一个 run 目录的 steps.jsonl 里抠出 budget 字段。

    ⚠ 同一个 call_index 可能在 jsonl 里出现多次：loop.py 把同一条 call_budget
      挂在这次调用产生的每一条 step 记录上（模型回复里多个动作 → 多条 step）。
      按 call_index 去重，否则同一次调用会被当成好几次，差分对里混进一堆 0。
    """
    path = run_dir / "steps.jsonl"
    if not path.exists():
        return []
    by_call: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        b = rec.get("budget") if isinstance(rec, dict) else None
        if isinstance(b, dict) and _is_int(b.get("call_index")):
            by_call.setdefault(b["call_index"], b)
    return list(by_call.values())


def check_drift(runs_root: pathlib.Path) -> dict:
    """读盘 + 调纯函数。读盘的部分只做这一件事，好让上面那个函数可单测。"""
    runs_root = pathlib.Path(runs_root)
    if not runs_root.is_dir():
        return {"pairs": 0, "note": f"{runs_root} 不是目录"}
    runs = [_budgets_of(d) for d in sorted(runs_root.iterdir()) if d.is_dir()]
    out = check_drift_from_budgets(runs)
    out["runs_scanned"] = len(runs)
    return out


# ============================ 前置断言（Step 2）============================

def preflight(runner: Runner) -> bool | None:
    """spec §6.2 的前置断言。不满足就打印原因并返回 False。

    这些断言不是「优化」，是**整套差分方法的前提**：任何一条不成立，后面探针
    量到的差值就不再只反映待测变量。
    """
    nonce = runner_nonce()
    base_txt = _repeat_to(PURE_ALPHABETS["han"], 400)
    full_tools = tool_defs(True)
    rev_tools = list(reversed(full_tools))
    img_a = _png_b64(256, 256)
    img_b = _png_b64(256, 256, noisy=True)

    plan = [
        Request("A 基线（无 tools）", _probe_messages(base_txt, nonce=nonce), []),
        Request("B 同 A 重发（usage 字段/确定性）", _probe_messages(base_txt, nonce=nonce), []),
        Request("C 全量 tools（tools 是否计入 prompt_tokens）",
                _probe_messages(base_txt, nonce=nonce), full_tools),
        Request("D 全量 tools 倒序（顺序是否影响计数）",
                _probe_messages(base_txt, nonce=nonce), rev_tools),
        Request("E 无图基线", _image_messages(base_txt, None), []),
        Request("F 256x256 平坦 PNG", _image_messages(base_txt, img_a), [], image_px=(256, 256)),
        Request("G 256x256 噪声 PNG（base64 长得多）",
                _image_messages(base_txt, img_b), [], image_px=(256, 256)),
        # H：跟 A 除 max_tokens 外逐字节相同 —— 专门验证简报 Step 2 点名的那条：
        # max_tokens=1 被接受，且**不改变 prompt 计数**。用一个明显更大的 max_tokens
        # 做对照，而不是拿 A 自己（A 本身就是拿 runner.max_tokens 跑的，不能自证）。
        Request("H max_tokens=64（与 A 同 messages，只变 max_tokens）",
                _probe_messages(base_txt, nonce=nonce), [], max_tokens=64),
    ]
    # 逐字节相同的检查是**本地**做的，一个请求都不用发。
    assert_only_one_difference(plan[0], plan[2], field="tools")
    assert_only_one_difference(plan[2], plan[3], field="tools")
    assert_only_one_difference(plan[4], plan[5], field="messages")
    assert_only_one_difference(plan[5], plan[6], field="messages")
    assert_only_one_difference(plan[0], plan[7], field="max_tokens")
    print("前置断言（本地部分）：同组请求除待测变量外逐字节相同 -> PASS")

    if not runner.confirm("前置断言（Step 2）", plan):
        # None：中止（dry-run 或用户在确认处敲了非 y）——**不是** FAIL。
        # dry-run 成功跑完、用户主动放弃都不该被 main() 当成断言失败去报退出码 1。
        return None

    r = {p.label: runner.send(p) for p in plan}
    ok = True

    def pt(label) -> int | None:
        v = r[label]["prompt_tokens"]
        return v if _is_int(v) else None

    a, b = pt("A 基线（无 tools）"), pt("B 同 A 重发（usage 字段/确定性）")
    if a is None or b is None:
        print("前置断言 usage: prompt_tokens 不是整数或拿不到 -> FAIL"
              "（usage 字段路径与类型在这个模型/区域不成立，后面全部免谈）")
        return False
    print(f"前置断言 usage: prompt_tokens={a} 是整数 -> PASS")

    c, d = pt("C 全量 tools（tools 是否计入 prompt_tokens）"), pt("D 全量 tools 倒序（顺序是否影响计数）")
    if c is None or d is None:
        print("前置断言 tools: 带 tools 的请求没拿到 usage -> FAIL")
        ok = False
    else:
        counted = c != a
        print(f"前置断言 tools 计入 prompt_tokens: 无 {a} / 有 {c}（差 {c - a}）-> "
              f"{'PASS（计入）' if counted else 'FAIL（不计入，tools_schema 段的 est 全是虚的）'}")
        ok &= counted
        same_order = c == d
        print(f"前置断言 tools 顺序无关: 正序 {c} / 倒序 {d} -> "
              f"{'PASS' if same_order else 'FAIL（顺序影响计数，tools_schema 必须按真实顺序序列化）'}")
        ok &= same_order

    e, f_, g = pt("E 无图基线"), pt("F 256x256 平坦 PNG"), pt("G 256x256 噪声 PNG（base64 长得多）")
    if None in (e, f_, g):
        print("前置断言 图片编码无关: 有请求没拿到 usage -> FAIL")
        ok = False
    else:
        same = (f_ - e) == (g - e)
        print(f"前置断言 图片编码无关: 平坦 +{f_ - e} / 噪声 +{g - e} -> "
              f"{'PASS' if same else 'FAIL（base64 长度影响计数 —— 图片不是按分辨率计费，'
                                     'estimate_image 的整个形状要改）'}")
        ok &= same

    h = pt("H max_tokens=64（与 A 同 messages，只变 max_tokens）")
    if a is None or h is None:
        print("前置断言 max_tokens 不改变 prompt 计数: 拿不到 usage -> FAIL")
        ok = False
    else:
        mt_free = a == h
        print(f"前置断言 max_tokens 不改变 prompt 计数: max_tokens={runner.max_tokens} 时 prompt={a} "
              f"/ max_tokens=64 时 prompt={h} -> "
              f"{'PASS' if mt_free else 'FAIL（max_tokens 本身被计入某种预算，所有探针的 '
                                        'max_tokens=1 假设作废，成本要重新算）'}")
        ok &= mt_free

    mv = sorted(runner.model_versions)
    enough = len(mv) == 1 and mv[0] and mv[0] != runner.model_id
    print(f"前置断言 model_version 足以绑定标定: {mv} -> "
          f"{'PASS' if enough else 'WARN'}")
    if not enough:
        # ⚠ 这条只警告不拦：qwen3.7-plus 是滚动别名（model/providers/alibaba.py:23），
        #   端点回的 model 字段如果就等于别名本身，这次标定绑不到具体快照 ——
        #   标定文档里必须写清楚「绑不到快照」，而不是假装绑上了。
        print("   ⚠ 返回的 model_version 等于别名本身或有多个：这次标定绑不到具体模型快照。"
              "在验收文档里明确写下这一点，别让后来人以为系数对得上某个固定版本。")

    print(f"\n前置断言：{'全部 PASS' if ok else '有 FAIL —— 先解决它们，别往下跑探针'}")
    runner.dump("preflight", {"ok": ok})
    return ok


# ============================ 三条硬门（Step 1）============================

def gate(runner: Runner, *, image_sizes=((256, 256), (624, 1388))) -> bool | None:
    """三条硬门。任一不过就返回 False，调用方打印「不许写系数」后退出 1。"""
    nonce = runner_nonce()
    ok = True

    # 先把整个门的成本一次报清楚：三条门加起来的请求都在这里。
    long_prefix_text = _repeat_to(PURE_ALPHABETS["han"], 6000)
    plan = [
        Request("T0-1 第一次", _probe_messages("确定性探针" + _repeat_to(PURE_ALPHABETS["han"], 300),
                                             nonce=nonce), []),
        Request("T0-1 第二次", _probe_messages("确定性探针" + _repeat_to(PURE_ALPHABETS["han"], 300),
                                             nonce=nonce), []),
        Request("T0-2 长前缀第一次", _probe_messages(long_prefix_text, nonce=nonce), []),
        Request("T0-2 长前缀第二次", _probe_messages(long_prefix_text, nonce=nonce), []),
    ]
    mid = _target_model_key(runner)
    rule = tokens.IMAGE_RULES.get(mid)
    img_plan: list[tuple[int, int, Request, Request]] = []
    if rule is not None:
        for w, h in image_sizes:
            b64 = _png_b64(w, h)
            base = Request(f"T0-3 {w}x{h} 无图", _image_messages("图片探针", None), [])
            with_img = Request(f"T0-3 {w}x{h} 有图", _image_messages("图片探针", b64), [],
                               image_px=(w, h))
            assert_only_one_difference(base, with_img, field="messages")
            img_plan.append((w, h, base, with_img))
            plan += [base, with_img]
    else:
        # 早说：dry-run 也要看得见这条，否则用户只看到「4 个请求」，不知道 T0-3 压根没排进计划。
        print(f"⚠ IMAGE_RULES 里没有 {mid} → T0-3 必然 FAIL，它的请求也不会排进计划。"
              "这不是 bug：先查官方文档把 ImageRule 填进 tokens.IMAGE_RULES，再回来跑这条**验证**它。")

    if not runner.confirm("三条硬门（T0）", plan):
        # None：中止（dry-run 或用户在确认处敲了非 y）——**不是** FAIL，理由同 preflight()。
        return None

    # T0-1：同一请求重发两次，prompt_tokens 相同。
    #   不成立 → §4.3 的 C2 失效，探针要改成多次取中位数。
    a = runner.prompt_tokens(plan[0])
    b = runner.prompt_tokens(plan[1])
    good = a is not None and a == b
    print(f"T0-1 重发确定性: {a} vs {b} -> {'PASS' if good else 'FAIL'}")
    if not good:
        print("   ⚠ 重发不确定 → C2 自然差分失效，所有受控探针必须改成同一请求多次取中位数，"
              "成本乘以采样次数。")
    ok &= good

    # T0-2：缓存命中不改变 prompt_tokens，只改变 cached_tokens。
    #   不成立 → 所有差分组必须彻底破缓存，探针成本上升。
    u1 = runner.send(plan[2])
    u2 = runner.send(plan[3])
    same = _is_int(u1["prompt_tokens"]) and u1["prompt_tokens"] == u2["prompt_tokens"]
    hit = _is_int(u2["cached_tokens"]) and u2["cached_tokens"] > 0
    print(f"T0-2 缓存不改 prompt_tokens: prompt {u1['prompt_tokens']}/{u2['prompt_tokens']}, "
          f"cached {u1['cached_tokens']}/{u2['cached_tokens']} -> "
          f"{'PASS' if same and hit else 'FAIL'}")
    if not hit:
        print("   ⚠ cached_tokens 一直是 0 或 null：可能前缀不够长、该区域不支持隐式缓存，"
              "或者这个端点根本不报这个字段。先跑 --probe cache 看阈值，再回来判这条。")
    if not same:
        print("   ⚠ 命中缓存改变了 prompt_tokens → 所有差分组必须彻底破缓存（每组换 nonce），"
              "探针成本上升。")
    ok &= (same and hit)

    # T0-3：官方图片规则成立。IMAGE_RULES 为空时这条必然不过 —— 那是对的：
    #   先查官方文档把规则填进 tokens.IMAGE_RULES，再跑这条验证。
    if rule is None:
        print(f"T0-3 图片规则: IMAGE_RULES 里没有 {mid} -> FAIL")
        print("   ⚠ 这不是 bug：IMAGE_RULES 故意留空。先查官方文档（每 token 覆盖多少像素、"
              "边长取整到多少的倍数、像素数上下限、每张图固定加项）把 ImageRule 填进 "
              "tokens.IMAGE_RULES，再跑这条来**验证**它。规则是抄来的，不是拟合出来的。")
        ok = False
    else:
        for w, h, base, with_img in img_plan:
            predicted = tokens.apply_image_rule(w, h, rule)
            measured = _image_only_tokens(runner, base, with_img)
            good = predicted == measured
            print(f"T0-3 {w}x{h}: 预测 {predicted} 实测 {measured} -> {'PASS' if good else 'FAIL'}")
            ok &= good

    print("\n" + ("GATE PASS" if ok else
                  "GATE FAIL —— **不许写系数**。上面 FAIL 的那条先解决，"
                  "在它成立之前 tokens.py 的常量和 CALIBRATION_ID 一个字都不许改。"))
    runner.dump("gate", {"ok": ok, "model_key": mid})
    return ok


def _image_only_tokens(runner: Runner, base: Request, with_img: Request) -> int | None:
    """「同一条极短文本 + 一张纯色图」减去「同一条极短文本、无图」。
    两次请求除图片外逐字节相同 —— 这是本脚本所有差分的通用规矩。"""
    a = runner.prompt_tokens(base)
    b = runner.prompt_tokens(with_img)
    return None if (a is None or b is None) else b - a


def _target_model_key(runner: Runner) -> str:
    """IMAGE_RULES 的键。用 profile 里的 model id 而不是返回的 model_version：
    budget.account 传给 estimate_image 的就是 model id，两边必须是同一个键，
    否则标定验的是 A、运行时查的是 B。"""
    return runner.model_id


# ============================ 探针（Step 3）============================

_NONCE = None


def runner_nonce() -> str:
    """整次调用共用一个 nonce。它只出现在最前面，用来把这次跑的前缀和上一次跑的岔开。"""
    global _NONCE
    if _NONCE is None:
        _NONCE = f"[calib {uuid.uuid4().hex}] "
    return _NONCE


def probe_text(runner: Runner) -> dict | None:
    """P1：文本系数。同一 role、同一 content 形状、同一 tools（这里固定为无 tools，
    让 intercept 只吸收 system + 消息包装），只变字符类与长度。

    两组语料、两个用途，输出也分两段打印：
      1. 纯语料 → 拟合每个桶的系数，附该桶纯语料上的斜率与残差。
      2. 真实语料 → 「预测 vs 实测」和相对误差。**这一段才是验收判据。**

    ⚠ 验收标准（2026-09-09 改）：不是「拟合残差小就算成功」。老标准下三桶模型
      在每条合成语料上都完美线性，套到真实文本上却差 2.6 倍。现在的标准是
      **真实语料上的相对误差 <= REAL_CORPUS_MAX_REL_ERROR**；不过就不许把系数
      填进 tokens.py（宁可 None 不可错数）。
    """
    nonce = runner_nonce()
    # n=0 对每一类都是同一条空 payload 请求，只发一次 —— 发多遍是同一个基线花多份钱，
    # 而且多条完全相同的请求会互相污染缓存命中的判断。
    base_req = Request("基线×0", _probe_messages("", nonce=nonce), [],
                       meta={"group": "base", "kind": "base", "chars": 0})
    plan = [base_req]
    plan += [Request(f"{kind}×{n}", _probe_messages(_repeat_to(PURE_ALPHABETS[kind], n),
                                                    nonce=nonce), [],
                     meta={"group": "pure", "kind": kind, "chars": n})
             for kind in FIT_KINDS
             for n in LENGTHS if n > 0]

    real = build_real_corpora()
    for name, text in real.items():
        plan.append(Request(f"真实:{name}", _probe_messages(text, nonce=nonce), [],
                            meta={"group": "real", "kind": name, "chars": len(text)}))

    if not runner.confirm("P1 文本系数（--probe text）", plan):
        return None

    fit_samples: list[tuple[str, int]] = []
    real_samples: list[tuple[str, str, int]] = []
    baseline: int | None = None
    for req in plan:
        r = runner.send(req)
        if not _is_int(r["prompt_tokens"]):
            continue
        payload = req.messages[1]["content"][len(nonce):]
        group = req.meta["group"]
        if group == "base":
            baseline = r["prompt_tokens"]
            fit_samples.append((payload, r["prompt_tokens"]))
        elif group == "pure":
            fit_samples.append((payload, r["prompt_tokens"]))
        else:
            real_samples.append((req.meta["kind"], payload, r["prompt_tokens"]))
        print(f"  {req.label:<24} chars={len(payload):<6} prompt_tokens={r['prompt_tokens']}")

    fit = fit_char_coefficients(fit_samples)

    # ── 1. 每个桶的系数 + 该桶纯语料上的斜率/残差 ──────────────────
    print("\nP1-1 拟合结果（只用纯语料）：")
    if "coefficients" in fit:
        slopes = _pure_corpus_slopes(runner, baseline)
        for b in fit["buckets"]:
            owner = _dominant_corpus_for(b)
            note = ""
            if owner and owner in slopes:
                sl, spread = slopes[owner]
                share = tokens.count_chars(PURE_ALPHABETS[owner]).get(b, 0) / len(PURE_ALPHABETS[owner])
                note = (f"   ← 纯语料 {owner}（该桶占 {share:.0%}）斜率 {sl:.4f} token/字符，"
                        f"跨长度极差 {spread:.4f}")
            print(f"  {b:<12} = {fit['coefficients'][b]:.4f}{note}")
        print(f"  {'intercept':<12} = {fit['intercept']:.4f}"
              f"（0 字基线实测 {baseline}；两者应当接近，差得多说明拟合被某条语料拽偏了）")
        print(f"  样本数 n = {fit['n']}，残差 rms = {fit['rms_residual']:.2f} token，"
              f"最大 |残差| = {fit['max_abs_residual']:.2f} token")
        print("\n  每条纯语料自己的斜率（极差 = 最长档与最短档斜率之差，衡量这条语料线不线性）：")
        for kind in FIT_KINDS:
            if kind not in slopes:
                continue
            sl, spread = slopes[kind]
            cc = tokens.count_chars(PURE_ALPHABETS[kind])
            top = max(cc, key=lambda b: cc[b])
            print(f"    {kind:<12} {sl:.4f} token/字符（极差 {spread:.4f}）"
                  f"  主桶 {top} 占 {cc[top] / len(PURE_ALPHABETS[kind]):.0%}")
        print("    ⚠ cjk_punct 与 west_punct 两条语料都 100% 落 cjk_punct 桶："
              "它们的斜率差多少，就是「把 — … “ ” 并进中文标点桶」这个假设错多少。"
              "差得明显（比如 >15%）就该把 0x2010-0x2027 单独立桶。")
        print("  ⚠ 残差小**不**等于模型成立 —— 每条合成语料单独看都是完美线性，"
              "判据在下面那一段。")
    else:
        print(f"  {fit.get('note')}")

    # ── 2. 真实语料上的预测 vs 实测（验收判据） ────────────────────
    ev = evaluate_on_real_corpora(fit, real_samples, baseline)
    print("\nP1-2 真实语料检验（**这一段才是判据**）：")
    print(f"  {'语料':<20}{'字符':>7}{'实测':>8}{'预测':>9}{'相对误差':>10}{'实测token/字符':>16}")
    for r in ev["rows"]:
        pred = "—" if r["predicted_content_tokens"] is None else f"{r['predicted_content_tokens']:.0f}"
        rel = "—" if r["rel_error"] is None else f"{r['rel_error'] * 100:.1f}%"
        tpc = "—" if r["measured_tokens_per_char"] is None else f"{r['measured_tokens_per_char']:.3f}"
        meas = "—" if r["measured_content_tokens"] is None else str(r["measured_content_tokens"])
        print(f"  {r['corpus']:<20}{r['chars']:>7}{meas:>8}{pred:>9}{rel:>10}{tpc:>16}")

    # ── 3. 一句明确的结论 ─────────────────────────────────────────
    print("\nP1-3 结论：")
    if ev["passes"] is None:
        print("  真实语料一条都没量到（语料取不到，或 usage 缺失）—— **不许填系数**。"
              "先把 .iphone/screenmap.json 和 runs/ 准备好再跑一遍。")
    elif ev["passes"]:
        print(f"  真实语料上最大相对误差 {ev['max_rel_error'] * 100:.1f}% "
              f"<= {REAL_CORPUS_MAX_REL_ERROR * 100:.0f}% 阈值 -> **可以填系数**。"
              "把上面 P1-1 的每个桶抄进 tokens.TOKENS_PER_CHAR，"
              "CALIBRATION_ID 改成验收文档名，并把 P1-2 这张表原样抄进验收文档。")
    else:
        print(f"  真实语料上最大相对误差 {ev['max_rel_error'] * 100:.1f}% "
              f"> {REAL_CORPUS_MAX_REL_ERROR * 100:.0f}% 阈值 -> **不许填系数**"
              "（宁可 None 不可错数）。下一步怎么判：")
        print("    a) 如果误差集中在某一两条语料、且那几条语料里某个桶占比特别高，"
              "多半是**桶还不够细** —— 把那个桶按上面 P1-2 的 buckets 明细再拆一层"
              "（比如 ascii_sym 里把空格/换行和括号分开），补一条对应的纯语料重跑。")
        print("    b) 如果各条真实语料**普遍**被低估、而纯语料个个完美线性，"
              "那就是**字符类模型本身不适合**：分词效率取决于词表命中密度，"
              "不是字符类。合成语料把「重复串压得极好」这件事标进了系数里。")
        print("       这种情况下别再加桶了，换路子：本地装一份该模型的 tokenizer"
              "（transformers 的 Qwen 词表）直接数 token，估算器退化成一层缓存；"
              "或者换成「按 token 边界密度」的特征（空格/标点/数字段落数），"
              "那才是分词器真正敏感的东西。两条路都要先在这套真实语料上验过再落地。")

    runner.dump("probe-text", {"fit": fit, "real_eval": ev, "baseline_tokens": baseline,
                               "threshold": REAL_CORPUS_MAX_REL_ERROR})
    return {"fit": fit, "real_eval": ev}


def _pure_corpus_slopes(runner: Runner, baseline: int | None) -> dict[str, tuple[float, float]]:
    """从本次已发出的记录里，按语料算「每字符 token」斜率和跨长度的极差。

    极差（最长档斜率 - 最短档斜率的绝对差）就是「这条语料自己线性不线性」的
    体检指标：2026-09-09 每条语料的极差都在 0.006 以内，线性得无可挑剔 ——
    这正是「单条语料完美线性 ≠ 模型成立」这句话的由来。
    """
    by_kind: dict[str, list[tuple[int, int]]] = {}
    for rec in runner.records:
        meta = rec.get("meta") or {}
        if meta.get("group") != "pure" or not _is_int(rec.get("prompt_tokens")):
            continue
        by_kind.setdefault(meta["kind"], []).append((meta["chars"], rec["prompt_tokens"]))
    out: dict[str, tuple[float, float]] = {}
    if not _is_int(baseline):
        return out
    for kind, pts in by_kind.items():
        sl = [(pt - baseline) / n for n, pt in sorted(pts) if n > 0]
        if sl:
            out[kind] = (sl[-1], max(sl) - min(sl))
    return out


def _dominant_corpus_for(bucket: str) -> str | None:
    """哪条纯语料主要落这个桶。用来把「桶系数」和「读得出斜率的那条语料」对上。"""
    best, best_share = None, 0.0
    for kind in FIT_KINDS:
        text = PURE_ALPHABETS[kind]
        share = tokens.count_chars(text).get(bucket, 0) / max(1, len(text))
        if share > best_share:
            best, best_share = kind, share
    return best if best_share >= 0.5 else None


def probe_wrap(runner: Runner) -> dict | None:
    """P2：消息包装。内容 token 近似相同，只变消息形状。

    ⚠ 2026-09-09 这套探针整个报 400：
        InvalidParameter: The provided messages input is invalid.
        The error info is [Can only get item pairs from a mapping.]
      七条请求里有形状是端点不收的，而当时是「一条抛、全组死」，既拿不到
      其余六条的数，也不知道是哪一条不合法。两处改动：

      1. 每条请求独立记账（Runner.try_send）——**照发不误**，被拒就把拒绝理由
         记下来继续。绝不能靠「不发那种形状」来绕过：那样就测不到它的包装开销，
         而这套探针存在的意义正是量这个。

      2. 工具形状那一组按**主循环真正发出去的样子**重排（对照
         harness/messages.py 的 MessageLog + tests/data/wire_window_golden.json）：
           - 主循环发 assistant tool_calls 时，请求里**一定带着 tools 表**
             （transport.request_kwargs 永远填 tools）。原来这一组 tools=[]，
             等于声明「我没有任何工具」却又给了一次工具调用，本身就自相矛盾。
           - 主循环里 assistant tool_calls **一定紧跟着对应的 tool 结果消息**
             （loop.py 推完 assistant_tool_call 就推 tool_result）。原来的
             W5 是一条**悬空的 tool call**：有调用、没结果。这是七条里唯一
             一条「MessageLog 产得出的每条消息形状都合法、但消息之间的配对
             关系是主循环从不会产生的」——**它是 400 的头号嫌疑**，错误里
             那句 "get item pairs" 也正是在讲配对。
         悬空那条现在仍然发（改名 W5X，排在最后），只是不再挡住别人：
         它被拒时，前面几条的包装开销已经量到了。

      ⚠ 这些是嫌疑排序，不是结论 —— 本文件不许离线断言端点收什么。跑完看
        输出里的「端点拒绝的形状」那一节，它会点名到底是哪几条。
    """
    nonce = runner_nonce()
    body = _repeat_to(PURE_ALPHABETS["han"], 300)
    half_a, half_b = body[:150], body[150:]
    tid = "call_calib_0"
    # 主循环真的会带的工具表。取 tap 一个就够：这一组内部所有请求的 tools
    # 逐字节相同，它在差分里整个消掉。
    one_tool = [t for t in tool_defs(True) if t.get("function", {}).get("name") == "tap"] \
        or tool_defs(True)[:1]

    one_part = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": nonce + half_a + half_b}]}]
    joined = [{"role": "system", "content": _SYSTEM},
              {"role": "user", "content": [{"type": "text",
                                            "text": nonce + half_a + "\n\n" + half_b}]}]
    two_parts = [{"role": "system", "content": _SYSTEM},
                 {"role": "user", "content": [{"type": "text", "text": nonce + half_a},
                                              {"type": "text", "text": half_b}]}]
    two_msgs = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": nonce + half_a}]},
                {"role": "user", "content": [{"type": "text", "text": half_b}]}]
    sys_string = [{"role": "system", "content": _SYSTEM + "\n" + nonce + half_a + half_b}]

    def _tool_call_msgs(args_json: str) -> list[dict]:
        """assistant 那条的形状与 MessageLog.assistant_tool_call 逐字段一致。"""
        return [{"role": "assistant", "content": None,
                 "tool_calls": [{"id": tid, "type": "function",
                                 "function": {"name": "tap", "arguments": args_json}}]}]

    args_json = json.dumps({"id": 1, "reason": half_b}, ensure_ascii=False)
    user_a = [{"role": "system", "content": _SYSTEM},
              {"role": "user", "content": [{"type": "text", "text": nonce + half_a}]}]
    # B 组基线：与下面几条的 tools 逐字节相同，文本合起来也相同。
    tool_base = [{"role": "system", "content": _SYSTEM},
                 {"role": "user", "content": [{"type": "text",
                                               "text": nonce + half_a + half_b}]}]
    tool_pair = user_a + _tool_call_msgs(args_json) + [
        {"role": "tool", "tool_call_id": tid, "content": "ok"}]
    tool_pair_long = user_a + _tool_call_msgs(args_json) + [
        {"role": "tool", "tool_call_id": tid, "content": half_b}]
    tool_dangling = user_a + _tool_call_msgs(args_json)

    plan = [
        Request("W0 user 单 part（基线）", one_part, []),
        Request("W1 user 单 part 内含 \\n\\n 分隔", joined, []),
        Request("W2 user 两个 text part", two_parts, []),
        Request("W3 两条 user 消息", two_msgs, []),
        Request("W4 全塞进 system string", sys_string, []),
        Request("W5B 带 tools 的基线（无 tool call）", tool_base, one_tool),
        Request("W5 assistant tool call + tool 结果（主循环的真实形状）", tool_pair, one_tool),
        Request("W6 同 W5，tool 结果换成长文本", tool_pair_long, one_tool),
        # 排最后：它是 400 的头号嫌疑，被拒也不影响前面几条已经量到的数。
        Request("W5X 悬空 tool call（无 tool 结果，主循环从不产生）", tool_dangling, one_tool),
    ]
    if not runner.confirm("P2 消息包装（--probe wrap）", plan):
        return None
    recs = {p.label: runner.try_send(p) for p in plan}
    got = {k: (v["prompt_tokens"] if _is_int(v.get("prompt_tokens")) else None)
           for k, v in recs.items()}
    rejected = {k: v["error"] for k, v in recs.items() if v.get("error")}
    for k, v in got.items():
        print(f"  {k:<48} prompt_tokens={v}")

    out = {"raw": got, "rejected": rejected}

    def delta(label, base_label):
        a, b = got.get(base_label), got.get(label)
        return (b - a) if _is_int(a) and _is_int(b) else None

    W0 = "W0 user 单 part（基线）"
    W5B = "W5B 带 tools 的基线（无 tool call）"
    out["OVERHEAD_PER_SEGMENT_JOIN_candidate"] = delta("W1 user 单 part 内含 \\n\\n 分隔", W0)
    out["extra_text_part"] = delta("W2 user 两个 text part", W0)
    out["OVERHEAD_PER_MESSAGE_candidate"] = delta("W3 两条 user 消息", W0)
    out["system_string_vs_user_part"] = delta("W4 全塞进 system string", W0)
    # ⚠ 下面三条的基线是 W5B 而不是 W0：B 组带着 tools，工具表本身几百 token，
    #   拿 W0 当基线会把工具表算进「包装开销」里。
    out["tool_call_plus_result_wrap"] = delta(
        "W5 assistant tool call + tool 结果（主循环的真实形状）", W5B)
    out["tool_result_grows_by_long_content"] = delta(
        "W6 同 W5，tool 结果换成长文本", "W5 assistant tool call + tool 结果（主循环的真实形状）")
    out["assistant_tool_call_wrap_dangling"] = delta(
        "W5X 悬空 tool call（无 tool 结果，主循环从不产生）", W5B)

    print("\nP2 产出（增量）：")
    for k in ("OVERHEAD_PER_SEGMENT_JOIN_candidate", "extra_text_part",
              "OVERHEAD_PER_MESSAGE_candidate", "system_string_vs_user_part",
              "tool_call_plus_result_wrap", "tool_result_grows_by_long_content",
              "assistant_tool_call_wrap_dangling"):
        print(f"  {k:<40} = {out[k]}")
    print("  ⚠ OVERHEAD_PER_MESSAGE 取 W3 的增量：W3 与 W0 的**文本逐字节相同**，"
          "差的只有多出来的一条消息包装。tool call / tool result 那几行的增量里还含"
          "工具名与 id 的字符，不能直接当 per-message 用。")

    print("\nP2 端点拒绝的形状：")
    if not rejected:
        print("  没有 —— 七种形状端点全收。2026-09-09 那个 400 已经不复现，"
              "在验收文档里记下这次的形状表。")
    else:
        for k, err in rejected.items():
            print(f"  ✗ {k}\n      {err[:400]}")
        print("  ⚠ 被拒的那种形状**没有包装开销数据**。budget 对它的估算是无依据的，"
              "验收文档里要写成「未测」，不要用相邻形状的数去顶。")
    runner.dump("probe-wrap", out)
    return out


def probe_tools(runner: Runner) -> dict | None:
    """P3：tools。固定 messages，只变工具表。"""
    nonce = runner_nonce()
    msgs = _probe_messages(_repeat_to(PURE_ALPHABETS["han"], 200), nonce=nonce)
    full = tool_defs(True)
    plan = [Request("T00 无 tools", msgs, [])]
    plan += [Request(f"T{i + 1:02d} 前 {i + 1} 个工具（真实顺序）", msgs, full[:i + 1])
             for i in range(len(full))]
    plan.append(Request("TREV 全量倒序", msgs, list(reversed(full))))
    for a, b in zip(plan, plan[1:]):
        assert_only_one_difference(a, b, field="tools")
    if not runner.confirm("P3 tools（--probe tools）", plan):
        return None
    got = [(p.label, runner.prompt_tokens(p)) for p in plan]
    base = got[0][1]
    per_tool = []
    for (la, va), (lb, vb) in zip(got, got[1:]):
        d = (vb - va) if _is_int(va) and _is_int(vb) else None
        per_tool.append({"from": la, "to": lb, "delta": d})
        print(f"  {lb:<32} prompt_tokens={vb}  Δ={d}")
    full_v = got[-2][1]
    rev_v = got[-1][1]
    order_free = _is_int(full_v) and full_v == rev_v
    print(f"\nP3 工具表总量: {(full_v - base) if _is_int(full_v) and _is_int(base) else None} token"
          f"（估算 {tokens.estimate_text(json.dumps(full, sort_keys=True, ensure_ascii=False))}）")
    print(f"P3 顺序无关: 正序 {full_v} / 倒序 {rev_v} -> "
          f"{'PASS' if order_free else 'FAIL（budget 的 tools_schema 必须按真实顺序序列化，'
                                      '现在用的是 sort_keys=True）'}")
    out = {"raw": got, "per_tool": per_tool, "order_free": order_free,
           "tools_total_measured": (full_v - base) if _is_int(full_v) and _is_int(base) else None}
    runner.dump("probe-tools", out)
    return out


def _image_ladder(rule) -> list[tuple[int, int]]:
    """分辨率梯子。**必须跨过官方规则里的缩放阈值** —— 只在阈值一侧取点，
    量到的全是同一条分支，等于没验。"""
    ar = 624 / 1388                                    # 真实截图的宽高比
    if rule is None:
        # 规则还没填：给一把常见分辨率，但这一组只能当参考，不能当验证。
        return [(256, 256), (448, 448), (624, 1388), (1170, 2532)]
    targets = []
    for base, ks in ((rule.min_pixels, (0.25, 0.5, 1.0, 2.0)),
                     (rule.max_pixels, (0.5, 1.0, 2.0))):
        if base:
            targets += [base * k for k in ks]
    sizes = []
    for t in sorted(set(targets)):
        w = max(rule.size_multiple, round(math.sqrt(t * ar)))
        h = max(rule.size_multiple, round(math.sqrt(t / ar)))
        # 故意**不**对齐到 size_multiple：取整那一步正是要验的分支之一。
        sizes.append((int(w), int(h)))
    sizes.append((624, 1388))                          # 真机实际用的那一档
    return sorted(set(sizes))


def probe_image(runner: Runner) -> dict | None:
    """图片：**验证** IMAGE_RULES，不拟合它。"""
    mid = _target_model_key(runner)
    rule = tokens.IMAGE_RULES.get(mid)
    if rule is None:
        print(f"⚠ IMAGE_RULES 里没有 {mid}：这一组只能量出实测值，量不出「对不对」。"
              "先查官方文档把规则填进去（见模块 docstring 第 1 步），再跑这条验证它。")
    ladder = _image_ladder(rule)
    plan: list[Request] = []
    groups = []
    for w, h in ladder:
        b64 = _png_b64(w, h)
        base = Request(f"I {w}x{h} 无图", _image_messages("图片探针", None), [])
        with_img = Request(f"I {w}x{h} 有图", _image_messages("图片探针", b64), [], image_px=(w, h))
        assert_only_one_difference(base, with_img, field="messages")
        groups.append((w, h, base, with_img))
        plan += [base, with_img]
    if not runner.confirm("图片规则验证（--probe image）", plan):
        return None
    rows = []
    ok = rule is not None
    for w, h, base, with_img in groups:
        measured = _image_only_tokens(runner, base, with_img)
        predicted = tokens.apply_image_rule(w, h, rule) if rule is not None else None
        good = (predicted == measured) if rule is not None else None
        rows.append({"w": w, "h": h, "measured": measured, "predicted": predicted, "match": good})
        print(f"  {w}x{h}: 实测 {measured} 预测 {predicted} -> "
              f"{'?' if good is None else ('PASS' if good else 'FAIL')}")
        if good is False:
            ok = False
    print("\n图片规则:", "全部吻合" if ok else "有不吻合的档位 —— 规则抄错了，或者这个模型不是这套规则")
    out = {"rows": rows, "rule_present": rule is not None, "ok": ok}
    runner.dump("probe-image", out)
    return out


def probe_cache(runner: Runner) -> dict | None:
    """缓存阈值：固定一段长前缀、只改末尾，前缀长度从明显低于 1024 token 扫到明显高于。

    ⚠ 不用运行中位数反推（spec §4.2）：命中率不等于阈值。未命中可能是前缀不够、
      首次请求、TTL 过期、区域不支持、请求不匹配 —— 中位数把这些混在一起，
      得到的数字回答不了「阈值在哪」这个问题。这里逐档看 cached_tokens 从哪一点开始非零。
    """
    nonce = runner_nonce()
    plan: list[Request] = []
    for n in CACHE_PREFIX_CHARS:
        prefix = nonce + _repeat_to(PURE_ALPHABETS["han"], n)
        # 同一档发两遍：第一遍写缓存，第二遍读。只看第二遍的 cached_tokens。
        # 两遍的**前缀逐字节相同**，只有末尾的 tail 不同 —— 末尾不同不影响前缀命中。
        plan.append(Request(f"C{n} 第一遍（写缓存）",
                            _probe_messages(prefix + "\n尾巴 A"), [], meta={"chars": n, "pass": 1}))
        plan.append(Request(f"C{n} 第二遍（读缓存）",
                            _probe_messages(prefix + "\n尾巴 B"), [], meta={"chars": n, "pass": 2}))
    if not runner.confirm("缓存阈值（--probe cache）", plan):
        return None
    rows = []
    for i in range(0, len(plan), 2):
        first = runner.send(plan[i])
        second = runner.send(plan[i + 1])
        n = plan[i].meta["chars"]
        rows.append({"prefix_chars": n,
                     "prompt_tokens_1": first["prompt_tokens"],
                     "prompt_tokens_2": second["prompt_tokens"],
                     "cached_1": first["cached_tokens"], "cached_2": second["cached_tokens"]})
        print(f"  前缀 {n:>5} 字 → prompt {first['prompt_tokens']}/{second['prompt_tokens']}, "
              f"cached {first['cached_tokens']}/{second['cached_tokens']}")
    hits = [r for r in rows if _is_int(r["cached_2"]) and r["cached_2"] > 0]
    if not hits:
        print("\n缓存阈值: 全程 cached_tokens 都是 0 或 null —— 结论是「这个端点/区域上"
              "隐式缓存没生效或不上报」，**不是**「阈值比 4000 字还高」。两者不要混为一谈。")
        threshold = None
    else:
        threshold = min(r["prefix_chars"] for r in hits)
        first_pt = next(r["prompt_tokens_2"] for r in rows if r["prefix_chars"] == threshold)
        print(f"\n缓存阈值: cached_tokens 从前缀 {threshold} 字开始非零"
              f"（那一档 prompt_tokens={first_pt}）。")
        print("  这个数才是 docs/19 §3.6 的答案。拿它和真实运行的冻结前缀长度比，"
              "判断前缀够不够 1024 token。")
        if threshold == CACHE_PREFIX_CHARS[0]:
            print("  ⚠ 最短的一档就命中了：阈值在梯子外面，把 CACHE_PREFIX_CHARS 往下再排几档重跑。")
    out = {"rows": rows, "threshold_prefix_chars": threshold}
    runner.dump("probe-cache", out)
    return out


def probe_segments(runner: Runner) -> dict | None:
    """P4：**按段**标定。产出直接就是 tokens.py 里那两张表要填的东西。

    为什么要有这个探针（2026-09-09 的结论）：字符类模型在真实语料上被证伪了，
    而且不是「桶不够细」—— tools 和 mixed 两条语料桶构成相近、误差方向却相反
    （高估 14% / 低估 19%）。同样的字符配比推不出同样的 token 数，驱动量是
    词表命中密度。加桶治不了，所以改成按段量：段内风格稳定，一个比值就够；
    确定性的段（system / tools_schema）干脆标成精确常量。

    三段输出：
      P4-1 精确常量  —— system 和 tools_schema。附**代码现算的 hash 键**，
                        可以直接抄进 tokens.EXACT_SEGMENT_TOKENS。
      P4-2 每段比值  —— 有真实语料的段，实测 token/字符 + 字符类兜底的误差。
      P4-3 没有语料的段 —— 明确报「无语料，未标定」，连原因一起报。

    ⚠ tools_schema 必须走 `tools=` 参数的差分，**不能**把 schema 当文本量：
      服务端把工具表渲染成自己的模板，和我们 json.dumps 出来的不是同一段字节。
      2026-09-09 两条路差 6.4%（4629 vs 4350）。这里两个都量，就是为了让这条
      差异留在留档里，别让后人再踩一次。
    """
    import hashlib as _hashlib

    from iphone_agent.harness.prompt import hash_of as _prompt_hash_of

    nonce = runner_nonce()
    corpora, missing = build_segment_corpora()
    full_tools = tool_defs(True)
    tools_blob = json.dumps(full_tools, sort_keys=True, ensure_ascii=False)

    base_req = Request("基线（空 payload，无 tools）", _probe_messages("", nonce=nonce), [],
                       meta={"group": "base"})
    # tools 差分：messages 与基线**逐字节相同**，只多一个 tools= 参数。
    tools_req = Request("tools= 参数（全量工具表）", _probe_messages("", nonce=nonce), full_tools,
                        meta={"group": "tools_param"})
    assert_only_one_difference(base_req, tools_req, field="tools")

    plan = [base_req, tools_req]
    # 把工具表**当文本**发一遍，量它作为第 2 层后备时的比值，同时把「两条路不等价」
    # 这件事量出来。
    plan.append(Request("tools_schema(当文本)", _probe_messages(tools_blob, nonce=nonce), [],
                        meta={"group": "seg", "seg": "tools_schema(当文本)"}))
    plan += [Request(f"段:{seg}", _probe_messages(text, nonce=nonce), [],
                     meta={"group": "seg", "seg": seg, "chars": len(text)})
             for seg, text in corpora.items()]

    if not runner.confirm("P4 按段标定（--probe segments）", plan):
        if runner.dry_run:
            # dry-run 也要把「哪些段没有语料」告诉人：这部分不花钱，而且正是
            # 跑之前该先修好的东西（把 IPHONE_CALIB_RUNS 指对、把 screenmap 备好）。
            print(f"\n  [dry-run] 本次会量 {len(corpora)} 个段，"
                  f"另有 {len(missing)} 个段无语料：")
            for seg, why in sorted(missing.items()):
                print(f"    - {seg:<24} 无语料，未标定 —— {why}")
        return None

    baseline = runner.prompt_tokens(base_req)
    tools_param = runner.prompt_tokens(tools_req)
    if not _is_int(baseline):
        print("  基线拿不到 prompt_tokens —— 所有差分作废，这一轮不产出任何数。")
        runner.dump("probe-segments", {"baseline_tokens": baseline})
        return None

    rows = []
    for req in plan[2:]:
        r = runner.send(req)
        seg = req.meta["seg"]
        text = corpora.get(seg, tools_blob)
        measured = (r["prompt_tokens"] - baseline) if _is_int(r["prompt_tokens"]) else None
        pred = tokens.estimate_text(text)
        rows.append({
            "segment": seg, "chars": len(text), "measured_tokens": measured,
            "tokens_per_char": (measured / len(text)) if measured and text else None,
            "char_class_predicted": pred,
            "rel_error": (abs(pred - measured) / measured) if measured else None,
        })
        print(f"  {req.label:<40} chars={len(text):<6} prompt_tokens={r['prompt_tokens']}")

    # ── P4-1 精确常量 ─────────────────────────────────────────────
    sys_text = corpora.get("system", "")
    sys_row = next((x for x in rows if x["segment"] == "system"), None)
    tools_text_row = next((x for x in rows if x["segment"] == "tools_schema(当文本)"), None)
    tools_exact = (tools_param - baseline) if _is_int(tools_param) else None
    sys_key = _prompt_hash_of(sys_text) if sys_text else None
    tools_key = _hashlib.sha256(tools_blob.encode("utf-8")).hexdigest()

    print("\nP4-1 精确常量（抄进 tokens.EXACT_SEGMENT_TOKENS，键是代码现算的，别手编）：")
    print("EXACT_SEGMENT_TOKENS = {")
    if sys_row and sys_row["measured_tokens"] is not None:
        print(f'    ("system", "{sys_key}"): {sys_row["measured_tokens"]},'
              f'   # {sys_row["chars"]} 字，system_prompt(True, True)')
    else:
        print("    # system 没量到 —— 不许填")
    if tools_exact is not None:
        print(f'    ("tools_schema",\n     "{tools_key}"): {tools_exact},'
              f'   # {len(full_tools)} 个工具，走 tools= 参数')
    else:
        print("    # tools_schema 没量到 —— 不许填")
    print("}")
    if tools_exact is not None and tools_text_row and tools_text_row["measured_tokens"]:
        d = tools_exact - tools_text_row["measured_tokens"]
        print(f"  ⚠ 同一份 schema 两条路：tools= 参数 {tools_exact} vs 当文本发 "
              f"{tools_text_row['measured_tokens']}，差 {d}"
              f"（{abs(d) / tools_exact * 100:.1f}%）。budget 记的是前者。")

    # ── P4-2 每段比值 ─────────────────────────────────────────────
    print("\nP4-2 每段 token/字符（抄进 tokens.SEGMENT_TOKENS_PER_CHAR）：")
    print(f"  {'段':<32}{'字符':>7}{'实测':>8}{'token/字符':>12}"
          f"{'字符类预测':>12}{'兜底误差':>10}")
    for x in sorted(rows, key=lambda r: -(r["chars"] or 0)):
        tpc = "—" if x["tokens_per_char"] is None else f"{x['tokens_per_char']:.3f}"
        rel = "—" if x["rel_error"] is None else f"{x['rel_error'] * 100:.1f}%"
        meas = "—" if x["measured_tokens"] is None else str(x["measured_tokens"])
        warn = "  ⚠样本太短" if (x["chars"] or 0) < SEGMENT_MIN_CHARS_FOR_RATIO else ""
        print(f"  {x['segment']:<32}{x['chars']:>7}{meas:>8}{tpc:>12}"
              f"{x['char_class_predicted']:>12}{rel:>10}{warn}")
    print(f"  ⚠ 标了「样本太短」的（< {SEGMENT_MIN_CHARS_FOR_RATIO} 字）只能当参考："
          "±1 token 的量化误差就能把比值挪好几个百分点。别填进表里。")
    print("  ⚠ 「兜底误差」是第 3 层字符类估算在这一段上错多少。它小的段，"
          "填不填第 2 层都行；它大的段（>10%）才是分层真正的收益所在。")

    # ── P4-3 没有语料的段 ─────────────────────────────────────────
    print("\nP4-3 无语料、未标定的段（它们继续走第 3 层字符类兜底）：")
    for seg, why in sorted(missing.items()):
        print(f"  {seg:<24} 无语料，未标定 —— {why}")
    if not missing:
        print("  （无）")
    print("\nP4-4 结论：把 P4-1 和 P4-2 抄进 tokens.py 之后，再把 CALIBRATION_ID "
          "改成验收文档名 —— 在那之前它必须一直是 uninitialized。")

    out = {"baseline_tokens": baseline, "tools_param_tokens": tools_param,
           "exact": {"system": {"key": sys_key,
                                "tokens": sys_row and sys_row["measured_tokens"]},
                     "tools_schema": {"key": tools_key, "tokens": tools_exact}},
           "rows": rows, "missing": missing,
           "min_chars_for_ratio": SEGMENT_MIN_CHARS_FOR_RATIO}
    runner.dump("probe-segments", out)
    return out


PROBES = {"text": probe_text, "wrap": probe_wrap, "tools": probe_tools,
          "image": probe_image, "cache": probe_cache, "segments": probe_segments}


# ============================ 入口 ============================

def _make_runner(args) -> Runner:
    from iphone_agent.model.config import resolve
    from iphone_agent.model.transports.chat_completions import ChatCompletionsTransport
    # dry-run 不需要密钥：它一个请求都不发，卡在「没配 key」上等于逼人为了看成本先去配密钥。
    resolved = resolve(args.model, need_key=not args.dry_run)
    for n in resolved.notices:
        print("提示：", n)
    print(f"模型: {resolved.spec}  key 来源: {resolved.api_key_source}  "
          f"CALIBRATION_ID(当前): {tokens.CALIBRATION_ID}")
    print(f"OVERHEAD_PER_MESSAGE(当前)={budget_mod.OVERHEAD_PER_MESSAGE} "
          f"OVERHEAD_PER_SEGMENT_JOIN(当前)={budget_mod.OVERHEAD_PER_SEGMENT_JOIN}")
    return Runner(ChatCompletionsTransport(resolved), dry_run=args.dry_run, yes=args.yes,
                  out_dir=pathlib.Path(args.out), max_tokens=args.max_tokens)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="calibrate_tokens.py",
        description="token 估算系数的离线标定。发真实请求、花真钱、人手动跑，不在 CI 跑。",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true", help="跑三条硬门（T0-1/2/3）")
    ap.add_argument("--preflight", action="store_true", help="跑前置断言（spec §6.2）")
    ap.add_argument("--probe", choices=sorted(PROBES), help="跑一套探针")
    ap.add_argument("--check-drift", metavar="RUNS_ROOT", nargs="?", const="runs",
                    help="C2 自然差分校验：读 runs/*/steps.jsonl，零成本不发请求")
    ap.add_argument("--model", default=None, help="provider:model，缺省用默认解析链")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要发的请求数与预估 token，一个请求都不发")
    ap.add_argument("--yes", action="store_true", help="跳过成本确认（知道自己在花钱时才用）")
    ap.add_argument("--out", default="runs/calibrate-tokens", help="原始数据落盘目录")
    ap.add_argument("--max-tokens", type=int, default=1,
                    help="每次请求的输出上限。默认 1 —— 但成本主要在输入，别指望靠它省钱")
    args = ap.parse_args(argv)

    if not (args.gate or args.preflight or args.probe or args.check_drift):
        ap.print_help()
        return 2

    # --check-drift 完全不碰模型层：没有 key、没有网也能跑。
    if args.check_drift:
        r = check_drift(pathlib.Path(args.check_drift))
        print(json.dumps(r, ensure_ascii=False, indent=2))
        if r.get("pairs", 0) < 30:
            print(f"⚠ pairs={r.get('pairs')}：样本太少，中位数不代表什么。"
                  "先多跑几个真机 state 模式任务再看。")
        if not (args.gate or args.preflight or args.probe):
            return 0

    runner = _make_runner(args)
    ok = True
    # preflight()/gate() 三态：True=PASS，False=真 FAIL，None=中止（dry-run 或用户在
    # 确认处敲了非 y）。None 不是失败——dry-run 成功跑完该是退出码 0，用户主动放弃
    # 也不该被当成断言/硬门没过来报错。只有 False 才计入 ok 和触发提前退出/跳过探针。
    preflight_failed = False
    if args.preflight:
        r = preflight(runner)
        if r is False:
            ok = False
            preflight_failed = True
    if args.gate:
        r = gate(runner)
        if r is False:
            ok = False
            if not runner.dry_run:
                return 1
    if args.probe:
        if preflight_failed:
            print("\n⚠ preflight FAIL —— 不跑探针。先解决前置断言里的 FAIL，"
                  "它们打印的时候已经说过「别往下跑探针」。")
        else:
            PROBES[args.probe](runner)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
