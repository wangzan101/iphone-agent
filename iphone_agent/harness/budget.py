"""分段 token 记账。不改输入、不写文件、不读 jsonl。

⚠ 这是观测设施：任何异常都由调用方降级成一个结构化错误字段，绝不顶掉任务
  （对齐 loop.py:262-267 记忆注入失败的处置）。本模块自身对坏数据一律降级
  不抛，让「算不出来」表现为 null 或 unknown 段，而不是表现为一个错数。
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics

from iphone_agent.harness import prompt as prompt_mod
from iphone_agent.harness import tokens

# 待标定（spec §4.1 探针 P2）。消息包装（role、type、data URL 前缀、
# tool_call 的 id/type/name）和段间的 "\n\n" 分隔符都算在这里。
OVERHEAD_PER_MESSAGE = 4
OVERHEAD_PER_SEGMENT_JOIN = 1

# ── 每段 est 的来源标签（写进 segments[名]["layer"]）─────────────────
# tokens.LAYER_EXACT / LAYER_SEGMENT_RATIO / LAYER_CHAR_CLASS 三个来自估算器，
# 下面两个是 budget 自己产生的量，估算器里没有对应的层：
#: 图片：走 tokens.estimate_image 的官方 smart_resize 规则。
#: 2026-09-09 在生产分辨率范围内逐 token 验过（见 tokens.apply_image_rule），
#: 所以它和第 1 层一样可信，但**来源不同**（是规则不是查表），单独一个标签。
LAYER_IMAGE_RULE = "image_rule"
#: 既没有实测也没有规则的量：protocol_residual 的 OVERHEAD_PER_* 就是这种，
#: 它自己的注释写着「待标定」。unknown 里那些 est=0 的占位也归这里。
LAYER_UNCALIBRATED = "uncalibrated"
#: 一个段里混进了不同层的贡献（理论上不该发生；发生了要看得见，不能被抹平）。
LAYER_MIXED = "mixed"
#: 这个段还没有任何一次带层的贡献（est 恒为 0 的纯占位段）。
LAYER_NONE = "none"


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _usage_int(usage, key: str) -> int | None:
    """拿不到就 None。⚠ 绝不返回 0 —— null 和 0 是不同的事实。"""
    if not isinstance(usage, dict):
        return None
    v = usage.get(key)
    return int(v) if _is_num(v) else None


def _bump(segs: dict, name, *, est: int = 0, chars: int = 0, layer: str | None = None) -> None:
    """累加一个段。`count` 的语义是「这个段被贡献了几次」。

    段名理论上都是代码写死的 str，但视图是外部数据：真出现不可哈希的段名
    （list/dict）时用 str() 兜底，而不是让记账模块抛出去顶掉任务。

    `layer` 说明这次贡献的 est 是哪一层给的（见模块顶部的 LAYER_*）。
    传 None 表示「这次贡献不表态」——est=0 的占位不该把一个精确段污染成 mixed。
    """
    if not isinstance(name, str):
        name = str(name)
    s = segs.setdefault(name, {"est": 0, "chars": 0, "count": 0, "layer": LAYER_NONE})
    s["est"] += est
    s["chars"] += chars
    s["count"] += 1
    if layer is not None:
        cur = s.get("layer", LAYER_NONE)
        # 同一个段被不同层贡献过就标 mixed：这时段的 est 是几层加出来的，
        # 不能宣称它整段精确，也不该悄悄按最后一次的层报。
        s["layer"] = layer if cur in (LAYER_NONE, layer) else LAYER_MIXED


def _claim(segs: dict, name: str, *, est: int, chars: int, count: int,
           layer: str = LAYER_NONE) -> dict:
    """写死名字、`count` 语义特殊的段（tools_schema / protocol_residual）。

    这些段的 count 不是「贡献次数」而是「工具条数」/「消息条数」，所以不能走
    _bump。但直接 `segs[name] = {...}` 会在段名撞车时静默清账 —— 被覆盖的
    est/chars 凭空消失而 unknown 仍是 0，正是原则 2 要防的「一本看起来正常的
    错账」。所以撞车时先把旧的量整体转记进 unknown 让它显形。
    """
    old = segs.get(name)
    if isinstance(old, dict):
        _bump(segs, "unknown", est=old.get("est", 0), chars=old.get("chars", 0))
    segs[name] = {"est": est, "chars": chars, "count": count, "layer": layer}
    return segs[name]


def _px(part: dict) -> tuple[int, int]:
    """图片 part 的分辨率，来自 `_px` 内部键。

    现状（Task 10 起）：loop.py 的两个调用点——`push_obs` 和
    `state_view(...)`——都已经传了 `px=(obs.width_px, obs.height_px)`，
    所以两种 context_mode 下的图片 part 都带着真实分辨率，
    estimate_image 拿得到尺寸就能算出 token 数，不再恒进 unknown。

    拿不到就退回 (0, 0)：estimate_image 对非正尺寸返回 None，会被记进
    unknown 段，而不是套一个假分辨率算出的假 token 数。
    """
    v = part.get("_px")
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return (0, 0)
    try:
        return (int(v[0]), int(v[1]))
    except (TypeError, ValueError):
        # 尺寸不是数：不猜，退回 (0, 0) 让它走 unknown。
        return (0, 0)


def _exact_key(seg, text: str) -> str | None:
    """这一段在 tokens.EXACT_SEGMENT_TOKENS 里的查表键；没有就 None（落下一层）。

    只有 system 走这条路：它的键是 prompt.hash_of(渲染后的文本)，和 run.json 里
    留档的 prompt_hash 同源 —— 提示词改一个字或 PROMPT_VERSION 一动，键就变、
    常量自动失效、这一段自己落回第 2 层。这正是我们要的：**宁可退回估算，
    也不要拿旧版提示词的实测值去报新版提示词的账。**

    tools_schema 不走这里 —— 它的键是 account() 里本来就在算的 schema_hash。

    ⚠ 观测设施绝不顶掉任务：算 hash 出任何岔子都返回 None 降级。
    """
    if seg != "system":
        return None
    try:
        return prompt_mod.hash_of(text)
    except Exception:       # noqa: BLE001 —— 算不出键就是查不到，降级不抛
        return None


def _est(seg, text: str) -> tuple[int, str]:
    """一段文本的 (est, layer)。分层选择全在 tokens.estimate_segment 里。"""
    return tokens.estimate_segment(seg, text, key=_exact_key(seg, text))


def account(view, tools, usage, *, context_mode: str, call_index: int, model_id: str) -> dict:
    segs: dict[str, dict] = {}
    n_messages = 0
    n_joins = 0

    for m in view if isinstance(view, list) else []:
        if not isinstance(m, dict):
            continue
        n_messages += 1
        content = m.get("content")
        if not isinstance(content, list):
            # system / assistant / tool：段名打在顶层，content 是字符串或 None
            text = content if isinstance(content, str) else ""
            seg = m.get("_seg") or "unknown"
            est, layer = _est(seg, text)
            _bump(segs, seg, est=est, chars=len(text), layer=layer)
            args = ""
            n_bad_args = 0
            tcs = m.get("tool_calls")
            for tc in tcs if isinstance(tcs, list) else []:
                if not isinstance(tc, dict):
                    n_bad_args += 1
                    continue
                fn = tc.get("function")
                a = fn.get("arguments") if isinstance(fn, dict) else None
                if isinstance(a, str):
                    args += a
                elif a is not None:
                    # 多个 provider/SDK 会把 arguments 反序列化成 dict 再回传。
                    # 这不是我们认识的形状：不 str() 它去猜字符数（原则 3），
                    # 但必须显形（原则 2）—— 进 unknown，est 记 0。
                    n_bad_args += 1
            for _ in range(n_bad_args):
                _bump(segs, "unknown", est=0)
            if args:
                est, layer = _est("assistant_tool_args", args)
                _bump(segs, "assistant_tool_args", est=est, chars=len(args), layer=layer)
            continue
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "image_url":
                est = tokens.estimate_image(*_px(p), model_id=model_id)
                if est is None:
                    # 不认识的模型或拿不到分辨率：进 unknown，不套默认系数（spec §2.6）。
                    # ⚠ 这一条同时给 image 和 unknown 各 +1 count：下游按 count
                    # 汇总「一共几张图 / 几处漏账」时会把同一张图数两次，需要按段名
                    # 分别看，不能把所有段的 count 直接相加。
                    _bump(segs, "image", est=0)
                    _bump(segs, "unknown", est=0)
                else:
                    # 与文本 part 同一套严格判定：顶层 `_seg` 缺失就是漏标本身，
                    # 进 unknown，不静默命名成 "image"（原则 2）。
                    _bump(segs, (p.get("_seg", None)) or "unknown",
                          est=est, layer=LAYER_IMAGE_RULE)
                continue
            # 文本 part：顶层 `_seg` 是 `_obs_parts` 细分的前提（state_view /
            # user_observation 都是这么打的：_seg 打包了 pairs[0][0]，
            # _obs_parts 才是完整的逐段列表）。用 `in` 判断而不是 .get()——
            # 顶层 _seg 缺失就是漏标本身，这时整段一起进 unknown，不能因为
            # _obs_parts 还在就假装知道怎么分段（那会让漏标显不出形）。
            if "_seg" not in p:
                text = p.get("text") if isinstance(p.get("text"), str) else ""
                est, layer = _est("unknown", text)
                _bump(segs, "unknown", est=est, chars=len(text), layer=layer)
                continue
            sub = p.get("_obs_parts")
            if isinstance(sub, list) and sub:
                n_joins += max(0, len(sub) - 1)
                for pair in sub:
                    # 视图是外部数据：不能直接 `for seg, text in sub` 解包，
                    # 形状不对会抛（原则「观测设施绝不顶掉任务」）。不是
                    # (seg, text) 二元组就不猜哪一半是文本，整条进 unknown。
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        _bump(segs, "unknown", est=0)
                        continue
                    seg, text = pair
                    text = text if isinstance(text, str) else ""
                    seg = seg or "unknown"
                    est, layer = _est(seg, text)
                    _bump(segs, seg, est=est, chars=len(text), layer=layer)
            else:
                text = p.get("text") if isinstance(p.get("text"), str) else ""
                seg = p["_seg"] or "unknown"
                est, layer = _est(seg, text)
                _bump(segs, seg, est=est, chars=len(text), layer=layer)

    if tools:
        if isinstance(tools, list):
            try:
                blob = json.dumps(tools, sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                blob = None
        else:
            # tools 不是 list：这不是我们认识的工具表形状，条数无从谈起，
            # 序列化出来的 est 也不代表 provider 真正会发的 payload。按原则 3
            # 不猜，按原则 2 进 unknown 显形（est 记 0，缺的量会体现在 residual）。
            blob = None
        if blob is None:
            _bump(segs, "unknown", est=0)
        else:
            # schema_hash 先算：它就是第 1 层的查表键（工具集一改 hash 就变，
            # 常量自动失效落回按 JSON 文本估的第 2 层）。
            schema_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()
            est, layer = tokens.estimate_segment("tools_schema", blob, key=schema_hash)
            _claim(segs, "tools_schema", est=est, chars=len(blob),
                   count=len(tools), layer=layer)["schema_hash"] = schema_hash

    residual_est = n_messages * OVERHEAD_PER_MESSAGE + n_joins * OVERHEAD_PER_SEGMENT_JOIN
    # ⚠ 这一段的 count 是「消息条数」，不是别的段那种「贡献次数」。下游汇总时
    # 要按段名单独解释，别和其他段的 count 混在一起加。
    _claim(segs, "protocol_residual", est=residual_est, chars=0, count=n_messages,
           layer=LAYER_UNCALIBRATED)
    segs.setdefault("unknown", {"est": 0, "chars": 0, "count": 0, "layer": LAYER_NONE})

    est_total = sum(s["est"] for s in segs.values())
    # 这一次调用的账里，多少 token 是精确的、多少是估的。est_total 一个数说不出
    # 「±20% 是压在 6000 token 的前缀上还是压在 200 token 的备忘上」——差别很大。
    est_by_layer: dict[str, int] = {}
    for s in segs.values():
        est_by_layer[s.get("layer", LAYER_NONE)] = \
            est_by_layer.get(s.get("layer", LAYER_NONE), 0) + s["est"]
    actual = _usage_int(usage, "prompt_tokens")
    cached = _usage_int(usage, "cached_tokens")
    if cached is None and isinstance(usage, dict):
        cached = _usage_int(usage.get("prompt_tokens_details"), "cached_tokens")

    if actual is not None and actual > 0:        # None / 0 / 负数都走 null 分支
        # 负数和 0 一样是「不可信」而不是「一个小的真值」：拿它算出来的
        # signed_bias / abs_pct_error 会是符号都反了的「看起来正常的错数」。
        residual = actual - est_total
        signed = residual / actual
        abs_pct = abs(residual) / actual
    else:
        residual = signed = abs_pct = None

    return {
        "context_mode": context_mode,
        "call_index": call_index,
        "segments": segs,
        "est_total": est_total,
        "est_by_layer": est_by_layer,
        "actual_prompt_tokens": actual,
        "cached_tokens": cached,
        "residual_tokens": residual,
        "signed_bias": signed,
        "abs_pct_error": abs_pct,
        "calibration": tokens.CALIBRATION_ID,
    }


def _pct(values: list[float], q: float) -> float | None:
    """values 上的分位数（就近取整索引，不插值——够用且没有插值带来的假精度）。"""
    if not values:
        return None
    xs = sorted(values)
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


def summarize(budgets: list[dict]) -> dict:
    """run 级汇总。纯函数，不读 jsonl。

    ⚠ 字段按**调用**命名不按 step：一次有效 step 可能对应多次模型调用
      （空回复重试、多工具被拒都有 usage 但不增 steps，loop.py:500 vs :654）。
    ⚠ per_call_context 描述「一次上下文的构成」（一次请求里各段占多少），
      cumulative_submitted_tokens 描述「重复发送的工作量」（整个 run 累计
      提交了多少），两者语义不同，不共用 total/share——混用会让人误判
      「这一段占比高」和「这一段被反复重发」两件不同的事。

    ⚠ 观测设施绝不顶掉任务：`budgets` 里任何一条形状不对（不是 dict、
      `segments` 缺失或不是 dict、`est` 是字符串/NaN/None……）一律降级为
      「当这条没有可用数据」，不抛异常，也不让一条坏数据污染其它条目的
      统计（sum/median 只看得到干净的那些）。
    """
    if not isinstance(budgets, list):
        budgets = []

    def is_error(b) -> bool:
        return isinstance(b, dict) and "budget_error" in b

    # ok：形状正常、可用于统计的条目。budget_error 条目和完全不是 dict 的
    # 条目都被排除在外，但仍计入 calls_total（它们确实发生过一次调用）。
    ok = [b for b in budgets if isinstance(b, dict) and not is_error(b)]

    per_seg: dict[str, list[int]] = {}
    seg_layer: dict[str, str] = {}
    by_layer: dict[str, int] = {}
    calls_with_unknown = 0
    for b in ok:
        segs = b.get("segments")
        if not isinstance(segs, dict):
            continue
        for name, s in segs.items():
            if not isinstance(s, dict):
                continue
            est = s.get("est")
            v = int(est) if _is_num(est) else 0
            per_seg.setdefault(name, []).append(v)
            # 层是外部数据（budget 可能来自 jsonl 里的老留档，那时还没有这个
            # 字段）：不是字符串就当 "none"，不猜、也不抛。
            lay = s.get("layer")
            lay = lay if isinstance(lay, str) else LAYER_NONE
            by_layer[lay] = by_layer.get(lay, 0) + v
            prev = seg_layer.get(name)
            seg_layer[name] = lay if prev in (None, lay) else LAYER_MIXED
        unk = segs.get("unknown")
        # 只看 unknown 段自己的 count（有多少次贡献进了 unknown，不管每次
        # est 是不是 0），回答「这次调用里出没出现过没被计量到的东西」——
        # 不是对各段 count 求和（那个会把 protocol_residual 的「消息条数」
        # 和 tools_schema 的「工具条数」错误地混进同一个数字）。
        if isinstance(unk, dict) and _is_num(unk.get("count")) and unk["count"] > 0:
            calls_with_unknown += 1

    grand = sum(sum(v) for v in per_seg.values()) or 1
    per_call_context = {
        name: {"min": min(v), "median": statistics.median(v), "max": max(v),
               "share": sum(v) / grand,
               # 这一段的 est 是哪一层给的。跨调用不一致（比如提示词中途变了、
               # 常量表命中率变了）就标 mixed —— 那本身是个值得看见的事实。
               "layer": seg_layer.get(name, LAYER_NONE)}
        for name, v in per_seg.items()
    }
    # 「这个 run 的账里有多少 token 是精确的、多少是估的」。
    # ⚠ share 的分母是 grand（所有段所有调用的 est 之和），和 per_call_context
    #   的 share 同一个分母 —— 两张表可以直接对着看。
    token_provenance = {
        lay: {"est": v, "share": v / grand} for lay, v in sorted(by_layer.items())
    }
    exact_est = by_layer.get(tokens.LAYER_EXACT, 0) + by_layer.get(LAYER_IMAGE_RULE, 0)

    def clean(key: str) -> list[float]:
        return [b[key] for b in ok if _is_num(b.get(key))]

    signed = clean("signed_bias")
    abs_pct = clean("abs_pct_error")

    calibration = "uninitialized"
    for b in ok:
        if isinstance(b.get("calibration"), str):
            calibration = b["calibration"]
            break

    return {
        "calls_total": len(budgets),
        "calls_with_error": sum(1 for b in budgets if is_error(b)),
        "calls_with_usage": sum(1 for b in ok if _is_num(b.get("actual_prompt_tokens"))),
        "calls_without_usage": sum(1 for b in ok if not _is_num(b.get("actual_prompt_tokens"))),
        "calls_with_unknown": calls_with_unknown,
        # prefix_drift 挂在 call_budget 上，不挂在 ok/error 的分类上——
        # account() 炸了的调用照样可能漂移（loop.py 两种 call_budget 都会
        # 挂这个字段），所以这里数全体 budgets，不是只数 ok。
        "calls_with_prefix_drift": sum(
            1 for b in budgets if isinstance(b, dict) and b.get("prefix_drift") is True),
        "per_call_context": per_call_context,
        "token_provenance": token_provenance,
        # 一个数字回答「这本账有多少可信」：查表实测 + 已验证的图片规则占多少。
        # 剩下的是按段比值和字符类估出来的，误差在 10~20% 量级。
        "measured_share": exact_est / grand,
        "cumulative_submitted_tokens": sum(
            int(b["est_total"]) for b in ok if _is_num(b.get("est_total"))
        ),
        "error_metrics": {
            "signed_bias_median": statistics.median(signed) if signed else None,
            "abs_pct_error_median": statistics.median(abs_pct) if abs_pct else None,
            "abs_pct_error_p90": _pct(abs_pct, 0.9),
            "abs_pct_error_max": max(abs_pct) if abs_pct else None,
        },
        # 逐调用留原始序列，**不做「命中率中位数」这种会被误读成阈值结论的
        # 汇总**（spec §4.2：前缀是否够触发隐式缓存由专门探针回答，运行中位数
        # 回答不了那个问题——中位数会把「稳定命中」和「时好时坏」抹成同一个数）。
        "cache": [{"call_index": b.get("call_index"),
                   "prompt_tokens": b.get("actual_prompt_tokens"),
                   "cached_tokens": b.get("cached_tokens")} for b in ok],
        "calibration": calibration,
    }
