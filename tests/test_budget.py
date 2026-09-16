import pathlib

import pytest

from iphone_agent.harness import budget, tokens
from iphone_agent.harness.messages import MessageLog, build_state_parts


def a_view():
    log = MessageLog()
    log.system("sys")
    log.user_text("任务：t", seg="task")
    log.freeze()
    parts = build_state_parts(history="H", memory="M", last_result=None,
                              transition=None, elements_text="E" * 100, step=1, max_steps=30)
    return log.state_view("\n\n".join(t for _, t in parts), "img", parts=parts)


def test_segments_are_accounted_by_name():
    b = budget.account(a_view(), None, {"prompt_tokens": 100},
                       context_mode="state", call_index=0, model_id="m")
    assert set(b["segments"]) >= {"system", "task", "state_history", "state_elements"}
    assert b["segments"]["state_elements"]["chars"] > b["segments"]["state_history"]["chars"]


def test_same_segment_appearing_twice_is_summed_with_count():
    log = MessageLog()
    log.system("s")
    log.user_text("app A", seg="app_note")
    log.user_text("app B", seg="app_note")
    b = budget.account(log.windowed(), None, None,
                       context_mode="window", call_index=0, model_id="m")
    assert b["segments"]["app_note"]["count"] == 2


def test_missing_usage_yields_null_not_zero():
    """null 和 0 是不同的事实：一个是没拿到，一个是真的零。"""
    b = budget.account(a_view(), None, None, context_mode="state", call_index=0, model_id="m")
    assert b["actual_prompt_tokens"] is None
    assert b["residual_tokens"] is None and b["signed_bias"] is None and b["abs_pct_error"] is None


def test_zero_prompt_tokens_also_yields_null_error_metrics():
    b = budget.account(a_view(), None, {"prompt_tokens": 0},
                       context_mode="state", call_index=0, model_id="m")
    assert b["signed_bias"] is None


def test_unknown_segment_goes_to_unknown_bucket_not_dropped():
    """漏标的必须显形（spec §8 R4）：静默归零会让账看起来是对的。"""
    view = a_view()
    view[-1]["content"][0].pop("_seg")
    b = budget.account(view, None, None, context_mode="state", call_index=0, model_id="m")
    assert b["segments"]["unknown"]["chars"] > 0


def test_unknown_model_image_goes_to_unknown():
    b = budget.account(a_view(), None, None, context_mode="state",
                       call_index=0, model_id="no-such-model")
    assert b["segments"]["image"]["est"] == 0
    assert b["segments"]["unknown"]["count"] >= 1


def test_tools_are_accounted_separately_with_a_hash():
    tools = [{"type": "function", "function": {"name": "tap", "parameters": {}}}]
    b = budget.account(a_view(), tools, None, context_mode="state", call_index=0, model_id="m")
    assert b["segments"]["tools_schema"]["count"] == 1
    assert len(b["segments"]["tools_schema"]["schema_hash"]) == 64


def test_calibration_status_is_reported():
    """2026-09-09 标定后 tokens.CALIBRATION_ID 不再是 "uninitialized"（见验收文档
    docs/superpowers/acceptance/2026-09-09-token标定.md）——budget 的 calibration
    字段就是原样透传它，这条测试钉住「透传」这条行为，不钉具体字符串（字符串
    随下一轮标定会变，钉字面值只会制造无意义的红）。"""
    b = budget.account(a_view(), None, None, context_mode="state", call_index=0, model_id="m")
    assert b["calibration"] == tokens.CALIBRATION_ID
    assert b["calibration"] != "uninitialized"


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": "x"}, {"prompt_tokens": float("nan")},
                                   {"prompt_tokens": float("inf")}, [1, 2], "nope"])
def test_bad_usage_degrades_without_raising(usage):
    b = budget.account(a_view(), None, usage, context_mode="state", call_index=0, model_id="m")
    assert b["actual_prompt_tokens"] is None


@pytest.mark.parametrize("view", [[], [{"role": "user", "content": "not a list"}],
                                  [{"role": "user", "content": [{"type": "image_url"}]}]])
def test_bad_view_degrades_without_raising(view):
    b = budget.account(view, None, None, context_mode="state", call_index=0, model_id="m")
    assert isinstance(b["est_total"], int)


# --- 坏数据一律降级不抛（模块 docstring 的承诺）---------------------------

BAD_VIEWS = [
    # _obs_parts 形状不对：三元组 / 一元组 / 裸字符串 / 非可迭代
    [{"role": "user", "content": [{"type": "text", "text": "x", "_seg": "a",
                                   "_obs_parts": [("a", "b", "c")]}]}],
    [{"role": "user", "content": [{"type": "text", "text": "x", "_seg": "a",
                                   "_obs_parts": [("a",)]}]}],
    [{"role": "user", "content": [{"type": "text", "text": "x", "_seg": "a",
                                   "_obs_parts": ["abc"]}]}],
    [{"role": "user", "content": [{"type": "text", "text": "x", "_seg": "a",
                                   "_obs_parts": [5]}]}],
    # 不可哈希的段名
    [{"role": "user", "content": [{"type": "text", "text": "x", "_seg": "a",
                                   "_obs_parts": [(["a"], "b")]}]}],
    [{"role": "user", "content": [{"type": "text", "text": "x", "_seg": ["a"]}]}],
    [{"role": "system", "content": "s", "_seg": {"a": 1}}],
    # _px 形状不对
    [{"role": "user", "content": [{"type": "image_url", "_seg": "image",
                                   "_px": ["a", "b"]}]}],
    [{"role": "user", "content": [{"type": "image_url", "_seg": "image",
                                   "_px": [None, None]}]}],
    # tool_calls 形状不对
    [{"role": "assistant", "content": "a", "_seg": "assistant_text", "tool_calls": "nope"}],
    [{"role": "assistant", "content": "a", "_seg": "assistant_text", "tool_calls": ["a"]}],
    [{"role": "assistant", "content": "a", "_seg": "assistant_text",
      "tool_calls": [{"function": "nope"}]}],
    # ⚠ 不是造出来的极端值：多个 provider/SDK 把 arguments 反序列化成 dict 再回传。
    [{"role": "assistant", "content": "a", "_seg": "assistant_text",
      "tool_calls": [{"function": {"arguments": 123}}]}],
    [{"role": "assistant", "content": "a", "_seg": "assistant_text",
      "tool_calls": [{"function": {"arguments": {"x": 1}}}]}],
]


@pytest.mark.parametrize("view", BAD_VIEWS)
def test_bad_view_shapes_degrade_without_raising(view):
    b = budget.account(view, None, None, context_mode="state", call_index=0, model_id="m")
    assert isinstance(b["est_total"], int)


@pytest.mark.parametrize("tools", [5, "nope", {"a": 1}, object()])
def test_bad_tools_degrade_without_raising(tools):
    b = budget.account(a_view(), tools, None, context_mode="state",
                       call_index=0, model_id="m")
    assert isinstance(b["est_total"], int)
    assert "tools_schema" not in b["segments"]
    assert b["segments"]["unknown"]["count"] >= 1


def test_deserialized_tool_arguments_go_to_unknown_not_concatenated():
    """arguments 是 dict（provider 反序列化过）：不猜字符数，但必须显形。"""
    view = [{"role": "assistant", "content": "", "_seg": "assistant_text",
             "tool_calls": [{"function": {"arguments": {"x": 1}}}]}]
    b = budget.account(view, None, None, context_mode="state", call_index=0, model_id="m")
    assert "assistant_tool_args" not in b["segments"]
    assert b["segments"]["unknown"]["count"] >= 1


def test_bad_obs_pair_goes_to_unknown():
    view = [{"role": "user", "content": [{"type": "text", "text": "x", "_seg": "a",
                                          "_obs_parts": [("a", "b", "c"), ("ok", "yyyy")]}]}]
    b = budget.account(view, None, None, context_mode="state", call_index=0, model_id="m")
    assert b["segments"]["unknown"]["count"] >= 1
    assert b["segments"]["ok"]["chars"] == 4


# --- Important 2：段名冲突不能静默清账 -------------------------------------

def test_segment_name_collision_does_not_silently_erase():
    """真实段名撞上写死的段名时，被覆盖的量必须进 unknown，不能凭空消失。"""
    view = [{"role": "system", "content": "P" * 40, "_seg": "protocol_residual"},
            {"role": "system", "content": "T" * 40, "_seg": "tools_schema"}]
    tools = [{"type": "function", "function": {"name": "tap"}}]
    b = budget.account(view, tools, None, context_mode="state", call_index=0, model_id="m")
    # tools_schema 有实测的段比值（第 2 层），protocol_residual 没有（走字符类兜底）——
    # 转记进 unknown 的量是各自当初记账时的量，所以这里也要按各自的层算。
    collided = (tokens.estimate_segment("protocol_residual", "P" * 40)[0]
                + tokens.estimate_segment("tools_schema", "T" * 40)[0])
    assert b["segments"]["unknown"]["est"] == collided
    assert b["segments"]["unknown"]["chars"] == 80


# --- Important 3：负的 prompt_tokens 也是不可信 ----------------------------

def test_negative_prompt_tokens_yields_null_error_metrics():
    b = budget.account(a_view(), None, {"prompt_tokens": -5},
                       context_mode="state", call_index=0, model_id="m")
    assert b["residual_tokens"] is None
    assert b["signed_bias"] is None and b["abs_pct_error"] is None


# --- Important 4：图片漏标与文本走同一套严格判定 ---------------------------

def test_image_without_seg_goes_to_unknown_not_named_image(monkeypatch):
    rule = tokens.ImageRule(pixels_per_token=750, constant_tokens=0, size_multiple=28,
                            min_pixels=None, max_pixels=None)
    monkeypatch.setitem(tokens.IMAGE_RULES, "m", rule)
    view = [{"role": "user", "content": [{"type": "image_url", "_px": [100, 100]}]}]
    b = budget.account(view, None, None, context_mode="state",
                       call_index=0, model_id="m")
    assert b["segments"]["unknown"]["est"] > 0
    assert "image" not in b["segments"]


# =========================== summarize() ===================================

def _b(est_total, actual, segs, mode="state", idx=0):
    return {"context_mode": mode, "call_index": idx, "segments": segs,
            "est_total": est_total, "actual_prompt_tokens": actual,
            "cached_tokens": None, "residual_tokens": None if not actual else actual - est_total,
            "signed_bias": None if not actual else (actual - est_total) / actual,
            "abs_pct_error": None if not actual else abs(actual - est_total) / actual,
            "calibration": "uninitialized"}


def test_share_sums_to_one():
    segs = {"a": {"est": 30, "chars": 0, "count": 1}, "b": {"est": 70, "chars": 0, "count": 1}}
    s = budget.summarize([_b(100, 100, segs), _b(100, 100, segs)])
    total = sum(v["share"] for v in s["per_call_context"].values())
    assert abs(total - 1.0) < 1e-9


def test_pct_never_below_median_for_small_n():
    """budget._pct 用 round()（不是 int() 截断），review 顺手排查过它是否有和
    calibrate_tokens.check_drift_from_budgets 同款「n=2 时截断成最小值」的问题。
    round(0.9*1)=1 取到的是最大值，不是最小值，所以 n=2..14 时 p90 都不会低于
    median（脚本核对过 n=1..14 全部成立）。这条测试只是把这个「没问题」钉成
    回归保护，不是修复。"""
    for n in range(2, 15):
        xs = [float(i) for i in range(1, n + 1)]
        median = xs[(n - 1) // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
        p90 = budget._pct(xs, 0.9)
        assert p90 is not None
        assert p90 >= median, f"n={n}: p90={p90} < median={median}"


def test_call_counts_are_by_call_not_by_step():
    s = budget.summarize([_b(10, 10, {}), _b(10, None, {}), _b(10, 12, {})])
    assert s["calls_total"] == 3
    assert s["calls_with_usage"] == 2 and s["calls_without_usage"] == 1


def test_signed_and_absolute_errors_are_both_reported():
    """只报 signed 会正负抵消：+50% 和 -50% 的中位数是 0，看起来完美。"""
    s = budget.summarize([_b(50, 100, {}), _b(150, 100, {})])
    assert abs(s["error_metrics"]["signed_bias_median"]) < 1e-9
    assert s["error_metrics"]["abs_pct_error_median"] == pytest.approx(0.5)


def test_cumulative_is_separate_from_per_call_context():
    s = budget.summarize([_b(100, 100, {"a": {"est": 100, "chars": 0, "count": 1}})] * 5)
    assert s["cumulative_submitted_tokens"] == 500
    assert s["per_call_context"]["a"]["median"] == 100


def test_calls_with_prefix_drift_counts_true_flags_including_on_error_entries():
    """任务 12：calls_with_prefix_drift 数的是 prefix_drift 为真的调用数——
    这个字段第 9 个任务时被删掉过（当时 account() 从不产出 prefix_drift，
    留着会恒为 0），现在 loop.py 真的会产出它了，必须钉住不再消失。

    ⚠ prefix_drift 挂在 call_budget 上，account() 炸了的错误占位符一样可能
      挂着这个字段（loop.py 两种 call_budget 都会挂）——所以这里也要覆盖
      budget_error 条目，不能只测正常条目。"""
    b1 = _b(100, 100, {"a": {"est": 100, "chars": 0, "count": 1}})
    b1["prefix_drift"] = True
    b2 = _b(100, 100, {"a": {"est": 100, "chars": 0, "count": 1}}, idx=1)
    b2["prefix_drift"] = False
    b3 = {"budget_error": "boom", "call_index": 2, "prefix_drift": True}
    s = budget.summarize([b1, b2, b3])
    assert s["calls_with_prefix_drift"] == 2


def test_empty_and_error_budgets_do_not_raise():
    assert budget.summarize([])["calls_total"] == 0
    s = budget.summarize([{"budget_error": "boom"}, _b(10, 10, {})])
    assert s["calls_total"] == 2 and s["calls_with_error"] == 1


# --- 加固：坏数据形状不能让 summarize 抛异常，也不能污染干净条目的统计 ------

def test_summarize_none_list_does_not_raise():
    assert budget.summarize(None)["calls_total"] == 0


def test_summarize_non_dict_entries_do_not_raise():
    s = budget.summarize([None, 42, "oops", _b(10, 10, {})])
    assert s["calls_total"] == 4
    assert s["calls_with_usage"] == 1


def test_summarize_segments_not_dict_falls_back():
    bad = _b(10, 10, {})
    bad["segments"] = "not a dict"
    s = budget.summarize([bad, _b(10, 10, {"a": {"est": 5, "chars": 0, "count": 1}})])
    assert s["per_call_context"] == {
        "a": {"min": 5, "median": 5, "max": 5, "share": 1.0, "layer": budget.LAYER_NONE}}


def test_summarize_est_string_or_nan_does_not_raise_or_pollute():
    segs_bad = {"a": {"est": "oops", "chars": 0, "count": 1},
                "b": {"est": float("nan"), "chars": 0, "count": 1}}
    segs_ok = {"a": {"est": 10, "chars": 0, "count": 1}}
    s = budget.summarize([_b(10, 10, segs_bad), _b(10, 10, segs_ok)])
    assert s["per_call_context"]["a"]["max"] == 10
    assert s["per_call_context"]["b"]["max"] == 0


def test_summarize_est_total_bad_type_excluded_from_cumulative():
    bad = _b(10, 10, {})
    bad["est_total"] = "oops"
    s = budget.summarize([bad, _b(10, 10, {})])
    assert s["cumulative_submitted_tokens"] == 10


def test_summarize_calls_with_unknown_counts_calls_not_occurrences():
    """unknown 段自身的 count（多少次贡献进了 unknown），不是跨段 count 求和。"""
    has_unknown = _b(10, 10, {"unknown": {"est": 0, "chars": 0, "count": 2}})
    no_unknown = _b(10, 10, {"a": {"est": 10, "chars": 0, "count": 1}})
    unknown_zero_count = _b(10, 10, {"unknown": {"est": 0, "chars": 0, "count": 0}})
    s = budget.summarize([has_unknown, no_unknown, unknown_zero_count])
    assert s["calls_with_unknown"] == 1


# --- 空洞 1：calls_with_error 是否被排除在 calls_with_usage 之外 ----------------

def test_calls_with_error_are_excluded_from_usage_counts():
    """error 条目计入 calls_total 和 calls_with_error，但不计入 calls_with_usage。

    这是 reviewer 在测试覆盖扫描中发现的空洞：现有测试没有同时包含
    正常条目和 error 条目，无法验证 error 条目是否真的被排除在外。
    这条测试特意构造一个既有 budget_error 又有 actual_prompt_tokens 的条目，
    确保即使代码错误地把 error 也计入 calls_with_usage，测试也能检测到。
    """
    normal_with_usage = _b(10, 100, {"a": {"est": 10, "chars": 0, "count": 1}})
    # error 条目：既有 budget_error 又有 actual_prompt_tokens（这是关键）
    error_with_usage = {
        "budget_error": "connection_timeout",
        "call_index": 1,
        "actual_prompt_tokens": 50,  # 关键：error 也有 usage，用来测试排除逻辑
        "cached_tokens": None
    }
    normal_without_usage = _b(20, None, {"a": {"est": 20, "chars": 0, "count": 1}})

    s = budget.summarize([normal_with_usage, error_with_usage, normal_without_usage])

    # calls_total：所有条目都算（包括 error）
    assert s["calls_total"] == 3
    # calls_with_error：只有 error 条目算
    assert s["calls_with_error"] == 1
    # calls_with_usage：只有有 usage 的正常条目算，error 被排除（不是 2）
    assert s["calls_with_usage"] == 1
    # calls_without_usage：只有正常但没 usage 的条目算，error 被排除
    assert s["calls_without_usage"] == 1


# --- 空洞 2：share 的分母到底是各段 est 之和还是 est_total -----

def test_share_uses_segment_sum_not_est_total_as_denominator():
    """per_call_context 的 share 分母是各段 est 之和，不是 est_total。

    这是 reviewer 在测试覆盖扫描中发现的空洞：现有 test_share_sums_to_one
    中两条 budget 的 est_total 恰好等于各段 est 之和，两种分母会算出同一个数。
    这条测试构造 est_total 与各段 est 之和显著不同的输入，钉住真正的分母。
    """
    # 构造：各段 est 之和 = 100，但 est_total = 1000（可能由于 residual 的存在）
    segs = {"a": {"est": 30, "chars": 0, "count": 1},
            "b": {"est": 70, "chars": 0, "count": 1}}

    budget_with_mismatch = {
        "context_mode": "state",
        "call_index": 0,
        "segments": segs,
        "est_total": 1000,  # 不等于 30+70=100，体现了 residual 等因素
        "actual_prompt_tokens": 1000,
        "cached_tokens": None,
        "residual_tokens": 0,
        "signed_bias": 0.0,
        "abs_pct_error": 0.0,
        "calibration": "uninitialized"
    }

    s = budget.summarize([budget_with_mismatch])

    # 如果分母是各段 est 之和（100），share 应该是：
    # a: 30/100 = 0.3, b: 70/100 = 0.7（总和 = 1.0）
    # 如果分母是 est_total（1000），share 应该是：
    # a: 30/1000 = 0.03, b: 70/1000 = 0.07（总和 = 0.1）
    assert s["per_call_context"]["a"]["share"] == pytest.approx(0.3)
    assert s["per_call_context"]["b"]["share"] == pytest.approx(0.7)

    # 验证总和为 1.0
    total_share = sum(v["share"] for v in s["per_call_context"].values())
    assert total_share == pytest.approx(1.0)


# ================= scripts/calibrate_tokens.py 的纯计算部分 =================
#
# ⚠ scripts/ 不是包（没有 __init__.py），而且 conftest 的 autouse fixture 把
#   工作目录换到 tmp_path 了，所以按**文件绝对路径**加载，不靠 cwd 也不靠 sys.path。

def _calib():
    import importlib.util
    import pathlib
    import sys
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "calibrate_tokens.py"
    mod = sys.modules.get("_calibrate_tokens_under_test")
    if mod is None:
        spec = importlib.util.spec_from_file_location("_calibrate_tokens_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_calibrate_tokens_under_test"] = mod
        spec.loader.exec_module(mod)
    return mod


def test_check_drift_skips_calls_with_prefix_drift():
    check_drift_from_budgets = _calib().check_drift_from_budgets
    good = [_b(100, 100, {}, idx=0), _b(150, 148, {}, idx=1)]
    bad = [dict(_b(100, 100, {}, idx=0), prefix_drift=True), _b(999, 100, {}, idx=1)]
    assert check_drift_from_budgets([good])["pairs"] == 1
    assert check_drift_from_budgets([bad])["pairs"] == 0


def test_check_drift_skips_calls_without_usage():
    check_drift_from_budgets = _calib().check_drift_from_budgets
    assert check_drift_from_budgets(
        [[_b(100, None, {}, idx=0), _b(150, 148, {}, idx=1)]])["pairs"] == 0


def test_check_drift_reports_pair_count():
    """样本少时中位数没有意义，pairs 必须报出来。"""
    check_drift_from_budgets = _calib().check_drift_from_budgets
    r = check_drift_from_budgets([[_b(100, 100, {}, idx=0), _b(150, 148, {}, idx=1)]])
    assert r["pairs"] == 1 and "abs_rel_error_median" in r


def test_check_drift_reports_pairs_even_when_zero():
    """空结果也得带 pairs：调用方对「没数据」和「误差很小」的处置完全不同。"""
    assert _calib().check_drift_from_budgets([])["pairs"] == 0


def test_check_drift_ignores_window_mode_calls():
    """window 模式的尾部构成是另一回事，混进来的中位数谁也代表不了。"""
    check_drift_from_budgets = _calib().check_drift_from_budgets
    rows = [_b(100, 100, {}, mode="window", idx=0), _b(150, 148, {}, mode="window", idx=1)]
    assert check_drift_from_budgets([rows])["pairs"] == 0


def test_check_drift_sorts_by_call_index_not_file_order():
    """jsonl 里的顺序不保证按 call_index 排 —— 排错了差分就是乱配对的。

    est != actual 在这三行里，且乱序配对和按 call_index 配对算出的
    abs_rel_error_median 必须不同 —— 否则这条测试就算把 sorted() 换成
    「按列表原样配对」也照样全绿，保护不了「按 call_index 排序」这件事。
    call_index 顺序应为 0,1,2（est=100/200/300, actual=90/250/270），
    但传入顺序是 idx=2,0,1（模拟 jsonl 乱序）。
      按 call_index 配对: (0->1) |100-160|/160=0.375；(1->2) |100-20|/20=4.0
                          median = 2.1875
      按文件原样配对    : (idx2->idx0) |(-200)-(-180)|/180=0.1111；
                          (idx0->idx1) |100-160|/160=0.375
                          median = 0.24305...（与上面不同）
    """
    check_drift_from_budgets = _calib().check_drift_from_budgets
    rows = [_b(300, 270, {}, idx=2), _b(100, 90, {}, idx=0), _b(200, 250, {}, idx=1)]
    r = check_drift_from_budgets([rows])
    assert r["pairs"] == 2
    assert r["abs_rel_error_median"] == 2.1875


def test_check_drift_survives_broken_rows():
    """观测设施读到坏数据要少算一对，不能抛出去。"""
    check_drift_from_budgets = _calib().check_drift_from_budgets
    rows = [None, "oops", {"budget_error": "boom", "call_index": 0},
            _b(100, 100, {}, idx=1), _b(150, 148, {}, idx=2)]
    assert check_drift_from_budgets([rows])["pairs"] == 1


def test_check_drift_does_not_pair_across_runs():
    """两个 run 的末尾和开头不是相邻调用，配对它们等于凭空造一个巨大的差分。"""
    check_drift_from_budgets = _calib().check_drift_from_budgets
    run_a = [_b(100, 100, {}, idx=0)]
    run_b = [_b(9000, 9000, {}, idx=0)]
    assert check_drift_from_budgets([run_a, run_b])["pairs"] == 0


def test_check_drift_error_is_relative_to_actual_delta():
    check_drift_from_budgets = _calib().check_drift_from_budgets
    # actual 增 100，est 增 110 → 相对误差 0.1
    rows = [_b(1000, 1000, {}, idx=0), _b(1110, 1100, {}, idx=1)]
    r = check_drift_from_budgets([rows])
    assert r["abs_rel_error_median"] == pytest.approx(0.1)


def test_check_drift_p90_never_below_median_with_two_pairs():
    """n=2 时 p90 的索引算法不能自相矛盾：p90 必须 >= median。

    用 int(0.9*(n-1)) 截断在 n=2 时会算出索引 0（取到最小误差），把
    p90 报得比 median 还小 —— reviewer 在真实标定里见过这种自相矛盾的输出
    （median 0.0310 / p90 0.0204）。这里构造 errs=[0.02, 0.05]（median=0.035），
    截断版会输出 p90=0.02（< median，错误），修复后应输出 p90=0.05（>= median）。
    """
    check_drift_from_budgets = _calib().check_drift_from_budgets
    rows = [_b(1000, 1000, {}, idx=0), _b(1102, 1100, {}, idx=1),
            _b(1312, 1300, {}, idx=2)]
    r = check_drift_from_budgets([rows])
    assert r["pairs"] == 2
    assert r["abs_rel_error_median"] == pytest.approx(0.035)
    assert r["abs_rel_error_p90"] == pytest.approx(0.05)
    assert r["abs_rel_error_p90"] >= r["abs_rel_error_median"]


def test_budgets_of_dedupes_repeated_call_index(tmp_path):
    """一次调用产生多条 step 时，同一条 budget 会被挂在每条 step 上（loop.py）。
    不去重的话同一次调用被当成好几次，差分对里混进一堆 0。"""
    import json
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "steps.jsonl").write_text("\n".join(
        json.dumps({"budget": _b(100, 100, {}, idx=i)}) for i in (0, 0, 1)), encoding="utf-8")
    assert [b["call_index"] for b in _calib()._budgets_of(run_dir)] == [0, 1]


#: 拟合器测试用的合成系数。桶名跟着 tokens.BUCKETS 走，加桶不用改这些测试。
_SYNTH = {"han": 0.6, "cjk_punct": 0.9, "west_punct": 1.1, "ascii_alpha": 0.25,
          "ascii_digit": 0.5, "ascii_sym": 0.3, "astral": 1.5, "other": 0.4}
#: 每个桶各一条「几乎 100% 落它」的纯语料 —— 这正是真探针要求的形状。
_PURE = {"han": "中", "cjk_punct": "，", "west_punct": "—", "ascii_alpha": "a",
         "ascii_digit": "7", "ascii_sym": "[", "astral": "😀", "other": "α"}


def _synth_samples():
    texts = [""]
    for ch in _PURE.values():
        texts += [ch * 100, ch * 400]
    def synth(t):
        c = tokens.count_chars(t)
        return round(sum(_SYNTH[b] * n for b, n in c.items()) + 12)
    return [(t, synth(t)) for t in texts]


def test_fit_char_coefficients_recovers_known_coefficients():
    """先验证拟合器本身是对的：喂进合成数据，每个桶的系数都要能还原。"""
    calib = _calib()
    fit = calib.fit_char_coefficients(_synth_samples())
    for bucket, expected in _SYNTH.items():
        assert fit["coefficients"][bucket] == pytest.approx(expected, abs=0.02), bucket
    assert fit["intercept"] == pytest.approx(12, abs=1.0)
    assert fit["max_abs_residual"] < 1.0


def test_fit_char_coefficients_reports_residuals_not_just_numbers():
    """系数只有在「按字符类线性叠加」成立时才有意义 —— 残差必须一起报出来。"""
    calib = _calib()
    fit = calib.fit_char_coefficients(_synth_samples())
    assert "rms_residual" in fit and "max_abs_residual" in fit


def test_fit_char_coefficients_refuses_when_underdetermined():
    """未知数比样本多 —— 报「解不出来」，不返回一组看起来正常的错数。"""
    calib = _calib()
    fit = calib.fit_char_coefficients([("中", 1), ("a", 1), ("😀", 2)])
    assert "coefficients" not in fit and "note" in fit


def test_fit_char_coefficients_names_the_buckets_that_have_no_corpus():
    """某个桶在所有样本里都是 0（漏了它的纯语料）—— 正规方程本来就奇异。
    必须点名缺哪个桶，而不是笼统报「奇异」让人自己猜。这是本次分桶从 3 个
    加到 7 个之后最容易犯的错：加了桶忘了加语料。"""
    calib = _calib()
    samples = [(t, v) for t, v in _synth_samples() if "😀" not in t]
    fit = calib.fit_char_coefficients(samples)
    assert "coefficients" not in fit
    assert fit["empty_buckets"] == ["astral"]


def test_real_corpus_evaluation_is_the_acceptance_criterion():
    """验收判据是**真实语料上的相对误差**，不是拟合残差。

    构造一个「拟合完美、真实语料上差很远」的局面 —— 这正是 2026-09-09 真机
    踩到的形状（合成语料完美线性，真实混排差 2.6 倍）。evaluate_on_real_corpora
    必须判 passes=False。
    """
    calib = _calib()
    fit = calib.fit_char_coefficients(_synth_samples())
    baseline = 12
    # 真实语料实测比预测贵一倍：模型不成立。
    text = "中" * 100
    predicted = calib.predict_tokens(text, fit)
    ev = calib.evaluate_on_real_corpora(fit, [("real", text, baseline + int(predicted * 2))],
                                        baseline)
    assert ev["passes"] is False
    assert ev["max_rel_error"] == pytest.approx(0.5, abs=0.02)


def test_every_fit_bucket_has_a_near_pure_corpus():
    """每个桶都得有一条「几乎 100% 落它」的纯语料，否则它的系数只能靠解方程
    从别的语料里挤出来 —— 那正是老三桶模型「残差看着还行、每个系数都错」的
    来源。加桶时忘了加语料，这条测试当场拦下。

    ⚠ 门槛设 0.85 而不是 1.0：emoji 语料里有一个 U+FE0F（落 other 桶）是
      emoji 本身的写法带来的，astral 占 8/9 ≈ 0.89 —— 那是真实 emoji 的样子，
      不该为了凑纯度把它改掉。other 桶由 other_bmp 那条语料撑着。
    """
    calib = _calib()
    covered = {}
    for kind in calib.FIT_KINDS:
        text = calib.PURE_ALPHABETS[kind]
        c = tokens.count_chars(text)
        top = max(c, key=lambda b: c[b])
        share = c[top] / len(text)
        assert share >= 0.85, f"语料 {kind} 不纯：{top} 只占 {share:.0%}，{c}"
        covered.setdefault(top, []).append(kind)
    missing = [b for b in tokens.BUCKETS if b not in covered]
    assert not missing, f"这些桶没有对应的纯语料：{missing}"


def test_mixed_corpus_is_not_used_for_fitting():
    """mixed 是检验组的一员。把它喂进拟合等于让判据和被判者混在一起 ——
    残差自然好看，但那时候「误差小」不再说明任何事情。"""
    calib = _calib()
    assert "mixed" in calib.PURE_ALPHABETS
    assert "mixed" not in calib.FIT_KINDS
    assert "mixed" in calib.build_real_corpora()


def test_real_corpora_degrade_when_a_source_is_missing(tmp_path, monkeypatch):
    """语料取不到就少一条，**不抛**：标定脚本是观测设施，坏输入降级不砸任务。
    也绝不能拿空串/合成品顶替 —— 那会让检验组里混进一条假的「真实语料」。"""
    calib = _calib()
    monkeypatch.setattr(calib, "SCREENMAP_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(calib, "RUNS_ROOT", tmp_path / "no-runs")
    real = calib.build_real_corpora()
    assert "real_elements" not in real and "real_memory" not in real
    # 不依赖外部数据的两条必须还在，而且都是主循环真的会发出去的字节。
    assert real["real_system_prompt"] and real["real_tools_json"]


def test_real_elements_corpus_looks_like_build_observation_output(tmp_path, monkeypatch):
    """真实元素语料的行格式必须和 perceive/elements.py 的 build_observation 一致：
    `[id] 文字 (cx,cy) 0.99`。格式一旦对不上，量到的就不是模型真正看到的那段。"""
    import json
    import re
    calib = _calib()
    sm = tmp_path / "screenmap.json"
    sm.write_text(json.dumps({"nodes": [{"representative": ["关于本机", "About"]}]}),
                  encoding="utf-8")
    monkeypatch.setattr(calib, "SCREENMAP_PATH", sm)
    text = calib._real_elements_text()
    lines = text.splitlines()
    assert lines[0].startswith("observation #")
    for line in lines[1:]:
        assert re.fullmatch(r"\[\d+\] .+ \(\d+,\d+\) \d\.\d\d", line), line


def test_real_corpus_evaluation_reports_unknown_not_pass_when_there_is_no_data():
    """一条真实语料都没量到时 verdict 是 None（不知道），不是 True ——
    「没数据」折成「通过」就是在编。"""
    calib = _calib()
    fit = calib.fit_char_coefficients(_synth_samples())
    assert calib.evaluate_on_real_corpora(fit, [], 12)["passes"] is None
    assert calib.evaluate_on_real_corpora(fit, [("r", "中", None)], None)["passes"] is None


def test_assert_only_one_difference_catches_second_variable():
    """差分组内出现第二个变化量 → 这一组的结论作废，必须当场炸。"""
    calib = _calib()
    a = calib.Request("a", [{"role": "user", "content": "x"}], [])
    b = calib.Request("b", [{"role": "user", "content": "y"}], [{"type": "function"}])
    with pytest.raises(AssertionError):
        calib.assert_only_one_difference(a, b, field="messages")


def test_assert_only_one_difference_catches_unchanged_variable():
    calib = _calib()
    a = calib.Request("a", [{"role": "user", "content": "x"}], [])
    b = calib.Request("b", [{"role": "user", "content": "x"}], [])
    with pytest.raises(AssertionError):
        calib.assert_only_one_difference(a, b, field="messages")


def test_estimate_plan_reports_unknown_image_cost_as_none():
    """IMAGE_RULES 没填时图片成本是「不知道」，不是 0 —— 写成 0 就是在瞒报成本。"""
    calib = _calib()
    plan = [calib.Request("i", [{"role": "user", "content": "x"}], [], image_px=(624, 1388))]
    est = calib.estimate_plan(plan, "no-such-model", 1)
    assert est["image_tokens"] is None and est["requests"] == 1


def test_dry_run_confirm_never_sends():
    """--dry-run 的核心承诺：一个请求都不发。"""
    calib = _calib()
    class Boom:
        resolved = None
        def decide(self, *a, **k):
            raise AssertionError("dry-run 下不该发请求")
    r = calib.Runner(Boom(), dry_run=True, yes=True, out_dir=pathlib.Path("/nonexistent"))
    assert r.confirm("x", [calib.Request("a", [{"role": "user", "content": "x"}], [])]) is False
    with pytest.raises(RuntimeError):
        r.send(calib.Request("a", [{"role": "user", "content": "x"}], []))


# =========================== 分层与自陈可信度 ==============================

def test_real_system_prompt_hits_the_exact_constant_layer():
    """真实前缀里最大的一块必须走第 1 层：它是确定性的，不该被估。"""
    from iphone_agent.harness.prompt import system_prompt
    from iphone_agent.harness.tools import tool_defs

    text = system_prompt(allow_coord_tap=True, has_skills=True)
    log = MessageLog()
    log.system(text)
    log.freeze()
    b = budget.account(log.raw(), tool_defs(True), None,
                       context_mode="state", call_index=0, model_id="m")
    sysdeg = b["segments"]["system"]
    assert sysdeg["layer"] == tokens.LAYER_EXACT
    assert sysdeg["est"] == 3051          # 2026-09-1x 按需看图：提示词版本 23（感知一节改为「默认只有 OCR」）
                                           # 后重标：5446 字。出处 docs/superpowers/acceptance/2026-09-15-token标定/
                                           # probe-segments-20260915-115841.json（上一版 2873）
    tsc = b["segments"]["tools_schema"]
    assert tsc["layer"] == tokens.LAYER_EXACT
    assert tsc["est"] == 6308             # 2026-09-1x 按需看图：observe 描述改为看全屏后重标：
                                           # 18 个工具，走 tools= 参数；出处
                                           # docs/superpowers/acceptance/2026-09-15-token标定/
                                           # probe-segments-20260915-115841.json（上一版 6265）
    # ⚠ 这两段合起来是冻结前缀的大头。分层的全部收益就在这一行里。
    assert b["est_by_layer"][tokens.LAYER_EXACT] == 3051 + 6308


def test_changed_prompt_falls_back_to_the_ratio_layer_not_the_stale_constant():
    """提示词改一个字，hash 就对不上 —— 必须降级报 segment_ratio，
    绝不能继续报 1920（那是「未改字」那份提示词的实测常量，是一个看起来
    正常的错数）。"""
    from iphone_agent.harness.prompt import system_prompt

    log = MessageLog()
    log.system(system_prompt(allow_coord_tap=True, has_skills=True) + "X")
    log.freeze()
    b = budget.account(log.raw(), None, None,
                       context_mode="state", call_index=0, model_id="m")
    assert b["segments"]["system"]["layer"] == tokens.LAYER_SEGMENT_RATIO
    assert b["segments"]["system"]["est"] != 1920


def test_every_segment_reports_which_layer_its_estimate_came_from():
    b = budget.account(a_view(), [{"type": "function", "function": {"name": "tap"}}],
                       {"prompt_tokens": 100}, context_mode="state",
                       call_index=0, model_id="qwen3.7-plus")
    known = {tokens.LAYER_EXACT, tokens.LAYER_SEGMENT_RATIO, tokens.LAYER_CHAR_CLASS,
             budget.LAYER_IMAGE_RULE, budget.LAYER_UNCALIBRATED,
             budget.LAYER_MIXED, budget.LAYER_NONE}
    for name, s in b["segments"].items():
        assert s["layer"] in known, (name, s["layer"])
    # 元素列表有实测比值 —— 2026-09-09 这轮标定连 state_history 也测到了
    # （见 SEGMENT_TOKENS_PER_CHAR），这条测试原本拿它当「没标定的段」的例子，
    # 现在改用 state_memo：它在 SEGMENTS_PENDING_CALIBRATION 的「完全没有语料」
    # 分组里，标定之后依然没有实测支撑，两段必须报不同的层这条断言才成立。
    assert b["segments"]["state_elements"]["layer"] == tokens.LAYER_SEGMENT_RATIO
    assert "state_memo" in tokens.SEGMENTS_PENDING_CALIBRATION
    assert b["segments"]["state_memo"]["layer"] == tokens.LAYER_CHAR_CLASS
    assert b["segments"]["protocol_residual"]["layer"] == budget.LAYER_UNCALIBRATED
    assert sum(b["est_by_layer"].values()) == b["est_total"]


def test_a_segment_fed_by_two_different_layers_is_marked_mixed_not_silently_relabelled():
    """同名段被不同层贡献过，est 就是几层加出来的 —— 不能宣称整段精确。"""
    segs = {}
    budget._bump(segs, "x", est=10, layer=tokens.LAYER_EXACT)
    budget._bump(segs, "x", est=5, layer=tokens.LAYER_CHAR_CLASS)
    assert segs["x"]["layer"] == budget.LAYER_MIXED
    # est=0 的占位不表态，不该把精确段污染成 mixed
    segs2 = {}
    budget._bump(segs2, "y", est=10, layer=tokens.LAYER_EXACT)
    budget._bump(segs2, "y", est=0)
    assert segs2["y"]["layer"] == tokens.LAYER_EXACT


def test_summarize_reports_how_much_of_the_run_is_measured_vs_estimated():
    segs = {"system": {"est": 1861, "chars": 3486, "count": 1,
                       "layer": tokens.LAYER_EXACT},
            "state_history": {"est": 139, "chars": 300, "count": 1,
                              "layer": tokens.LAYER_CHAR_CLASS}}
    s = budget.summarize([_b(2000, 2000, segs)])
    prov = s["token_provenance"]
    assert prov[tokens.LAYER_EXACT]["est"] == 1861
    assert prov[tokens.LAYER_CHAR_CLASS]["est"] == 139
    assert abs(sum(v["share"] for v in prov.values()) - 1.0) < 1e-9
    assert s["measured_share"] == pytest.approx(1861 / 2000)
    assert s["per_call_context"]["system"]["layer"] == tokens.LAYER_EXACT


def test_summarize_tolerates_old_budgets_without_a_layer_field():
    """jsonl 里的历史留档没有这个字段：当 none，不猜也不抛。"""
    s = budget.summarize([_b(100, 100, {"a": {"est": 100, "chars": 0, "count": 1}})])
    assert s["token_provenance"][budget.LAYER_NONE]["est"] == 100
    assert s["measured_share"] == 0.0


# =========================== --probe segments =============================

def test_segment_corpora_degrade_instead_of_raising_when_sources_are_missing(tmp_path,
                                                                            monkeypatch):
    """语料源全缺（干净的 worktree 就是这样）时，构造器要如实报「无语料」，
    不能抛、更不能拿别的段的文本顶替。"""
    calib = _calib()
    monkeypatch.setattr(calib, "RUNS_ROOT", tmp_path / "nope")
    monkeypatch.setattr(calib, "SCREENMAP_PATH", tmp_path / "nope.json")
    corpora, missing = calib.build_segment_corpora()
    # system 不依赖任何外部文件，永远量得到；元素列表依赖 screenmap，这时必须缺席。
    assert "system" in corpora
    assert "state_elements" not in corpora and "state_elements" in missing
    assert all(isinstance(v, str) and v for v in missing.values()), "缺语料必须说清原因"
    # 一个段不能同时「有语料」和「无语料」——两边都在会让人以为它还没量。
    assert not (set(corpora) & set(missing))


def test_probe_segments_is_registered_and_needs_no_network_to_plan():
    calib = _calib()
    assert "segments" in calib.PROBES
