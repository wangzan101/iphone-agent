"""最近运行的摘要。

「昨天干了什么」由 runlog 回答，不另建存储 —— 失败信息本来就在 runs/ 里，
给它一个读取入口比复制一份数据薄（spec §4.3）。

⚠ 摘要只含程序能零歧义生成的字段。「卡在主屏翻页来回打转」那种是语义判断，
程序没有页面分类能力，不能写进来。
"""
import json
import os
from datetime import datetime as _real_datetime

import pytest

from iphone_agent.harness import recap
from iphone_agent.memory import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory")

# root 绕过文件权限位，chmod 造不出 EACCES —— 那两条测试在 root 下无意义。
_needs_non_root = pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root 绕过 rwx 权限位，chmod 0o000 不会产生 PermissionError，这条测不出东西",
)


def _make_run(root, name, task, end_reason, steps, started):
    d = root / name
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "task": task, "end_reason": end_reason, "steps": steps,
        "started_at": started, "ended_at": started + 10,
    }, ensure_ascii=False), encoding="utf-8")
    return d


def test_summaries_are_newest_first(tmp_path):
    _make_run(tmp_path, "a", "任务一", "done_success", 8, 1000)
    _make_run(tmp_path, "b", "任务二", "max_steps", 30, 2000)
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    assert len(lines) == 2
    assert "任务二" in lines[0] and "任务一" in lines[1]
    assert skipped == 0


def test_summary_contains_only_mechanical_fields(tmp_path):
    _make_run(tmp_path, "a", "打开设置", "max_steps", 30, 1000)
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    line = lines[0]
    assert "打开设置" in line
    assert "max_steps" in line
    assert "30" in line
    assert skipped == 0


def test_limit_is_respected(tmp_path):
    for i in range(7):
        _make_run(tmp_path, f"r{i}", f"任务{i}", "done_success", i, 1000 + i)
    lines, skipped = recap.run_summaries(tmp_path, limit=3)
    assert len(lines) == 3
    assert skipped == 0


def test_broken_run_json_is_skipped_not_fatal(tmp_path):
    _make_run(tmp_path, "good", "好的", "done_success", 5, 2000)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "run.json").write_text("{ 半个 json", encoding="utf-8")
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    assert len(lines) == 1 and "好的" in lines[0]
    assert skipped == 1


def test_missing_runs_root_is_empty(tmp_path):
    assert recap.run_summaries(tmp_path / "nope", limit=5) == ([], 0)


def test_unfinished_run_is_skipped(tmp_path):
    """还在跑的运行 end_reason 是 null，摘要里写它没意义。"""
    d = tmp_path / "running"
    d.mkdir()
    (d / "run.json").write_text(json.dumps(
        {"task": "在跑", "end_reason": None, "steps": 0, "started_at": 1}), encoding="utf-8")
    assert recap.run_summaries(tmp_path, limit=5) == ([], 0)


def test_long_task_text_is_truncated(tmp_path):
    """注入是有预算的，任务原话可能很长。"""
    _make_run(tmp_path, "a", "很长的任务" * 20, "done_success", 3, 1000)
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    line = lines[0]
    assert len(line) < 120
    assert skipped == 0


def test_wrong_type_started_at_does_not_kill_the_good_rows(tmp_path):
    """合法 JSON 不等于字段类型对：started_at 是字符串会让 sort 抛 TypeError，
    如果不在 try 里规整类型，好的那行会跟着坏行一起消失（全灭，不是单条跳过）。
    必须是「坏行 + 好行」两行同时存在，才能测出这种全灭，只放坏行测不出来。
    """
    _make_run(tmp_path, "good", "好的", "done_success", 5, 2000)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "run.json").write_text(json.dumps({
        "task": "坏的", "end_reason": "done_success", "steps": 1,
        "started_at": "not-a-number",
    }, ensure_ascii=False), encoding="utf-8")
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    assert len(lines) == 1 and "好的" in lines[0]
    assert skipped == 1


def test_wrong_type_task_does_not_kill_the_good_rows(tmp_path):
    """task 是 int 时旧代码 task.replace(...) 抛 AttributeError，把整批（含好行）
    一起带走。规整之后 int 是可以安全转成字符串的标量（"123"，不是容器 repr），
    所以这里不要求那一行被排除 —— 只要求它不再拖累好行、不再抛异常。
    """
    _make_run(tmp_path, "good", "好的", "done_success", 5, 2000)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "run.json").write_text(json.dumps({
        "task": 123, "end_reason": "done_success", "steps": 1,
        "started_at": 1000,
    }, ensure_ascii=False), encoding="utf-8")
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    assert any("好的" in line for line in lines)


def test_wrong_type_steps_does_not_leak_python_repr(tmp_path):
    """steps 是 list 时 int(...) 会抛 TypeError，整行按坏数据处理 —— 输出里
    不该出现 [1, 2, 3] 这种 Python repr，好行也不能被这一行拖累。
    """
    _make_run(tmp_path, "good", "好的", "done_success", 5, 2000)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "run.json").write_text(json.dumps({
        "task": "怪的", "end_reason": "done_success", "steps": [1, 2, 3],
        "started_at": 1000,
    }, ensure_ascii=False), encoding="utf-8")
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    joined = "\n".join(lines)
    assert "[1, 2, 3]" not in joined
    assert any("好的" in line for line in lines)
    assert skipped == 1


def test_wrong_type_end_reason_does_not_leak_python_repr(tmp_path):
    """end_reason 是非空 dict 时是「真值」，`x or ""` 不会短路成空字符串，
    单纯 str() 会把 {'x': 1} 这种 Python repr 原样吐进摘要 —— 必须整行按
    坏数据处理，跟 steps 是同一个缺陷类，不能只堵会崩的那两个字段。
    """
    _make_run(tmp_path, "good", "好的", "done_success", 5, 2000)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "run.json").write_text(json.dumps({
        "task": "怪的", "end_reason": {"x": 1}, "steps": 1,
        "started_at": 1000,
    }, ensure_ascii=False), encoding="utf-8")
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    joined = "\n".join(lines)
    assert "{'x': 1}" not in joined
    assert any("好的" in line for line in lines)
    assert skipped == 1


def test_started_at_nan_does_not_kill_the_good_rows(tmp_path):
    """NaN 是 json 模块默认接受的合法 token（json.loads('{"x": NaN}') 不报错），
    float(nan) 本身也不报错，会悄悄混过类型规整，真正炸的是排序/格式化阶段。
    """
    _make_run(tmp_path, "good", "好的", "done_success", 5, 2000)
    _make_run(tmp_path, "bad", "坏的", "done_success", 1, float("nan"))
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    assert len(lines) == 1 and "好的" in lines[0]
    assert skipped == 1


def test_started_at_inf_does_not_kill_the_good_rows(tmp_path):
    """Infinity 同样是合法 token 且 float(inf) 不报错，真正炸的是
    datetime.fromtimestamp(inf) —— OverflowError，不在旧 except 元组里。
    """
    _make_run(tmp_path, "good", "好的", "done_success", 5, 2000)
    _make_run(tmp_path, "bad", "坏的", "done_success", 1, float("inf"))
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    assert len(lines) == 1 and "好的" in lines[0]
    assert skipped == 1


def test_steps_inf_does_not_kill_the_good_rows(tmp_path):
    """int(float('inf')) 抛的是 OverflowError，不是 ValueError，
    旧的 except (..., ValueError) 接不住，会漏网炸穿整批。
    """
    _make_run(tmp_path, "good", "好的", "done_success", 5, 2000)
    _make_run(tmp_path, "bad", "坏的", "done_success", float("inf"), 1000)
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    assert len(lines) == 1 and "好的" in lines[0]
    assert skipped == 1


def test_format_stage_unexpected_exception_is_caught_not_fatal(tmp_path, monkeypatch):
    """验证保护网本身：格式化阶段的宽捕获必须能兜住三条 NaN/Infinity 用例
    都覆盖不到的、完全没想到过的异常类型（这里用 RuntimeError 模拟），
    而不是只对已知的 OverflowError/ValueError 生效。
    """
    _make_run(tmp_path, "bad", "坏的", "done_success", 1, 999)
    _make_run(tmp_path, "good", "好的", "done_success", 2, 2000)

    class _FlakyDatetime:
        @staticmethod
        def fromtimestamp(ts):
            if ts == 999:
                raise RuntimeError("模拟一个格式化阶段没人想到过的异常")
            return _real_datetime.fromtimestamp(ts)

    monkeypatch.setattr(recap, "datetime", _FlakyDatetime)
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    assert len(lines) == 1 and "好的" in lines[0]
    assert skipped == 1


def test_runs_root_that_is_a_file_returns_empty_not_crash(tmp_path):
    """root.exists() 对文件也返回 True，检查通过后直接进 root.iterdir()，
    而这一行不在任何 try 里 —— 一行 run.json 都没参与就整函数死了，
    跟「坏数据拖累好行」是同一类失败：一个和数据无关的输入就能让函数
    整体炸穿。必须用 is_dir() 挡在最前面。
    """
    p = tmp_path / "not_a_dir"
    p.write_text("not a dir", encoding="utf-8")
    assert recap.run_summaries(p, limit=5) == ([], 0)


def test_skipped_rows_are_counted(tmp_path):
    """跳过的行数是调用方唯一能用来判断「摘要功能本身是否正常」的信号，
    必须精确：2 个坏行必须让 skipped 恰好等于 2，不多不少。
    """
    _make_run(tmp_path, "good", "好的", "done_success", 5, 2000)
    bad1 = tmp_path / "bad1"
    bad1.mkdir()
    (bad1 / "run.json").write_text("{ 半个 json", encoding="utf-8")
    bad2 = tmp_path / "bad2"
    bad2.mkdir()
    (bad2 / "run.json").write_text(json.dumps({
        "task": "坏的二", "end_reason": "done_success", "steps": 1,
        "started_at": "not-a-number",
    }, ensure_ascii=False), encoding="utf-8")
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    assert len(lines) == 1
    assert skipped == 2


def test_a_programming_bug_is_distinguishable_from_no_history(tmp_path, monkeypatch):
    """这次改动的全部意义：宽 except 会把「摘要功能自己写错了」和「确实没有
    历史运行」都变成空列表 —— 调用方原本无法区分。有了 skipped 之后，
    真正没有历史运行是 ([], 0)，功能内部炸了是 ([], N>0)，两者不再是同一个
    返回值。这里 monkeypatch _clean_str 模拟它自己写错（必定抛异常），只放
    一条本来完全正常的数据，断言返回的是 ([], 1)，而不是看起来人畜无害的
    ([], 0)。
    """
    _make_run(tmp_path, "normal", "正常任务", "done_success", 3, 1000)

    def _always_broken(value):
        raise RuntimeError("模拟 _clean_str 自己写错了")

    monkeypatch.setattr(recap, "_clean_str", _always_broken)
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    assert lines == []
    assert skipped == 1


def test_over_long_runs_root_returns_empty_not_crash(tmp_path):
    """`is_dir()` 只吞 ENOENT/ENOTDIR/EBADF/ELOOP，别的 OSError 原样往外抛。
    路径名超过文件系统上限时它抛 OSError [Errno 63] File name too long ——
    这一整段目录访问当时不在任何 try 里，一行 run.json 都还没碰到就炸穿了。
    """
    too_long = tmp_path / ("a" * 300)
    assert recap.run_summaries(too_long, limit=5) == ([], 0)


@_needs_non_root
def test_unreadable_runs_root_returns_empty_not_crash(tmp_path):
    """目录能 stat（is_dir() 说 True）不等于能列 —— iterdir() 会抛
    PermissionError [Errno 13]。这一行是「换成 is_dir() 更安全」那轮完全
    没保护到的位置：检查通过了，紧接着的遍历照样炸。
    """
    d = tmp_path / "runs"
    d.mkdir()
    os.chmod(d, 0o000)
    try:
        assert recap.run_summaries(d, limit=5) == ([], 0)
    finally:
        # 不恢复权限的话 tmp_path 清理会失败，把失败算到别的测试头上。
        os.chmod(d, 0o755)


@_needs_non_root
def test_runs_root_with_unreadable_parent_returns_empty_not_crash(tmp_path):
    """父目录没有 execute 权限时连 stat 都做不到，is_dir() 自己就抛
    PermissionError [Errno 13] —— 又一个 is_dir() 不吞、会原样冒出来的 errno。
    """
    parent = tmp_path / "parent"
    parent.mkdir()
    d = parent / "runs"
    d.mkdir()
    os.chmod(parent, 0o000)
    try:
        assert recap.run_summaries(d, limit=5) == ([], 0)
    finally:
        os.chmod(parent, 0o755)


def test_dir_without_run_json_is_not_counted_as_skipped(tmp_path):
    """runs/ 下的普通目录（没有 run.json）不是缺陷，只是个普通目录 ——
    必须 continue 而不计入 skipped。skipped 是「本该能读却读不了」的信号，
    把正常目录混进去会毁掉它区分「程序坏了」和「正常运作」的全部意义。
    """
    _make_run(tmp_path, "good", "好的", "done_success", 5, 2000)
    (tmp_path / "not_a_run").mkdir()
    (tmp_path / "散落文件").write_text("x", encoding="utf-8")
    lines, skipped = recap.run_summaries(tmp_path, limit=5)
    assert len(lines) == 1 and "好的" in lines[0]
    assert skipped == 0


@_needs_non_root
def test_unreadable_run_dir_does_not_kill_the_good_rows(tmp_path):
    """`f.is_file()` 和上一轮删掉的 `is_dir()` 是同一套源码：只吞
    ENOENT/ENOTDIR/EBADF/ELOOP，EACCES 原样往外抛。它当时写在 try 外面，
    一个 chmod 000 的子目录就能让整个函数炸穿，好的那行也一起没 ——
    跟根目录那层是同一个模式，只是下沉了一层。
    """
    _make_run(tmp_path, "good", "好的", "done_success", 5, 2000)
    bad = tmp_path / "bad"
    bad.mkdir()
    os.chmod(bad, 0o000)
    try:
        lines, skipped = recap.run_summaries(tmp_path, limit=5)
        assert len(lines) == 1 and "好的" in lines[0]
        # 这是一个本该能读却读不了的运行目录，是缺陷信号，必须计数。
        assert skipped == 1
    finally:
        os.chmod(bad, 0o755)


def test_empty_memory_and_no_runs_injects_nothing(store, tmp_path):
    """索引为空就什么都不注入。塞一句「记忆为空」是纯噪声。"""
    msg, _recent, skipped = recap.build_memory_message(store, tmp_path / "runs")
    assert msg is None and skipped == 0


def test_message_marks_provenance_mechanically(store, tmp_path):
    """✓/✗ 由 source_outcome 生成，是机械标记。

    记忆是模型的断言不是已验证的事实，出处的成败是判断它的唯一机械依据 ——
    设计 v1 想用「成功才落盘」保证质量，但那次震荡运行本身就是 done_success，
    规则拦不住它自己举的例子（spec §0）。
    """
    store.write("good-one", "来自成功", "x", "runs/1", "done_success")
    store.write("bad-one", "来自失败", "y", "runs/2", "done_failed")
    store.write("by-hand", "人工写的", "z", "manual", "manual")
    msg, _recent, _skipped = recap.build_memory_message(store, tmp_path / "runs")
    assert "good-one — 来自成功" in msg and "○未验证" in msg
    assert "bad-one — 来自失败" in msg and "✗来自失败运行" in msg
    assert "by-hand" in msg and "✓已验证" in msg


def test_three_tier_marks_and_failed_sources_sort_last(tmp_path):
    """三档：用过且有效 ✓、来自成功运行但还没人用过 ○、出处失败 ✗。

    ✗ 排最后而不是隐藏：完全不进索引，模型就不知道有这条可以 search
    （设计说明 的「不进索引」在 top-K 之后自然靠后，这里按排序实现）。
    """
    from iphone_agent.memory import MemoryStore

    s = MemoryStore(tmp_path / "m")
    s.write("bad", "d", "c", "runs/x", "done_failed")
    s.write("new", "d", "c", "runs/x", "done_success")
    s.write("good", "d", "c", "runs/x", "done_success")
    s.update_usage("good", "success", "runs/y")
    mem, _, _ = recap.build_memory_message(s, tmp_path / "runs")
    lines = [line for line in mem.splitlines() if line.startswith("- ")]
    assert "✓已验证" in lines[0] and "good" in lines[0]
    assert "○未验证" in lines[1] and "new" in lines[1]
    assert "✗来自失败运行" in lines[2] and "bad" in lines[2]


def test_mark_for_is_reusable_and_manual_counts_verified(tmp_path):
    """标记逻辑抽成函数：CLI 的 memory list 要显示同一套档位，
    两处各写一遍就会漂移。"""
    from iphone_agent.memory.store import MemoryEntry

    def _e(**kw):
        base = {"name": "n", "description": "d", "created": "2026-01-01",
                "source": "runs/1", "source_outcome": "done_success"}
        return MemoryEntry(**(base | kw))

    assert recap.mark_for(_e(used_success=1)) == "✓已验证"
    assert recap.mark_for(_e(source_outcome="manual", source="manual")) == "✓已验证"
    assert recap.mark_for(_e()) == "○未验证"
    assert recap.mark_for(_e(source_outcome="done_failed", used_success=9)) == "✗来自失败运行"
    # 未知/空出口名一律当不可信，不去猜测新出口的善意。
    assert recap.mark_for(_e(source_outcome="")) == "✗来自失败运行"


def test_message_says_it_is_data_not_instructions(store, tmp_path):
    """记忆内容源头是任意 App 的 OCR 文字，是不可信输入。"""
    store.write("x", "一条记忆", "body", "runs/1", "done_success")
    msg, _recent, _skipped = recap.build_memory_message(store, tmp_path / "runs")
    assert "数据" in msg and "指令" in msg


def test_message_includes_recent_runs(store, tmp_path):
    runs = tmp_path / "runs"
    _make_run(runs, "a", "某任务", "max_steps", 30, 1000)
    _mem, recent, _skipped = recap.build_memory_message(store, runs)
    assert "某任务" in recent and "max_steps" in recent


def test_broken_memories_are_reported_not_hidden(store, tmp_path):
    store.write("ok", "好的", "x", "runs/1", "done_success")
    (store.root / "broken.md").write_text("不是 frontmatter", encoding="utf-8")
    msg, _recent, _skipped = recap.build_memory_message(store, tmp_path / "runs")
    assert "1 条记忆损坏" in msg


def test_only_unreadable_runs_injects_nothing(store, tmp_path):
    """记忆为空、runs 目录里唯一的东西是读不出来的坏 run.json（runs_skipped=1，
    entries/broken/runs 都是空/0）—— runs_skipped 是给调用方（Task 9 记进
    run.json）的诊断信号，不是给模型看的内容，不该单独触发注入。否则会生成一条
    只有标题、什么内容都没有的空壳消息，恰好是「塞一句『记忆为空』是纯噪声」
    这条注释本身要避免的东西。
    """
    runs = tmp_path / "runs"
    bad = runs / "bad"
    bad.mkdir(parents=True)
    (bad / "run.json").write_text("{ 半个 json", encoding="utf-8")
    msg, _recent, skipped = recap.build_memory_message(store, runs)
    assert msg is None
    # ⚠ 判空返回 None 时 skipped 仍要如实带出去：它正是「摘要代码自己坏了」
    #   与「确实没有历史运行」的唯一区分信号，咽下去这条链就断了。
    assert skipped == 1


def test_memory_index_and_recent_runs_are_separate_messages(store, tmp_path):
    """记忆索引只在写记忆时变、最近运行每次任务都变——拼在一条里会让前缀
    缓存在第一个字就断（设计说明），所以必须是两条各自独立的消息。"""
    store.write("settings-entry", "设置入口", "在主屏第一页", "runs/x", "done_success")
    runs = tmp_path / "runs"
    _make_run(runs, "20260101-000000-aaaa", "t", "done_success", 3, 1000)
    mem, recent, skipped = recap.build_memory_message(store, runs)
    assert mem.startswith(recap.HEADER) and "settings-entry" in mem and "最近的运行" not in mem
    assert recent.startswith(recap.HEADER) and "最近的运行" in recent and "settings-entry" not in recent
    assert skipped == 0


def test_index_is_full_when_small_and_topk_by_relevance_when_large(tmp_path, monkeypatch):
    """条数 ≤ MEMORY_INJECT_TOPK 时索引照旧全量、跟 task 无关；超过阈值才按
    任务相关度截断，并附一句「只列 K 条」的说明（计划 B Task 3）。"""
    from iphone_agent import config

    monkeypatch.setattr(config, "MEMORY_INJECT_TOPK", 3)
    s = MemoryStore(tmp_path / "m")
    for i in range(3):
        s.write(f"m{i}", f"条目{i}", "c", "runs/x", "done_success")
    mem, _, _ = recap.build_memory_message(s, tmp_path / "runs", task="随便")
    assert mem.count("\n- ") == 3 and "只列" not in mem
    s.write("wechat-groups", "微信置顶群在首页顶部", "c", "runs/x", "done_success")
    s.write("pay-code", "支付宝付款码在首页右上", "c", "runs/x", "done_success")
    mem, _, _ = recap.build_memory_message(s, tmp_path / "runs", task="看微信群消息")
    assert mem.count("\n- ") == 3 and "wechat-groups" in mem and "共 5 条记忆，这里只列 3 条" in mem


def test_only_runs_no_memory(store, tmp_path):
    runs = tmp_path / "runs"
    _make_run(runs, "20260101-000000-aaaa", "t", "done_success", 3, 1000)
    mem, recent, _skipped = recap.build_memory_message(store, runs)
    assert mem is None and recent is not None


def test_run_summaries_prefers_similar_tasks_then_recent(tmp_path):
    """query 给了：按相关度选，不足用最近补齐（B7 —— 「最近」≠「相关」）。"""
    runs = tmp_path / "runs"
    _make_run(runs, "20260101-000001-a", "打开设置看版本", "done_success", 3, 1)
    _make_run(runs, "20260101-000002-b", "微信群消息摘要", "no_progress", 9, 2)
    _make_run(runs, "20260101-000003-c", "计算器算乘法", "done_success", 5, 3)
    lines, _ = recap.run_summaries(runs, 2, query="看看微信群")
    assert "微信群" in lines[0] and "计算器" in lines[1]  # 相关的在前，不足用最近补
    lines2, _ = recap.run_summaries(runs, 2)
    assert "计算器" in lines2[0]  # 无 query 仍按时间


def test_run_summaries_scans_only_recent_dirs(tmp_path, monkeypatch):
    """C5：目录名倒序只扫前 RECAP_SCAN_MAX 个，不管全目录有多少个。"""
    from iphone_agent import config

    monkeypatch.setattr(config, "RECAP_SCAN_MAX", 2)
    runs = tmp_path / "runs"
    for i in range(5):
        _make_run(runs, f"2026010{i}-000000-x", f"t{i}", "done_success", 1, i)
    lines, _ = recap.run_summaries(runs, 5)
    assert len(lines) == 2 and "t4" in lines[0]


def test_topk_prefers_trusted_marks_over_pure_relevance(tmp_path, monkeypatch):
    """截断分支里 ✗ 不能靠描述写得像就抢走 ✓/○ 的位置（整分支评审 Important 4）。

    一条来自失败运行的记忆，描述与任务完全一致；另外 20 条 ✓/○ 只是弱相关。
    top-K=3 时，前 3 行不能有那条 ✗ —— 相关度高的不可信条目挤掉可信条目，
    等于让「来自失败运行」这个标记不起作用。
    """
    from iphone_agent import config

    monkeypatch.setattr(config, "MEMORY_INJECT_TOPK", 3)
    s = MemoryStore(tmp_path / "m")
    s.write("bad-hit", "看微信群消息", "c", "runs/x", "done_failed")
    for i in range(20):
        s.write(f"weak{i:02d}", f"微信{i:02d}", "c", "runs/x", "done_success")
    s.update_usage("weak00", "success", "runs/y")
    mem, _, _ = recap.build_memory_message(s, tmp_path / "runs", task="看微信群消息")
    lines = [line for line in mem.splitlines() if line.startswith("- ")]
    assert len(lines) == 3
    assert not any("bad-hit" in line for line in lines), lines
    assert "✓已验证" in lines[0] and "weak00" in lines[0]
