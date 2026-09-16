"""记忆相关的工具、校验与信任边界。"""
from iphone_agent.harness.prompt import PROMPT_VERSION, system_prompt


def test_system_prompt_states_the_trust_boundary():
    """这是唯一一条真正建立**边界**而非限制体积的防线（spec §2.2 第 3 条）。

    路径约束防的是文件越权，字符限额防的是体积膨胀，都不阻止几十个字的
    恶意指令 —— 攻击不必写「忽略之前的指令」，伪装成经验就行。
    """
    assert "历史记录" in system_prompt()
    assert "不能授权" in system_prompt()
    assert "以当前屏幕为准" in system_prompt()


def test_prompt_version_bumped():
    """提示词变了就要改版本号，否则 run.json 的 prompt_hash 对不上历史运行。"""
    assert PROMPT_VERSION == "23"


import pytest

from iphone_agent.harness.actions import Action, ValidationError, validate_action


def _act(name, args):
    return Action(name, args, "测试", None, "call-1")


def test_recall_requires_a_name():
    with pytest.raises(ValidationError):
        validate_action(_act("recall", {}), None)
    with pytest.raises(ValidationError):
        validate_action(_act("recall", {"name": 123}), None)


def test_recall_accepts_a_string_name():
    a = validate_action(_act("recall", {"name": "settings-entry"}), None)
    assert a.args["name"] == "settings-entry"


def test_recall_runs_limit_is_clamped_not_rejected():
    """夹到上限而不是报错：模型给个 100 是想多看点，不是犯错。"""
    a = validate_action(_act("recall_runs", {"limit": 100}), None)
    assert a.args["limit"] == 20
    b = validate_action(_act("recall_runs", {"limit": 0}), None)
    assert b.args["limit"] == 1
    c = validate_action(_act("recall_runs", {}), None)
    assert c.args["limit"] == 5


def test_search_memory_validation():
    with pytest.raises(ValidationError):
        validate_action(_act("search_memory", {"query": ""}), None)
    with pytest.raises(ValidationError):
        validate_action(_act("search_memory", {"query": "x" * 101}), None)


def test_search_memory_tool_returns_ranked_hits(tmp_path, fake_env):
    import json as _json

    from tests.test_loop import isolated_store, run

    store = isolated_store(tmp_path)
    store.write("wechat-groups", "微信置顶群在首页顶部", "c", "runs/x", "done_success")
    store.write("settings-entry", "设置入口", "c", "runs/x", "done_success")
    r, dev, m = run(fake_env, [["a"]] * 4,
                    [[("search_memory", {"query": "微信群"})],
                     [("done", {"status": "success", "result": "x"})]],
                    tmp_path, store=store)
    tool = [x for x in m.seen[1] if x["role"] == "tool"][-1]
    hits = _json.loads(tool["content"])["hits"]
    assert hits[0]["name"] == "wechat-groups" and "mark" in hits[0]


def test_recall_returns_the_full_body_through_the_message_window(tmp_path):
    """⚠ 必须测模型**实际收到**的消息，不能只测 store.read()。

    messages.py 的滑窗会截断工具结果。它现在分两档：最新一条 20000 字符、更老的 500
    （见 config.TOOL_RESULT_MAX_CHARS）。这条用例压的是**刚返回时不被截断**这一档 ——
    body 取正文上限（config.MEMORY_BODY_MAX）本身，不管这个上限具体是多少，
    只要它没超过 20000 那一档，就应该整段原样送达。
    """
    from iphone_agent import config
    from iphone_agent.harness.executor import Executor
    from iphone_agent.harness.messages import MessageLog
    from iphone_agent.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory")
    body = "字" * config.MEMORY_BODY_MAX
    store.write("long-one", "很长的一条", body, "runs/1", "done_success")

    ex = Executor(None, None, store=store, runs_root=tmp_path / "runs")
    res, _ = ex.run(validate_action(_act("recall", {"name": "long-one"}), None), None)
    assert res.ok

    msgs = MessageLog()
    msgs.tool_result("call-1", res.to_json())
    delivered = msgs.windowed()[-1]["content"]
    assert body in delivered, "正文过一遍滑窗后被截断了"


def test_recall_missing_says_so_and_does_not_guess(tmp_path):
    from iphone_agent.harness.executor import Executor
    from iphone_agent.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory")
    store.write("settings-entry", "d", "x", "runs/1", "done_success")
    ex = Executor(None, None, store=store, runs_root=tmp_path / "runs")
    res, _ = ex.run(validate_action(_act("recall", {"name": "setting-entry"}), None), None)
    assert res.ok is False and res.error == "memory_not_found"


def test_recall_runs_returns_summaries(tmp_path):
    import json

    from iphone_agent.harness.executor import Executor
    from iphone_agent.memory import MemoryStore

    runs = tmp_path / "runs"
    d = runs / "a"
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps(
        {"task": "某任务", "end_reason": "done_success", "steps": 6, "started_at": 1000},
        ensure_ascii=False), encoding="utf-8")

    ex = Executor(None, None, store=MemoryStore(tmp_path / "memory"), runs_root=runs)
    res, _ = ex.run(validate_action(_act("recall_runs", {"limit": 5}), None), None)
    assert res.ok and "某任务" in res.to_json()


def test_done_without_remember_defaults_to_empty():
    a = validate_action(_act("done", {"status": "success", "result": "ok"}), None)
    assert a.args["remember"] == []


def test_done_accepts_up_to_three_memories():
    items = [{"name": f"m{i}", "description": "d", "content": "c"} for i in range(3)]
    a = validate_action(
        _act("done", {"status": "success", "result": "ok", "remember": items}), None)
    assert len(a.args["remember"]) == 3


def test_done_rejects_more_than_three():
    items = [{"name": f"m{i}", "description": "d", "content": "c"} for i in range(4)]
    with pytest.raises(ValidationError) as e:
        validate_action(
            _act("done", {"status": "success", "result": "ok", "remember": items}), None)
    assert "3" in e.value.message


def test_done_rejects_malformed_memory_items():
    for bad in [
        "不是列表",
        [{"name": "m"}],                               # 缺字段
        [{"name": "m", "description": "d"}],
        [{"name": 1, "description": "d", "content": "c"}],
        [["name", "m"]],
    ]:
        with pytest.raises(ValidationError):
            validate_action(
                _act("done", {"status": "success", "result": "ok", "remember": bad}), None)


def test_done_failed_may_also_remember():
    """失败的经验也是经验。v2 的立场是把出处成败摆明面上让人权衡，
    不是替它筛掉（spec §4.2）。"""
    items = [{"name": "m0", "description": "d", "content": "c"}]
    a = validate_action(
        _act("done", {"status": "failed", "result": "没找到", "remember": items}), None)
    assert len(a.args["remember"]) == 1


def test_done_accepts_kind_and_used_memories():
    """kind 缺省是 knowledge，used_memories 缺省是空列表 —— 老模型不给这两个字段
    也要能照常结束任务；kind 只认白名单里的两个值，别的写法当成校验失败。"""
    a = validate_action(_act("done", {"status": "success", "result": "r",
                                      "remember": [{"name": "a", "description": "d",
                                                    "content": "c", "kind": "playbook"}],
                                      "used_memories": ["a", "b"]}), None)
    assert a.args["remember"][0]["kind"] == "playbook" and a.args["used_memories"] == ["a", "b"]
    a2 = validate_action(_act("done", {"status": "success", "result": "r",
                                       "remember": [{"name": "a", "description": "d",
                                                     "content": "c"}]}), None)
    assert a2.args["remember"][0]["kind"] == "knowledge" and a2.args["used_memories"] == []
    with pytest.raises(ValidationError):
        validate_action(_act("done", {"status": "success", "result": "r",
                                      "remember": [{"name": "a", "description": "d",
                                                    "content": "c", "kind": "recipe"}]}), None)


def test_done_used_memories_must_be_short_list_of_nonempty_strings():
    with pytest.raises(ValidationError):
        validate_action(_act("done", {"status": "success", "result": "r",
                                      "used_memories": "a"}), None)
    with pytest.raises(ValidationError):
        validate_action(_act("done", {"status": "success", "result": "r",
                                      "used_memories": ["a", ""]}), None)
    with pytest.raises(ValidationError):
        validate_action(_act("done", {"status": "success", "result": "r",
                                      "used_memories": ["a"] * 11}), None)


# ---- Task 9：接线（注入、落盘、留档、查询次数上限） ----

def test_runlog_records_injected_text_and_results(tmp_path):
    """注入原文必须留档。记忆后来被覆盖后，就再也还原不出当时模型看见了什么，
    验收无从复盘（spec §5.4）。"""
    import json

    from iphone_agent.harness.runlog import RunLog

    log = RunLog(tmp_path, task="t", model="m", config_snapshot={}, prompt_hash="h")
    log.set_memory(injected="【历史记录】\n- a — b   ✓来自成功运行",
                   recalled=["a"],
                   written=[{"name": "a", "ok": True}])
    log.finish("done_success", 3, "success", "r", "v", {})
    data = json.loads((log.dir / "run.json").read_text(encoding="utf-8"))
    assert "【历史记录】" in data["memory"]["injected"]
    assert data["memory"]["recalled"] == ["a"]
    assert data["memory"]["written"] == [{"name": "a", "ok": True}]


def test_set_memory_after_finish_keeps_what_finish_wrote(tmp_path):
    """两者都会重写 run.json。set_memory 排在 finish 之后（写记忆要等设备释放），
    它不能把 end_reason/done 抹掉。"""
    import json

    from iphone_agent.harness.runlog import RunLog

    log = RunLog(tmp_path, task="t", model="m", config_snapshot={}, prompt_hash="h")
    log.finish("done_success", 3, "success", "r", "v", {"model_calls": 2})
    log.set_memory(injected=None, recalled=[], written=[{"name": "a", "ok": True}])
    data = json.loads((log.dir / "run.json").read_text(encoding="utf-8"))
    assert data["end_reason"] == "done_success" and data["done"]["result"] == "r"
    assert data["memory"]["written"] == [{"name": "a", "ok": True}]


def test_partial_write_keeps_the_earlier_ones(tmp_path):
    """3 条里第 2 条失败，第 1 条要留着 —— 记忆之间没有事务关系（spec §4.2）。"""
    from iphone_agent.harness.loop import commit_memories
    from iphone_agent.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory")
    items = [
        {"name": "first", "description": "第一条", "content": "a"},
        {"name": "BAD NAME", "description": "第二条", "content": "b"},
        {"name": "third", "description": "第三条", "content": "c"},
    ]
    written = commit_memories(store, items, source="runs/x", outcome="done_success",
                              audit_clean=True)
    assert store.read("first") == "a"
    assert store.read("third") == "c"
    assert [w["ok"] for w in written] == [True, False, True]
    assert "invalid_name" in written[1]["error"]


def test_playbook_downgraded_when_run_not_clean(tmp_path):
    """playbook 是「照着做」的操作步骤，比陈述性知识危险得多：只有这次运行既报成功
    又通过事后审计，才允许按 playbook 落盘。不干净就降级成 knowledge 并在 written
    里留痕 —— 不是丢掉这条记忆，而是把它降到「参考」这一档。"""
    from iphone_agent.harness.loop import commit_memories
    from iphone_agent.memory import MemoryStore

    s = MemoryStore(tmp_path / "m")
    w = commit_memories(s, [{"name": "p", "description": "d", "content": "c",
                             "kind": "playbook"}],
                        source="runs/x", outcome="done_success", audit_clean=False)
    assert w[0]["ok"] and w[0]["downgraded"] == "playbook→knowledge"
    assert s.index()[0][0].kind == "knowledge"

    w2 = commit_memories(s, [{"name": "q", "description": "d", "content": "c",
                              "kind": "playbook"}],
                         source="runs/x", outcome="done_success", audit_clean=True)
    assert "downgraded" not in w2[0] and s.read("q") is not None
    assert [e.kind for e in s.index()[0] if e.name == "q"] == ["playbook"]

    # 失败运行里的 playbook 一样要降级：audit_clean 为真也救不了它。
    w3 = commit_memories(s, [{"name": "r", "description": "d", "content": "c",
                              "kind": "playbook"}],
                         source="runs/x", outcome="done_failed", audit_clean=True)
    assert w3[0]["downgraded"] == "playbook→knowledge"


def test_commit_respects_the_cap(tmp_path, monkeypatch):
    """测规则（满了就拒），不钉死具体的条数上限 —— 那是 config.MEMORY_MAX_ITEMS
    的事，它已经从「设计假设」改成了「保险丝」，具体数值不该被测试焊死。"""
    from iphone_agent import config
    from iphone_agent.harness.loop import commit_memories
    from iphone_agent.memory import MemoryStore

    monkeypatch.setattr(config, "MEMORY_MAX_ITEMS", 3)
    store = MemoryStore(tmp_path / "memory")
    for i in range(3):
        store.write(f"m{i:02d}", "d", "x", "runs/1", "done_success")
    written = commit_memories(store, [{"name": "one-more", "description": "d",
                                       "content": "c"}],
                              source="runs/x", outcome="done_success", audit_clean=True)
    assert written[0]["ok"] is False and "3" in written[0]["error"]
    assert store.read("one-more") is None


def test_commit_of_an_existing_name_is_allowed_at_the_cap(tmp_path, monkeypatch):
    """满额时覆盖已有的一条不该被拒 —— 总数并没有增加。"""
    from iphone_agent import config
    from iphone_agent.harness.loop import commit_memories
    from iphone_agent.memory import MemoryStore

    monkeypatch.setattr(config, "MEMORY_MAX_ITEMS", 3)
    store = MemoryStore(tmp_path / "memory")
    for i in range(3):
        store.write(f"m{i:02d}", "d", "旧", "runs/1", "done_success")
    written = commit_memories(store, [{"name": "m00", "description": "d",
                                       "content": "新"}],
                              source="runs/x", outcome="done_success", audit_clean=True)
    assert written[0]["ok"] is True
    assert store.read("m00") == "新"


# ---- Task 9：主循环上的行为（都走 run_task，不测内部函数） ----

from tests.test_loop import INITIAL_SETTLE_FRAMES, ScriptedModel


def _run(fake_env, specs, script, tmp_path, store=None, **kw):
    """跟 test_loop.run 同构，只是多了一个隔离的 store 和 runs 根目录。"""
    from iphone_agent.harness.loop import run_task
    from iphone_agent.memory import MemoryStore

    dev, per, _frames = fake_env([specs[0]] * INITIAL_SETTLE_FRAMES + list(specs))
    model = ScriptedModel(script)
    runs = tmp_path / "runs"
    store = store if store is not None else MemoryStore(tmp_path / "memory")
    r = run_task("t", dev, per, model, runs, store=store, **kw)
    return r, dev, model, store


def _run_json(r):
    import json
    return json.loads((r.run_dir / "run.json").read_text(encoding="utf-8"))


def test_memory_is_injected_as_a_user_message_before_the_task(fake_env, tmp_path):
    """必须是用户消息、且排在任务描述之前。

    记忆内容源头是任意 App 屏幕上的 OCR 文字，是不可信输入；进系统提示等于
    给它系统级权威（spec §5.2）。
    """
    from iphone_agent.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory")
    store.write("settings-entry", "设置在第二屏", "往左划一页", "runs/1", "done_success")

    r, _dev, m, _s = _run(fake_env, [["a"], ["a"]],
                          [[("done", {"status": "success", "result": "x"})]],
                          tmp_path, store=store)
    first = m.seen[0]
    texts = ["".join(p.get("text", "") for p in msg["content"])
             if isinstance(msg["content"], list) else msg["content"] for msg in first]
    roles = [msg["role"] for msg in first]
    inj = next(i for i, t in enumerate(texts) if "settings-entry" in t)
    task = next(i for i, t in enumerate(texts) if "任务：t" in t)
    assert roles[inj] == "user", "记忆被注入成了系统提示"
    assert inj < task
    assert "settings-entry" not in texts[roles.index("system")]
    assert "settings-entry" in _run_json(r)["memory"]["injected"]


def test_empty_memory_injects_nothing(fake_env, tmp_path):
    """一句「记忆为空」是纯噪声，不该占一条消息。"""
    r, _dev, m, _s = _run(fake_env, [["a"], ["a"]],
                          [[("done", {"status": "success", "result": "x"})]], tmp_path)
    assert _run_json(r)["memory"]["injected"] is None
    assert not any("历史记录" in "".join(p.get("text", "") for p in msg["content"])
                   for msg in m.seen[0] if isinstance(msg["content"], list))


def test_done_remember_lands_on_disk_with_the_run_dir_as_source(fake_env, tmp_path):
    """source 必须是运行目录这种由代码生成的值 —— store.validate() 不校验 source，
    frontmatter 的解析靠「字段值不含换行」这条不变式撑着。

    source 写的是相对 workspace root 的路径，不是 log.dir 的绝对路径 —— 绝对路径
    会把用户主目录写进每条记忆的 frontmatter（log.dir 现在是绝对路径，见 loop.py）。
    """
    items = [{"name": "settings-entry", "description": "设置在第二屏", "content": "往左划一页"}]
    r, _dev, _m, store = _run(
        fake_env, [["a"], ["a"]],
        [[("done", {"status": "success", "result": "x", "remember": items})]], tmp_path)
    assert store.read("settings-entry") == "往左划一页"
    text = (store.root / "settings-entry.md").read_text(encoding="utf-8")
    expected_source = str(r.run_dir.relative_to(tmp_path))
    assert f"source: {expected_source}" in text
    assert not expected_source.startswith("/")
    assert "source_outcome: done_success" in text
    assert _run_json(r)["memory"]["written"] == [{"name": "settings-entry", "ok": True}]


def test_memory_is_written_only_after_the_device_is_released(fake_env, tmp_path):
    """设备释放优先：写记忆再慢再出错，都不能让按住的键留在设备上。"""
    from iphone_agent.memory import MemoryStore

    seen = {}

    class SpyStore(MemoryStore):
        def write(self, *a, **kw):
            seen["released"] = self._dev.released
            return super().write(*a, **kw)

    store = SpyStore(tmp_path / "memory")
    items = [{"name": "m0", "description": "d", "content": "c"}]

    from iphone_agent.harness.loop import run_task
    dev, per, _f = fake_env([["a"]] * (INITIAL_SETTLE_FRAMES + 2))
    store._dev = dev
    model = ScriptedModel([[("done", {"status": "success", "result": "x", "remember": items})]])
    run_task("t", dev, per, model, tmp_path / "runs", store=store)
    assert seen["released"] is True


def test_a_failing_memory_write_does_not_change_the_end_reason(fake_env, tmp_path):
    """写记忆炸了只记日志。落进循环那个兜底 except 会被误标成 device_error。"""
    from iphone_agent.memory import MemoryStore

    class BoomStore(MemoryStore):
        def write(self, *a, **kw):
            raise RuntimeError("磁盘满了")

    items = [{"name": "m0", "description": "d", "content": "c"}]
    r, dev, _m, _s = _run(
        fake_env, [["a"], ["a"]],
        [[("done", {"status": "success", "result": "x", "remember": items})]],
        tmp_path, store=BoomStore(tmp_path / "memory"))
    assert r.end_reason == "done_success" and r.error is None
    assert dev.released
    written = _run_json(r)["memory"]["written"]
    assert written and written[0]["ok"] is False and "磁盘满了" in written[0]["error"]


def test_a_failing_injection_does_not_kill_the_run(fake_env, tmp_path):
    """读记忆同理：注入拼不出来，任务照跑，设备照样释放。"""
    from iphone_agent.harness.runlog import RunLog
    from iphone_agent.memory import MemoryStore

    class BoomStore(MemoryStore):
        def index(self):
            raise RuntimeError("记忆目录读不了")

    r, dev, _m, _s = _run(fake_env, [["a"], ["a"]],
                          [[("done", {"status": "success", "result": "x"})]],
                          tmp_path, store=BoomStore(tmp_path / "memory"))
    assert r.end_reason == "done_success" and dev.released
    assert _run_json(r)["memory"]["injected"] is None
    assert any(s.get("error_phase") == "memory_inject" for s in RunLog.read_steps(r.run_dir))


def test_recall_is_capped_at_three_per_run(fake_env, tmp_path):
    """查询也要有预算，否则模型能把一整轮步数耗在翻记忆上。
    超出走既有的连续拒绝熔断，不另发明机制。"""
    from iphone_agent.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory")
    for i in range(4):
        store.write(f"m{i}", "d", f"正文{i}", "runs/1", "done_success")

    script = [[("recall", {"name": f"m{i}"})] for i in range(3)]
    script += [[("recall_runs", {"limit": 5})],
               [("recall", {"name": "m3"})],
               [("done", {"status": "success", "result": "x"})]]
    r, _dev, m, _s = _run(fake_env, [["a"]] * 20, script, tmp_path, store=store)

    tool_msgs = [x["content"] for x in m.seen[-1] if x["role"] == "tool"]
    assert "recall_budget_exhausted" in tool_msgs[-1]
    # recall_runs 有自己的额度，不该被 recall 用掉的三次连累
    assert "runs" in tool_msgs[-2]
    assert r.end_reason == "done_success"
    assert _run_json(r)["memory"]["recalled"] == ["m0", "m1", "m2"]
    assert store.read("m3") == "正文3"   # 只是没让模型读到，记忆本身还在


def test_recall_budget_exhaustion_feeds_the_rejection_breaker(fake_env, tmp_path):
    """连续拒绝到上限就停 —— 用的是既有熔断，不是新机制。"""
    from iphone_agent.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory")
    for i in range(10):
        store.write(f"m{i}", "d", "c", "runs/1", "done_success")
    script = [[("recall", {"name": f"m{i}"})] for i in range(10)]
    r, _dev, _m, _s = _run(fake_env, [["a"]] * 30, script, tmp_path, store=store)
    assert r.end_reason == "model_error" and "拒绝" in r.error


# ---- Task 9 补轮：recall 不计熔断；skipped 要有路径到 run.json ----

def test_recall_does_not_count_toward_no_progress():
    """查记忆不改变画面 —— 它本来就不该改，这正是 observe/wait 在 NON_COUNTING 里的理由。

    实测过的缺陷：RECALL_PER_RUN 和 NO_PROGRESS_WARN 都是 3，模型规规矩矩地用满
    查询预算，就正好撞上「连续多步无进展」的警告，被劝去换一条路。
    步数预算（steps += 1）和熔断计数是两件事，recall 该消耗前者、不该进后者。
    """
    from iphone_agent.harness.guard import ActionGuard

    g = ActionGuard()
    for i in range(3):
        assert g.record_outcome(_act("recall", {"name": f"m{i}"}), False, None) is None
    assert g.record_outcome(_act("recall_runs", {"limit": 5}), False, None) is None
    assert g.no_progress == 0


def test_recall_of_the_same_name_on_the_same_screen_is_not_rejected_either():
    """进了 NON_COUNTING，同屏去重也不再拦它（guard 两处都判了 NON_COUNTING）。

    同屏读同一条记忆两次确实没意义，但那该由查询预算管，不该由「这个动作做过了
    且画面没变」这条为屏幕操作设计的规则管 —— 它对 recall 永远成立。
    """
    from iphone_agent.harness.guard import ActionGuard

    g = ActionGuard()
    a = _act("recall", {"name": "m0"})
    g.record_executed(a, 123, 400, 800)
    assert g.check_repeat(a, 123, 400, 800) is False


def test_run_json_records_skipped_run_summaries(fake_env, tmp_path):
    """坏掉的 run.json 数必须有一条路径到达 run.json。

    Task 4 给 run_summaries 加 skipped 的理由就是「让调用方区分『确实没有历史』
    和『摘要代码本身坏了』」—— 值只生成不传递，那条链是断的。
    """
    import json

    runs = tmp_path / "runs"
    good = runs / "good"
    good.mkdir(parents=True)
    (good / "run.json").write_text(json.dumps(
        {"task": "某任务", "end_reason": "done_success", "steps": 6, "started_at": 1000},
        ensure_ascii=False), encoding="utf-8")
    (runs / "bad").mkdir()
    (runs / "bad" / "run.json").write_text("{ 半个 json", encoding="utf-8")

    r, _dev, _m, _s = _run(fake_env, [["a"], ["a"]],
                           [[("done", {"status": "success", "result": "x"})]], tmp_path)
    assert _run_json(r)["memory"]["runs_skipped"] == 1
    assert "某任务" in _run_json(r)["memory"]["injected"]


def test_a_broken_file_does_not_block_writing_new_memories(tmp_path):
    """记忆目录里有一个非 UTF-8 文件，之后所有写入全部失败 —— 实测到的真实故障。

    链条：commit_memories → store.count() → index() → read_text("utf-8") 抛出。
    count() 对每个新名字都会被调用，所以第一条就炸、整批全失败，而且这个状态会
    一直持续到有人手动找到并清掉那个文件。
    """
    from iphone_agent.harness.loop import commit_memories
    from iphone_agent.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory")
    store.write("seed", "占位", "x", "runs/1", "done_success")
    (store.root / "broken.md").write_bytes("---\nname: x\n---\n中文正文".encode("gbk"))

    written = commit_memories(store, [
        {"name": "first", "description": "第一条", "content": "a"},
        {"name": "second", "description": "第二条", "content": "b"},
    ], source="runs/x", outcome="done_success", audit_clean=True)
    assert [w["ok"] for w in written] == [True, True]
    assert store.read("first") == "a" and store.read("second") == "b"


def test_a_broken_file_with_the_same_name_does_not_block_the_write(tmp_path):
    """坏文件的名字恰好是这次要写的名字：仍然两条都要写成功。

    这是 index() 那一轮修完之后剩下的更窄的口子 —— read() 抛出会穿过
    commit_memories（它只接 MemoryRejected/OSError），整批失败。
    """
    from iphone_agent.harness.loop import commit_memories
    from iphone_agent.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory")
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "first.md").write_bytes("---\nname: first\n---\n中文".encode("gbk"))

    written = commit_memories(store, [
        {"name": "first", "description": "第一条", "content": "a"},
        {"name": "second", "description": "第二条", "content": "b"},
    ], source="runs/x", outcome="done_success", audit_clean=True)
    assert [w["ok"] for w in written] == [True, True]
    assert store.read("first") == "a" and store.read("second") == "b"
    assert len(list(store.trash.glob("*.md"))) == 1   # 坏文件保住了，没有硬删


# ---- Task 10：iphone memory list/show/rm/prune ----

def test_memory_cli_list_show_rm(tmp_path, capsys):
    """文件格式的兑现：出问题时你能直接看见它记了什么、手动删掉。"""
    from iphone_agent.cli.commands import Session, cmd_memory
    from iphone_agent.workspace import Workspace

    # 记忆目录跟着工作区走（原来靠 monkeypatch config.MEMORY_DIR）——
    # 显式给一个 tmp 工作区，别去碰开发机上真实的 .iphone/memory。
    session = Session(Workspace(tmp_path))
    from iphone_agent.memory import MemoryStore
    MemoryStore(session.workspace.memory_dir).write(
        "one", "第一条", "正文内容", "runs/1", "done_success")

    assert cmd_memory(session, ["list"]) == 0
    assert "one" in capsys.readouterr().out

    assert cmd_memory(session, ["show", "one"]) == 0
    assert "正文内容" in capsys.readouterr().out

    assert cmd_memory(session, ["rm", "one"]) == 0
    assert cmd_memory(session, ["show", "one"]) == 1      # 已经不在了


def test_memory_cli_unknown_subcommand_returns_2(tmp_path):
    from iphone_agent.cli.commands import Session, cmd_memory
    from iphone_agent.workspace import Workspace

    session = Session(Workspace(tmp_path))
    assert cmd_memory(session, ["nonsense"]) == 2
    assert cmd_memory(session, []) == 2


@pytest.mark.parametrize("escape", ["../private", "/etc/passwd", "..\\private"])
def test_recall_cannot_escape_the_memory_dir(tmp_path, escape):
    """⚠ 走完整的 recall 路径（校验 → Executor.run），不是只测 store.read()。

    设计稿 §2.2 的第 1 条防线是「写入永远经过代码，模型给不了路径」。
    写入路径靠 validate() 兑现了，读取路径以前直接把名字拼进 self.root：
    recall("../private") 于是能读到记忆目录之外的任意 .md，内容进模型上下文，
    而模型有 type 工具能把它打进任意 App（2026-09-07 最终 review 实测复现）。
    """
    from iphone_agent.harness.executor import Executor
    from iphone_agent.memory import MemoryStore

    secret = "私密笔记：银行卡尾号 1234"
    store = MemoryStore(tmp_path / "memory")
    store.write("settings-entry", "正常的一条", "正文", "runs/1", "done_success")
    # 越权目标放在记忆目录**外面**
    (tmp_path / "private.md").write_text(secret, encoding="utf-8")
    # POSIX 上 `..\private.md` 不越权、只是个奇怪的文件名，
    # 按字面名字造一个，这条参数才会在修复前真的红。
    (store.root / "..\\private.md").write_text(secret, encoding="utf-8")

    a = validate_action(_act("recall", {"name": escape}), None)
    ex = Executor(None, None, store=store, runs_root=tmp_path / "runs")
    res, _ = ex.run(a, None)

    assert res.ok is False
    assert res.error == "memory_not_found"
    assert secret not in res.to_json(), "记忆目录之外的内容漏进了模型上下文"


def test_prompt_does_not_lump_tab_bars_with_labelled_icons():
    """⚠ 提示词原来把「底部 tab 栏」和「主屏幕 App 图标」归成一类，都让用 icon_above。
    但 tab 栏是**纯图标没有标签** —— OCR 把图标本身读成乱码单字（'◎'、'曲'、'G'），
    那个乱码就在图标位置上，直接 tap 就中；用 icon_above 会往上推到 tab 栏上方的内容里。

    真机踩过三次：记账本的底部 tab 栏连点 icon_above，每次都打开某笔账单详情页，
    任务因「原地打转」被熔断。是提示词把模型教错了，不是模型的问题。
    """
    from iphone_agent.harness.prompt import system_prompt
    SYSTEM_PROMPT = system_prompt()
    # 只看讲 icon_above 的那一条（· 开头），不是整个「二、你的手」—— 版本 20 起
    # tab 栏在同一节里另有一条，明说「绝对不要用 icon_above」，那不是归回一类。
    icon_above_para = next(p for p in SYSTEM_PROMPT.split("· ") if 'target="icon_above"' in p)
    assert "tab 栏" not in icon_above_para, "tab 栏又被归回 icon_above 那一类了"
    assert "绝对不要**用 icon_above" in SYSTEM_PROMPT or "绝对不要" in SYSTEM_PROMPT
    assert "乱码" in SYSTEM_PROMPT, "没说清 OCR 会把纯图标读成乱码"


def test_prompt_tells_it_to_read_the_whole_list_before_counting():
    """⚠ 第四次假成功的形状：模型在**已经滚下去**的列表里回答「今天有几笔」，
    视野顶端是「昨天」，它就答了「0 笔」—— 而账本里今天确实有一笔 1.50。

    前三次假成功是环境不干净或工具误导；这次是它**自己没确认看到的是列表开头**。
    解药本来就有（collect 一步滚到底并汇总全表），它没用。
    """
    from iphone_agent.harness.prompt import system_prompt
    SYSTEM_PROMPT = system_prompt()
    assert "先用 collect 通读整个列表" in SYSTEM_PROMPT
    assert "不是真正的开头" in SYSTEM_PROMPT, "得说清「你看到的开头可能不是开头」"


def test_prompt_tells_it_camera_and_mic_are_unavailable():
    """不知道这条的话，模型会去试扫一扫、语音输入这些注定失败的入口，白烧步数。
    这是硬边界，正确做法是 done(failed)，不是换着法子试。"""
    from iphone_agent.harness.prompt import system_prompt
    SYSTEM_PROMPT = system_prompt()
    assert "相机和麦克风" in SYSTEM_PROMPT
    for k in ("扫一扫", "语音输入", "done(failed)"):
        assert k in SYSTEM_PROMPT, k


def test_prompt_steers_app_opening_to_open_app():
    """⚠ 退路只在 open_app 里。真机批跑：模型有 10 次自己手搓
    `key spotlight` + `type`，那些用不上退路，全卡死；而同一批里
    open_app 用了 7 次，在打字 0/10 全败的情况下把 9/12 的任务扛下来了。"""
    from iphone_agent.harness.prompt import system_prompt
    SYSTEM_PROMPT = system_prompt()
    assert "一律用 open_app" in SYSTEM_PROMPT
    assert "不要自己按 spotlight 再打字" in SYSTEM_PROMPT
    assert "没有退路" in SYSTEM_PROMPT, "得说清手搓那条路的代价"


def test_done_used_memories_are_stripped_and_length_capped():
    """条目 strip 后存回，长度上限与 MEMORY_NAME_RE 一致（48，评审 Minor 7）。

    以前 strip 只用于判空、存回的还是原串，" k \n" 这种永远匹配不上任何记忆名；
    而没有长度上限意味着模型能把一整段文本塞进 used_memories。
    """
    from iphone_agent import config

    a = validate_action(_act("done", {"status": "success", "result": "r",
                                      "used_memories": ["  k  ", "b\n"]}), None)
    assert a.args["used_memories"] == ["k", "b"]
    with pytest.raises(ValidationError):
        validate_action(_act("done", {"status": "success", "result": "r",
                                      "used_memories": ["x" * 49]}), None)
    assert config.MEMORY_NAME_RE.match("x" * 48)


def test_done_schema_limits_come_from_config():
    """schema 描述里的数字必须来自 config，不能是写死的字面量（评审 Important 3）。"""
    from iphone_agent import config
    from iphone_agent.harness.tools import tool_defs

    done = next(t for t in tool_defs() if t["function"]["name"] == "done")
    props = done["function"]["parameters"]["properties"]
    assert props["used_memories"]["maxItems"] == config.MEMORY_USED_PER_RUN
    assert props["remember"]["maxItems"] == config.MEMORY_WRITE_PER_RUN
    content_desc = props["remember"]["items"]["properties"]["content"]["description"]
    assert str(config.MEMORY_BODY_MAX) in content_desc
