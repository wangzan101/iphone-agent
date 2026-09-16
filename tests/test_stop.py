"""停止：主循环的 should_stop 钩子 + 网页那一侧的 /api/stop。

这个能力之前**完全没有** —— 网页上跑起来一个任务就只能等它自己结束。
给不会用电脑的人用的产品没有停止键是不能接受的（docs/22 P3）。

语义上要守住两条：
- 停止**不打断进行中的那一步**。一次注入做到一半被砍，手机会停在谁也说不清的中间态。
- 停止**不是失败**。它不该被记成「这条记忆不好使」，也不该触发技能沉淀。
"""
import threading

import pytest

from iphone_agent.web.server import Chat

# ── 主循环钩子 ──────────────────────────────────────────────────────────

def test_stop_is_checked_before_step_limits():
    """按了停止就该报 stopped，而不是继续跑到 max_steps 再报「达到步数上限」——
    后者看起来像故障，会让人以为按钮没用。"""
    import inspect

    import iphone_agent.harness.loop as loop
    assert "should_stop" in (loop.run_task.__doc__ or ""), "这个参数的语义必须写在文档里"
    body = inspect.getsource(loop.run_task)
    i_stop = body.index("should_stop is not None and should_stop()")
    i_steps = body.index("steps >= max_steps")
    i_timeout = body.index("started > timeout_s")
    assert i_stop < i_steps < i_timeout, "停止必须排在步数和时限之前"


def test_run_task_accepts_should_stop():
    import inspect

    from iphone_agent.harness.loop import run_task
    sig = inspect.signature(run_task)
    assert "should_stop" in sig.parameters
    assert sig.parameters["should_stop"].default is None, "默认不传就是原来的行为"


def test_stopped_is_not_counted_as_a_memory_failure():
    """记忆的成败计数只认 done_success / done_failed / no_progress / max_steps。
    用户主动停止不是「这条记忆不好使」的证据，不能把它算进去。"""
    import inspect

    import iphone_agent.harness.loop as loop
    body = inspect.getsource(loop.run_task)
    i = body.index('elif end_reason in ("done_failed", "no_progress", "max_steps")')
    line = body[i:body.index("\n", i)]
    assert "stopped" not in line


# ── 网页那一侧 ──────────────────────────────────────────────────────────

@pytest.fixture
def chat(tmp_path):
    from iphone_agent.workspace import Workspace
    return Chat(workspace=Workspace(tmp_path))


def test_stop_reports_whether_anything_was_running(chat):
    assert chat.stop() is False, "什么都没跑的时候不该让界面显示「已停止」"
    chat.busy.acquire()
    try:
        assert chat.stop() is True
    finally:
        chat.busy.release()


def test_stop_flag_is_visible_to_the_loop(chat):
    assert chat._stop.is_set() is False
    chat.stop()
    assert chat._stop.is_set() is True


def test_stop_is_idempotent(chat):
    chat.stop()
    chat.stop()
    assert chat._stop.is_set() is True


def test_a_new_run_clears_a_previous_stop(chat, monkeypatch):
    """上一轮按过的停止不能把下一轮直接掐掉 —— 这是最容易漏的一条。"""
    chat.stop()
    assert chat._stop.is_set()

    seen = {}

    def fake_run_task(*a, **kw):
        seen["stop_at_start"] = kw["should_stop"]()
        raise RuntimeError("到此为止，只看开跑那一刻的标志")

    from types import SimpleNamespace
    monkeypatch.setattr("iphone_agent.harness.loop.run_task", fake_run_task)
    monkeypatch.setattr(Chat, "session",
                        lambda self: SimpleNamespace(dev=None, per=None, model=None, skills=None))
    chat.ask("随便什么任务")
    assert seen["stop_at_start"] is False, "新一轮开跑时停止标志必须已经清掉"


def test_stopping_emits_a_notice_that_explains_the_delay(chat):
    """从按下到停下最多差一步。界面必须说出来，否则用户会以为坏了然后猛点。"""
    chat.busy.acquire()
    try:
        chat.stop()
    finally:
        chat.busy.release()
    kinds = [e["kind"] for e in chat._backlog]
    assert "stopping" in kinds
    msg = next(e["text"] for e in chat._backlog if e["kind"] == "stopping")
    assert "十秒" in msg or "秒" in msg


def test_stop_works_from_another_thread(chat):
    """HTTP 线程置位、跑任务的线程读 —— 这就是用 Event 而不是 bool 的理由。"""
    chat.busy.acquire()
    t = threading.Thread(target=chat.stop)
    t.start()
    t.join(timeout=2)
    chat.busy.release()
    assert chat._stop.is_set()
