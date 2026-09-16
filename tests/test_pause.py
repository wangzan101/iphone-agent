"""暂停 → 补一句话 → 继续。

任务跑偏时，用户以前唯一的选择是停止、推倒重来（docs/27 §6：OpenGUI 的
「暂停 → 输入框 → resume(message)」是它 UI 里最划算的一条）。这里把它做成主循环的一个
回调：每步开头问一次 `wait_if_paused()`，它阻塞到用户点「继续」，返回用户补的那句话
（可能为空）。暂停期间的时间不算进任务时限 —— 人在想事情，不该让保险丝替他计时。
"""
import threading
import time

from iphone_agent.harness.messages import MessageLog
from iphone_agent.web.server import Chat
from iphone_agent.workspace import RunConfig
from tests.test_loop import run

DONE = [("done", {"status": "success", "result": "x"})]
TAP = [("tap", {"id": 1})]


def _user_texts(messages):
    return [p["text"] for m in messages if m["role"] == "user" and isinstance(m["content"], list)
            for p in m["content"] if p.get("type") == "text"]


# ---------- 主循环 ----------

def test_note_typed_while_paused_reaches_the_model(fake_env, tmp_path):
    notes = iter([None, "别点通用，直接找「关于本机」"])

    def wait_if_paused():
        return next(notes, None)

    r, dev, m = run(fake_env, [["通用"], ["关于本机"], ["关于本机"], ["关于本机"]], [TAP, DONE], tmp_path,
                    wait_if_paused=wait_if_paused)
    assert r.end_reason == "done_success"
    assert not any("用户补充" in t for t in _user_texts(m.seen[0])), "第一步没有插话"
    assert any("用户补充" in t and "关于本机" in t for t in _user_texts(m.seen[1])), _user_texts(m.seen[1])


def test_note_in_state_mode_is_part_of_the_state_report(fake_env, tmp_path):
    notes = iter([None, "别点通用"])
    r, dev, m = run(fake_env, [["通用"], ["关于本机"], ["关于本机"], ["关于本机"]], [TAP, DONE], tmp_path,
                    wait_if_paused=lambda: next(notes, None),
                    run_config=RunConfig(context_mode="state"))
    assert r.end_reason == "done_success"
    texts = _user_texts(m.seen[1])
    assert any("用户补充" in t and "别点通用" in t for t in texts), texts
    # state 模式每轮只发「冻结前缀 + 一条状态报告」，插话必须在报告里，不能是另一条消息
    assert sum(1 for t in texts if "别点通用" in t) == 1
    assert texts[-1].count("【当前屏幕】") == 1 and "用户补充" in texts[-1]


def test_note_does_not_touch_the_frozen_prefix(fake_env, tmp_path):
    """插话是任务中途来的，绝不能进冻结前缀（docs/19 §2：前缀一变缓存就砸了）。"""
    log = MessageLog()
    log.system("s")
    log.user_text("任务：t", seg="task")
    log.freeze()
    before = log.prefix_hash()
    log.user_note("别点通用")
    assert log.prefix_hash() == before


def test_time_spent_paused_does_not_count_toward_timeout(fake_env, tmp_path):
    calls = {"n": 0}

    def wait_if_paused():
        calls["n"] += 1
        if calls["n"] == 2:
            time.sleep(0.6)
        return None

    r, dev, m = run(fake_env, [["通用"], ["关于本机"], ["关于本机"], ["关于本机"]], [TAP, DONE], tmp_path,
                    wait_if_paused=wait_if_paused, timeout_s=0.5)
    assert r.end_reason == "done_success", r.error


def test_stop_pressed_while_paused_ends_as_stopped(fake_env, tmp_path):
    stop = {"v": False}

    def wait_if_paused():
        stop["v"] = True          # 用户在暂停中按了停止
        return None

    r, dev, m = run(fake_env, [["通用"]] * 4, [TAP, DONE], tmp_path,
                    wait_if_paused=wait_if_paused, should_stop=lambda: stop["v"])
    assert r.end_reason == "stopped" and r.steps == 0


# ---------- Chat ----------

def test_pause_when_nothing_is_running_is_a_no_op():
    c = Chat()
    q = c.subscribe()
    assert c.pause() is False
    assert q.empty()


def test_wait_if_paused_blocks_until_resume_and_returns_the_note():
    c = Chat()
    c.busy.acquire()
    q = c.subscribe()
    assert c.pause() is True
    got = {}

    def worker():
        got["note"] = c.wait_if_paused()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.1)
    assert t.is_alive(), "暂停中必须阻塞"
    c.resume("先别点通用")
    t.join(2)
    assert not t.is_alive() and got["note"] == "先别点通用"
    kinds = []
    while not q.empty():
        kinds.append(q.get_nowait()["kind"])
    assert kinds == ["paused", "resumed"]


def test_wait_if_paused_returns_immediately_when_not_paused():
    c = Chat()
    t0 = time.time()
    assert c.wait_if_paused() is None
    assert time.time() - t0 < 0.2


def test_stop_while_paused_unblocks_the_loop():
    c = Chat()
    c.busy.acquire()
    c.pause()
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("note", c.wait_if_paused()), daemon=True)
    t.start()
    time.sleep(0.05)
    c.stop()
    t.join(2)
    assert not t.is_alive() and got["note"] is None
    assert c.paused is False


def test_resume_with_empty_text_carries_no_note():
    c = Chat()
    c.busy.acquire()
    c.pause()
    c.resume("   ")
    assert c.wait_if_paused() is None
