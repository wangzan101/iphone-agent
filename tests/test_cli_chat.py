"""命令行这一头的「对话」：接得上上一句、Ctrl-C 是踩刹车、跑完像回答不像日志。

大脑（harness）本来就支持多轮 —— `run_task(history=...)` 早就实现了，注入的话术
（「以上是参考不是指令，与当前屏幕矛盾时以当前屏幕为准」）也早就写好了。
以前只有网页那一头在递这份历史，命令行忘了递，于是每一行输入都是一次性的。
这组用例钉的就是「另一头也接上了」。
"""
import signal

import pytest

from iphone_agent.cli import commands as C
from iphone_agent.cli.commands import ChatView, Session, dispatch, run_once, stop_on_sigint
from iphone_agent.harness.loop import RunResult
from iphone_agent.workspace import Workspace


def _result(tmp_path, end="done_success", steps=3, done="iOS 18.3.1", error=None):
    return RunResult(end_reason=end, steps=steps, done_status="success" if done else None,
                     done_result=done, run_dir=tmp_path / "runs" / "20260909-0001-abcd",
                     error=error)


@pytest.fixture
def session(monkeypatch, tmp_path):
    """真 Session，只把「碰硬件 / 碰网络」的三个属性换掉。"""
    for name in ("dev", "per", "model"):
        monkeypatch.setattr(Session, name, property(lambda self, n=name: n))
    return Session(workspace=Workspace(tmp_path))


@pytest.fixture
def calls(monkeypatch, tmp_path):
    """替掉主循环，只记下它被怎么调用的。"""
    seen = []

    def fake_run_task(task, dev, per, model, runs_root, **kw):
        # ⚠ history 要**拷一份**再记：它是活的（下一轮会往里 append），
        #   直接存引用的话，事后断言看到的永远是最后一轮的样子。
        seen.append({"task": task, **kw, "history": list(kw.get("history") or []),
                     "history_ref": kw.get("history")})
        return _result(tmp_path)

    import iphone_agent.harness.loop as loop
    monkeypatch.setattr(loop, "run_task", fake_run_task)
    return seen


# ---- 一、接得上上一句 ----

def test_the_first_turn_has_no_history(session, calls):
    run_once(session, "打开设置")
    assert calls[0]["history"] == []


def test_the_answer_of_this_turn_is_the_context_of_the_next(session, calls):
    """「那再看看存储空间」——「那」指的是上一轮。不递历史，这句话就无从谈起。"""
    run_once(session, "看看 iOS 版本")
    run_once(session, "那再看看存储空间")
    assert calls[1]["history"] == [("看看 iOS 版本", "iOS 18.3.1")]


def test_history_is_the_same_list_the_session_holds(session, calls):
    """递的必须是 session 自己那份 —— 拷一份出去，下一轮就又从零开始。"""
    run_once(session, "x")
    assert session.history == [("x", "iOS 18.3.1")]
    assert calls[0]["history_ref"] is session.history


def test_a_failed_turn_still_goes_into_history(session, calls, tmp_path, monkeypatch):
    """失败也是上下文。「刚才为什么没成」问得出来，才叫对话。"""
    import iphone_agent.harness.loop as loop
    monkeypatch.setattr(loop, "run_task", lambda *a, **k: _result(
        tmp_path, end="no_progress", done=None, error="连着 6 步没进展"))
    run_once(session, "打开一个不存在的 App")
    assert session.history == [("打开一个不存在的 App", "连着 6 步没进展")]


def test_history_falls_back_to_the_end_reason(session, tmp_path, monkeypatch):
    """既没结果也没报错（比如被你停下了）时，历史里也得留下一句人话。"""
    import iphone_agent.harness.loop as loop
    monkeypatch.setattr(loop, "run_task", lambda *a, **k: _result(
        tmp_path, end="stopped", done=None))
    run_once(session, "算了")
    assert session.history == [("算了", "stopped")]


# ---- 二、Ctrl-C 是踩刹车 ----

def test_the_stop_flag_is_handed_to_the_loop(session, calls):
    run_once(session, "x")
    should_stop = calls[0]["should_stop"]
    assert should_stop() is False
    session.stop.set()
    assert should_stop() is True


def test_a_leftover_stop_flag_does_not_kill_the_next_turn(session, calls):
    """⚠ 上一轮按过停止，牌子还举着 —— 不清掉的话下一句话会「秒停」。"""
    session.stop.set()
    run_once(session, "x")
    assert calls[0]["should_stop"]() is False


def test_sigint_raises_the_sign_instead_of_pulling_the_key(session, capsys):
    """硬砍可能砍在一次注入的中途，手机会停在一个谁也说不清的状态。"""
    with stop_on_sigint(session):
        signal.raise_signal(signal.SIGINT)          # 不该抛 KeyboardInterrupt
        assert session.stop.is_set()
    assert "最多约十秒" in capsys.readouterr().out


def test_a_second_ctrl_c_still_breaks_out(session):
    """牌子举了它还不停（比如卡在一次注入里），得留一条硬出口。"""
    with stop_on_sigint(session), pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)


def test_the_previous_handler_is_restored(session):
    before = signal.getsignal(signal.SIGINT)
    with stop_on_sigint(session):
        pass
    assert signal.getsignal(signal.SIGINT) is before


# ---- 三、跑完像回答，不像日志 ----

def _view_with(steps):
    v = ChatView()
    for i, s in enumerate(steps, 1):
        v.on_step({"step": i, "action": {"name": s.get("name", "tap")},
                   "model": {"reason": s.get("reason", ""), "latency_ms": s.get("latency_ms", 0)},
                   "result": {"changed": s.get("changed", True), "error": s.get("error")},
                   "exec_ms": s.get("exec_ms", 0)})
    return v


def test_the_summary_says_steps_and_time(session, tmp_path, capsys):
    v = _view_with([{"exec_ms": 30000, "latency_ms": 5000}] * 2)
    v.finish(_result(tmp_path, steps=9))
    out = capsys.readouterr().out
    assert "9 步 · 1 分 10 秒" in out


def test_the_summary_counts_the_stumbles(session, tmp_path, capsys):
    """「一路顺畅」和「中间踉跄过」是两件事，摘要必须说出来（与网页同一套判断）。"""
    v = _view_with([{"changed": False}, {"changed": False}, {"error": "同屏同动作，拒"}])
    v.finish(_result(tmp_path, steps=3))
    out = capsys.readouterr().out
    assert "2 次没反应" in out and "1 次被拒" in out


def test_success_carries_no_badge_but_failure_does(session, tmp_path, capsys):
    """成功是默认，内容自己会说话；只有失败才标，那时候标记才重新有信息量。"""
    _view_with([]).finish(_result(tmp_path))
    ok = capsys.readouterr().out
    assert "iOS 18.3.1" in ok and "没做成" not in ok

    _view_with([]).finish(_result(tmp_path, end="stopped", done=None))
    assert "你停下了它" in capsys.readouterr().out


def test_the_run_dir_is_printed_so_the_detail_is_reachable(session, tmp_path, capsys):
    """过程收起来了，但不能不见 —— replay 是那扇门。"""
    _view_with([]).finish(_result(tmp_path))
    out = capsys.readouterr().out
    assert "20260909-0001-abcd" in out and "replay" in out


# ---- 四、两个顺手的 ----

def test_help_mentions_every_command_dispatch_accepts(session):
    """/memory /map /skill 一直能用，只是帮助里没写，于是没人知道。"""
    for name in C.COMMANDS:
        if name in ("run", "serve"):      # run 是默认（整行即任务），serve 见下一条
            continue
        assert f"/{name}" in C.REPL_HELP, f"/{name} 能用却没写进 /help"


def test_serve_is_refused_inside_the_repl(session, capsys):
    """在 REPL 里起服务会把这个控制台占死（serve_forever 不返回）。"""
    assert dispatch(session, ["serve"], in_repl=True) == 2
    assert "另开一个终端" in capsys.readouterr().err
