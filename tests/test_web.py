"""网页对话界面的纯逻辑。

「接得上话」现在两头都有了（命令行那份在 test_cli_chat.py）。这个界面剩下的
理由是命令行给不了的那部分：手机画面实时可见、每次运行逐帧回放、写类剧本在
页面上点头批准、密钥有地方填。
"""
import json
from pathlib import Path

from iphone_agent import config
from iphone_agent.web.server import Chat


def test_events_fan_out_to_every_subscriber():
    """⚠ 第一版只有一个 queue.Queue —— queue.get() 是破坏性的，
    开两个标签页就会互相抢事件。真机上撞到过：第二个连接收不到 user 事件。"""
    c = Chat()
    a, b = c.subscribe(), c.subscribe()
    c.emit("step", n=1)
    assert a.get_nowait()["n"] == 1
    assert b.get_nowait()["n"] == 1


def test_a_new_subscriber_catches_up_on_recent_events():
    """刷新页面不该一片空白。"""
    c = Chat()
    c.emit("user", text="你好")
    c.emit("done", result="嗯")
    q = c.subscribe()
    assert [q.get_nowait()["kind"] for _ in range(2)] == ["user", "done"]


def test_backlog_is_bounded():
    c = Chat()
    for i in range(config.WEB_BACKLOG + 50):
        c.emit("step", n=i)
    q = c.subscribe()
    got = []
    while not q.empty():
        got.append(q.get_nowait())
    assert len(got) == config.WEB_BACKLOG
    assert got[-1]["n"] == config.WEB_BACKLOG + 49, "留下的该是最近的那些"


def test_unsubscribe_stops_delivery():
    c = Chat()
    q = c.subscribe()
    c.unsubscribe(q)
    c.emit("step", n=1)
    assert q.empty()


def test_a_second_ask_while_busy_is_refused_not_queued():
    """一台手机同一时刻只能做一件事。排队会让人以为它在做，其实在等。"""
    c = Chat()
    c.busy.acquire()
    q = c.subscribe()
    c.ask("再来一个")
    ev = q.get_nowait()
    assert ev["kind"] == "error" and "上一轮还在跑" in ev["text"]


def test_frame_paths_must_stay_inside_runs():
    """⚠ 这是个本机 HTTP 服务，路径拼接必须校验 containment。"""
    root = Path("runs").resolve()
    for bad in ("../../etc/passwd", "../iphone_agent/config.py", "/etc/passwd"):
        p = (Path("runs") / bad).resolve()
        assert root not in p.parents, f"{bad} 逃出了 runs/"


def test_the_page_is_self_contained():
    """界面从「一个字符串常量」搬到了 web/static/（docs/22 P1）——
    175 行的字符串撑不起向导、设置和技能页。

    但**没有构建步骤**这一条不变：这些文件原样发出去，改完刷新就看得见。
    也**没有外部资源**：这是个离线的本机工具，断网也得能用。
    """
    from iphone_agent.web import static_files as sf
    html = sf.read_asset("/")[0].decode()
    js = sf.read_asset("/app.js")[0].decode()
    css = sf.read_asset("/app.css")[0].decode()
    assert "EventSource" in js

    # ⚠ 查的是「有没有引用外部资源」，不是「有没有出现 https:// 这几个字」——
    #   后者会把输入框的占位符（https://…/v1）也判成违规。断言要贴着意图写，
    #   不然它就会在正确的改动上报假警，然后被人顺手删掉。
    for blob in (html, js, css):
        for bad in ('src="http', "src='http", 'href="http', "href='http",
                    "url(http", "@import", "cdn.", "unpkg", "jsdelivr", "googleapis",
                    'fetch("http', "fetch('http"):
            assert bad not in blob, f"不该引用外部资源：{bad}"


def test_serve_announces_a_missing_key_up_front_but_still_starts(tmp_path, monkeypatch, capsys):
    """原来的意图仍然成立：**报错要出现在人还没投入之前**。
    Session 是第一次对话才构造的 —— 不在启动时查，人打完字发出去才看到
    「没有 API key」，中间那段时间全是白等的。

    但「起不来」这个做法在产品化之后是错的：**填密钥的地方就在这个界面里**，
    起不来就永远进不去，新用户只能回去 vim config.toml —— 那正是要消灭的一步
    （docs/22 P2）。所以改成照常起，但立刻在终端说清楚，界面也能从
    /api/state 问出来同一件事。
    """
    from iphone_agent.web import server
    monkeypatch.chdir(tmp_path)          # 没有 .iphone/config.toml
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "IPHONE_USE_MODEL", "IPHONE_USE_COORD_MODE"):
        monkeypatch.delenv(k, raising=False)

    class FakeServer:
        def __init__(self, addr, handler):
            pass

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(server, "_Server", FakeServer)
    assert server.serve(0) == 0
    err = capsys.readouterr().err
    assert "还没配好" in err and "设置" in err


def test_client_disconnects_are_not_reported_as_errors():
    """⚠ 浏览器关标签页、SSE 重连、curl 超时都会触发 ConnectionResetError。
    每关一个标签页吐一坨堆栈，会把真正的错误淹掉，也会让人以为出事了。"""
    import sys
    from unittest.mock import patch

    from iphone_agent.web.server import _Server
    srv = _Server.__new__(_Server)
    for exc in (ConnectionResetError(), BrokenPipeError(), TimeoutError()):
        with patch.object(sys, "exc_info", return_value=(type(exc), exc, None)), \
             patch("http.server.ThreadingHTTPServer.handle_error") as up:
            srv.handle_error(None, ("127.0.0.1", 1))
            assert not up.called, f"{type(exc).__name__} 不该被当成错误"
    boom = ValueError("真错误")
    with patch.object(sys, "exc_info", return_value=(ValueError, boom, None)), \
         patch("http.server.ThreadingHTTPServer.handle_error") as up:
        srv.handle_error(None, ("127.0.0.1", 1))
        assert up.called, "真错误必须照旧报出来"


def test_ask_emits_memory_event_for_ok_written_names(tmp_path, monkeypatch):
    """run 结束后，网页要能看见「本次新增了哪些记忆」，才谈得上撤销。
    只有 ok 为真的条目才展示——写失败的那条本来就没落盘，撤销无从谈起。"""
    from unittest.mock import patch

    from iphone_agent.harness.loop import RunResult
    from iphone_agent.workspace import Workspace

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")   # ask() 里 s.model 是懒加载属性，
    # run_task 本身被 mock 掉了，但取 s.model 这一步还在真跑，要有 key 才不炸。
    ws = Workspace(tmp_path)
    c = Chat(ws)
    q = c.subscribe()

    run_dir = tmp_path / "runs" / "20260101-000000-abcd"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "memory": {"written": [
            {"name": "a-good-name", "ok": True},
            {"name": "b-downgraded", "ok": True, "downgraded": "playbook→knowledge"},
            {"name": None, "ok": False, "error": "invalid_name: 坏"},
        ]}
    }), encoding="utf-8")

    fake_res = RunResult(end_reason="done_success", steps=1, done_status="ok",
                         done_result="完成", run_dir=run_dir)
    with patch("iphone_agent.harness.loop.run_task", return_value=fake_res):
        c.ask("干点什么")

    events = []
    while not q.empty():
        events.append(q.get_nowait())
    kinds = [e["kind"] for e in events]
    assert "memory" in kinds
    mem = next(e for e in events if e["kind"] == "memory")
    assert mem["names"] == ["a-good-name", "b-downgraded"]
    assert mem["run"] == run_dir.name


def test_ask_emits_no_memory_event_when_nothing_written(tmp_path, monkeypatch):
    """没有新记忆时不该多出一条空事件——前端不用为「空列表」专门写一支逻辑。"""
    from unittest.mock import patch

    from iphone_agent.harness.loop import RunResult
    from iphone_agent.workspace import Workspace

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    ws = Workspace(tmp_path)
    c = Chat(ws)
    q = c.subscribe()

    run_dir = tmp_path / "runs" / "20260101-000001-abcd"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"memory": {"written": []}}), encoding="utf-8")

    fake_res = RunResult(end_reason="done_success", steps=1, done_status="ok",
                         done_result="完成", run_dir=run_dir)
    with patch("iphone_agent.harness.loop.run_task", return_value=fake_res):
        c.ask("干点什么")

    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert "memory" not in [e["kind"] for e in events]


def _live_server(tmp_path):
    """起一个真的、监听 127.0.0.1:0（系统分配端口）的服务，在后台线程跑。

    undo 端点要走真实的 do_POST 分支（404/400/200 三条路径都靠 HTTP 状态码
    区分），照抄 handler 内部方法拿不到这些——直接打真请求最贴近浏览器会
    发生的事。"""
    import threading

    from iphone_agent.web.server import Chat, _handler, _Server
    from iphone_agent.workspace import Workspace

    ws = Workspace(tmp_path)
    chat = Chat(ws)
    srv = _Server(("127.0.0.1", 0), _handler(chat))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, chat


def test_undo_moves_memory_to_trash_and_is_gone_from_store(tmp_path):
    from iphone_agent.memory import MemoryStore

    srv, chat = _live_server(tmp_path)
    try:
        store = MemoryStore(chat.workspace.memory_dir)
        store.write("a-good-name", "desc", "content", "runs/x", "done_success")
        assert store.read("a-good-name") is not None

        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
        conn.request("POST", "/api/memory/undo",
                     body=json.dumps({"name": "a-good-name"}),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()

        assert resp.status == 200
        assert body == {"ok": True}
        assert store.read("a-good-name") is None
        assert list(store.trash.glob("*.md")), "撤销是移进 trash，不是永久删除"
    finally:
        srv.shutdown()
        srv.server_close()   # 不关会留一个没关的 socket，pytest 把这当成资源泄露警告报错


def test_undo_of_missing_name_returns_ok_false_not_error(tmp_path):
    srv, chat = _live_server(tmp_path)
    try:
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
        conn.request("POST", "/api/memory/undo",
                     body=json.dumps({"name": "does-not-exist"}),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()
        assert resp.status == 200
        assert body == {"ok": False}
    finally:
        srv.shutdown()
        srv.server_close()   # 不关会留一个没关的 socket，pytest 把这当成资源泄露警告报错


def test_undo_rejects_names_outside_memory_name_re_with_400(tmp_path):
    """越权名字（`_path` 已经挡过一次）在这里必须是 400，不能悄悄变成
    200 {"ok": false}——那会让调用方以为「合法地查了一下，没这条」，
    而不是「你传的东西本身不合法」。"""
    srv, chat = _live_server(tmp_path)
    try:
        import http.client
        for bad in ("../escape", "a/b", "UPPER", ""):
            conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
            conn.request("POST", "/api/memory/undo",
                         body=json.dumps({"name": bad}),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            resp.read()
            conn.close()
            assert resp.status == 400, f"{bad!r} 应该 400"
    finally:
        srv.shutdown()
        srv.server_close()   # 不关会留一个没关的 socket，pytest 把这当成资源泄露警告报错


# ── 新对话 ─────────────────────────────────────────────────────────────

def test_new_chat_clears_history_and_last_step():
    c = Chat()
    c.history.append(("你好", "你好呀"))
    c.last_step = {"n": 1, "name": "tap"}
    assert c.new_chat() is True
    assert c.history == []
    assert c.last_step is None


def test_new_chat_refuses_while_busy_and_changes_nothing():
    """一台手机同一时刻只能做一件事——跑到一半清 history 会让正在跑的这一轮
    和已经发生的事件对不上。"""
    c = Chat()
    c.history.append(("你好", "你好呀"))
    c.busy.acquire()
    assert c.new_chat() is False
    assert c.history == [("你好", "你好呀")]


def test_new_chat_endpoint_returns_200_and_clears_history(tmp_path):
    srv, chat = _live_server(tmp_path)
    try:
        chat.history.append(("你好", "你好呀"))
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
        conn.request("POST", "/api/new-chat")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()
        assert resp.status == 200
        assert body == {"ok": True}
        assert chat.history == []
    finally:
        srv.shutdown()
        srv.server_close()   # 不关会留一个没关的 socket，pytest 把这当成资源泄露警告报错


def test_new_chat_endpoint_returns_409_while_a_task_is_running(tmp_path):
    srv, chat = _live_server(tmp_path)
    try:
        chat.busy.acquire()
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
        conn.request("POST", "/api/new-chat")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()
        assert resp.status == 409
        assert body["ok"] is False
        assert body["error_message"]["code"] == "api.newChatBusy"
    finally:
        srv.shutdown()
        srv.server_close()   # 不关会留一个没关的 socket，pytest 把这当成资源泄露警告报错
