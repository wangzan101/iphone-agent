"""一个能跟 agent 对话的网页。

为什么要有它：命令行里 `iphone run "..."` 每次都是一次性的，看不到过程、
接不上话，也没法体验「对话」。而这个 agent 本来就有跨任务记忆和屏幕图，
对话的连续性是它已经具备的能力，只是没有一个能感受到它的地方。

**零新依赖**：标准库的 http.server + Server-Sent Events。这个项目一直很克制，
不为一个界面引入 web 框架。

⚠ 只监听 127.0.0.1。这个界面能操作你的真手机，不该暴露到网络上。
"""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from iphone_agent import config
from iphone_agent.web import api, static_files
from iphone_agent.workspace import Workspace


class Chat:
    """一场对话。串行执行 —— 一台手机同一时刻只能做一件事。"""

    def __init__(self, workspace: Workspace | None = None,
                 model_spec: str | None = None, resolved=None) -> None:
        # 工作区显式传进来：路径不再是模块级常量，一个进程能开两场对话（docs/20 D3）。
        self.workspace = workspace or Workspace.default()
        # resolved：serve 启动时那次解析的结果。不接过来的话，第一次对话会再解析一遍，
        # 同一批 notices 在 stderr 上打两遍。
        self._model_spec = model_spec
        self._resolved = resolved
        self.history: list[tuple[str, str]] = []
        self.busy = threading.Lock()
        # 停止标志。threading.Event 而不是 bool：HTTP 线程置位、跑任务的线程读，
        # 两个线程各一个。每轮开跑前 clear —— 否则上一轮按过的停止会把下一轮直接掐掉。
        self._stop = threading.Event()
        # 暂停标志 + 暂停期间用户补的那句话。主循环每步开头调 wait_if_paused()，
        # 置位期间它阻塞在那里；resume(text) 清位并把话交给它。停止也能解开阻塞。
        self._paused = threading.Event()
        self._note: str | None = None
        # 挂起的安全确认（harness/safety.py）：主循环等在这个队列上，网页 /api/confirm 往里放答案。
        # 没人应答就按拒绝 —— 「无人只读」在网页这头的形态就是超时。
        self._confirm_q: queue.Queue | None = None
        self._confirm_text: str | None = None
        self._confirm_message: dict | None = None
        self._confirm_deadline = 0.0
        # 状态条要显示「它现在在干嘛」。SSE 是推的，但新开的标签页要能立刻问出当前状态，
        # 不能等下一个事件才知道。
        self.last_step: dict | None = None
        self._session = None
        # ⚠ 每个连接一个队列，事件**扇出**给所有订阅者。
        #   第一版只有一个 queue.Queue —— queue.get() 是破坏性的，
        #   开两个标签页就会互相抢事件，刷新一次对话内容全丢。
        self._subs: list[queue.Queue] = []
        self._subs_lock = threading.Lock()
        self._backlog: list[dict] = []      # 新连上来的补看最近这些，刷新不至于空白

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._subs_lock:
            for ev in self._backlog:
                q.put(ev)
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._subs_lock:
            if q in self._subs:
                self._subs.remove(q)

    def session(self):
        if self._session is None:
            from iphone_agent.cli.commands import Session
            self._session = Session(self.workspace, model_spec=self._model_spec, resolved=self._resolved)
        return self._session

    def emit(self, kind: str, **data) -> None:
        ev = {"kind": kind, **data}
        with self._subs_lock:
            self._backlog.append(ev)
            del self._backlog[:-config.WEB_BACKLOG]
            subs = list(self._subs)
        for q in subs:
            q.put(ev)

    @staticmethod
    def _written_memory_names(run_dir) -> list[str]:
        """从 run.json 里读 commit_memories 落的 written 列表，只要 ok 的名字。

        读不出来（并发写入没落盘完、文件不存在这类边角情况）就当没写——
        网页少提示一次「新增记忆」不是事故，抛出去把整场对话的事件流打断才是。
        """
        try:
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        except Exception:                            # noqa: BLE001 —— 见上
            return []
        written = ((run.get("memory") or {}).get("written")) or []
        return [w["name"] for w in written if w.get("ok")]

    def stop(self) -> bool:
        """请求停止。返回「当时是不是真有东西在跑」——没在跑就别让界面显示「已停止」。

        ⚠ 它只是置个位。主循环在**每一步开头**才会看到，所以从按下到停下最多差一步
        （约十秒）。界面要把这句话说出来，否则用户会以为按钮坏了、然后猛点。
        """
        running = self.busy.locked()
        self._stop.set()
        self._paused.clear()                 # 暂停中按停止：解开主循环的阻塞，让它看到停止位
        self._note = None
        self.answer(False)                   # 正等人确认的写操作：停止 = 拒绝
        if running:
            self.emit("stopping", text="正在停止 —— 当前这一步做完就收手（最多约十秒）。",
                      message={"code": "event.stopping"})
        return running

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def pause(self) -> bool:
        """请求暂停。没在跑就什么都不做（返回 False）。

        和停止一样只是置位：主循环在**每一步开头**才停下，所以「已暂停」不等于手机已经不动了，
        最多还差一步。界面要说清楚。
        """
        if not self.busy.locked():
            return False
        self._note = None
        self._paused.set()
        self.emit("paused", text="已暂停 —— 当前这一步做完就停。想补一句话就打在下面，然后点继续。",
                  message={"code": "event.paused"})
        return True

    def resume(self, text: str | None = None) -> bool:
        """继续，可带一句补充。没暂停就什么都不做。"""
        if not self._paused.is_set():
            return False
        note = (text or "").strip()
        self._note = note or None
        self._paused.clear()
        self.emit("resumed", text=note)
        return True

    def new_chat(self) -> bool:
        """开一场新对话：清空 history、last_step。有任务在跑就什么都不做，返回 False——

        history 是每轮任务的上下文，跑到一半清掉会让正在跑的那一轮和已经产生的事件对不上。
        """
        if self.busy.locked():
            return False
        self.history = []
        self.last_step = None
        return True

    @property
    def pending_confirm(self) -> str | None:
        return self._confirm_text

    @property
    def pending_confirm_message(self) -> dict | None:
        return self._confirm_message

    @property
    def confirm_timeout_remaining(self) -> int:
        return max(0, int(self._confirm_deadline - time.monotonic())) if self._confirm_text else 0

    def confirm(self, text: str) -> bool:
        """主循环遇到写类动作时调（跑任务的线程）。阻塞到网页回答或超时；超时 = 拒绝。"""
        q: queue.Queue = queue.Queue(maxsize=1)
        self._confirm_q, self._confirm_text = q, text
        self._confirm_message = getattr(text, "message", {"code": "confirm.legacy", "params": {"detail": text}})
        self._confirm_deadline = time.monotonic() + config.WEB_CONFIRM_TIMEOUT_S
        self.emit("confirm", text=text, message=self._confirm_message, timeout_s=config.WEB_CONFIRM_TIMEOUT_S)
        try:
            ok = bool(q.get(timeout=config.WEB_CONFIRM_TIMEOUT_S))
        except queue.Empty:
            ok = False
        finally:
            self._confirm_q, self._confirm_text = None, None
            self._confirm_message = None
            self._confirm_deadline = 0.0
        self.emit("confirmed", ok=ok)
        return ok

    def answer(self, ok: bool) -> bool:
        """网页的回答。没有挂起的问题就什么都不做（返回 False）。"""
        q = self._confirm_q
        if q is None:
            return False
        try:
            q.put_nowait(bool(ok))
        except queue.Full:
            return False
        return True

    def handover(self, need: str, reason: str) -> str | None:
        """模型把一步交给人（harness handover）。复用暂停机制：发 handover 事件、置暂停位、
        阻塞到用户点「继续」。返回用户补的话（没补就 ""）；用户按了停止返回 None。"""
        if not self.busy.locked():
            return None
        self._note = None
        self._paused.set()
        self.emit("handover", need=need, reason=reason,
                  text=f"它需要你来做一步：{need}。做完后点「继续」（可以补一句话）。",
                  message={"code": "event.handover", "params": {"need": need}})
        note = self.wait_if_paused()
        if self._stop.is_set():
            return None
        return note or ""

    def wait_if_paused(self) -> str | None:
        """主循环每步开头调。没暂停立刻回 None；暂停了就等到 resume / stop。

        返回用户在暂停期间补的那句话（没有就 None）。⚠ 只等**这一次**暂停：
        取走 note 之后清空，同一句话不会在下一步再交一遍。
        """
        while self._paused.is_set():
            self._paused.wait(0.2)
        note, self._note = self._note, None
        return note

    def ask(self, text: str) -> None:
        """跑一轮。⚠ 整个函数不能抛 —— 抛了网页就再也收不到事件了。"""
        if not self.busy.acquire(blocking=False):
            self.emit("error", text="上一轮还在跑，等它结束再说。一台手机同时只能做一件事。",
                      message={"code": "event.busy"})
            return
        try:
            self._stop.clear()
            self._paused.clear()
            self._note = None
            self.emit("user", text=text)
            from iphone_agent.harness.loop import run_task
            s = self.session()

            def on_step(rec):
                a = rec.get("action") or {}
                r = rec.get("result") or {}
                self.last_step = {"n": rec.get("step"), "name": a.get("name"),
                                  "error": r.get("error")}
                self.emit("step",
                          n=rec.get("step"),
                          name=a.get("name", "?"),
                          args=a.get("args", a.get("args_raw", {})),
                          reason=(rec.get("model") or {}).get("reason", ""),
                          changed=r.get("changed"),
                          error=r.get("error"),
                          hint=(r.get("hint") or "")[:400],
                          # 动作之后那一帧才是「当前画面」；没有就退回动作之前那张
                          frame=(rec.get("after_frame_file")
                                 or (rec.get("observation") or {}).get("frame_file")),
                          # 回放的节奏靠这两个。固定节拍看不出「哪一步卡了很久」。
                          exec_ms=rec.get("exec_ms"),
                          latency_ms=(rec.get("model") or {}).get("latency_ms"))

            res = run_task(text, s.dev, s.per, s.model, self.workspace.runs,
                           workspace=self.workspace,
                           on_step=on_step, history=self.history,
                           on_start=lambda d: self.emit("start", run=d),
                           on_frame=lambda f: self.emit("frame", frame=f),
                           skill_store=s.skills,
                           should_stop=self._stop.is_set,
                           wait_if_paused=self.wait_if_paused,
                           confirm=self.confirm,
                           on_handover=self.handover)
            answer = res.done_result or res.error or res.end_reason
            self.history.append((text, str(answer)))
            self.emit("done", end=res.end_reason, steps=res.steps,
                      result=res.done_result, error=res.error,
                      error_message={"code": "event.error", "params": {"detail": res.error}} if res.error else None,
                      run=res.run_dir.name)
            # ⚠ 这次跑真写进去的记忆，人得能看见才谈得上撤销（也不能删）。
            #   run.json 是本次唯一权威来源——不重算，直接读 commit_memories
            #   落下的 written 列表；只挑 ok 的：写失败那条本来就没落盘。
            names = self._written_memory_names(res.run_dir)
            if names:
                self.emit("memory", names=names, run=res.run_dir.name)
        except Exception as e:                      # noqa: BLE001 —— 见 docstring
            self.emit("error", text=f"{type(e).__name__}: {e}",
                      message={"code": "event.error", "params": {"detail": f"{type(e).__name__}: {e}"}},
                      detail=traceback.format_exc()[-1500:])
        finally:
            self.busy.release()


def _handler(chat: Chat):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):        # 别把访问日志混进任务输出里
            pass

        def _send(self, code, ctype, body: bytes, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        # 只读的数据面。都不带参数 —— 有参数的走 POST。
        GETS = {"/api/doctor": api.doctor, "/api/state": api.state,
                "/api/models": api.models, "/api/config": api.get_config,
                "/api/learned": api.learned, "/api/runs": api.runs}

        def _json(self, pair):
            code, data = pair
            return self._send(code, "application/json; charset=utf-8",
                              json.dumps(data, ensure_ascii=False).encode())

        def _api(self, fn, *args):
            from iphone_agent.model.errors import ConfigError
            try:
                return self._json(fn(chat, *args))
            except ConfigError as e:
                return self._json((400, {"error": str(e), "error_message": e.as_message()}))

        def do_GET(self):                                      # noqa: N802
            if self.path == "/api/stream":
                return self._stream()
            if self.path.startswith("/api/frame/"):
                return self._frame(self.path[len("/api/frame/"):])
            if self.path.split("?", 1)[0] == "/api/screen.png":
                return self._screen()
            fn = self.GETS.get(self.path)
            if fn is not None:
                return self._api(fn)
            if self.path.startswith("/api/runs/"):
                # 回放：某一次运行的每一步。id 的合法性在 api 那边校验（它是路径的一部分）。
                return self._json(api.run_detail(chat, self.path[len("/api/runs/"):]))
            # 静态资源。⚠ 只发 web/static/ 里的东西 —— 这个服务是在用户的项目目录里
            # 跑起来的，把 cwd 暴露出去不可接受（static_files.py 的模块注释）。
            hit = static_files.read_asset(self.path.split("?", 1)[0])
            if hit is not None:
                body, ctype = hit
                # 不缓存：没有构建步骤也就没有指纹文件名，缓存住了改完刷新看不见。
                return self._send(200, ctype, body, {"Cache-Control": "no-store"})
            return self._send(404, "text/plain", b"not found")

        def do_POST(self):                                     # noqa: N802
            if self.path == "/api/memory/undo":
                return self._memory_undo()
            if self.path == "/api/stop":
                # 不带请求体，也不需要。停止是幂等的：多按几次不会更停。
                return self._json((200, {"ok": True, "running": chat.stop()}))
            if self.path == "/api/pause":
                return self._json((200, {"ok": True, "paused": chat.pause()}))
            if self.path == "/api/new-chat":
                if chat.new_chat():
                    return self._json((200, {"ok": True}))
                return self._json((409, {
                    "ok": False, "error": "有任务正在跑，先停下来再开新对话。",
                    "error_message": {"code": "api.newChatBusy"}}))
            if self.path == "/api/confirm":
                body = self._body()
                if body is None:
                    return self._send(400, "text/plain", b"bad json")
                return self._json((200, {"ok": True, "answered": chat.answer(bool(body.get("ok")))}))
            if self.path == "/api/resume":
                body = self._body()
                if body is None:
                    return self._send(400, "text/plain", b"bad json")
                return self._json((200, {"ok": True, "resumed": chat.resume(body.get("text"))}))
            if self.path == "/api/model/test":
                return self._api(api.test_model)
            if self.path == "/api/config":
                body = self._body()
                if body is None:
                    return self._send(400, "text/plain", b"bad json")
                return self._api(api.put_config, body)
            if self.path != "/api/send":
                return self._send(404, "text/plain", b"not found")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, "text/plain", b"bad json")
            text = (body.get("text") or "").strip()
            if not text:
                return self._send(400, "text/plain", b"empty")
            threading.Thread(target=chat.ask, args=(text,), daemon=True).start()
            return self._send(200, "application/json", b'{"ok":true}')

        def _body(self):
            """读 JSON 请求体。读不出来返回 None，调用方回 400。"""
            n = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return None

        def _memory_undo(self):
            # 撤销 = 移进 trash，永不 unlink——这是网页给的一个「后悔药」端口，
            # 不是删除端口。move_to_trash 本身对不存在的名字返回 False，
            # 但它对越权名字（含 / 或 ..）也是返回 False——那会被网页当成
            # 「合法查了一下，没这条」。这里在调用它之前先用 MEMORY_NAME_RE
            # 单独挡一次，让越权名字变成 400（客户端传错），跟「确实没有这条
            # 记忆」（200 {"ok": false}）分开。
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, "text/plain", b"bad json")
            name = body.get("name")
            if not isinstance(name, str) or not config.MEMORY_NAME_RE.match(name):
                return self._send(400, "text/plain", b"bad name")
            from iphone_agent.memory import MemoryStore
            store = MemoryStore(chat.workspace.memory_dir)
            ok = store.move_to_trash(name, "网页撤销")
            return self._send(200, "application/json",
                              json.dumps({"ok": ok}).encode())

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = chat.subscribe()
            try:
                while True:
                    try:
                        payload = json.dumps(q.get(timeout=15), ensure_ascii=False)
                    except queue.Empty:
                        payload = None
                    try:
                        if payload is None:
                            self.wfile.write(b": keepalive\n\n")   # 别让代理掐断连接
                        else:
                            self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return          # 页面关了，正常退出
            finally:
                chat.unsubscribe(q)

        def _screen(self):
            """手机**此刻**的画面。只抓帧，不注入 —— 纯只读。

            为什么要有它：进来第一眼那栏不该是一块黑的。用户对这个产品的全部信心
            都来自「我看见它在动我的手机」，而这份信心得在他还没发出第一条指令之前
            就建立起来（docs/22 第三节）。

            ⚠ 任务跑起来之后不要再问这里 —— 那时候画面由 SSE 的 step 事件驱动，
            两边同时抓帧会互相抢镜像窗口。
            """
            if chat.busy.locked():
                return self._send(409, "text/plain", b"busy")
            try:
                import io
                frame = chat.session().dev.capture()
                buf = io.BytesIO()
                frame.image.save(buf, format="PNG")
            except Exception:                        # noqa: BLE001 —— 镜像没连是常态，不是故障
                return self._send(404, "text/plain", b"no mirror")
            return self._send(200, "image/png", buf.getvalue(), {"Cache-Control": "no-store"})

        def _frame(self, rel: str):
            # ⚠ 只允许 runs/ 里的 png，且必须真的落在 runs/ 下面 ——
            #   这是个本机 HTTP 服务，路径拼接必须校验 containment。
            p = (chat.workspace.runs / rel).resolve()
            root = chat.workspace.runs.resolve()
            if p.suffix != ".png" or root not in p.parents or not p.is_file():
                return self._send(404, "text/plain", b"not found")
            return self._send(200, "image/png", p.read_bytes(),
                              {"Cache-Control": "max-age=86400"})
    return H


class _Server(ThreadingHTTPServer):
    """把「客户端断开」这类正常事件从错误输出里拿掉。

    ⚠ 浏览器关标签页、SSE 重连、curl 超时，都会让 socketserver 打一整坨
    ConnectionResetError 堆栈到 stderr。这个服务是要长期开着的 ——
    每关一个标签页吐一坨堆栈，会把真正的错误淹掉，也会让人以为出事了。
    真错误照旧打出来。
    """
    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def serve(port: int = config.WEB_PORT, model_spec: str | None = None) -> int:
    # ⚠ 启动时就把 key 查掉。这里的 Session 是**第一次对话才构造**的，
    #   不查的话服务能正常起来、网页也能打开，人打完字发出去才看到
    #   「没有 API key」—— 中间那段时间全是白等的。
    #   报错要出现在人还没投入之前。
    from iphone_agent.model.config import resolve
    from iphone_agent.model.errors import ConfigError
    ws = Workspace.default()
    r = None
    try:
        r = resolve(model_spec)
    except ConfigError as e:
        # ⚠ 以前这里 return 2 —— 服务根本起不来。
        #   但**填密钥的地方就在这个界面里**：起不来就永远进不去，新用户被锁在门外，
        #   只能回去 vim config.toml。这正是产品化要消灭的那一步（docs/22 P2）。
        #   所以照常起，把问题交给界面去说，并引导到设置页。
        print(f"还没配好：{e}", file=__import__("sys").stderr)
        print("界面照常打开，去「设置」里填上密钥就能用。", file=__import__("sys").stderr)
    if r is not None:
        for n in r.notices:
            print(n, file=__import__("sys").stderr)
        print(f"模型 {r.spec}")
    chat = Chat(ws, model_spec, resolved=r)
    srv = _Server(("127.0.0.1", port), _handler(chat))
    print(f"打开 http://127.0.0.1:{port}   （Ctrl-C 停止）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n停止")
    finally:
        srv.server_close()
        if chat._session is not None:
            try:
                chat._session.dev.release_all()
            except Exception:      # noqa: BLE001 —— 收尾失败不该盖住退出
                pass
    return 0
