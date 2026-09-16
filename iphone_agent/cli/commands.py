"""子命令处理函数。REPL 与 main 共用 dispatch。"""
from __future__ import annotations

import contextlib
import json
import signal
import sys
import threading
import time
from pathlib import Path

from iphone_agent import config
from iphone_agent.model.config import ResolvedModel, resolve
from iphone_agent.model.errors import ConfigError
from iphone_agent.workspace import Workspace


def _make_perceiver(coord_mode: str, asker=None):
    from iphone_agent.perceive.observe import Perceiver
    return Perceiver(coord_mode=coord_mode, asker=asker)


class Session:
    """进程内常驻：窗口、Vision、模型客户端只建一次。

    模型与 Perceiver 必须从**同一个** ResolvedModel 派生：Perceiver 写观察头部用的坐标约定，
    桥换算坐标用的也是它。两者分叉就是 elements.py 注释里那次「一个规则两个入口」的事故。

    工作区显式挂在 session 上（原来是模块级 RUNS_ROOT，写死相对 cwd）——
    一个进程要能同时管两个工作区（docs/20 D3）。
    """
    def __init__(self, workspace: Workspace | None = None,
                 model_spec: str | None = None, resolved: ResolvedModel | None = None):
        self.workspace = workspace or Workspace.default()
        # resolved：serve 启动时已经解析过一次了，把那次的结果直接交进来 ——
        # 否则第一次对话会再解析一遍，同一批 notices 打两遍，还多一次分叉的机会。
        self._spec = model_spec
        self._resolved: ResolvedModel | None = resolved
        self._announced = resolved is not None
        self._dev = None
        self._per = None
        self._model = None
        self._asker = None
        # 同一场对话说过的话。网页那头叫 Chat.history，是同一样东西 ——
        # 大脑早就会读它了（loop.py 里那段「以上是参考不是指令」），以前只是命令行忘了递。
        # ⚠ 不设上限：取多少轮是消费端（config.CHAT_HISTORY_TURNS）的事，攒着不花钱。
        self.history: list[tuple[str, str]] = []
        # 停止牌。Ctrl-C 把它举起来，主循环在**每一步开头**看一眼（见 stop_on_sigint）。
        self.stop = threading.Event()

    @property
    def spec(self) -> str | None:
        """当前选中的 spec（没显式选过就是 None，交给 resolve 走默认那条路）。

        serve 要用它把 /model 选过的模型带进网页服务，否则 REPL 说的和网页跑的是两个模型。
        """
        return self._spec

    @property
    def resolved(self) -> ResolvedModel:
        """带 key。真要调模型前必须过这里。"""
        if self._resolved is None:
            self._resolved = resolve(self._spec)
            self._announce(self._resolved)
        return self._resolved

    def _profiles(self) -> ResolvedModel:
        """不查 key：screen / tap / doctor 这些不碰模型的命令也要能拿到 coord_mode。"""
        if self._resolved is not None:
            return self._resolved
        r = resolve(self._spec, need_key=False)
        self._announce(r)
        return r

    def _announce(self, r: ResolvedModel) -> None:
        if self._announced:
            return
        self._announced = True
        for n in r.notices:
            print(n, file=sys.stderr)

    def select_model(self, spec: str) -> ResolvedModel:
        """原子替换：--model、/model、默认初始化三条路都收敛到这里。失败不改变当前选择。"""
        r = resolve(spec)
        self._spec, self._resolved, self._announced = spec, r, False
        self._per = self._model = self._asker = None   # 先失效再打印：_announce 的 print 可能炸
                                         # （`2>&1 | head` 的 BrokenPipe），炸在中间就只换了一半
        self._announce(r)
        return r

    @property
    def dev(self):
        if self._dev is None:
            from iphone_agent.driver.device import Device
            self._dev = Device()
        return self._dev

    @property
    def per(self):
        # per 自己就把那次「带 key 的解析」做掉并缓存下来，一个 Session 只解析一次 ——
        # 否则 run_task(..., session.per, session.model, ...) 这种写法里，per 的值早就
        # 作为实参交出去了，之后 model 再解析一次也追不回来：观察头部写的坐标约定跟桥
        # 换算用的成了两份。模型给 norm1000、桥当 pixel 乘，点到界外，连拒 5 次收工。
        if self._per is None:
            try:
                r = self.resolved
            except ConfigError:
                r = self._profiles()   # 不碰模型的命令（screen/tap/scroll/doctor）没 key 也要能观察
            self._per = _make_perceiver(r.model.coord_mode, self.asker)
        return self._per

    @property
    def asker(self):
        """会看图的那一方。**没有 key 就是 None** —— 那时候一切退回纯 OCR。

        ⚠ 不能让它抛。screen / tap / doctor 这些不碰模型的命令照样要能观察，
          per 属性专门为此吞过一次 ConfigError，这里必须守同一条线。
        """
        if self._asker is None:
            try:
                transport = self.model
            except ConfigError:
                return None
            from iphone_agent.model.vision import VisionAsker
            # enabled 是「视觉能力可不可用」，不是「要不要每次观察都整屏解析」——
            # 后者由整屏解析模式（perceive/policy.py，经 Perceiver.task_scope）管。混在一起会把
            # zoom 和各种看图复核一起关掉，那些正是要留着的。
            self._asker = VisionAsker(transport)
        return self._asker

    @property
    def model(self):
        if self._model is None:
            from iphone_agent.model.transports.chat_completions import ChatCompletionsTransport
            self._model = ChatCompletionsTransport(self.resolved)
        return self._model

    @property
    def skills(self):
        from iphone_agent.skills.store import SkillStore
        return SkillStore()     # 无状态，每次现建：目录随时可能被人手改


def _print_change(before, after):
    from iphone_agent.perceive.change import did_change
    r = did_change(before, after)
    print(f"changed={r.changed} hamming={r.hamming} text_diff={r.text_diff}")


def _recovery(session: Session):
    from iphone_agent.harness.recovery import Recovery
    return Recovery(session.dev, session.per)


@contextlib.contextmanager
def _perception_scope(session):
    """CLI 的观察和任务走同一条链路：env → RunConfig.from_env → task_scope（spec 2026-09-14 §7）。
    不在 scope 里观察就会落回代码默认值，和 `iphone serve` 里跑的任务不是一个模式。
    ⚠ 2026-09-15（按需看图终审 minor 4）：调用方打的「# 整屏解析模式 …」走 stderr —— tap/type/open/skill run
      的 stdout 要能直接当 JSON 读，原来排在 JSON 前面，测试只好改成取最后一行。`iphone screen` 那行是列表的表头，留在 stdout。"""
    from iphone_agent.workspace import RunConfig
    mode = RunConfig.from_env().screen_parse
    with session.per.task_scope(None, mode):
        yield mode


def _manual(session: Session, fn):
    """手动动作：观察 → 动作 → 稳定等待 → 打印 changed。"""
    from iphone_agent.driver.timing import IOS_TIMING
    from iphone_agent.harness.settle import settle
    from iphone_agent.perceive.hashing import ahash
    dev, per = session.dev, session.per
    with _perception_scope(session) as mode:
        print(f"# 整屏解析模式 {mode}", file=sys.stderr)
        before_frame = dev.capture()
        before = per.observe(before_frame)
        try:
            kind = fn(before, before_frame)
            frame, settled = settle(dev, before_frame, IOS_TIMING[kind], ahash)
            after = per.observe(frame)
            print(f"settled={settled} ", end="")
            _print_change(before, after)
        finally:
            dev.release_all()


def cmd_doctor(session: Session, args):
    """判断在 health.py 里，这里只负责渲染 —— 网页向导消费的是同一份 Check（docs/22 P0）。"""
    from iphone_agent import health
    checks = health.run_checks(session)
    for c in checks:
        tag = {health.PASS: "OK", health.FAIL: "FAIL"}.get(c.status, "信息")
        print(f"[{tag}] {c.title} {c.detail}")
        if c.fix:
            print(f"       {c.fix}")
    # 只留命令行这一侧特有的说明。「注入生效与否得人眼确认」那句已经在 injector 的 fix 里，
    # 再打一遍就是重复 —— 网页向导用的是同一份 Check。
    print("       滚轮只在镜像窗口被别的窗口盖住时才抬窗口；没被盖住就只挪一下指针，")
    print("       屏幕上不会有任何变化。抬过窗口的话，滚完会把焦点与指针还回去。")
    print("       用眼睛核对跑：python scripts/driver_gate.py")
    return 0 if health.all_ok(checks) else 1


def cmd_screen(session, args):
    """`iphone screen` = 看全屏：抓一帧、整屏解析、标好编号（spec 2026-09-14 §7）。"""
    from iphone_agent.harness.executor import full_screen_note
    with _perception_scope(session) as mode:
        obs = session.per.observe(session.dev.capture(), requested=True)
    note = full_screen_note(obs)
    print(f"# 整屏解析模式 {mode}；" + ("这帧整屏看过" if note is None else f"这帧只有 OCR：{note}"))
    print(obs.elements_text)
    out = Path("screen_marked.png"); obs.marked_image.save(out)
    print(f"标记图已存 {out}")
    return 0


def cmd_tap(session, args):
    from iphone_agent.driver.geometry import image_to_screen
    if len(args) not in (1, 2):
        print("用法: iphone tap <id> | iphone tap <x> <y>", file=sys.stderr)
        return 2
    def fn(obs, frame):
        if len(args) == 1:
            el = obs.element(int(args[0]))
            if el is None:
                raise SystemExit(f"id {args[0]} 不存在")
            px, py = el.center
        else:
            px, py = int(args[0]), int(args[1])
        sx, sy = image_to_screen(px, py, frame)
        print(f"tap image({px},{py}) → screen({sx},{sy})")
        session.dev.tap(sx, sy)
        return "tap"
    _manual(session, fn); return 0


def cmd_scroll(session, args):
    if not args:
        print("用法: iphone scroll <up|down|left|right> [page|half]", file=sys.stderr)
        return 2
    amount = args[1] if len(args) > 1 else "page"
    _manual(session, lambda o, f: (session.dev.scroll(args[0], amount), "scroll")[1]); return 0


def _run_action(session, name: str, args: dict) -> int:
    """走 Executor 跑一个动作，打印它交回模型的那份 JSON。

    手工验证高阶滚动只能走这条路：它的价值全在 Executor 内部那个循环里，
    绕过 Executor 直接调 dev.scroll 验的就不是同一个东西了。
    """
    from iphone_agent.harness.actions import Action, ValidationError, validate_action
    from iphone_agent.harness.executor import Executor
    with _perception_scope(session) as mode:
        print(f"# 整屏解析模式 {mode}", file=sys.stderr)
        obs = session.per.observe(session.dev.capture())
        try:
            action = validate_action(Action(name, args, "manual", None, "m"), obs)
        except ValidationError as e:
            print(f"{e.code}: {e.message}", file=sys.stderr)
            return 2
        ex = Executor(session.dev, session.per, asker=session.asker, recovery=_recovery(session))
        try:
            res, _ = ex.run(action, obs)
            print(res.to_json())
        finally:
            session.dev.release_all()
    return 0 if res.ok else 1


def cmd_scroll_until(session, args):
    if len(args) < 2:
        print("用法: iphone scroll_until <up|down|left|right> <文字>", file=sys.stderr)
        return 2
    return _run_action(session, "scroll_until",
                       {"direction": args[0], "text": " ".join(args[1:])})


def cmd_collect(session, args):
    if not args:
        print("用法: iphone collect <up|down|left|right>", file=sys.stderr)
        return 2
    return _run_action(session, "collect", {"direction": args[0]})


def cmd_type(session, args):
    """走 Executor 而非 _manual：只有这样才能拿到 `_verify_type` 的落屏校验（spec §10.3）。"""
    from iphone_agent.harness.actions import Action
    from iphone_agent.harness.executor import Executor
    with _perception_scope(session) as mode:
        print(f"# 整屏解析模式 {mode}", file=sys.stderr)
        frame = session.dev.capture(); obs = session.per.observe(frame)
        ex = Executor(session.dev, session.per, asker=session.asker, recovery=_recovery(session))
        try:
            res, _ = ex.run(Action("type", {"text": " ".join(args)}, "manual", None, "m"), obs)
            print(res.to_json())
        finally:
            session.dev.release_all()
    return 0


def cmd_key(session, args):
    if not args:
        print("用法: iphone key <home|switcher|spotlight>", file=sys.stderr)
        return 2
    name = {"switcher": "app_switcher"}.get(args[0], args[0])
    _manual(session, lambda o, f: (session.dev.key(name), "key")[1]); return 0


def _layout(session):
    """CLI 手动命令也查布局表；空或读不出就 None（和主循环同一个入口 Layout.load_or_none）。"""
    from iphone_agent.twin.layout import Layout
    return Layout.load_or_none(session.workspace.twin_device / "layout.json")


def cmd_open(session, args):
    from iphone_agent.harness.actions import Action
    from iphone_agent.harness.executor import Executor
    with _perception_scope(session) as mode:
        print(f"# 整屏解析模式 {mode}", file=sys.stderr)
        frame = session.dev.capture(); obs = session.per.observe(frame)
        ex = Executor(session.dev, session.per, asker=session.asker, recovery=_recovery(session),
                      layout=_layout(session), layout_path=session.workspace.twin_device / "layout.json")
        try:
            res, _ = ex.run(Action("open_app", {"name": " ".join(args)}, "manual", None, "m"), obs)
            print(res.to_json())
        finally:
            session.dev.release_all()
    return 0


def cmd_model(session, args) -> int:
    """`iphone model` 看当前；`iphone model provider:model` 切换。REPL 里是 /model。"""
    if args:
        try:
            session.select_model(args[0])
        except ConfigError as e:
            print(str(e), file=sys.stderr)
            return 2
    try:
        r = session.resolved
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    from iphone_agent.model.describe import describe
    d = describe(r)
    print(f"模型 {d['spec']}  key={d['api_key_source']}  base={d['base_url']}")
    print(f"坐标约定 {d['coord_mode']}  坐标点击={'开' if d['allow_coord_tap'] else '关'}  "
          f"已标定={'是' if d['calibrated'] else '否'}  max_tokens={d['max_tokens']}")
    return 0


END_WORD = {"stopped": "你停下了它", "max_steps": "步数用完了", "timeout": "超时了",
             "no_progress": "它绕不出来了", "interrupted": "被中断了"}


def human_time(ms: float) -> str:
    s = round((ms or 0) / 1000)
    if not s:
        return ""
    return f"{s // 60} 分 {s % 60} 秒" if s >= 60 else f"{s} 秒"


class ChatView:
    """把一次运行渲染成**回答**，不是日志。

    判断跟网页那头是同一套（`web/static/app.js` 的摘要行）：过程收成一行
    「几步 · 用时 ·（几次没反应 / 几次被拒）」，回答是主体，细节留一扇门（replay）。
    成功不加标签 —— 每条前面写「✓ 完成」等于每句话前加一句「我做完了」；
    只有失败才标，那时候标记才重新有信息量。

    ⚠ 这里是**逐行往下滚**，不是就地更新那一行。REPL 底下还有 prompt_toolkit 的
      输入行，抢光标就会把它撞花；而且 `iphone run` 的输出经常被重定向进文件，
      转义序列在那儿是垃圾。
    """

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def on_step(self, rec: dict) -> None:
        a = rec.get("action") or {}
        m = rec.get("model") or {}
        r = rec.get("result") or {}
        self.steps.append({"changed": r.get("changed"), "error": r.get("error"),
                           "exec_ms": rec.get("exec_ms") or 0,
                           "latency_ms": m.get("latency_ms") or 0})
        # 「它当时想的」那一行才是人要看的；动作名和参数是给 replay 的。
        line = f"  {str(rec.get('step', '')):>2} {a.get('name', '?'):<9} {m.get('reason', '')}"
        if r.get("error"):
            line += f"   ← {r['error']}"
        elif r.get("changed") is False:
            line += "   ← 画面没变"
        print(line.rstrip())

    def finish(self, result) -> None:
        nc = sum(1 for s in self.steps if s["changed"] is False and not s["error"])
        er = sum(1 for s in self.steps if s["error"])
        ms = sum(s["exec_ms"] + s["latency_ms"] for s in self.steps)
        bits = [f"{result.steps or len(self.steps)} 步", human_time(ms)]
        if nc:
            bits.append(f"{nc} 次没反应")
        if er:
            bits.append(f"{er} 次被拒")
        print("\n—— " + " · ".join(b for b in bits if b))
        if result.end_reason != "done_success":
            print(END_WORD.get(result.end_reason, "没做成"))
        answer = result.done_result or result.error or ""
        if answer:
            print(answer)
        print(f"（记录 {result.run_dir.name} · `iphone replay {result.run_dir}` 看每一步）")


@contextlib.contextmanager
def stop_on_sigint(session: Session):
    """Ctrl-C = 踩刹车，不是拔钥匙。

    ⚠ 硬砍可能砍在一次注入的**中途** —— 键按下去没松开、页面滑到一半，
      手机停在一个谁也说不清的状态。所以第一次 Ctrl-C 只是举个牌子，
      主循环在每一步开头看见了才收手：从按下到真停下**最多差一步（约十秒）**。
      这句话必须打出来，否则人会以为按钮坏了然后猛按 —— 又回到硬砍。
      第二次按仍然硬中断：牌子举了它还不停时得有出口。
    """
    def handler(signum, frame):
        if session.stop.is_set():
            raise KeyboardInterrupt
        session.stop.set()
        print("\n正在停止 —— 当前这一步做完就收手（最多约十秒）。"
              "再按一次 Ctrl-C 立刻中断，但手机可能停在一个中间态。")

    try:
        old = signal.signal(signal.SIGINT, handler)
    except ValueError:              # 不在主线程就装不上；那就退回默认行为
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)


def run_once(session: Session, task: str, *, max_steps=None, timeout_s=None, on_step=None):
    """跑一轮，并且把这一轮**接进对话**：上文递进去、回答记下来、停止牌一直举着。

    这是 `iphone run` 和 REPL 共用的那一层。网页那头对应的是 `Chat.ask`。
    """
    from iphone_agent.harness.loop import run_task
    from iphone_agent.harness.safety import osascript_confirm, osascript_handover
    session.stop.clear()        # ⚠ 上一轮按过停止，牌子还举着：不清掉下一句话会秒停
    # 命令行是人敲的，算「有人在」：写类动作弹系统对话框问一次（默认取消、超时拒绝）；
    # 模型交接（handover）也弹对话框，人做完点「我做完了」。
    result = run_task(task, session.dev, session.per, session.model, session.workspace.runs,
                      max_steps=max_steps, timeout_s=timeout_s, on_step=on_step,
                      history=session.history, workspace=session.workspace,
                      skill_store=session.skills, should_stop=session.stop.is_set,
                      confirm=osascript_confirm,
                      on_handover=lambda need, reason: osascript_handover(need))
    # 失败和「被你停下」也进历史：「刚才为什么没成」问得出来，才叫对话。
    session.history.append((task, str(result.done_result or result.error or result.end_reason)))
    return result


def chat_turn(session: Session, task: str, *, max_steps=None, timeout_s=None) -> int:
    """一轮对话，从话进来到回答出去。返回退出码。"""
    view = ChatView()
    with stop_on_sigint(session):
        result = run_once(session, task, max_steps=max_steps, timeout_s=timeout_s,
                          on_step=view.on_step)
    view.finish(result)
    if result.error:
        print(f"原因：{result.error}", file=sys.stderr)
    return 0 if result.end_reason == "done_success" else 1


def cmd_run(session, args):
    import argparse
    p = argparse.ArgumentParser(prog="iphone run")
    p.add_argument("task")
    # 默认留 None：由 RunConfig 在**调用时**取，才改得动（docs/20 D8）。
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--model", help="provider:model，如 deepseek:deepseek-v3；覆盖 IPHONE_USE_MODEL 与 config.toml")
    ns = p.parse_args(args)
    if ns.model:
        try:
            session.select_model(ns.model)
        except ConfigError as e:
            print(str(e), file=sys.stderr)
            return 2
    return chat_turn(session, ns.task, max_steps=ns.max_steps, timeout_s=ns.timeout)


def cmd_replay(session, args):
    from iphone_agent.harness.runlog import RunLog
    d = Path(args[0])
    run = json.loads((d / "run.json").read_text())
    print(f"任务：{run['task']}\n模型：{run['model']} {run.get('model_version','')}\n结束：{run['end_reason']} 步数：{run['steps']}\n")
    for s in RunLog.read_steps(d):
        a = s.get("action", {}); m = s.get("model", {}); r = s.get("result", {})
        shown_args = a.get("args", a.get("args_raw", {}))
        if s.get("kind") == "procedure_step":
            mark = "✓" if s.get("expect_ok", True) else "✗"
            print(f"    ↳ {a.get('name')} {json.dumps(shown_args, ensure_ascii=False)}  "
                  f"changed={r.get('changed')} {r.get('error', '')} expect{mark}")
            continue
        print(f"#{s.get('step')} {a.get('name')} {json.dumps(shown_args, ensure_ascii=False)}")
        print(f"    reason: {m.get('reason','')}")
        if m.get("expect"): print(f"    expect: {m['expect']}")
        print(f"    result: changed={r.get('changed')} hamming={r.get('hamming')} {r.get('error','')} {r.get('hint','')}")
    return 0


def cmd_memory(session, args) -> int:
    """看得见、改得掉 —— 这是选文件格式而不是数据库的兑现。

    自动失效做不到的部分，由「你能看见」兜底：模型不能删记忆（一次失败
    不足以否掉一条），删除只有你这一条路径。
    """
    from iphone_agent.harness import recap
    from iphone_agent.memory import MemoryStore
    store = MemoryStore(session.workspace.memory_dir)
    sub = args[0] if args else ""
    if sub == "list":
        entries, broken = store.index()
        for e in entries:
            # 三档标记跟 recap 用同一个函数（mark_for）——CLI 上人看到的
            # 和 recall 里模型看到的必须是同一件事，不能各算各的。
            mark = recap.mark_for(e)
            print(f"{mark} {e.kind:<9} {e.name:<24} {e.description} "
                  f"(用 {e.used_success}/{e.used_failed})")
        print(f"共 {len(entries)} 条" + (f"，另有 {broken} 条损坏" if broken else ""))
        return 0
    if sub == "show" and len(args) == 2:
        body = store.read(args[1])
        if body is None:
            print(f"没有名叫 {args[1]!r} 的记忆", file=sys.stderr)
            return 1
        print(body)
        return 0
    if sub == "rm" and len(args) == 2:
        if not store.move_to_trash(args[1], "手动删除"):
            print(f"没有名叫 {args[1]!r} 的记忆", file=sys.stderr)
            return 1
        print(f"已移入 {store.trash}")
        return 0
    if sub == "prune":
        n = len(list(store.trash.glob("*.md"))) if store.trash.exists() else 0
        for p in store.trash.glob("*.md"):
            p.unlink()
        print(f"清掉 {n} 条")
        return 0
    print("用法: iphone memory <list|show <name>|rm <name>|prune>", file=sys.stderr)
    return 2


def cmd_serve(session, args) -> int:
    """网页对话界面 —— 命令行看不到过程也接不上话。"""
    import argparse
    p = argparse.ArgumentParser(prog="iphone serve")
    p.add_argument("--port", type=int, default=config.WEB_PORT)
    p.add_argument("--model", default=None, help="provider:model")
    ns = p.parse_args(args)
    from iphone_agent.web import serve
    # 不给 --model 就继承本会话当前选的那个：REPL 里 /model 换完再 /serve，
    # 网页却起在默认模型上，还什么都不说 —— 这正是这一版要消灭的那类静默分叉。
    return serve(ns.port, model_spec=ns.model or session.spec)


def cmd_map(session, args) -> int:
    """屏幕图：agent 从既往运行里自己长出来的那张地图。

    看得见很重要 —— 这张图现在会影响模型每一步的判断（每轮观察后面跟着的
    【位置】和【路线】两段就是它），所以你得能查它到底知道些什么。
    """
    from iphone_agent.harness.whereami import _describe
    from iphone_agent.memory.screenmap import build_from_runs, find_nodes, route

    ws = session.workspace
    m = build_from_runs(ws.runs, cache=ws.screenmap_cache)
    nodes = m.stable_nodes()
    sub = args[0] if args else "list"

    if sub == "list":
        print(f"{len(nodes)} 个稳定节点（访问 >= 2 次才有指纹）、{len(m.edges)} 条边\n")
        for n in sorted(nodes, key=lambda n: -n.visits):
            print(f"  #{n.key:<3} 到过 {n.visits:>3} 次  {_describe(m, n.key)}")
        return 0

    if sub == "show" and len(args) == 2:
        try:
            key = int(args[1])
        except ValueError:
            print("用法: iphone map show <节点号>", file=sys.stderr)
            return 2
        n = next((x for x in m.nodes if x.key == key), None)
        if n is None:
            print(f"没有节点 {key}", file=sys.stderr)
            return 1
        print(f"节点 #{n.key}  到过 {n.visits} 次")
        print(f"指纹（{len(n.fingerprint)} 个稳定词）：{'、'.join(sorted(n.fingerprint))}")
        print(f"样本：{'、'.join(n.samples)}")
        out = sorted(m.out_edges(n.key), key=lambda e: -e.count)
        print(f"\n从这儿走出去过 {len(out)} 条：")
        for e in out:
            print(f"  {e.action}「{e.target}」→ #{e.dst}（{_describe(m, e.dst)}）x{e.count}")
        return 0

    if sub == "route" and len(args) >= 2:
        want = " ".join(args[1:])
        hit = find_nodes(m, want)
        if not hit:
            print(f"图上没有含「{want}」的屏")
            return 1
        home = max(nodes, key=lambda n: n.visits) if nodes else None
        for n in hit[:3]:
            print(f"#{n.key}（到过 {n.visits} 次）：{_describe(m, n.key)}")
            if home is not None:
                r = route(m, home.key, n.key)
                path = " → ".join(f"{e.action}「{e.target}」" for e in r) if r else "（从最常去的那屏没有路）"
                print(f"   从 #{home.key} 过去：{path}")
        return 0

    print("用法: iphone map [list | show <节点号> | route <文字>]", file=sys.stderr)
    return 2


def _skill_ref(ref: str) -> tuple[str, str | None]:
    """`app` 或 `app/proc` 拆成 (app, proc|None)。"""
    app, _, proc = ref.partition("/")
    return app, (proc or None)


def _skill_list(store) -> int:
    """App（知识）/ 带剧本的技能 / 纯文字技能，一次性看完（spec §5 的「看得见」）。
    带剧本的技能挂在它 apps[0] 那个 App 下面显示，纯文字的平铺在后面。"""
    from iphone_agent.skills import model as M
    cat = store.load()
    print(f"App {len(cat.apps)} 个、技能 {len(cat.skills)} 个"
          + (f"、损坏 {len(cat.broken)} 个文件" if cat.broken else ""))
    for a in sorted(cat.apps.values(), key=lambda a: a.name):
        marks = ["✓" if a.status == "verified" else "草稿", a.risk]
        if a.name in cat.shadowed:
            marks.append("遮蔽了结构层")
        print(f"  {a.name:<20} {a.display:<10} {' '.join(marks)}")
        for p in cat.procedures_of(a.name):
            st = {"verified": "✓", "draft": "草稿", "stale": "✗失效"}[p.status]
            extra = []
            if p.status == "draft" and p.provenance.verified_count >= 2:
                extra.append("可批准")
            if p.weak_steps:
                extra.append("指纹弱")
            print(f"      {p.name:<28} {st} {p.risk} 走通{p.provenance.verified_count}次 {' '.join(extra)}")
    for s in sorted(cat.skills.values(), key=lambda s: s.name):
        if s.procedure is not None:
            continue        # 上面已经在 App 下面列过了
        computed = M.scenario_risk(s.risk, [p.risk for a in s.apps for p in cat.eligible_procedures([a])], s.body)
        marks = [s.status] + (["风险低报"] if M.RISK_RANK[computed] > M.RISK_RANK[s.risk] else [])
        if s.name in cat.shadowed_scenarios:
            marks.append("遮蔽了结构层")     # 同 App：被遮蔽而人看不见，是最贵的一种静默失败
        print(f"  技能 {s.name:<20} {s.description}  {' '.join(marks)}")
    for path, why in cat.broken:
        print(f"  损坏 {path}: {why}")
    return 0


def _skill_show(store, ref: str) -> int:
    from iphone_agent.skills import model as M
    app, proc = _skill_ref(ref)
    cat = store.load()
    if proc is None:
        a = cat.app(app)
        if a is None:
            print(f"没有 App {app}", file=sys.stderr)
            return 1
        print(f"{a.name} {a.display}  open={a.open} risk={a.risk} status={a.status} updated={a.updated}\n{a.body}")
        print(f"剧本 {len(cat.procedures_of(app))} 条；子图 {'有' if store.read_map(app) else '无'}")
        return 0
    p = cat.procedure(app, proc)
    if p is None:
        print(f"没有剧本 {app}/{proc}", file=sys.stderr)
        return 1
    print(M.procedure_to_json(p))
    return 0


def _skill_approve(store, ref: str) -> int:
    app, proc = _skill_ref(ref)
    if proc is not None:
        p = store.approve_procedure(app, proc)
        print(f"{app}/{proc} → {p.status}")
        return 0
    cat = store.load()
    if app in cat.scenarios:
        store.approve_scenario(app)
        print(f"技能 {app} → manual")
        return 0
    if app in cat.apps:
        store.approve_app(app)
        print(f"App {app} → verified（请确认 APP.md 里的 risk）")
        return 0
    print(f"没有 {app}", file=sys.stderr)
    return 1


def _skill_mv(store, ref: str, new_name: str) -> int:
    app, proc = _skill_ref(ref)
    store.mv_procedure(app, proc, new_name)
    print(f"{app}/{proc} → {app}/{new_name}")
    return 0


def _skill_fork(store, app: str) -> int:
    print(f"已复制到 {store.fork(app)}")
    return 0


def _skill_export(store, app: str, dest: str) -> int:
    print(f"已导出到 {store.export(app, Path(dest))}（进结构层前请人工检查指纹词里有没有私人信息）")
    return 0


def _skill_rm(store, ref: str) -> int:
    app, proc = _skill_ref(ref)
    ok = store.trash_procedure(app, proc, "手动删除") if proc else store.trash_scenario(app, "手动删除")
    if not ok:
        print(f"没有 {ref}", file=sys.stderr)
        return 1
    print(f"已移入 {store.trash}")
    return 0


def _skill_sync(session, store) -> int:
    from iphone_agent.skills.extract import sync
    rep = sync(store, session.workspace.runs)
    print(f"{len(rep['apps'])} 个 App、{rep['maps']} 张子图；草稿：{json.dumps(rep['drafts'], ensure_ascii=False)}")
    return 0


def _skill_run(session, store, ref: str, kv: list[str]) -> int:
    """人在场时手动跑一条剧本 —— 写类剧本唯一的执行入口。走 ProcedureRunner，留档同任务运行。"""
    import time

    from iphone_agent.harness.executor import Executor
    from iphone_agent.harness.guard import ActionGuard
    from iphone_agent.harness.procedure import COUNTED_FAILURES, ProcedureRunner
    from iphone_agent.harness.runlog import RunLog
    app, _, name = ref.partition("/")
    cat = store.load()
    p = cat.procedure(app, name)
    if p is None:
        print(f"没有剧本 {ref}", file=sys.stderr)
        return 1
    kv_args = dict(s.split("=", 1) for s in kv if "=" in s)
    missing = set(p.params) - set(kv_args)
    if missing:
        print(f"缺参数 {sorted(missing)}", file=sys.stderr)
        return 2
    log = RunLog(session.workspace.runs, task=f"skill run {ref}", model="manual", config_snapshot={}, prompt_hash="manual")
    ex = Executor(session.dev, session.per, catalog=cat, asker=session.asker, recovery=_recovery(session))
    runner = ProcedureRunner(ex, ActionGuard(), log, deadline=time.time() + config.TASK_TIMEOUT_S,
                             on_step=lambda rec: print(f"  ↳ {rec['action']['name']} {rec['result'].get('error', 'ok')}"))
    res = None
    with _perception_scope(session) as mode:
        print(f"# 整屏解析模式 {mode}", file=sys.stderr)
        try:
            obs = session.per.observe(session.dev.capture())
            res, _ = runner.run(p, kv_args, obs, "manual", p.tool_name)
            print(res.to_json())
            store.record_run(app, name, ok=res.ok, counted=res.error in COUNTED_FAILURES, run_id=log.dir.name)
        finally:
            session.dev.release_all()
            ok = res is not None and res.ok
            log.finish("done_success" if ok else "done_failed", 1, None, None, "manual", {})
    return 0 if ok else 1


def cmd_skill(session, args) -> int:
    """App / 剧本 / 场景三层知识：看得见、批得了、删得掉（spec §5）。写只写个人层。"""
    from iphone_agent.skills import model as M
    store = session.skills
    sub = args[0] if args else "list"
    usage = ("用法: iphone skill [list | show <app>[/<proc>] | approve <app>[/<proc>] | approve <scenario> | "
             "run <app>/<proc> [k=v …] | mv <app>/<proc> <new> | fork <app> | export <app> <dir> | "
             "rm <app>/<proc>|<scenario> | sync]")
    try:
        if sub == "list":
            return _skill_list(store)
        if sub == "show" and len(args) == 2:
            return _skill_show(store, args[1])
        if sub == "approve" and len(args) == 2:
            return _skill_approve(store, args[1])
        if sub == "mv" and len(args) == 3:
            return _skill_mv(store, args[1], args[2])
        if sub == "fork" and len(args) == 2:
            return _skill_fork(store, args[1])
        if sub == "export" and len(args) == 3:
            return _skill_export(store, args[1], args[2])
        if sub == "rm" and len(args) == 2:
            return _skill_rm(store, args[1])
        if sub == "sync":
            return _skill_sync(session, store)
        if sub == "run" and len(args) >= 2:
            return _skill_run(session, store, args[1], args[2:])
    except M.SkillError as e:
        print(f"{e.code}: {e.message}", file=sys.stderr)
        return 1
    print(usage, file=sys.stderr)
    return 2


def cmd_twin(session, args) -> int:
    """设备层孪生：主屏布局表。见 docs/32 §1.5。"""
    from iphone_agent.twin.layout import Layout
    usage = ("用法: iphone twin <scan|show|rebuild|report|bench>\n"
             "  scan     回主屏第一页，一页页往右翻到底，把每一页的 App 位置按页序写进布局表（只看不点）\n"
             "  show     打印布局表\n"
             "  rebuild  从全部留档重建 App 层孪生（屏文件）\n"
             "  report   离线报告：每个 App 认出了多少屏、五态占比\n"
             "  bench    合入闸门：错合、错归必须为 0（标注在 evalset/twin/）")
    what = args[0] if args else ""
    path = session.workspace.twin_device / "layout.json"
    if what == "scan":
        from iphone_agent.twin.scan import scan_home
        try:
            lay = scan_home(session.dev, session.per, path)
        finally:
            session.dev.release_all()
        if lay is None:
            return 1
        n = sum(len(p.cells) for p in lay.pages)
        print(f"布局表 {path}：{len(lay.pages)} 页，{n} 个格子")
        return 0
    if what == "show":
        lay = Layout.load_or_none(path)
        if lay is None:
            print(f"布局表为空（{path}）。先跑 iphone twin scan。")
            return 0
        for p in sorted(lay.pages, key=lambda p: p.order):
            print(f"第 {p.order} 页（{'完整' if p.complete else '不完整'}，{p.scanned}）")
            rows: dict[int, list[str]] = {}
            for c in p.cells:
                rows.setdefault(c.row, []).append(f"{c.label or '?'}" if not c.unknown else "?")
            for r in sorted(rows):
                print("  " + " | ".join(rows[r]))
        return 0
    if what == "rebuild":
        from iphone_agent.twin.record import LockTimeout, rebuild
        try:
            st = rebuild(session.workspace)
        except LockTimeout:
            print("孪生正被另一个任务写入，稍后再试。", file=sys.stderr)
            return 1
        print(f"重建完成：{st.runs} 个运行，建屏 {st.screens_created}，转正 {st.screens_confirmed}，"
              f"转移 {st.transitions}；撞车 {st.collisions}，坏文件 {st.corrupt}，孤儿 {st.orphans}")
        return 0
    if what == "report":
        from iphone_agent.twin.report import report_text
        print(report_text(session.workspace))
        return 0
    if what == "bench":
        from iphone_agent.twin import bench
        from iphone_agent.twin.record import simulate
        # 标注和其他评测集一样按当前目录找，留档按工作区找
        pairs, owners = bench.load_labels(Path("evalset") / "twin")
        if not pairs and not owners:
            print("evalset/twin/ 下没有标注（pairs.json / owners.json）。", file=sys.stderr)
            return 2
        r = bench.run_bench(simulate(session.workspace), pairs, owners)
        print(bench.format_result(r))
        return 0 if r.passed else 1
    print(usage, file=sys.stderr)
    return 2


def cmd_eval(session: Session, args: list[str]) -> int:
    """离线回放评测：拿历史运行当尺子。见 iphone_agent/eval/replay.py 的模块注释。"""
    from iphone_agent.eval import replay
    usage = ("用法: iphone eval <bench|diff|tasks|tasks-diff|verify|curve|ab|elements|effect> …\n"
             "  bench [--label-agree] [--label-agree-limit N]  跑评测集（evalset/），存 results/<ts>.json，打印摘要。"
             "几秒，不调模型；--label-agree 另比两种标注（第一次要真调视觉）；"
             "--label-agree-limit N 把 label_agree 限到最多 N 帧（按 App 分层轮询，确定性）\n"
             "  diff <a> <b>     两次 bench 结果对比，列出翻转的样本\n"
             "  tasks [--n 3] [--only <id>]  真机上每题跑 n 次再判「办成了没」，存 results/tasks-<ts>.json\n"
             "  tasks-diff <a> <b>  两次 tasks 结果逐题对比，列出翻转的题\n"
             "  verify <id> <run_dir>  离线：拿一次历史运行对着某道题判\n"
             "  curve [--reps 5] [--only <id>]  学习曲线：开孪生提示 vs 无位置提示，ABBA 交替，存 results/curve-<ts>.json\n"
             "  ab [--reps 3] [--only <id>]  按需看图 A/B：always 对 on_demand，A B B A A B，硬闸题多一层无坐标，存 results/ab-<ts>.json（要人在场）\n"
             "  elements  同一张历史截图，纯 OCR 与加上屏幕解析各认出多少元素\n"
             "  effect    当时判成「没有变化」的步，让看图的那一方重判一次\n"
             "  --limit N 只跑前 N 步（后两项都要真调模型，先用小样本试花销）\n"
             "  --no-vision 完全不调模型，只验证纯 OCR 那一路没被改坏")
    what = args[0] if args else ""
    if what == "ab":
        from iphone_agent.eval import ab as AB
        root = Path("evalset")
        regular, hard, errors = AB.load_ab_tasks(root)
        for e in errors:
            print(f"⚠ 任务文件有问题，已跳过：{e}", file=sys.stderr)
        not_ready = [msg for t in hard if (msg := AB.ready(t))]
        if not_ready:
            print("硬闸题还不能跑（spec 2026-09-14 §10.3）：" + "；".join(not_ready), file=sys.stderr)
            return 2
        reps = 3
        if "--reps" in args:
            i = args.index("--reps")
            if i + 1 >= len(args) or not args[i + 1].isdigit():
                print("--reps 后面要跟一个数字", file=sys.stderr)
                return 2
            reps = int(args[i + 1])
        if "--only" in args:
            i = args.index("--only")
            keep = args[i + 1] if i + 1 < len(args) else ""
            regular = [t for t in regular if t.id == keep]
            hard = [t for t in hard if t.id == keep]
        if not regular and not hard:
            print("没有可跑的题", file=sys.stderr)
            return 2
        ws_root = root / "ab-ws" / time.strftime("%Y%m%d-%H%M%S")
        res = AB.run_ab(session, regular, hard, reps, ws_root)
        path = AB.save(res, root / "results")
        for line in AB.table(res):
            print(line)
        print(f"# 存到 {path}；两组工作区在 {ws_root}（on = always，off = on_demand）")
        return 0
    if what == "curve":
        from iphone_agent.eval import curve as C
        from iphone_agent.eval import verify as V
        root = Path("evalset")
        tasks, errors = V.load_tasks(root / "tasks" / "curve")
        for e in errors:
            print(f"⚠ 任务文件有问题，已跳过：{e}", file=sys.stderr)
        reps = 5
        if "--reps" in args:
            i = args.index("--reps")
            if i + 1 >= len(args) or not args[i + 1].isdigit():
                print("--reps 后面要跟一个数字", file=sys.stderr)
                return 2
            reps = int(args[i + 1])
        if "--only" in args:
            i = args.index("--only")
            tasks = [t for t in tasks if i + 1 < len(args) and t.id == args[i + 1]]
        if not tasks:
            print("没有可跑的题（evalset/tasks/curve/*.json）", file=sys.stderr)
            return 2
        ws_root = root / "curve-ws" / time.strftime("%Y%m%d-%H%M%S")
        res = C.run_curve(session, tasks, reps, ws_root)
        path = C.save(res, root / "results")
        for line in C.table(res):
            print(line)
        print(f"# 存到 {path}；两组工作区在 {ws_root}")
        return 0
    if what in ("tasks", "tasks-diff", "verify"):
        from iphone_agent.eval import verify as V
        root = Path("evalset")
        if what == "tasks-diff":
            if len(args) < 3:
                print("用法: iphone eval tasks-diff <a.json> <b.json>", file=sys.stderr)
                return 2
            a = json.loads(Path(args[1]).read_text(encoding="utf-8"))
            b = json.loads(Path(args[2]).read_text(encoding="utf-8"))
            for line in V.diff(a, b):
                print(line)
            return 0
        tasks, errors = V.load_tasks(root / "tasks")
        for e in errors:
            print(f"⚠ 任务文件有问题，已跳过：{e}", file=sys.stderr)
        if what == "verify":
            if len(args) < 3:
                print("用法: iphone eval verify <task-id> <run_dir>", file=sys.stderr)
                return 2
            t = next((t for t in tasks if t.id == args[1]), None)
            if t is None:
                print(f"没有这道题：{args[1]}（有：{[t.id for t in tasks]}）", file=sys.stderr)
                return 2
            v = V.verify_run(t, Path(args[2]))
            print(f"{v.status}  {v.run}  安全违规尝试 {v.safety_attempts}")
            for c in v.checks:
                # ok=None：坐标点击反查不到元素，待人工审查——不算失败，不打 ✗（m9 终审）
                mark = "待审" if c["ok"] is None else ("✓" if c["ok"] else "✗")
                print(f"  {mark} {c['type']}  {c['why']}")
            for r in v.reasons:
                print(f"  · {r}")
            return 0 if v.status in ("pass", "review") else 1
        n = 3
        if "--n" in args:
            i = args.index("--n")
            if i + 1 >= len(args) or not args[i + 1].isdigit():
                print("--n 后面要跟一个数字", file=sys.stderr)
                return 2
            n = int(args[i + 1])
        if "--only" in args:
            i = args.index("--only")
            tasks = [t for t in tasks if i + 1 < len(args) and t.id == args[i + 1]]
        if not tasks:
            print("没有可跑的任务（evalset/tasks/*.json）", file=sys.stderr)
            return 2
        res = V.run_tasks(session, tasks, n=n)
        path = V.save(res, root)
        for line in V.summary(res):
            print(line)
        print(f"# 存到 {path}")
        return 0
    if what == "bench":
        from iphone_agent.eval import bench as B
        root = Path("evalset")
        asker = session.asker
        if asker is not None:
            asker.cache_dir = root / "vision-cache"    # 视觉回复缓存在这，bench 才能天天跑
        res = B.bench(root, session.per)
        if "--label-agree" in args:
            if asker is None:
                print("没有配看图的模型，label_agree 跑不了", file=sys.stderr)
                return 2
            from iphone_agent.twin.context import ScreenIdentityContext
            from iphone_agent.twin.live import LiveTwin
            from iphone_agent.twin.record import known_apps
            limit = None
            if "--label-agree-limit" in args:
                i = args.index("--label-agree-limit")
                if i + 1 >= len(args) or not args[i + 1].isdigit():
                    print("--label-agree-limit 后面要跟一个数字", file=sys.stderr)
                    return 2
                limit = int(args[i + 1])
            ws = session.workspace
            # 用当前工作区的孪生挑真实候选（只读；LiveTwin 不写盘）。先记下孪生的版本再跑。
            rev = B.twin_fingerprint(ws.twin_apps)
            ctx = ScreenIdentityContext(LiveTwin(ws.twin_apps, "bench", known_apps=known_apps(ws)))
            res["label_agree"] = B.label_agree(root, session.per, ctx, rev, limit=limit)
        path = B.save(res, root)
        for line in B.summary(res):
            print(line)
        # 每个指标最多列 3 条错例，看个方向；全部在 results 文件里
        for k, v in res["metrics"].items():
            for x in v["misses"][:3]:
                print(f"    · {k}: {x}")
        print(f"# 存到 {path}")
        return 0
    if what == "diff":
        from iphone_agent.eval import bench as B
        if len(args) < 3:
            print("用法: iphone eval diff <a.json> <b.json>", file=sys.stderr)
            return 2
        a = json.loads(Path(args[1]).read_text(encoding="utf-8"))
        b = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        for line in B.diff(a, b):
            print(line)
        return 0
    if what not in ("elements", "effect"):
        print(usage, file=sys.stderr)
        return 2
    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        if i + 1 >= len(args) or not args[i + 1].isdigit():
            print("--limit 后面要跟一个数字", file=sys.stderr)
            return 2
        limit = int(args[i + 1])
    asker = None if "--no-vision" in args else session.asker
    runs_root = session.workspace.runs
    if not runs_root.exists():
        print(f"没有 {runs_root}，先跑几个任务再来评", file=sys.stderr)
        return 2
    if what == "elements":
        steps = replay.iter_steps(runs_root, limit)
        print(f"# {len(steps)} 步可评" + ("（--no-vision：只有 OCR）" if asker is None else ""))
        rep = replay.eval_elements(steps, asker)
    else:
        # ⚠ 先全量扫再挑，不能把 limit 交给 iter_steps —— 那样 limit 会砍在
        #   「所有步」上，挑完可能一步不剩。limit 要砍在**挑出来的那些**上面。
        pool = replay.pick_no_change(replay.iter_steps(runs_root, need_after=True))
        steps = pool[:limit] if limit else pool
        print(f"# {len(pool)} 步当时判成没变化，这次评 {len(steps)} 步")
        rep = replay.eval_effect(steps, asker)
    for line in rep.lines():
        print(line)
    if asker is not None:
        print(f"# 视觉调用 {asker.calls} 次，失败 {asker.failures} 次")
    return 0


COMMANDS = {"doctor": cmd_doctor, "eval": cmd_eval, "screen": cmd_screen, "tap": cmd_tap, "scroll": cmd_scroll,
            "scroll_until": cmd_scroll_until, "collect": cmd_collect,
            "type": cmd_type, "key": cmd_key, "open": cmd_open,
            "model": cmd_model, "run": cmd_run, "replay": cmd_replay, "memory": cmd_memory,
            "map": cmd_map, "serve": cmd_serve, "skill": cmd_skill, "twin": cmd_twin}


# REPL 的 /help。⚠ 它就写在 COMMANDS 边上，因为它说的就是 COMMANDS ——
# 之前它是 repl.py 里的一个字符串，命令加了三个（memory / map / skill）没人想起来改它，
# 于是那三个「一直能用、但没人知道」。test_cli_chat.py 里有一条用例钉着这件事。
REPL_HELP = """整行输入 = 一个任务，接得上上一句。Ctrl-C 踩刹车，Ctrl-D 退出。
  看      /screen 抓一帧标好编号  /show 打开那张图  /doctor 权限与连接自检
  动      /tap <编号|x y>  /type <文字>  /key <home|switcher|spotlight>  /open <app>
          /scroll <up|down|left|right> [page|half]  /scroll_until <方向> <文字>  /collect <方向>
  回看    /replay <运行目录>  /eval <elements|effect> 拿历史运行量一次改动好了多少
  它记得  /memory 跨任务记忆  /map 它自己画的屏幕图  /skill 知识（写类剧本在这儿批准）  /twin <scan|show> 数字孪生设备层
  模型    /model [provider:model]
  /help  /quit"""


def dispatch(session: Session, argv: list[str], *, in_repl: bool = False) -> int:
    if not argv or argv[0] not in COMMANDS:
        print("用法: iphone <" + "|".join(COMMANDS) + "> ...", file=sys.stderr)
        return 2
    if in_repl and argv[0] == "serve":
        # serve_forever 不返回：在 REPL 里敲它，这个控制台就再也回不来了。
        print("在 REPL 里起服务会把这个控制台占死。另开一个终端跑 `iphone serve`。",
              file=sys.stderr)
        return 2
    try:
        return COMMANDS[argv[0]](session, argv[1:])
    except ConfigError as e:
        # 配置错误是给人看的一句话，不是给人看的一坨堆栈。screen / tap 这些不碰模型的命令
        # 现在也会读 config.toml（要 coord_mode），644 的文件、不认识的 provider 都从这儿出来。
        print(str(e), file=sys.stderr)
        return 2
