"""执行器健康：动作验证失败 → 恢复阶梯 → 重做。对模型透明。

## 这是哪一类问题

四个执行器（tap / scroll / type / key）各自都会整体失效：点击死（09-08）、打字死（09-08、09-10）、
连接暂停、窗口缩条、镜像重启后 pid 过期。以前每种各打一个补丁，其中三种是**把设备故障交给模型**
（「先 tap 输入框」「换目标没有用」）—— 模型解决不了设备故障，只会烧步数：09-10 凌晨两次任务
33 步、15 步全废在打字死掉之后。

## 原则

模型只看到两种结果：动作成功了，或者设备确定坏了（任务以 device_error 结束，附一句人能读懂的话）。
中间所有「救一救」在执行层内部做完，模型不知道发生过。设备的事是系统的事；
人只管付钱和发出去的内容。

## 阶梯（按代价排，不按症状排）

    ① resolve   release_all、重新解析窗口 / pid（镜像重启过、pid 过期 —— 一秒修好）
    ② connect   认「连接暂停 / 使用中」，点「继续」，等握手（reconnect.ensure_connected）
    ③ restart   退出镜像 App → 重启 → 等窗口 → 等握手（苹果自己的重连机制）
    每一级之后 redo() 一次；成功即止。全失败 → DeviceChannelDead。

打字死和点击死走同一个梯子，只是入口不同。将来屏幕键盘做出来，就是 type 在 ③ 之前多一级
「换通道」，梯子本身不变。每次爬梯都留痕（events），「打字死的频率」「重启平均几秒」才能量出来。
"""
from __future__ import annotations

import time
from collections.abc import Callable

from iphone_agent import config


class DeviceChannelDead(RuntimeError):
    """恢复阶梯爬完了还是不行。loop 把它记成 device_error 结束任务。"""

    def __init__(self, channel: str, tried: list[dict]):
        self.channel = channel
        self.tried = tried
        rungs = " → ".join(t["rung"] for t in tried) or "（一级都没试）"
        super().__init__(f"{channel} 通道失效，恢复阶梯全部试过（{rungs}）仍不行；请检查镜像连接")


CHANNEL_SAYS = {"keyboard": "打字", "tap": "点击"}


class Recovery:
    def __init__(self, dev, per, *, restart: Callable[[], float] | None = None,
                 ensure: Callable | None = None, max_restarts: int | None = None):
        self.dev = dev
        self.per = per
        self._restart = restart
        self._ensure = ensure
        self.max_restarts = config.RECOVERY_MAX_RESTARTS if max_restarts is None else max_restarts
        self.restarts = 0
        self.events: list[dict] = []          # 每次爬梯的每一级，loop 抄进 run 记录

    # ---- 三级 ----
    def _rung_resolve(self) -> str:
        self.dev.release_all()
        resolve = getattr(self.dev, "_resolve", None)
        if resolve is None:
            return "no-op"
        before = getattr(self.dev, "win", None)
        win = resolve()
        changed = before is not None and (getattr(win, "pid", None) != getattr(before, "pid", None)
                                          or getattr(win, "window_id", None) != getattr(before, "window_id", None))
        return "窗口/pid 变了，已重新解析" if changed else "窗口没变"

    def _rung_connect(self) -> str:
        ensure = self._ensure
        if ensure is None:
            from iphone_agent.harness.reconnect import ensure_connected
            ensure = ensure_connected
        ok, why, _ = ensure(self.dev, self.per)
        return why if ok else f"未恢复：{why}"

    def _rung_restart(self) -> str:
        if self.restarts >= self.max_restarts:
            raise _Skip(f"本次任务已重启镜像 {self.restarts} 次，不再重启")
        restart = self._restart
        if restart is None:
            from iphone_agent.driver.session import restart_mirror
            restart = restart_mirror
        self.restarts += 1
        dt = restart()
        why = self._rung_connect()
        return f"镜像重启，窗口 {dt:.1f}s 回来；{why}"

    RUNGS = ("resolve", "connect", "restart")

    def climb(self, channel: str, redo: Callable[[], bool] | None = None) -> bool:
        """一级一级救。redo 给了就每级之后重做一次，成功即返回 True；
        redo 没给（点击那种没法重做的），只要某一级**真的做了事**就返回 True，让调用方清计数再看。
        全部失败抛 DeviceChannelDead —— 不返回 False，免得调用方忘了处理。"""
        tried: list[dict] = []
        for rung in self.RUNGS:
            t0 = time.time()
            try:
                detail = getattr(self, f"_rung_{rung}")()
                skipped = False
            except _Skip as e:
                detail, skipped = str(e), True
            except Exception as e:                 # noqa: BLE001 —— 一级炸了不该拦住下一级
                detail, skipped = f"{type(e).__name__}: {e}", True
            ev = {"channel": channel, "rung": rung, "detail": detail,
                  "ms": int((time.time() - t0) * 1000), "skipped": skipped}
            if skipped:
                ev["ok"] = False
                tried.append(ev); self.events.append(ev)
                continue
            if redo is not None:
                ok = bool(redo())
            else:
                ok = rung == "restart" or ("变了" in detail or "恢复" in detail)
            ev["ok"] = ok
            tried.append(ev); self.events.append(ev)
            if ok:
                return True
        raise DeviceChannelDead(channel, tried)

    def drain(self) -> list[dict]:
        out, self.events = self.events, []
        return out


class _Skip(Exception):
    pass
