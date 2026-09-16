"""按运行阶段计时（spec 2026-09-14 §8.2）。

每轮 while 迭代一个 StepTimer，放在 contextvar 里；各阶段只在唯一的挂点上累加（CLAUDE.md §7）：
    capture  Device.capture        settle  harness.settle.settle（含它内部的截图）
    ocr / parse / label / zoom     Perceiver 里对应的调用（parse 包含兜底）
    judge    harness.judge._ask（四个 judge 函数都经过它）
阶段可以嵌套（settle 里有 capture），各记各的；同名嵌套不重复计。没有计时器时 phase 什么都不做 ——
CLI、bench、测试里的零散调用不受影响。contextvar 按线程隔离：serve 里每个任务在自己的线程里计。
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar

PHASES = ("capture", "settle", "ocr", "parse", "label", "zoom", "judge")
WAITS = ("pause", "confirm", "handover")


class StepTimer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.phases: dict[str, dict[str, int]] = {p: {"ms": 0, "n": 0} for p in PHASES}
        self.wait_s: dict[str, float] = dict.fromkeys(WAITS, 0.0)
        self.model_ms = 0
        self.exec_ms = 0
        self.parse_by = {"model": 0, "always": 0, "fallback": 0}
        self.model_called = False      # 这一轮调过模型没有：没调就结束的轮不产出 timing
        self.stamped = False           # 已经挂到某条记录上了：一轮只挂一次
        self._active: list[str] = []

    def add(self, name: str, ms: int) -> None:
        d = self.phases.setdefault(name, {"ms": 0, "n": 0})
        d["ms"] += ms
        d["n"] += 1

    def wait(self, kind: str, seconds: float) -> None:
        """等人的时间（暂停 / 确认 / 交接）：从 step_ms 里扣掉。"""
        self.wait_s[kind] = self.wait_s.get(kind, 0.0) + max(0.0, seconds)

    def result(self) -> dict:
        waits = {k: int(v * 1000) for k, v in self.wait_s.items()}
        wall = int((time.perf_counter() - self.t0) * 1000)
        return {"step_ms": max(0, wall - sum(waits.values())), "model_ms": self.model_ms,
                "exec_ms": self.exec_ms, "wait_ms": waits,
                "phases": {k: dict(v) for k, v in self.phases.items()}, "parse_by": dict(self.parse_by)}


_current: ContextVar[StepTimer | None] = ContextVar("iphone_step_timer", default=None)


def open_step() -> StepTimer:
    t = StepTimer()
    _current.set(t)
    return t


def close_step() -> None:
    _current.set(None)


def current() -> StepTimer | None:
    return _current.get()


@contextmanager
def phase(name: str):
    t = _current.get()
    if t is None or name in t._active:
        yield
        return
    t._active.append(name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t._active.remove(name)
        t.add(name, int((time.perf_counter() - t0) * 1000))


def note_parse(reason: str) -> None:
    t = _current.get()
    if t is not None and reason in t.parse_by:
        t.parse_by[reason] += 1
