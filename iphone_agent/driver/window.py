"""按镜像 App 的 PID 与竖屏比例选窗（spec §4.1）。"""
from __future__ import annotations

import time
from dataclasses import dataclass

import Quartz
from AppKit import NSDate, NSRunLoop, NSRunningApplication

from iphone_agent.driver.geometry import Rect

MIRROR_BUNDLE_ID = "com.apple.ScreenContinuity"  # iPhone 镜像；doctor 打印实际匹配到的 App 名以核对
# 窗口靠**纵横比**认，不认具体尺寸 —— 换电脑、拉大窗口都不受影响。
# ⚠ 下界必须能容下带 Home 键的机型：iPhone SE / 8 是 375x667 = **1.78**，
#   原来的 1.8 会把它们整个筛掉。上界 2.3 容得下 iPhone X 之后的 2.16-2.22。
#   取 1.7 留一点余量，同时离常见的横向窗口（比例 < 1）足够远。
ASPECT_MIN, ASPECT_MAX = 1.7, 2.3


class WindowNotFound(RuntimeError):
    pass


@dataclass(frozen=True)
class MirrorWindow:
    window_id: int
    pid: int
    rect: Rect


def _mirror_pids() -> list[int]:
    """镜像 App 的进程号。

    ⚠ 读之前必须泵一次 run loop。NSRunningApplication 的进程表是通知驱动的，
    而我们是个没有事件循环的命令行进程 —— 不泵就可能读到陈旧结果。
    2026-09-07 真机踩到：App 明明在跑（pid 前后一致、窗口一直在），
    这个查询却返回空列表，于是报「镜像未启动」。
    同一个根因之前已经在 NSWorkspace.frontmostApplication 上咬过一次
    （见 injector._frontmost_pid），所以这里一并按同样的方式处理。
    """
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.02))
    apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(MIRROR_BUNDLE_ID)
    return [int(a.processIdentifier()) for a in apps]


def _pids_owning_windows() -> set[int]:
    """从窗口列表反推「有窗口的进程」。

    窗口列表是每次都走系统调用的新鲜数据，不会陈旧 —— 拿它给上面那个查询兜底。
    """
    return {int(w.get("kCGWindowOwnerPID", -1)) for w in _window_list()}


def _window_list() -> list[dict]:
    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    return list(Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or [])


def _rect_of(info: dict) -> Rect:
    b = info["kCGWindowBounds"]
    return Rect(float(b["X"]), float(b["Y"]), float(b["Width"]), float(b["Height"]))


def find_mirror_window(retries: int = 3, gap_s: float = 0.35) -> MirrorWindow:
    """⚠ 短重试，因为窗口列表是**快照**，会瞬时读不到。

    2026-09-08 实测：连着跑探针时 `WindowNotFound` 会随机冒出来，
    可窗口其实一直在（隔一秒再查就是 312x694）。
    这和文件开头那条「进程表可能是陈旧的空结果」是同一类问题。

    重试是**只读**的，没有副作用，所以可以放心重试；三次仍然找不到才是真没有。
    """
    last: Exception | None = None
    for i in range(max(1, retries)):
        try:
            return _find_mirror_window_once()
        except WindowNotFound as e:
            last = e
            if i < retries - 1:
                time.sleep(gap_s)
    raise last if last else WindowNotFound("未找到镜像窗口")


def _find_mirror_window_once() -> MirrorWindow:
    pids = _mirror_pids()
    if not pids:
        # 兜底：进程表可能是陈旧的空结果，用窗口列表里 owner 名带 iPhone 的进程再试一次。
        pids = [int(w["kCGWindowOwnerPID"]) for w in _window_list()
                if "iPhone" in str(w.get("kCGWindowOwnerName", ""))]
        pids = sorted(set(pids))
    if not pids:
        raise WindowNotFound(f"未找到运行中的 {MIRROR_BUNDLE_ID}（iPhone 镜像未启动？）")
    candidates = []
    for info in _window_list():
        if int(info.get("kCGWindowOwnerPID", -1)) not in pids:
            continue
        if int(info.get("kCGWindowLayer", 0)) != 0:
            continue
        r = _rect_of(info)
        if r.w < 100 or r.h < 100:
            continue
        if not (ASPECT_MIN <= r.h / r.w <= ASPECT_MAX):
            continue
        candidates.append(MirrorWindow(int(info["kCGWindowNumber"]), int(info["kCGWindowOwnerPID"]), r))
    if not candidates:
        raise WindowNotFound("镜像 App 在运行，但没有竖屏比例的层 0 窗口（未连接？也可能窗口在别的 Space 或已最小化）")
    if len(candidates) > 1:
        raise WindowNotFound(f"找到多个候选窗口：{candidates}")
    return candidates[0]


def current_rect(window_id: int) -> Rect:
    for info in _window_list():
        if int(info.get("kCGWindowNumber", -1)) == window_id:
            return _rect_of(info)
    raise WindowNotFound(f"窗口 {window_id} 已消失")
