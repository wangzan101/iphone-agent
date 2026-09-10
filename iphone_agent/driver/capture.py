"""按 window id 抓帧（spec §4.2）。走 `screencapture` 子进程。

## 为什么不用 CGWindowListCreateImage（2026-09-05 真机结论，推翻了当初的 R6 裁决）

第一阶段最初选的是 Quartz 的 `CGWindowListCreateImage`：进程内调用、无子进程开销。
spike 阶段它返回空图，当时归因为录屏权限未授予 —— 这个归因是对的，授权后它连续几小时
正常工作（644x1436、倍率 2.0、BGRA 解码正确、中文 OCR 正常）。

但在同一台机器、权限齐全的情况下，它之后变成**稳定地挂起 30 秒然后返回空图**，
抓镜像窗口、抓别的 App 的窗口、全屏抓，无一例外；同一个 shell 里 spawn 的
`screencapture` 始终 150ms 出图。该 API 自 macOS 14 起标记废弃。

关键在于**这个挂起无法设上限**：它阻塞在 mach_msg 里，SIGALRM 只能等调用返回后
才被 Python 转成异常，实测设 2 秒和设 10 秒都是 30 秒才回来。而 settle 每个动作要抓
好几帧，一次 30 秒的挂起就能吃掉整个任务预算（spec 的总时限是 10 分钟）。

一个无法设置上限的调用不能放在 agent 主循环里，所以哪怕它大多数时候更快，也不留。

## screencapture 的代价与取舍

- 每次一个子进程，实测 133–186ms（vs 快路径正常时的几十毫秒）。
- 这个开销可以接受：时序表里每个动作后本来就至少等 300ms，settle 的轮询间隔是 150ms，
  抓帧耗时和轮询间隔同量级。标定 poll_ms 时把这一点算进去。
- `-x` 不播快门声，`-o` 不要窗口阴影（阴影会让图比窗口大一圈，破坏坐标换算）。
- 输出是 Retina 像素。Frame.scale 一律由实际图像宽度除以窗口宽度算出，不写死倍率。
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time

from PIL import Image

from iphone_agent import config
from iphone_agent.driver.geometry import Frame
from iphone_agent.driver.window import MirrorWindow, current_rect


class CaptureFailed(RuntimeError):
    pass


def _screencapture(window_id: int, timeout_s: float) -> Image.Image | None:
    """抓一个窗口。失败返回 None，由调用方决定怎么报。"""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        r = subprocess.run(["screencapture", "-x", "-o", "-l", str(window_id), path],
                           capture_output=True, timeout=timeout_s)
        if r.returncode != 0 or not os.path.getsize(path):
            return None
        with Image.open(path) as im:
            return im.convert("RGB")
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def capture_window(win: MirrorWindow, frame_id: int,
                   timeout_s: float | None = None) -> Frame:
    timeout_s = config.OP_TIMEOUT_S if timeout_s is None else timeout_s
    rect = current_rect(win.window_id)
    img = _screencapture(win.window_id, timeout_s)
    if img is None:
        raise CaptureFailed(
            f"抓帧失败（window {win.window_id}，超时 {timeout_s}s）。"
            "检查：录屏权限、镜像是否已连接、窗口是否在别的 Space 或已最小化。")
    return Frame(image=img, width_px=img.width, height_px=img.height, window_rect=rect,
                 ts=time.time(), frame_id=frame_id)
