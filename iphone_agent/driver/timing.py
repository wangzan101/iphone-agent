"""时序表。数值先估后标（spec §4.5、§11）。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Timing:
    min_wait_ms: int      # 动作后最短等待，再开始稳定轮询
    poll_ms: int          # 轮询间隔
    stable_span_ms: int   # 连续多久相邻帧都稳定才算稳定
    settle_max_ms: int    # 封顶


IOS_TIMING: dict[str, Timing] = {
    # settle_max 从 2500 提到 4500：2026-09-07 实测，主屏幕上 tap 打开 App 那一步
    # settled=False、耗时 3035ms 撞到了上限，观察落在动画中途。tap 也用于页内导航
    # （那种一秒内就稳），而 settle 一旦稳定就立即返回，所以放宽上限只在真的没稳时多等。
    "tap":      Timing(300, 150, 400, 4500),
    "scroll":   Timing(400, 150, 500, 2500),
    "type":     Timing(300, 150, 400, 2500),
    # 退格：跟打字同量级，屏幕上变的就是那几个字符
    "erase":    Timing(300, 150, 400, 2500),
    "key":      Timing(500, 150, 500, 3000),
    # 切输入法：屏幕上通常只有候选栏那一条会变，和 key 同量级就够
    "switch_ime": Timing(500, 150, 500, 3000),
    "open_app": Timing(1500, 200, 600, 5000),
    "wait":     Timing(0, 150, 300, 1000),
    "observe":  Timing(0, 0, 0, 0),
}
# 手势参数（待标定）
# 滚轮行数，待标定。滚轮是镜像里唯一能做出 iOS 滑动的通道（拖拽只会变成长按+拖物体）。
SCROLL_LINES = {"page": 30, "half": 15}
# 横向翻一页（主屏幕分页、轮播）用的行数，待标定
SCROLL_LINES_H = {"page": 10, "half": 5}
