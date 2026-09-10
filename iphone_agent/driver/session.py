"""镜像会话的插页：连接暂停 / 手机被拿起。

镜像不是一直在的。两种插页会挡在手机画面前面：

    「连接暂停」+ 一个「继续」按钮        —— 超时导致，点一下就回来
    「iPhone 使用中」+「锁定 iPhone 以连接」 —— 手机被拿起了，**只能人去解决**

第一种以前是不处理的（spec §1 明确排除）。但要无人值守跑任务，它必须能自愈 ——
否则一次超时就把整晚废掉。第二种仍然处理不了，那是物理世界的事，只能如实报告。

这里只做**判定**，是纯函数：给一组屏幕文字，说这是哪种插页。
恢复动作要同时用到 driver 和 perceive，按分层规矩放在 harness（`harness/reconnect.py`）。
"""
from __future__ import annotations

from collections.abc import Iterable

PAUSED = "paused"    # 「连接暂停」—— 点「继续」能回来
IN_USE = "in_use"    # 手机被拿起 —— 人的事

# ⚠ 只用**插页独有**的词。「继续」两个字在很多 App 里都有（引导页、协议页），
#   单看它会把正常画面误判成断线，然后去点一个不该点的按钮。
#   所以判定必须以「连接暂停」这种插页专有的标题为准，「继续」只用来找按钮。
_PAUSED_TITLES = ("连接暂停",)
_IN_USE_TITLES = ("iPhone 使用中", "iPhone使用中", "锁定 iPhone", "锁定iPhone")

RESUME_BUTTON = "继续"


def _hit(texts: Iterable[str], needles: tuple[str, ...]) -> bool:
    joined = "".join(t.replace(" ", "") for t in texts)
    return any(n.replace(" ", "") in joined for n in needles)


def detect(texts: Iterable[str]) -> str | None:
    """屏幕文字 → 插页类型；正常画面返回 None。

    先判「使用中」：手机被拿起时也可能同时出现别的字样，而这一种恢复不了，
    早点认出来比误判成 paused 然后徒劳地点半天强。
    """
    texts = list(texts)
    if _hit(texts, _IN_USE_TITLES):
        return IN_USE
    if _hit(texts, _PAUSED_TITLES):
        return PAUSED
    return None


MIRROR_BUNDLE = "com.apple.ScreenContinuity"


def reopen_mirror_window(timeout_s: float = 6.0) -> bool:
    """把缩成小条的镜像窗口叫回来。

    ⚠ 2026-09-08 实测：用 osascript 激活别的 App（`tell application "Finder" to activate`）
    之后，镜像窗口从 312x694 **缩成了 31x114 的一个小条**。这时
    `find_mirror_window` 直接找不到它 —— 比例 3.68 超出竖屏筛选范围，
    报的是「镜像 App 在运行，但没有竖屏比例的层 0 窗口」。

    `open -b com.apple.ScreenContinuity` 能把它恢复成正常尺寸（实测回到 312x694）。
    无人值守时这个必须自动做，否则窗口一缩后面全废。

    注意这**不是**「失去焦点就会缩」—— 整晚批跑时终端一直在前台，窗口都是好的。
    触发条件没查清，所以这里只做恢复，不做预防。
    """
    import subprocess
    import time

    from iphone_agent.driver.window import WindowNotFound, find_mirror_window
    subprocess.run(["open", "-b", MIRROR_BUNDLE], capture_output=True, check=False)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            find_mirror_window()
            return True
        except WindowNotFound:
            time.sleep(0.4)
    return False


def restart_mirror(timeout_s: float = 20.0) -> float:
    """退出镜像 App 再启动。返回窗口回来用了几秒；超时抛 RuntimeError。

    这是苹果自己的重连机制，程序调它没有越界 —— 2026-09-10 实测：打字通道时断时续
    （人和程序同死同活），用户手动重连两次都恢复了。手机侧不重启、画面停在原处、
    重连不要求重新认证（设计说明）。调用方在这之后要重新 ensure_connected 等握手。
    """
    import subprocess
    import time
    t0 = time.time()
    subprocess.run(["osascript", "-e", f'tell application id "{MIRROR_BUNDLE}" to quit'],
                   capture_output=True, check=False)
    time.sleep(2.0)
    if not reopen_mirror_window(timeout_s=timeout_s):
        raise RuntimeError(f"镜像退出后重启，{timeout_s:.0f}s 内没等到窗口")
    return time.time() - t0
