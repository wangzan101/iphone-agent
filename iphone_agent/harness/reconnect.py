"""连接暂停时自己点「继续」。

为什么放在 harness：判定在 driver（纯函数），但恢复要**观察 → 找按钮 → 点 → 再观察**，
同时用到 driver 和 perceive，按分层规矩只能在这一层编排。

两级尝试，因为「继续」是个**原生 macOS 按钮**，不是镜像过来的手机像素：
后台注入（SkyLight）对手机画面验证过，对原生控件没验过。所以先后台点，
不行再把窗口抬到前台点一次。抬窗口会抢焦点，但断线的时候本来也没在干别的。
"""
from __future__ import annotations

import time

from iphone_agent.driver.geometry import OutOfWindow, image_to_screen
from iphone_agent.driver.session import (
    IN_USE,
    RESUME_BUTTON,
    detect,
    reopen_mirror_window,
)
from iphone_agent.driver.window import WindowNotFound


def _look(dev, per):
    return per.observe(dev.capture())


def _frame_of(obs):
    from iphone_agent.driver.geometry import Frame
    return Frame(obs.image, obs.width_px, obs.height_px, obs.window_rect, 0.0, obs.frame_id)


def _tap_resume(dev, obs, front: bool) -> bool:
    btn = next((e for e in obs.elements if e.text.strip() == RESUME_BUTTON), None)
    if btn is None:
        return False
    if front:
        try:
            dev.inj.activate()
        except Exception:                     # noqa: BLE001 —— 抬不起来就退回后台点，不该因此放弃
            pass
    try:
        dev.tap(*image_to_screen(*btn.center, _frame_of(obs)))
    except OutOfWindow:
        return False
    return True


def ensure_connected(dev, per, obs=None, handshake_s: float = 45.0, poll_s: float = 2.5):
    """确认镜像可用；「连接暂停」就自己点回来。

    返回 `(能不能用, 说明, 当前观察)`。

    ⚠ `obs` 传进来就复用，不另抓帧。**正常情况下这个检查必须是零成本的** ——
    它每次任务开始都要跑，而调用方（loop）本来就要观察一次。第一版自己抓了一帧，
    把测试里精心设计的帧序列整个错位了，那不是测试的问题，是这个函数多做了事。

    手机被拿起返回 False —— 那是人的事，别徒劳点。

    ⚠ 点完「继续」之后**只看不点**，看满 `handshake_s` 还没回来才允许第二次。
    iphone-use 真机记录（`reference/iphone-use/crates/server/src/main.rs:465-469`，2026-06-12）：
    重连握手要 10–30 秒，握手中途落下的任何一次点击都会**取消**它 ——
    急着重试会把「能连上」变成「连上就断」。第一版这里只等 2.5 秒就点第二次，
    很可能正是我们看到「连接反复暂停」的一部分成因。45 秒取自它的 COOLDOWN。
    """
    if obs is None:
        try:
            obs = _look(dev, per)
        except WindowNotFound:
            # 窗口可能缩成了小条（实测 31x114）—— 叫回来再试一次。见 session.reopen_mirror_window。
            if not reopen_mirror_window():
                return False, "镜像窗口找不到，也叫不回来 —— 检查镜像 App 是不是退出了", None
            obs = _look(dev, per)
    state = detect(e.text for e in obs.elements)
    if state is None:
        return True, "连接正常", obs
    if state == IN_USE:
        return False, "手机被拿起了（「iPhone 使用中」）—— 只能等人把它锁上，点什么都没用", obs

    for front in (False, True):               # 先后台，再抬前台
        if not _tap_resume(dev, obs, front):
            return False, "认出了「连接暂停」，但屏幕上找不到「继续」按钮", obs
        deadline = time.monotonic() + handshake_s
        while True:                           # 握手窗口内：只看，不点
            time.sleep(poll_s)
            obs = _look(dev, per)
            state = detect(e.text for e in obs.elements)
            if state is None:
                return True, f"连接暂停已恢复（{'抬前台' if front else '后台'}点的「继续」）", obs
            if state == IN_USE:
                return False, "点完变成「iPhone 使用中」—— 手机被拿起了", obs
            if time.monotonic() >= deadline:
                break
    return False, f"点了两次「继续」、各等了 {handshake_s:.0f} 秒都没回来", obs
