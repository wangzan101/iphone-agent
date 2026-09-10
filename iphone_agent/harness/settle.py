"""稳定等待（spec §5.3）：先最短等待，再要求 stable_span 内相邻帧都稳定；封顶返回 settled=False。"""
from __future__ import annotations

import time

from iphone_agent import config
from iphone_agent.driver.geometry import Frame
from iphone_agent.driver.timing import Timing
from iphone_agent.perceive.hashing import hamming


def settle(device, before: Frame, timing: Timing, hasher) -> tuple[Frame, bool]:
    """等画面稳定，返回 (最后一帧, 是否在封顶前稳定)。

    `before` 从不被读：稳定与否只比较**相邻两次** `device.capture()`，跟动作之前那一帧无关。
    保留这个参数是为了 25 个调用点的签名兼容；删掉它是另一个纯重构，别顺手在功能改动里做。
    所以调用方传哪一帧都不影响行为（executor.return_to_first_home_page 曾为此专门写过一条 ⚠）。
    """
    time.sleep(timing.min_wait_ms / 1000)
    start = time.time()
    last = device.capture()
    last_h = hasher(last.image, config.STATUS_BAR_CROP_RATIO)
    stable_since = None
    while (time.time() - start) * 1000 < timing.settle_max_ms:
        time.sleep(timing.poll_ms / 1000)
        cur = device.capture()
        h = hasher(cur.image, config.STATUS_BAR_CROP_RATIO)
        if hamming(h, last_h) <= config.STABLE_THRESHOLD:
            stable_since = stable_since or time.time()
            if (time.time() - stable_since) * 1000 >= timing.stable_span_ms:
                return cur, True
        else:
            stable_since = None
        last, last_h = cur, h
    return last, False
