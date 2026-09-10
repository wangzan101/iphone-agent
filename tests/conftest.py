import re
import time

import pytest
from PIL import Image, ImageDraw

from iphone_agent.driver.geometry import Frame, Rect
from iphone_agent.driver.timing import Timing
from iphone_agent.perceive.observe import Perceiver
from iphone_agent.perceive.ocr import RawBox


def frame_with_text(texts, frame_id, w=400, h=800, rect=Rect(0, 0, 200, 400), fill="white"):
    """图案由文字决定：不同文字 → 不同图 → aHash 不同；相同文字 → 相同图。"""
    img = Image.new("RGB", (w, h), fill)
    d = ImageDraw.Draw(img)
    for i, t in enumerate(texts):
        code = sum(map(ord, t))
        off = (code % 5) * 120
        width = 100 + code % 90
        d.rectangle([20, off + 50 + i * 60, 20 + width, off + 80 + i * 60], fill="black")
    return Frame(img, w, h, rect, time.time(), frame_id), texts


class FakeDevice:
    """帧序列可编程；记录所有动作调用。"""
    def __init__(self, frames):
        self._frames = list(frames)      # [(Frame, texts)]
        self.calls = []
        self._i = 0
        self.rect = frames[0][0].window_rect
        self.released = False
        self.raise_on = {}

    def _next(self):
        f = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        return f[0]

    def capture(self):
        if "capture" in self.raise_on:
            raise self.raise_on["capture"]
        return self._next()

    @property
    def actions(self):
        """排除 loop 前置的 key home 后的动作调用。"""
        return [c for c in self.calls if c != ("key", "home")]

    def window_rect(self):
        return self.rect

    def ensure_frame_valid(self, frame):
        if not self.rect.same_as(frame.window_rect):
            from iphone_agent.driver.device import WindowMoved
            raise WindowMoved("moved")

    def _act(self, name, *a):
        if name in self.raise_on:
            raise self.raise_on[name]
        self.calls.append((name, *a))

    def tap(self, x, y): self._act("tap", x, y)
    def scroll(self, d, a): self._act("scroll", d, a)
    def swipe(self, k): self._act("swipe", k)
    def type(self, t): self._act("type", t)
    def key(self, n): self._act("key", n)
    def toggle_ime(self): self._act("toggle_ime")
    def backspace(self, n): self._act("backspace", n)
    def select_all_and_delete(self): self._act("select_all_and_delete")
    def enter(self): self._act("enter")
    def release_all(self): self.released = True


_CAND_LIKE = re.compile(r"^\s*[1-9]\s*\S")


def perceiver_for(frames):
    """按 frame_id 返回预设文字的假 OCR。

    ⚠ 以数字开头的文字（输入法候选）**横着排**，不跟其他元素一样竖着堆。
    2026-09-08 之前这里是竖排的，而真机上候选栏是一条水平带 —— y 几乎相同、
    x 依次排开（实测 '1设置'@(168,1218) '2 摄制'@(303,1217) '3 摄制组'@(441,1218)，
    图像 624x1388）。竖排的假数据让候选栏识别的测试在建模一个不存在的形态。
    这里的 y=0.10（下边原点）折算到图像上约 0.88 高度处，和实测的 0.877 对得上。
    """
    texts = {f.frame_id: t for f, t in frames}
    def ocr(img):
        fid = getattr(ocr, "_current_fid", None)
        out, col = [], 0
        for i, t in enumerate(texts.get(fid, [])):
            if _CAND_LIKE.match(t):
                out.append(RawBox(t, 0.99, 0.08 + col * 0.22, 0.10, 0.18, 0.04))
                col += 1
            else:
                # ⚠ 起点 0.85 不是 0.9：RawBox 是下边原点，0.9 折算到图像上是 0.08H，
                #   落在状态栏裁剪区（0.11）里，会被变化判定的文字集合丢掉。
                out.append(RawBox(t, 0.99, 0.1, 0.85 - i * 0.055, 0.4, 0.04))
        return out
    per = Perceiver(ocr=ocr)
    orig = per.observe
    def observe(frame):
        ocr._current_fid = frame.frame_id
        return orig(frame)
    per.observe = observe
    return per


@pytest.fixture
def fake_env():
    def make(frame_specs):
        frames = [frame_with_text(t, i + 1) for i, t in enumerate(frame_specs)]
        return FakeDevice(frames), perceiver_for(frames), frames
    return make


@pytest.fixture(autouse=True)
def fast_timing(monkeypatch):
    """裁决四：把 executor 用的 IOS_TIMING 各项换成接近零的值，测试验证的是控制流不是真实时序。
    poll_ms 与 stable_span_ms 保持 1:2 比例 —— 与真实 IOS_TIMING 各项的比例一致，
    使得「一段内容不再变化后需要多少次轮询才能判定 settled」这件事在测试和生产代码路径下一致
    （见 test_executor.py 里各帧序列设计的注释）。
    """
    import iphone_agent.harness.executor as executor_mod
    import iphone_agent.harness.loop as loop_mod
    fast = Timing(min_wait_ms=0, poll_ms=10, stable_span_ms=20, settle_max_ms=1000)
    table = dict.fromkeys(executor_mod.IOS_TIMING, fast)
    monkeypatch.setattr(executor_mod, "IOS_TIMING", table)
    # loop 里也用 IOS_TIMING（回主屏之后的初始 settle），不patch 的话整套测试会慢一个数量级
    monkeypatch.setattr(loop_mod, "IOS_TIMING", table)


@pytest.fixture(autouse=True)
def _no_real_config(monkeypatch, tmp_path):
    """⚠ 测试永远不该读到开发机上真实的 `.iphone/config.toml`。

    2026-09-08 撞到：把真实密钥写进配置文件之后，当时的「无 key 报错」测试
    立刻挂了 —— 它假设「没有配置文件」，而现在有了。挂得对，但暴露的是
    **测试和真实凭据文件耦合**：结果取决于开发机上有没有那个文件。

    统一把工作目录换到 tmp，需要配置文件的测试自己在里面造。
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _never_restart_the_real_mirror(monkeypatch):
    """⚠ 测试永远不能碰开发机上真实的镜像 App。

    2026-09-10 撞到：loop 里的恢复阶梯第 ③ 级是 session.restart_mirror（osascript 退出镜像再启动）。
    test_no_progress_stops_after_six 连点十次「没反应」，攒到 DEAD_TAPS_ALERT 就爬梯 ——
    一跑测试，开发机上正在连着的镜像被悄悄重启了一次（pid 变了）。
    这里把它换成只记不做的假函数；要验真重启，用 scripts/mirror_restart_probe.py。
    """
    import iphone_agent.driver.session as session_mod
    calls = []
    monkeypatch.setattr(session_mod, "restart_mirror", lambda timeout_s=20.0: calls.append(1) or 0.0)
    return calls


class SeesApps:
    """假视觉：按截图认出现在是哪个 App，只回答 judge.APP_PROMPT 那一问；别的问题一律「没问成」（None）。

    按图片字节认帧：frame_with_text 的图案由文字决定，Observation.image 就是 frame.image 原样
    （perceive/elements.py 构造 Observation 时 image=frame.image）。
    """

    def __init__(self, frames, app_of):
        self._texts = {f.image.tobytes(): tuple(t) for f, t in frames}
        self._app_of = app_of
        self.asked = []

    def ask_json(self, prompt, images, max_tokens=None):
        m = re.search(r"要打开的 App 是「(.+?)」", prompt)
        if m is None:
            return None
        seen = self._app_of(self._texts.get(images[0].tobytes(), ()))
        self.asked.append((m.group(1), seen))
        return {"is_app": seen == m.group(1), "confidence": 0.95, "seen": seen, "why": "假视觉"}
