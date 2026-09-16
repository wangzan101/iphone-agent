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
    """帧序列可编程；记录所有动作调用。

    ⚠ 2026-09-16：帧是按 **capture 次数** 往后走的，而 settle 轮询几次由墙钟决定
      （判稳看的是 stable_span_ms，慢机器上一次轮询就够）—— 用「前 N 帧是旧页、后面是新页」
      编排「动作前后画面不同」的测试，在 CI 的 macOS runner 上会读到前后同一帧，changed=False。
      要验「动作让画面变了」，用 ActionDrivenDevice / action_env：页面只随动作切，截图次数再怎么变都不影响。
      这个类留给**真的要逐帧动画**（渐变、加载中）的测试。
    """
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


class ActionDrivenDevice(FakeDevice):
    """页面只随指定动作切换；重复截图、轮询和启动时的 home 不消耗页面。"""
    def __init__(self, frames, transitions):
        super().__init__(frames)
        self._state = 0
        self._transitions = transitions

    def _next(self):
        self._i += 1
        return self._frames[self._state][0]

    def _act(self, name, *args):
        super()._act(name, *args)
        self._state = self._transitions.get((self._state, name), self._state)


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
    _track_frame_ids(per, ocr, "_current_fid")
    return per


def _track_frame_ids(per, ocr, attr):
    """假 OCR 按 frame_id 吐文字，所以每个观察入口都要先把「现在看的是哪一帧」告诉它。

    ⚠ 2026-09-14：Perceiver 多了一个只跑 OCR 的 observe_text（翻主屏找 App 用）。原来这里只包
      observe，observe_text 拿到的是上一帧的 frame_id —— 假 OCR 吐的是旧画面的字，按帧编排的测试
      全在建模一个不存在的画面。两个入口一起包。
    """
    for meth in ("observe", "observe_text"):
        orig = getattr(per, meth)

        def wrapped(frame, *a, _orig=orig, **kw):
            setattr(ocr, attr, frame.frame_id)
            return _orig(frame, *a, **kw)
        setattr(per, meth, wrapped)


GRID_COLS = (105, 243, 381, 519)


def grid_perceiver_for(frames, asker=None):
    """像 perceiver_for，但把每帧的文字排成主屏那样的 4 列网格（infer_page 认得出）。
    位置取自真机标注：列 x 105/243/381/519、行 y 从 290 起每 150 一行（624×1388），框 80×22。"""
    texts = {f.frame_id: t for f, t in frames}

    def ocr(img):
        out = []
        for i, t in enumerate(texts.get(ocr.fid, [])):
            cx, cy = GRID_COLS[i % 4], 290 + (i // 4) * 150
            out.append(RawBox(t, 0.99, (cx - 40) / 624, 1 - (cy + 11) / 1388, 80 / 624, 22 / 1388))
        return out

    ocr.fid = None
    per = Perceiver(ocr=ocr, asker=asker)
    _track_frame_ids(per, ocr, "fid")
    return per


def page_frame(texts, frame_id):
    """一页主屏 / 一个 App 画面：624×1388，和真机截图同尺寸（grid_perceiver_for 的网格按它标定）。"""
    return frame_with_text(texts, frame_id, w=624, h=1388, rect=Rect(0, 0, 312, 694))


class PagedDevice(FakeDevice):
    """主屏若干页，画面只随动作切换（不靠数帧数）：

    · key home → 第 1 页；key spotlight → spotlight 那一帧（给了的话）
    · 在主屏页上 scroll right → 下一页，最后一页再翻不动；scroll left → 上一页
    · type → after_type 那一帧（给了的话）
    · tap → on_tap(dev, x, y) 返回的那一帧（返回 None = 点了画面不变）
    """

    def __init__(self, pages, start=None, spotlight=None, after_type=None, on_tap=None):
        self.pages = list(pages)
        self.page = 0
        self.cur = start if start is not None else self.pages[0]
        self.spotlight, self.after_type, self.on_tap = spotlight, after_type, on_tap
        super().__init__([self.cur])

    def _next(self):
        self._i += 1
        return self.cur[0]

    def _show(self, f):
        if f is not None:
            self.cur = f

    def on_home_page(self, index):
        return self.cur is self.pages[index]

    def key(self, n):
        super().key(n)
        if n == "home":
            self.page = 0
            self._show(self.pages[0])
        elif n == "spotlight":
            self._show(self.spotlight)

    def scroll(self, d, a):
        super().scroll(d, a)
        if self.cur is self.pages[self.page]:
            if d == "right" and self.page + 1 < len(self.pages):
                self.page += 1
            elif d == "left" and self.page > 0:
                self.page -= 1
            self._show(self.pages[self.page])

    def type(self, t):
        super().type(t)
        self._show(self.after_type)

    def tap(self, x, y):
        super().tap(x, y)
        if self.on_tap is not None:
            self._show(self.on_tap(self, x, y))


class CountingAsker:
    """会被整屏解析问到的假视觉：只记次数，一律「没问成」（None）。用来证明某段路上没做整屏解析。"""

    def __init__(self):
        self.calls = 0

    def ask_json(self, prompt, images, max_tokens=None):
        self.calls += 1
        return None


@pytest.fixture
def long_sleeps(monkeypatch):
    """记下所有 ≥1 秒的 time.sleep（不真睡）；短的照睡 —— settle 的轮询要靠真实时钟走。

    用来证明「只看不点时不硬等 AFTER_PAGE_FLIP_S，要点之前才等」（2026-09-14）。
    events 是共享的：测试可以把设备动作也记进来，看等待落在哪两个动作之间。
    """
    real = time.sleep
    events: list = []

    def fake(s):
        if s >= 1:
            events.append(("sleep", s))
        else:
            real(s)
    monkeypatch.setattr(time, "sleep", fake)
    return events


@pytest.fixture
def fake_env():
    def make(frame_specs):
        frames = [frame_with_text(t, i + 1) for i, t in enumerate(frame_specs)]
        return FakeDevice(frames), perceiver_for(frames), frames
    return make


@pytest.fixture
def action_env():
    """用于验证动作结果的状态机；保留原 fake_env 给需要逐帧动画的测试。"""
    def make(frame_specs, transitions):
        frames = [frame_with_text(t, i + 1) for i, t in enumerate(frame_specs)]
        return ActionDrivenDevice(frames, transitions), perceiver_for(frames), frames
    return make


@pytest.fixture
def settle_clock(monkeypatch, request):
    """只替换 settle 模块的时钟，可模拟轮询超时唤醒，不改全局 time 或真实睡眠。"""
    import iphone_agent.harness.settle as settle_mod

    class Clock:
        now_ms = 1000
        overshoot_ms = request.param
        polls = 0

        def time(self):
            return self.now_ms / 1000

        def sleep(self, seconds):
            self.now_ms += round(seconds * 1000)
            if seconds > 0:
                self.now_ms += self.overshoot_ms
                self.polls += 1

    clock = Clock()
    monkeypatch.setattr(settle_mod, "time", clock)
    return clock


@pytest.fixture
def loop_clock(monkeypatch):
    """冻结 loop 看到的 `time.time()`，只由测试显式推进（别的时间函数照旧走真实 time）。

    ⚠ 2026-09-16：验「时限刚好用完」的用例原来靠模型 `time.sleep(0.5)` 加上前五步的真实耗时
      去逼近 timeout_s=0.6 —— 前五步在开发机上几十毫秒、在 CI 的 macOS runner 上就超了 0.6s，
      时限在第 6 步之前先炸，end_reason 成了 timeout。改成推进这只钟：loop 比的是
      `time.time() - started`，钟不自己走，「第几次决策时花掉多少预算」就是测试写死的那一个数，
      跟机器快慢无关。
    """
    import iphone_agent.harness.loop as loop_mod

    class Clock:
        def __init__(self):
            self.now = time.time()

        def time(self):
            return self.now

        def advance(self, seconds):
            self.now += seconds

        def __getattr__(self, name):        # sleep / perf_counter 等一律照旧
            return getattr(time, name)

    clock = Clock()
    monkeypatch.setattr(loop_mod, "time", clock)
    return clock


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
    # ⚠ 2026-09-14：twin/scan.walk_home_pages 翻页后自己 settle，也用 IOS_TIMING —— 漏了它，每翻一页按真机
    #   时序等，整套测试从 59 秒涨到 116 秒。翻页后点之前的 AFTER_PAGE_FLIP_S 同理置 0；
    #   要验这段等待的测试自己 monkeypatch 一个值（见 long_sleeps）。
    import iphone_agent.twin.scan as scan_mod
    from iphone_agent import config
    monkeypatch.setattr(scan_mod, "IOS_TIMING", table)
    monkeypatch.setattr(config, "AFTER_PAGE_FLIP_S", 0.0)


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
