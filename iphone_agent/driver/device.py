"""Device：上层唯一的手。收全局屏幕坐标与方向枚举；每次注入前激活确认 + 窗口矩形核对（spec §4.3）。"""
from __future__ import annotations

import time

from AppKit import NSPasteboard, NSPasteboardTypeString

from iphone_agent import config, timing
from iphone_agent.driver.capture import capture_window
from iphone_agent.driver.geometry import Frame, OutOfWindow, Rect, content_bounds
from iphone_agent.driver.injector import (
    KEY,
    SHIFT_KEYCODE,
    BackgroundInjector,
    ForegroundInjector,
    Injector,
    char_keystroke,
    modifier_mask,
)
from iphone_agent.driver.skylight import available as skylight_available
from iphone_agent.driver.timing import SCROLL_LINES, SCROLL_LINES_H
from iphone_agent.driver.window import (
    MirrorWindow,
    WindowNotFound,
    current_rect,
    find_mirror_window,
)


class WindowMoved(RuntimeError):
    pass


def _find_window_or_reopen() -> MirrorWindow:
    """找镜像窗口；找不到就把它叫回来再找一次。

    ⚠ 为什么放在驱动层最底下：镜像窗口会**自己缩成一个 38x110 的小条**
    （2026-09-08 多次实测，触发条件不明），这时 `find_mirror_window` 按纵横比
    直接筛掉它，`Device()` **在构造时**就抛 WindowNotFound ——
    于是 harness 里那套「连接自愈」根本没机会跑，脚本、CLI、任务全部当场死掉。

    `open -b com.apple.ScreenContinuity` 实测能把它恢复成 312x694。
    这就是人遇到这种情况会做的事，让代码自己做一次。
    """
    from iphone_agent.driver.session import reopen_mirror_window
    try:
        return find_mirror_window()
    except WindowNotFound:
        if not reopen_mirror_window():
            raise
        return find_mirror_window()


class Device:
    def __init__(self, win: MirrorWindow | None = None, injector: Injector | None = None):
        # 显式传进来的窗口/注入器视为「钉住」，不再自动重新发现（测试用）。
        self._pinned_win = win
        self._pinned_inj = injector
        self.win = win or _find_window_or_reopen()
        self.inj: Injector = injector or self._make_injector(self.win)
        self._frame_seq = 0
        self._content_cache: tuple[Rect, Rect] | None = None  # (窗口矩形, 内容矩形)

    def _make_injector(self, win: MirrorWindow) -> Injector:
        """按配置选注入路径。

        后台路径（SkyLight）点击不抢焦点，用户可以一边工作一边看它跑；
        滚轮和键盘仍走前台（见 BackgroundInjector 的说明）。
        私有框架不可用时如实退回前台，并让 doctor 报告实际走的是哪条。
        """
        fb = ForegroundInjector(win.pid)
        if config.INJECT_MODE == "foreground" or not skylight_available():
            return fb
        return BackgroundInjector(
            win.pid, win.window_id,
            window_origin=lambda: (self.window_rect().x, self.window_rect().y),
            fallback=fb)

    def injector_mode(self) -> str:
        return "background(SkyLight)" if isinstance(self.inj, BackgroundInjector) else "foreground(CGEvent)"

    def _resolve(self) -> MirrorWindow:
        """每次触碰设备之前重新发现窗口。

        镜像 App 一重启，window id 和 pid 全变，缓存住的句柄会让之后所有操作以
        「pid 31029 不存在」这种看不懂的方式失败（2026-09-05 真机实际踩到）。
        实测这次查询平均 8.8ms，比一次抓帧的 90ms 便宜一个数量级，
        不值得为省它而冒握着陈旧句柄的风险。
        """
        if self._pinned_win is not None:
            return self.win
        win = _find_window_or_reopen()
        if win.window_id != self.win.window_id or win.pid != self.win.pid:
            self.win = win
            self._content_cache = None  # 换了窗口，黑边要重量
            if self._pinned_inj is None:
                self.inj = self._make_injector(win)
        return self.win

    # ---- 观察 ----
    def capture(self) -> Frame:
        with timing.phase("capture"):
            self._frame_seq += 1
            return capture_window(self._resolve(), self._frame_seq)

    def window_rect(self) -> Rect:
        return current_rect(self._resolve().window_id)

    def content_rect(self) -> Rect:
        """手机屏幕在全局屏幕坐标里的矩形 —— 不是窗口矩形。

        镜像 App 在窗口内画了带圆角的机身，四周留黑边（实测左右各 6 点、上 38 点、下 6 点）。
        按几何算起终点的动作必须用这个，用窗口矩形会把起点算到黑边里，iOS 收不到触摸
        （2026-09-05 实测：边缘返回手势因此完全没反应）。

        黑边只随窗口几何变，与屏幕内容无关，所以按窗口矩形缓存，窗口一动就重算。
        """
        rect = self.window_rect()
        if self._content_cache is not None and self._content_cache[0].same_as(rect):
            return self._content_cache[1]
        frame = self.capture()
        b = content_bounds(frame.image)
        sc = frame.scale
        content = Rect(rect.x + b.x / sc, rect.y + b.y / sc, b.w / sc, b.h / sc)
        self._content_cache = (rect, content)
        return content

    def ensure_frame_valid(self, frame: Frame) -> None:
        now = self.window_rect()
        if not now.same_as(frame.window_rect):
            raise WindowMoved(f"窗口从 {frame.window_rect} 移到 {now}")

    def _pre_inject(self):
        self._resolve()
        self.inj.activate()

    def _require_inside(self, sx, sy):
        if not self.window_rect().contains(sx, sy):
            raise OutOfWindow(f"({sx},{sy}) 在窗口外")

    # ---- 动作 ----
    def tap(self, sx: int, sy: int) -> None:
        self._require_inside(sx, sy)
        self._pre_inject()
        self.inj.mouse_move(sx, sy)
        self.inj.mouse_down(sx, sy)
        time.sleep(0.05)
        self.inj.mouse_up(sx, sy)

    def scroll(self, direction: str, amount: str = "page") -> None:
        """滚动。direction 说的是**你想看到的内容在哪个方向**：
        down = 看下面的内容，right = 看右边的内容（主屏幕下一页）。

        四个方向全部走滚轮。鼠标拖拽在镜像里会被 iOS 当成长按+拖动物体，做不出滑动
        （2026-09-05 实测：主屏幕拖拽进编辑模式、图标被拖走），所以驱动层没有拖拽手势。

        ⚠ **主屏幕上滚轮完全不响应**（2026-09-09 评测集跑出来的）：纵向横向都不动，
          前台/后台注入都一样，滚轮行数 10~80 都试过。App 内的 scrollView 照常。
          历史 runs 里主屏翻页成功过 17 次，说明它曾经能用 —— 之后 iOS 或镜像
          可能改成只认带 phase 的触控板手势。后果：open_app 的「回主屏翻页找图标」
          退路对第一页以外的 App 整条是死的。要修得给滚轮事件加 scrollPhase，
          那是 injector 层的事，先记在这儿。
        """
        if direction in ("up", "down"):
            n = SCROLL_LINES[amount]
            v, h = (-n if direction == "down" else n), 0
        elif direction in ("left", "right"):
            n = SCROLL_LINES_H[amount]
            v, h = 0, (-n if direction == "right" else n)
        else:
            raise ValueError(f"未知方向 {direction!r}")
        # macOS 把滚轮路由给**指针下方**的窗口，所以发滚轮前必须 warp 指针并触发 hit-test。
        # 这套序列由注入器自己完成：前台实现直接做，后台实现要先临时切到前台再还回去
        # （它的 mouse_move 是空操作，指望调用方 warp 会丢掉 hit-test —— 踩过）。
        self._pre_inject()
        content = self.content_rect()
        # region 让后台注入器在镜像**还露着的部分**里挑落点：中心被别的窗口
        # （多半是终端）盖住时，不必为了滚一下就把镜像抬到最前面。
        self.inj.scroll_at(tuple(map(int, content.center)), v=v, h=h, region=content)

    def press(self, keycode: int, modifiers: list[int] = ()) -> None:
        self._pre_inject()
        down_mods: list[int] = []
        main_down = False
        mask = modifier_mask(modifiers)
        try:
            for m in modifiers:
                self.inj.key_down(m, flags=modifier_mask([m]))
                down_mods.append(m)
            time.sleep(0.02)
            # 主键事件必须带上修饰键的标志位：Cmd+1/2/3 是镜像 App 自己的 Mac 菜单快捷键，
            # 走 Mac 层而不是转发给 iOS。只「真按住」Cmd 而事件上没有标志位，Mac 层收不到
            # 这个组合键（2026-09-05 实测 key home 完全没反应）。真按住是给转发给 iOS 的
            # 那条路准备的（镜像会丢 flag 掩码），两条路并存所以两者都要。
            self.inj.key_down(keycode, flags=mask)
            main_down = True
            time.sleep(0.02)
        finally:
            # 零残留按键是硬指标：中间任何一步抛异常，已经按下的键都要在这里抬起，
            # 不能让异常把修饰键留在按住状态（后续所有 tap 都会变成 Cmd+点击）。
            if main_down:
                self.inj.key_up(keycode)
            for m in reversed(down_mods):
                self.inj.key_up(m)

    # 系统键。前三个是 cmd+数字，回车不带修饰键 —— 所以这里存 (keycode, 修饰键)
    # 而不是光存 keycode。
    _KEY_COMBOS = {
        "home":         (KEY["1"], [KEY["cmd"]]),
        "app_switcher": (KEY["2"], [KEY["cmd"]]),
        "spotlight":    (KEY["3"], [KEY["cmd"]]),
        # ⚠ 2026-09-09 补：回车的 keycode 从项目第一天就在 KEY 表里，enter() 也写好了，
        #   但**工具面从来没接过它，enter() 全项目零调用点**。后果是模型完全没有换行、
        #   没有「确认/发送」的手段：真机上它想在备忘录里另起一行，工具面上唯一沾边的
        #   只有 tap，于是反复点文本区试图把光标挪到位，9 步熔断。
        #   那不是模型笨，是手里没这把工具。
        "return":       (KEY["return"], []),
        # 光标键（harness/actions.KEY_NAMES 的后八个，一个测试钉着两边一致）。
        # 光标看不见，tap 文字只能把它放到被点的字上 —— 没有这组键，「在行末接着写」就得靠运气。
        "line_start":   (KEY["left"], [KEY["cmd"]]),
        "line_end":     (KEY["right"], [KEY["cmd"]]),
        "text_start":   (KEY["up"], [KEY["cmd"]]),
        "text_end":     (KEY["down"], [KEY["cmd"]]),
        "left":         (KEY["left"], []),
        "right":        (KEY["right"], []),
        "up":           (KEY["up"], []),
        "down":         (KEY["down"], []),
    }

    def key(self, name: str) -> None:
        code, mods = self._KEY_COMBOS[name]
        self.press(code, mods)

    def type(self, text: str) -> str:
        """输入文字。返回实际走的路径：'keystrokes'、'shifted' 或 'paste'。

        ## 打字母必须只发 key-down，不发 key-up（2026-09-07 真机实测）

        | 发法 | 结果 |
        |---|---|
        | down + up 成对 | **大小写按位置交替**：`abcdef` → `aBcDeF` |
        | **只发 down** | **全部正确**：`abcdef` → `abcdef` |
        | 只发 up | 什么都不出 |

        排除过的原因都不是：事件标志位、down→up 停留 0.02–0.35s、字符间隔、
        私有事件源、显式敲 Shift、双 up、每字符补 Shift、输入法解析。
        「有 up 就交替、没 up 就正确」这条规律本身足够指导实现，机制不再深挖。
        取证：`runs/pinyin4-*`、`runs/pinyin6-*`，详见 `docs/14`。

        ⚠ 键按下后不松，**再按同一个键就不出字符**。所以不是「相邻重复」要处理，
        是**整串里任何重复字母**：`shezhi` 会丢第二个 `h` 变成 `shezi`（实测）。
        算法因此是「要按的键若还按着，先松开」，而不是「看它和前一个是否相同」。

        ⚠ 需要 Shift 的字符（大写、部分符号）走不了这条路 —— Shift 本身是修饰键，
        不能只按不放。它们退回成对发送，因而**会受那个交替问题影响**，
        返回值标成 'shifted' 让调用方知道结果可能不可靠。

        ⚠ 粘贴四种情形实测全败（Spotlight 三种等待时长、先点输入框聚焦、
        备忘录正文里光标就位）。⚠ **但通用剪贴板需要「接力」开着且两台设备同一
        iCloud，我们从没查过这些前提** —— 判它坏可能判早了，见 `docs/15`。
        """
        strokes = [char_keystroke(ch) for ch in text]
        if any(k is None for k in strokes):
            pb = NSPasteboard.generalPasteboard()
            pb.clearContents()
            pb.setString_forType_(text, NSPasteboardTypeString)
            time.sleep(0.3)
            self.press(KEY["v"], [KEY["cmd"]])
            return "paste"

        if any(shift for _, shift in strokes):
            for code, shift in strokes:
                self.press(code, [SHIFT_KEYCODE] if shift else [])
                time.sleep(0.04)
            return "shifted"

        self._type_downs_only([code for code, _ in strokes])
        return "keystrokes"

    def _type_downs_only(self, codes: list[int]) -> None:
        """只发 down；要按的键若还按着，先松开再按。结束时全部松开。"""
        self._pre_inject()
        held: set[int] = set()
        try:
            for code in codes:
                if code in held:
                    self.inj.key_up(code)
                    held.discard(code)
                    time.sleep(0.03)
                self.inj.key_down(code)
                held.add(code)
                time.sleep(0.06)
        finally:
            # 零残留按键是硬指标：中途抛异常也要把按下的键全部抬起。
            for code in list(held):
                self.inj.key_up(code)

    def toggle_ime(self) -> None:
        """切换 iOS 输入法（中文拼音 ⇄ 英文）。

        ⚠ 没有任何 API 能查当前是哪个模式 —— 只能从证据推：打一串拼音字母，
        出候选栏就是中文，没候选就是英文。所以调用方拿到的永远是**信念**，
        不是事实，切完必须重新验证。

        2026-09-08 实测 fn/地球键（keycode 63）双向都能切：
        中文 → 英文 → 中文 → 英文，四次全对，窗口全程 312x694 没受影响。
        ctrl+space、caps lock、ctrl+shift+space 都没反应；
        cmd+space 会把输入整个清掉。
        """
        self.press(KEY["fn"])

    def backspace(self, n: int) -> None:
        """退格 n 次。

        ⚠ 用来撤销**我们自己刚打进去的那么多个字符**，绝不多删。
        盲目连按会删掉输入框里本来就有的内容（备忘录、聊天框里就是灾难）。

        为什么需要它：`select_all_and_delete()`（Cmd+A + Delete）**清不掉输入法
        待上屏的拼音缓冲区** —— 2026-09-08 实测，打完 'shezhi' 再调它，
        搜索框从 'Q shezhi' 只变成 'Q shezh'，只掉了一个字符。
        于是下一次输入会叠在残留上：连着两次 'shezhi' 的候选是「则设置设置」。
        """
        for _ in range(max(0, n)):
            self.press(KEY["delete"])
            time.sleep(0.03)

    def select_all_and_delete(self) -> None:
        self.press(KEY["a"], [KEY["cmd"]])
        self.press(KEY["delete"])

    def enter(self) -> None:
        """回车。和 key("return") 同一件事，留一个名字是给 executor 内部编排用的
        （那里想按回车时不该去翻工具名的 enum）。"""
        self.key("return")

    def release_all(self) -> None:
        self.inj.release_all()
