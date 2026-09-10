"""前台 CGEvent 注入（spec §4.3）。行为事实（phone-harness README/注释）：镜像丢 flag 掩码，修饰键必须真按住；
释放时先清标志位再抬起；滚动前 warp + MouseMoved 触发 hit-test。"""
from __future__ import annotations

import time
from typing import Protocol

import Quartz
from AppKit import (
    NSApplicationActivateIgnoringOtherApps,
    NSDate,
    NSRunLoop,
    NSRunningApplication,
    NSWorkspace,
)

from iphone_agent.driver import skylight as sl
from iphone_agent.driver.geometry import scroll_anchor_candidates

KEY = {"cmd": 55, "a": 0, "v": 9, "return": 36, "delete": 51,
       "1": 18, "2": 19, "3": 20,
       # fn/地球键：iOS 上切换输入法。2026-09-08 实测双向都能切
       # （中文→英文→中文→英文，判据是打 shezhi 之后有没有候选栏），
       # 而 ctrl+space、caps lock、ctrl+shift+space 都没反应。
       "fn": 63,
       # 方向键。iOS 接硬件键盘：Cmd+←/→ 行首/行末、Cmd+↑/↓ 文首/文末（2026-09-10 镜像里实测通，
       # Spotlight 框 abc → Cmd+← x → Cmd+→ y 得 xabcy）。
       "left": 123, "right": 124, "down": 125, "up": 126}

# 修饰键 keycode → CGEvent 标志位。
# 为什么两者都要：镜像把转发给 iOS 的按键的 flag 掩码丢掉，所以修饰键必须**真按住**那个键；
# 但 Cmd+1/2/3 这类是镜像 App 自己的 Mac 菜单快捷键，走的是 Mac 层，需要事件上**带标志位**。
# 两条路径并存，所以既真按住、也设标志位。
MODIFIER_FLAGS = {
    55: Quartz.kCGEventFlagMaskCommand,
    56: Quartz.kCGEventFlagMaskShift,
    58: Quartz.kCGEventFlagMaskAlternate,
    59: Quartz.kCGEventFlagMaskControl,
}


# ASCII 字符 → (keycode, 是否需要 shift)。美式键盘布局。
# 为什么需要它：粘贴这条路在真机上是坏的（2026-09-07 实测：Spotlight 三种等待时间、
# 先点输入框聚焦、以及备忘录正文里光标就位，四种情况全部失败），而逐键输入两处全成
# —— 键盘通道本身是好的，坏的是 Mac→iPhone 的剪贴板同步。
_ROW = "asdfhgzxcv§bqweryt123465=97-80]ou[ip\rlj'k;\\,/nm.`"
CHAR_KEYCODE: dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26,
    "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35,
    "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45,
    "m": 46, ".": 47, "`": 50, " ": 49,
}
# 需要 shift 的字符 → 对应的无 shift 键
SHIFTED = {"!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7",
           "*": "8", "(": "9", ")": "0", "_": "-", "+": "=", "{": "[", "}": "]",
           ":": ";", '"': "'", "<": ",", ">": ".", "?": "/", "~": "`", "|": "\\"}
SHIFT_KEYCODE = 56


def char_keystroke(ch: str) -> tuple[int, bool] | None:
    """字符 → (keycode, 是否按 shift)。打不出来的返回 None（中文等非 ASCII）。"""
    if ch in CHAR_KEYCODE:
        return CHAR_KEYCODE[ch], False
    low = ch.lower()
    if ch.isalpha() and low in CHAR_KEYCODE:
        return CHAR_KEYCODE[low], ch.isupper()
    if ch in SHIFTED:
        return CHAR_KEYCODE[SHIFTED[ch]], True
    return None


def modifier_mask(codes) -> int:
    m = 0
    for c in codes:
        m |= MODIFIER_FLAGS.get(c, 0)
    return m


class ActivateFailed(RuntimeError):
    pass


class Injector(Protocol):
    def mouse_down(self, x: int, y: int) -> None: ...
    def mouse_up(self, x: int, y: int) -> None: ...
    def mouse_drag(self, x: int, y: int) -> None: ...
    def mouse_move(self, x: int, y: int) -> None: ...
    def warp(self, x: int, y: int) -> None: ...
    def scroll_lines(self, v: int = 0, h: int = 0) -> None: ...
    def scroll_at(self, anchor, v: int = 0, h: int = 0,
                  raise_window: bool = True, region=None) -> None: ...
    def key_down(self, code: int, flags: int = 0) -> None: ...
    def key_up(self, code: int) -> None: ...
    def release_all(self) -> None: ...


class ForegroundInjector:
    def __init__(self, pid: int):
        self.pid = pid
        self._held_keys: list[int] = []
        self._mouse_down_at: tuple[int, int] | None = None

    # ---- 激活并确认前台（spec §4.3）----
    def _frontmost_pid(self, pump_s: float = 0.02) -> int | None:
        """当前前台 App 的 pid。

        ⚠ 必须先泵一次 run loop 再读。NSWorkspace 的 frontmostApplication 靠通知更新，
        而我们是个没有事件循环的命令行进程 —— 用 time.sleep 轮询的话，读到的永远是
        进程启动那一刻的陈旧值，App 早就切过去了也看不到（2026-09-05 实测：
        activateWithOptions_ 明明成功，用 time.sleep 轮询 3 秒始终读到旧 pid；
        换成泵 run loop 后 26-41ms 就能读到正确值，连测 5 次全中）。
        """
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(pump_s))
        front = NSWorkspace.sharedWorkspace().frontmostApplication()
        return int(front.processIdentifier()) if front is not None else None

    def _running_app(self):
        return NSRunningApplication.runningApplicationWithProcessIdentifier_(self.pid)

    def _do_activate(self, app) -> None:
        app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)

    def activate(self, timeout_s: float = 0.5) -> None:
        app = self._running_app()
        if app is None:
            raise ActivateFailed(
                f"镜像 App 进程 {self.pid} 已不存在 —— App 多半退出或重启过。"
                "Device 会在下次动作前自动重新发现窗口；若持续失败，检查镜像是否已连接。")
        if self._frontmost_pid() == self.pid:  # 已在前台，不必再折腾一次窗口切换
            return
        self._do_activate(app)
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            last = self._frontmost_pid()
            if last == self.pid:
                return
        raise ActivateFailed(
            f"激活后 frontmost 仍不是镜像 App（目标 pid={self.pid}，实际 pid={last}，"
            f"等了 {timeout_s}s）")

    # ---- 鼠标 ----
    def _post_mouse(self, kind, x, y):
        ev = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    def mouse_move(self, x, y):
        self._post_mouse(Quartz.kCGEventMouseMoved, x, y)

    def mouse_down(self, x, y):
        self._post_mouse(Quartz.kCGEventLeftMouseDown, x, y)
        self._mouse_down_at = (x, y)

    def mouse_drag(self, x, y):
        self._post_mouse(Quartz.kCGEventLeftMouseDragged, x, y)
        # 异常中断时 release_all 要在光标实际位置抬起，而不是起点；
        # 没按下就拖拽是异常调用，不该凭空造出一个「按下位置」
        if self._mouse_down_at is not None:
            self._mouse_down_at = (x, y)

    def mouse_up(self, x, y):
        self._post_mouse(Quartz.kCGEventLeftMouseUp, x, y)
        self._mouse_down_at = None

    def warp(self, x, y):
        Quartz.CGWarpMouseCursorPosition((x, y))

    def scroll_at(self, anchor, v: int = 0, h: int = 0, raise_window: bool = True,
                  region=None):
        """在 anchor 处滚轮：warp 指针 → MouseMoved 触发 hit-test → 发滚轮。

        raise_window=False 时不抬窗口：只要锚点处目标本来就是最上面的窗口，
        滚轮就能命中它 —— **不要求它是前台**。

        macOS 把滚轮路由给**真实指针下方**的窗口，不看事件里的坐标字段，
        所以这三步缺一不可。整套动作放在注入器里，不让调用方拼装 ——
        后台注入器的 mouse_move 是空操作，调用方拼装会静默丢掉 hit-test（踩过）。
        """
        if raise_window:
            self.activate()
        if anchor is not None:
            ax, ay = anchor
            self.warp(ax, ay)
            self.mouse_move(ax, ay)
            time.sleep(0.03)
        self.scroll_lines(v=v, h=h)

    def scroll_lines(self, v: int = 0, h: int = 0):
        """滚轮。两个轴：v 竖直、h 水平。

        滚轮是镜像里**唯一**能做出 iOS 滑动手势的通道 —— 鼠标拖拽会被当成长按+拖动
        物体（2026-09-05 实测：主屏幕进编辑模式、图标被拖走），做不出滑动。
        横向滚轮实测可以翻主屏幕页。
        """
        ev = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitLine, 2, int(v), int(h))
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    # ---- 键盘 ----
    def key_down(self, code: int, flags: int = 0):
        ev = Quartz.CGEventCreateKeyboardEvent(None, code, True)
        if flags:
            Quartz.CGEventSetFlags(ev, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        self._held_keys.append(code)

    def key_up(self, code: int):
        ev = Quartz.CGEventCreateKeyboardEvent(None, code, False)
        Quartz.CGEventSetFlags(ev, 0)  # 先清标志位再抬起
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        if code in self._held_keys:
            self._held_keys.remove(code)

    # ---- 清理（finally 调用，硬指标：零残留按键）----
    def release_all(self):
        for code in list(reversed(self._held_keys)):
            self.key_up(code)
        if self._mouse_down_at is not None:
            self.mouse_up(*self._mouse_down_at)


class BackgroundInjector:
    """点击走 SkyLight 不抢焦点；其余动作委托给前台实现。

    真机实测（2026-09-07，macOS 15.6.1）：

    | 做法 | 手机有反应 | 前台 | 用户还能打字 |
    |---|---|---|---|
    | SkyLight 点击，**不调** SetFrontProcess | ✅ | 不变 | ✅ |
    | SkyLight 点击，调 SetFrontProcess | ❌ | 变成镜像 | 要重新点回去 |
    | CGEventPostToPid 点击 | ❌ | 不变 | ✅ |

    所以 `_SLPSSetFrontProcessWithOptions` 对我们不但没用还有害 —— 那是 yabai 为了
    「把焦点给窗口」而做的事，我们只想送达事件。不调它，窗口既不抬升也不夺焦点，
    用户屏幕自始至终没有任何变化。

    键盘同样后台化了，而且更简单 —— 实测直接 `CGEventPostToPid` 发键即可，
    连 make_key 都不需要（做了也不坏事，只是多余）。

    ⚠ **只剩滚轮抢焦点**：macOS 按真实指针位置路由滚轮，不看事件里的坐标字段；
    实测 CGEventPostToPid 发滚轮无效（与 phone-harness 记录一致）。
    所以滚轮会抢焦点，但滚完立刻把前台 App 和指针位置还回去 —— 从「永久夺走」
    降成「闪一下」。
    """

    def __init__(self, pid: int, window_id: int, window_origin,
                 fallback: ForegroundInjector):
        self.pid = pid
        self.window_id = window_id
        self._origin = window_origin      # 可调用对象，返回当前窗口左上角 (x, y)
        self._fb = fallback
        self._down_at: tuple[int, int] | None = None
        self._held_keys: list[int] = []

    # ---- 鼠标：走 SkyLight，不抢焦点 ----
    def _win_xy(self, x, y):
        ox, oy = self._origin()
        return (x - ox, y - oy)

    def _record(self, event_type, x, y):
        # 全局坐标字段是必需的：实测只填窗口内坐标时手机毫无反应。
        sl.post(self.pid, self.window_id, event_type,
                global_xy=(x, y), window_xy=self._win_xy(x, y), set_front=False)

    def mouse_move(self, x, y):
        pass  # 后台路径不需要移动指针；真实指针不该被动

    def mouse_down(self, x, y):
        self._record(sl.EVENT_DOWN, x, y)
        self._down_at = (x, y)

    def mouse_drag(self, x, y):
        self._record(sl.EVENT_DRAGGED, x, y)
        if self._down_at is not None:
            self._down_at = (x, y)

    def mouse_up(self, x, y):
        self._record(sl.EVENT_UP, x, y)
        self._down_at = None

    # ---- 键盘：CGEventPostToPid，同样不抢焦点 ----
    # 实测（2026-09-07）：直接 PostToPid 发 Cmd+2 就能打开多任务，**什么前置都不需要**；
    # 先用 SkyLight 空位置记录做 make_key 也能用，但那一步是多余的。
    # （另试过用 SkyLight 记录直接发键盘事件，猜事件类型 10/11 —— 不成立。）
    def key_down(self, code: int, flags: int = 0):
        ev = Quartz.CGEventCreateKeyboardEvent(None, code, True)
        if flags:
            Quartz.CGEventSetFlags(ev, flags)
        Quartz.CGEventPostToPid(self.pid, ev)
        self._held_keys.append(code)

    def key_up(self, code: int):
        ev = Quartz.CGEventCreateKeyboardEvent(None, code, False)
        Quartz.CGEventSetFlags(ev, 0)   # 先清标志位再抬起，否则修饰键会锁死
        Quartz.CGEventPostToPid(self.pid, ev)
        if code in self._held_keys:
            self._held_keys.remove(code)

    # ---- 滚轮：唯一还要抢焦点的动作 ----
    def warp(self, x, y):
        self._fb.warp(x, y)

    def scroll_lines(self, v: int = 0, h: int = 0):
        self._fb.scroll_lines(v=v, h=h)

    @staticmethod
    def _onscreen_windows():
        opts = (Quartz.kCGWindowListOptionOnScreenOnly
                | Quartz.kCGWindowListExcludeDesktopElements)
        return [w for w in (Quartz.CGWindowListCopyWindowInfo(
            opts, Quartz.kCGNullWindowID) or [])
            if int(w.get("kCGWindowLayer", -1)) == 0]

    def _is_topmost_at(self, x, y, windows=None) -> bool:
        """锚点处镜像是不是最上面的普通窗口。

        窗口列表按前后顺序排，第一个覆盖该点的 layer-0 窗口就是 hit-test 会命中的那个。

        `windows` 可传入预取的列表：挑落点时要试十几个候选，每个都去查一次系统
        既慢又会拿到互相不一致的快照。
        """
        for w in (self._onscreen_windows() if windows is None else windows):
            b = w["kCGWindowBounds"]
            if b["X"] <= x <= b["X"] + b["Width"] and b["Y"] <= y <= b["Y"] + b["Height"]:
                return int(w.get("kCGWindowNumber", -1)) == self.window_id
        return False

    def pick_clear_anchor(self, region, default):
        """在 region 里挑一个镜像没被盖住的落点；全被盖住则返回 default。

        中心被盖住不等于整个镜像被盖住 —— 用户的终端往往只压住一角。
        只要镜像还露出一小块，就在那一块上滚，窗口不必抬起来，屏幕上什么都不变。
        """
        if region is None:
            return default
        windows = self._onscreen_windows()
        for cx, cy in scroll_anchor_candidates(region):
            if self._is_topmost_at(cx, cy, windows):
                return (int(cx), int(cy))
        return default

    def scroll_at(self, anchor, v: int = 0, h: int = 0, region=None):
        """滚轮不能用事件记录后台化（macOS 按真实指针位置路由，实测 PostToPid 发滚轮
        无效），但**大多数情况下也不需要抬窗口**。

        关键：滚轮命中的是「指针下方**最上面**的窗口」，不要求它是前台。
        用户为了看手机本来就让镜像露着，所以锚点处它通常就是最上面的 ——
        这时只 warp 指针、发滚轮，窗口既不抬升也不夺焦点，屏幕上什么都不会变。
        只有被别的窗口盖住时才退回「抬升 → 滚 → 还原」（phone-harness 记的
        「Chrome 遮住镜像时滚动会滚到 Chrome 里」正是这种情形）。
        """
        ax, ay = anchor
        clear = self._is_topmost_at(ax, ay)
        if not clear:
            # 中心被盖住不等于整个镜像被盖住，换个还露着的落点就不必抬窗口。
            picked = self.pick_clear_anchor(region, None)
            if picked is not None:
                ax, ay = picked
                clear = True
        prev_app = None
        # ⚠ 读之前必须泵 run loop。NSWorkspace 的前台信息是通知驱动的，
        # 在没有事件循环的命令行进程里会返回陈旧值 —— 读到的若已经是镜像自己，
        # 下面的还原判断就会被跳过，焦点/窗口层级永远还不回去。
        # 这是同一个根因第三次咬人（前两次：injector._frontmost_pid、window._mirror_pids）。
        if not clear:
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.02))
            prev_app = NSWorkspace.sharedWorkspace().frontmostApplication()
        prev_pos = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        try:
            if not clear:
                # ⚠ 抬窗口失败**不等于**滚不了。
                #   2026-09-08 真机撞到：窗口确实被抬到了最上面（用户亲眼看到），
                #   但 activate() 报 activate_failed（frontmost 仍是浏览器），
                #   于是滚轮**根本没发出去**就被闸门拦了。
                #
                #   「窗口在最上面」和「App 是活跃应用」是两件事，macOS 可以只做前者。
                #   而滚轮要的恰恰是前者 —— 它按真实指针下方最上面的窗口路由，
                #   **不要求那个 App 活跃**。用错的尺子去量，就会把能用的情况判死。
                try:
                    self._fb.activate()
                except ActivateFailed:
                    pass
                clear = self._is_topmost_at(ax, ay)     # 以真实条件为准，重新量一次
                if not clear:
                    raise ActivateFailed(
                        f"锚点 ({ax:.0f},{ay:.0f}) 处镜像仍被别的窗口盖住，"
                        f"抬窗口也没能让它到最上面 —— 把盖住镜像的窗口挪开一点再试。")
            self._fb.scroll_at((ax, ay), v=v, h=h, raise_window=False)
            # ⚠ 发完滚轮要留一点时间让镜像处理，再去还原焦点和指针。
            # 立刻把指针挪走、把前台切回去，会把还没被消费的滚动冲掉 ——
            # 2026-09-07 踩到：主屏幕横向翻页正常，而设置里的竖直滚动 changed=false。
            time.sleep(0.35)
        finally:
            Quartz.CGWarpMouseCursorPosition(prev_pos)
            if prev_app is not None and int(prev_app.processIdentifier()) != self.pid:
                for _ in range(3):
                    prev_app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                    NSRunLoop.currentRunLoop().runUntilDate_(
                        NSDate.dateWithTimeIntervalSinceNow_(0.05))
                    now = NSWorkspace.sharedWorkspace().frontmostApplication()
                    if now is not None and int(now.processIdentifier()) == int(
                            prev_app.processIdentifier()):
                        break

    def activate(self, timeout_s: float = 0.5) -> None:
        """点击和键盘路径都不需要激活 —— 这正是后台注入的意义。"""

    def release_all(self):
        """零残留按键是硬指标，后台路径也要守。"""
        for code in list(reversed(self._held_keys)):
            self.key_up(code)
        self._fb.release_all()
        if self._down_at is not None:
            self.mouse_up(*self._down_at)
