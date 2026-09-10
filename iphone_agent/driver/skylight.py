"""后台注入：不抢焦点地把鼠标事件投给镜像窗口。

## 为什么需要它

普通的 `CGEventPost(kCGHIDEventTap, ...)` 打在**当前活跃 App** 上，所以每次动作前
都得把镜像窗口拉到最前 —— 用户在电脑上做别的事就被打断。而用户只需要「能看见」，
镜像窗口本来就一直可见，不需要它占着前台。

先试过更便宜的 `CGEventPostToPid`（把事件直接投给进程）：2026-09-07 真机实测
**焦点确实没被抢，但手机完全没反应**（hamming=0）。这与 设计说明 记录的
phone-harness 观测一致 —— 那条路对鼠标点击是不通的。

## 机制来源（按 D12 的参考边界）

机制事实来自 **yabai 的公开源码**（MIT，`src/window_manager.c` 的
`window_manager_make_key_window`）与苹果公开的 Quartz/AppKit 文档，
已整理在 `设计说明-技术依赖与上游.md`。本文件按那份规格自己写，不看第三方实现。

macOS 的窗口服务器接受一种**合成事件记录**，经私有框架 SkyLight 的
`SLPSPostEventRecordTo(psn, buffer)` 直投给进程。yabai 用它「聚焦窗口但不抬起」；
把记录里的位置字段填上真实坐标，同一通道就变成带位置的点击。

事件记录 = **0xf8 字节的裸缓冲区**，布局：

| 偏移 | 内容 |
|---|---|
| `0x04` | 记录长度 `0xf8` |
| `0x08` | 事件类型：1 按下、2 抬起、6 拖拽 |
| `0x10` | 全局屏幕坐标（两个 double） |
| `0x20` | 窗口内坐标（两个 double） |
| `0x3a` | yabai 要求的标志字节 `0x10` |
| `0x3c` | 目标窗口 id（uint32） |

所有写入都在缓冲区内 —— 但**不等于填错就安全**：`0xC0` 之后有指针字段，
往那儿写整数会让我们自己的进程段错误，实测还把镜像 App 搞崩过一次
（2026-09-08）。只写这里列出的那几个偏移。

⚠ `SLPSPostEventRecordTo` 是私有 API，任何 macOS 版本都可能改。上游是 yabai
（跟系统更新很紧，布局一变几天内修）。前台路径（公开 CGEvent）保留作兜底，
`doctor` 会报告当前走的是哪条。
"""
from __future__ import annotations

import ctypes
import struct

SKYLIGHT = "/System/Library/PrivateFrameworks/SkyLight.framework/SkyLight"
APPSERVICES = "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"

RECORD_SIZE = 0xF8
OFF_LENGTH = 0x04
OFF_TYPE = 0x08
OFF_LOCATION = 0x10
OFF_WINDOW_LOCATION = 0x20
OFF_FLAG = 0x3A
OFF_WINDOW_ID = 0x3C
FLAG_BYTE = 0x10

# CGSEventType
EVENT_DOWN = 1
EVENT_UP = 2
EVENT_DRAGGED = 6
EVENT_SCROLL = 22

# 竖直滚轮的 delta（int32，符号即方向：负数向下看、正数向上看）。
#
# 2026-09-08 真机扫出来的。做法：给记录的类型字段填 22，然后在 0x40 之后逐格写 delta，
# 看手机滚不滚。0x6c 命中，0x60-0x68 和 0x70-0x7c 都没反应。
#
# ⚠ **这一条没有上游参考** —— yabai 不做滚轮注入，点击那条至少有它的公开做法背书。
#   所以它比点击更可能在 macOS 大版本更新后悄悄失效，驱动准入里必须验它。
OFF_SCROLL_DELTA = 0x6C

# ⚠ 0xC0 之后有指针字段：往那儿写整数会让**我们自己的进程段错误**，
#   而且实测把镜像 App 也搞崩过一次（pid 变了、窗口缩成 38x104）。
#   所以只写 0x6c 这一格，别再对这个缓冲区做任何"反正写错也是 no-op"的假设 ——
#   文件开头那句话是错的。
UNSAFE_OFFSET_FROM = 0xC0

# _SLPSSetFrontProcessWithOptions 的 mode
kCPSUserGenerated = 0x200


class ProcessSerialNumber(ctypes.Structure):
    _fields_ = [("highLongOfPSN", ctypes.c_uint32),
                ("lowLongOfPSN", ctypes.c_uint32)]


class SkyLightUnavailable(RuntimeError):
    """私有框架或所需符号不可用 —— 调用方应退回前台路径。"""


class _Libs:
    """惰性加载，且把「符号不存在」变成一个明确的异常而不是 AttributeError。"""

    _loaded = None

    @classmethod
    def get(cls):
        if cls._loaded is None:
            try:
                sky = ctypes.CDLL(SKYLIGHT)
                app = ctypes.CDLL(APPSERVICES)
            except OSError as e:
                raise SkyLightUnavailable(f"加载框架失败：{e}") from e
            for lib, name in ((sky, "SLPSPostEventRecordTo"),
                              (sky, "_SLPSSetFrontProcessWithOptions"),
                              (app, "GetProcessForPID")):
                if not hasattr(lib, name):
                    raise SkyLightUnavailable(f"符号 {name} 不存在（macOS 版本变了？）")

            sky.SLPSPostEventRecordTo.argtypes = [ctypes.POINTER(ProcessSerialNumber),
                                                  ctypes.c_char_p]
            sky.SLPSPostEventRecordTo.restype = ctypes.c_int
            sky._SLPSSetFrontProcessWithOptions.argtypes = [ctypes.POINTER(ProcessSerialNumber),
                                                            ctypes.c_uint32, ctypes.c_uint32]
            sky._SLPSSetFrontProcessWithOptions.restype = ctypes.c_int
            app.GetProcessForPID.argtypes = [ctypes.c_int,
                                             ctypes.POINTER(ProcessSerialNumber)]
            app.GetProcessForPID.restype = ctypes.c_int
            cls._loaded = (sky, app)
        return cls._loaded


def available() -> bool:
    try:
        _Libs.get()
        return True
    except SkyLightUnavailable:
        return False


def psn_for_pid(pid: int) -> ProcessSerialNumber:
    _, app = _Libs.get()
    psn = ProcessSerialNumber()
    status = app.GetProcessForPID(pid, ctypes.byref(psn))
    if status != 0:
        raise SkyLightUnavailable(f"GetProcessForPID({pid}) 返回 {status}")
    return psn


def build_record(event_type: int, window_id: int,
                 global_xy: tuple[float, float] | None,
                 window_xy: tuple[float, float] | None) -> bytes:
    """按 yabai 逆出的布局造一条事件记录。

    位置传 None 表示「空位置」（该字段填满 0xFF）—— 这正是 yabai 用来
    「给焦点但不产生点击」的形态。填真实坐标则变成带位置的点击。
    """
    buf = bytearray(RECORD_SIZE)
    buf[OFF_LENGTH] = RECORD_SIZE
    buf[OFF_TYPE] = event_type
    buf[OFF_FLAG] = FLAG_BYTE
    struct.pack_into("<I", buf, OFF_WINDOW_ID, window_id)

    for off, xy in ((OFF_LOCATION, global_xy), (OFF_WINDOW_LOCATION, window_xy)):
        if xy is None:
            buf[off:off + 0x10] = b"\xff" * 0x10
        else:
            struct.pack_into("<dd", buf, off, float(xy[0]), float(xy[1]))
    return bytes(buf)


def post(pid: int, window_id: int, event_type: int,
         global_xy: tuple[float, float] | None = None,
         window_xy: tuple[float, float] | None = None,
         set_front: bool = True) -> None:
    """把一条事件记录投给进程。

    set_front 控制要不要先调 `_SLPSSetFrontProcessWithOptions`。
    ⚠ yabai 调它是因为它的目的就是「把焦点给这个窗口」；我们的目的只是**送达事件**。
    实测（2026-09-07）带着这个调用时，NSWorkspace 报告的前台进程会变成镜像 —— 
    虽然窗口不会视觉抬升（所以人看不出来），但键盘焦点在窗口服务器层面已经转移了，
    用户接着敲字就会打进镜像里。所以这个开关的默认值将由真机诊断决定。
    """
    sky, _ = _Libs.get()
    psn = psn_for_pid(pid)
    if set_front:
        sky._SLPSSetFrontProcessWithOptions(ctypes.byref(psn), window_id, kCPSUserGenerated)
    rec = build_record(event_type, window_id, global_xy, window_xy)
    sky.SLPSPostEventRecordTo(ctypes.byref(psn), rec)


def make_key_window(pid: int, window_id: int, set_front: bool = True) -> None:
    """把窗口设为 key window，但不把 App 抬到前台。

    用空位置的一对 down/up —— yabai 正是用这个给焦点而不产生点击。
    键盘事件要送到 key window 的文本响应者，所以打字前需要它。
    """
    post(pid, window_id, EVENT_DOWN, set_front=set_front)
    post(pid, window_id, EVENT_UP, set_front=set_front)


def post_scroll(pid: int, window_id: int, delta: int,
                global_xy: tuple[float, float] | None = None,
                window_xy: tuple[float, float] | None = None) -> None:
    """⚠⚠ **会让 iPhone 镜像 App 崩溃。不要接进任何代码路径。** ⚠⚠

    保留它只是为了留住 2026-09-08 的实验结论，将来有更好的线索时能接着做。

    ## 已经验证成立的部分

    · CGSEventType **22** 的记录确实能被投递，手机确实会滚（hamming 29）。
    · 竖直 delta 在偏移 **0x6c**（int32，负数向下看、正数向上看）。
      0x60-0x68 和 0x70-0x7c 都没反应。
    · **位置字段要填**：同样的 delta，填位置 hamming=29，不填只有 2。
      位置只是记录里的一个数，填它**不需要动真实指针**。
    · **路由认的是记录里的 window id，不认指针位置** —— 指针停在 (20,20)
      压在别的窗口上时照样滚。所以窗口被盖住、被压到后台、挪到屏幕外都不影响。
      这正是"完全后台化"想要的性质。

    ## 为什么还是不能用

    发这种记录会让镜像 App **直接崩溃**，用户看到的是反复「正在连接」。
    2026-09-08 一个下午崩了 **17 次**，全部落在实验的三个时间窗内，
    窗口之外一次都没有（当天所有点击、键盘操作都没崩过）。

    崩溃签名（`~/Library/Logs/DiagnosticReports/iPhone Mirroring-*.ips`）：

        EXC_BREAKPOINT / SIGTRAP
        UniversalHIDKit  <-  ScreenSharingKit  <-  ScreenContinuityUI
        栈里有 Swift 的 Sequence.reduce

    EXC_BREAKPOINT 在 Swift 里是**运行时断言失败**（越界、强解包 nil、溢出），
    不是内存错误。UniversalHIDKit 的活是把 Mac 事件翻译成 iOS HID 事件，
    而我们这条记录 248 字节里几乎全是 0 —— 真实滚轮事件还带着轴的数量、
    每轴 delta、滚动阶段、惯性阶段。`Sequence.reduce` 基本坐实了它在
    遍历"若干个轴"，而数量字段是 0 或垃圾值，于是断言炸掉。
    这也解释了"有时先滚一下再崩"：事件被部分处理了，然后在后续字段上炸。

    ## 再做需要什么

    **不要继续盲猜字段** —— 每猜错一次的代价是用户的镜像 App 崩一次，
    而且就算试出一组不崩的，失败模式也是"App 挂掉"而不是"滚不动"，
    macOS 一更新就可能开始随机崩溃。

    要接着做，得先拿到**一条真实滚轮事件的完整记录**做对照
    （比如从事件流里截获再 dump），照着填字段，而不是扫偏移量。
    """
    sky, _ = _Libs.get()
    psn = psn_for_pid(pid)
    buf = bytearray(build_record(EVENT_SCROLL, window_id, global_xy, window_xy))
    struct.pack_into("<i", buf, OFF_SCROLL_DELTA, int(delta))
    sky.SLPSPostEventRecordTo(ctypes.byref(psn), bytes(buf))
