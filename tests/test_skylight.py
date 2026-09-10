"""SkyLight 事件记录的构造。

按 yabai 逆出的布局（设计说明，来源是 yabai 公开源码 + 苹果文档，符合 D12 的
参考边界）。私有 API 的投递没法在没有真机的情况下测，这里锁住**记录本身的字节布局**
—— 布局错了投递必然无效，而且是静默无效（所有写入都在缓冲区内，填错是 no-op）。
"""
import struct

from iphone_agent.driver import skylight as sl


def test_record_is_exactly_the_expected_size():
    r = sl.build_record(sl.EVENT_DOWN, 1, (0, 0), (0, 0))
    assert len(r) == 0xF8


def test_fixed_fields_match_the_yabai_layout():
    r = sl.build_record(sl.EVENT_DRAGGED, 0xABCDEF01, (1.0, 2.0), (3.0, 4.0))
    assert r[0x04] == 0xF8, "0x04 必须是记录长度"
    assert r[0x08] == sl.EVENT_DRAGGED, "0x08 是事件类型"
    assert r[0x3A] == 0x10, "0x3a 是 yabai 要求的标志字节"
    assert struct.unpack_from("<I", r, 0x3C)[0] == 0xABCDEF01, "0x3c 是目标窗口 id"


def test_coordinates_are_two_doubles_at_their_offsets():
    r = sl.build_record(sl.EVENT_DOWN, 1, (100.5, 200.25), (10.0, 20.0))
    assert struct.unpack_from("<dd", r, 0x10) == (100.5, 200.25), "0x10 是全局坐标"
    assert struct.unpack_from("<dd", r, 0x20) == (10.0, 20.0), "0x20 是窗口内坐标"


def test_none_position_means_all_ff():
    """空位置正是 yabai 用来「给焦点但不产生点击」的形态。"""
    r = sl.build_record(sl.EVENT_DOWN, 1, None, None)
    assert r[0x10:0x20] == b"\xff" * 0x10
    assert r[0x20:0x30] == b"\xff" * 0x10


def test_event_types_are_the_documented_values():
    assert (sl.EVENT_DOWN, sl.EVENT_UP, sl.EVENT_DRAGGED) == (1, 2, 6)


def test_everything_outside_the_named_fields_stays_zero():
    """所有写入都必须落在已知字段里；越界写入在私有 API 上是静默的未定义行为。"""
    r = sl.build_record(sl.EVENT_DOWN, 0xFFFFFFFF, (1.0, 1.0), (1.0, 1.0))
    known = set(range(0x04, 0x05)) | set(range(0x08, 0x09)) | set(range(0x10, 0x30)) \
        | set(range(0x3A, 0x3B)) | set(range(0x3C, 0x40))
    stray = [i for i in range(0xF8) if i not in known and r[i] != 0]
    assert not stray, f"这些偏移不该被写：{stray}"


# --- 滚轮那条路是封着的 ---
#
# 2026-09-08：SkyLight 发滚轮技术上成立（类型 22、偏移 0x6c、路由认 window id
# 不认指针），但**会让镜像 App 崩溃** —— 一个下午崩了 17 次，签名是
# UniversalHIDKit 里的 Swift 断言失败。所以它只作为实验结论留着，不接进任何路径。

def test_scroll_is_not_wired_into_any_injection_path():
    """⚠ 这条守的是「别哪天顺手接上」。失败模式是用户的镜像 App 挂掉，
    不是滚动不生效 —— 比现在这条路的失败模式严重一个量级。"""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "iphone_agent"
    for f in root.rglob("*.py"):
        if f.name == "skylight.py":
            continue
        assert "post_scroll" not in f.read_text(encoding="utf-8"), f"{f} 引用了 post_scroll"


def test_the_danger_is_written_where_someone_would_read_it():
    from iphone_agent.driver import skylight as sl
    doc = sl.post_scroll.__doc__ or ""
    assert "崩溃" in doc and "UniversalHIDKit" in doc, "崩溃签名没写进文档字符串"
    assert "不要接进" in doc


def test_the_unsafe_region_is_marked():
    """0xC0 之后有指针字段，写整数会让**我们自己的进程**段错误。"""
    from iphone_agent.driver import skylight as sl
    assert sl.UNSAFE_OFFSET_FROM == 0xC0
    assert sl.OFF_SCROLL_DELTA < sl.UNSAFE_OFFSET_FROM
