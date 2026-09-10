"""抓帧路径。

真实缺陷（2026-09-05 真机）：CGWindowListCreateImage 在权限齐全的情况下变成稳定挂起
30 秒后返回空图，且 SIGALRM 掐不断（阻塞在 mach_msg 里，设 2 秒和 10 秒都是 30 秒才回来）。
一个无法设上限的调用不能放在主循环里，所以改用 screencapture 子进程。
"""
import subprocess

import pytest
from PIL import Image

from iphone_agent.driver import capture as cap
from iphone_agent.driver.geometry import Rect
from iphone_agent.driver.window import MirrorWindow

WIN = MirrorWindow(window_id=7, pid=1, rect=Rect(0, 0, 100, 200))
RECT = Rect(10, 20, 100, 200)


@pytest.fixture(autouse=True)
def _stub_rect(monkeypatch):
    monkeypatch.setattr(cap, "current_rect", lambda wid: RECT)


def test_returns_frame_with_scale_derived_from_image(monkeypatch):
    monkeypatch.setattr(cap, "_screencapture", lambda wid, t: Image.new("RGB", (200, 400)))
    f = cap.capture_window(WIN, frame_id=3)
    assert f.frame_id == 3 and (f.width_px, f.height_px) == (200, 400)
    assert f.window_rect is RECT
    assert f.scale == 2.0  # 200/100，由图算出而非写死


def test_scale_follows_actual_image_not_a_hardcoded_retina_factor(monkeypatch):
    monkeypatch.setattr(cap, "_screencapture", lambda wid, t: Image.new("RGB", (100, 200)))
    assert cap.capture_window(WIN, frame_id=1).scale == 1.0


def test_failure_names_the_window_and_the_likely_causes(monkeypatch):
    monkeypatch.setattr(cap, "_screencapture", lambda wid, t: None)
    with pytest.raises(cap.CaptureFailed) as ei:
        cap.capture_window(WIN, frame_id=1, timeout_s=3.0)
    msg = str(ei.value)
    assert "7" in msg and "3.0" in msg and "录屏权限" in msg


def test_screencapture_invoked_with_no_shutter_and_no_shadow(monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        seen["timeout"] = kw.get("timeout")
        Image.new("RGB", (4, 4)).save(argv[-1])
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(cap.subprocess, "run", fake_run)
    assert cap._screencapture(42, timeout_s=5.0) is not None
    argv = seen["argv"]
    assert argv[0] == "screencapture"
    assert "-x" in argv          # 不播快门声
    assert "-o" in argv          # 不要阴影，否则图比窗口大一圈、坐标换算就错了
    assert "-l" in argv and "42" in argv
    assert seen["timeout"] == 5.0


def test_nonzero_exit_and_empty_file_both_treated_as_failure(monkeypatch):
    monkeypatch.setattr(cap.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, b"", b"boom"))
    assert cap._screencapture(42, timeout_s=5.0) is None

    def wrote_nothing(argv, **kw):
        open(argv[-1], "wb").close()
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(cap.subprocess, "run", wrote_nothing)
    assert cap._screencapture(42, timeout_s=5.0) is None


def test_subprocess_timeout_returns_none_not_raise(monkeypatch):
    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw.get("timeout", 1))

    monkeypatch.setattr(cap.subprocess, "run", boom)
    assert cap._screencapture(42, timeout_s=0.1) is None


# --- 窗口认的是比例，不是尺寸 ---

def test_aspect_range_covers_home_button_iphones():
    """⚠ iPhone SE / 8 是 375x667 = 1.78，原来的下界 1.8 会把它们整个筛掉。"""
    from iphone_agent.driver.window import ASPECT_MAX, ASPECT_MIN
    for name, w, h in (("iPhone SE / 8", 375, 667),
                       ("iPhone X 及以后", 375, 812),
                       ("iPhone 16 Pro Max", 440, 956)):
        r = h / w
        assert ASPECT_MIN <= r <= ASPECT_MAX, f"{name} 比例 {r:.2f} 落在 [{ASPECT_MIN},{ASPECT_MAX}] 之外"


def test_aspect_range_still_rejects_landscape_windows():
    from iphone_agent.driver.window import ASPECT_MIN
    assert ASPECT_MIN > 1440 / 900, "普通横向窗口不该被当成镜像"
