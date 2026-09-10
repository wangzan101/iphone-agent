"""镜像会话插页的判定。

两种插页会挡在手机画面前面：
  「连接暂停」+「继续」            —— 点一下能回来
  「iPhone 使用中」+「锁定 iPhone」 —— 手机被拿起，只能人去解决
"""
from iphone_agent.driver.session import IN_USE, PAUSED, detect


def test_paused_page_recognised():
    assert detect(["连接暂停", "继续"]) == PAUSED


def test_phone_in_use_recognised():
    assert detect(["iPhone 使用中", "锁定 iPhone 以连接"]) == IN_USE


def test_in_use_wins_over_paused():
    """两种字样同屏时按「使用中」算 —— 那一种恢复不了，
    早点认出来比误判成 paused 然后徒劳地点半天强。"""
    assert detect(["连接暂停", "继续", "iPhone 使用中"]) == IN_USE


def test_a_continue_button_alone_is_not_a_disconnect():
    """⚠ 这是最重要的一条。「继续」两个字在引导页、协议页里到处都是。
    只看它，正常画面会被误判成断线，然后去点一个不该点的按钮。"""
    assert detect(["欢迎使用", "继续", "跳过"]) is None
    assert detect(["同意并继续"]) is None


def test_normal_screen_is_none():
    assert detect(["设置", "飞行模式", "无线局域网"]) is None
    assert detect([]) is None


def test_spacing_and_variants_tolerated():
    """OCR 时有时没有空格，两种写法都得认。"""
    assert detect(["iPhone使用中"]) == IN_USE
    assert detect(["锁定iPhone以连接"]) == IN_USE


def test_reopen_is_exposed_for_the_shrunken_window_case():
    """窗口会缩成一个小条（实测 31x114），这时 find_mirror_window 直接找不到它
    —— 比例 3.68 超出竖屏筛选。无人值守必须能自己叫回来。"""
    from iphone_agent.driver.session import MIRROR_BUNDLE, reopen_mirror_window
    assert MIRROR_BUNDLE == "com.apple.ScreenContinuity"
    assert callable(reopen_mirror_window)
