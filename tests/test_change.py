"""变化判定。样本数值贴着 2026-09-07 的真机标定走
（runs/calibration-20260907-121553.json）：
  文本集合差异  噪声 ≤2，真实变化 27-51 → 阈值 8
  aHash 汉明    噪声 0，真实变化 0-34    → 阈值 2
两个判据是「或」的关系，因为它们互补：文字密集页滚动 aHash 可能为 0 而文本差 36，
图片页滚动则 aHash 34。
"""
from types import SimpleNamespace

from iphone_agent import config
from iphone_agent.perceive.change import did_change


def obs(ahash, texts):
    return SimpleNamespace(ahash=ahash, text_set=set(texts))


def bits(n):
    return (1 << n) - 1


def test_static_screen_is_not_a_change():
    """真机静止 35 个样本：aHash 恒 0，文本最多差 2。"""
    r = did_change(obs(0b111, {"a", "b"}), obs(0b111, {"a", "b", "c"}))
    assert not r.changed and r.hamming == 0 and r.text_diff == 1


def test_text_dense_page_scroll_detected_by_text_alone():
    """关于本机滚半屏的真实形状：aHash=0，文本=36。图像判据在这里是瞎的。"""
    before = obs(0, {f"t{i}" for i in range(20)})
    after = obs(0, {f"u{i}" for i in range(16)})
    r = did_change(before, after)
    assert r.hamming == 0
    assert r.text_diff == 36
    assert r.changed, "文本判据必须独自兜住文字密集页的滚动"


def test_image_page_scroll_detected_by_ahash_alone():
    """图片页滚动可能一个字都不变（相册、视频列表），这时全靠 aHash。"""
    r = did_change(obs(0, {"相册"}), obs(bits(34), {"相册"}))
    assert r.text_diff == 0 and r.hamming == 34
    assert r.changed, "aHash 必须独自兜住只换图不换字的页面"


def test_noise_sized_differences_stay_below_both_thresholds():
    """噪声上限：aHash 0、文本 2。两者都不该触发。"""
    r = did_change(obs(0, {"a", "b", "c"}), obs(0, {"a", "b", "d"}))
    assert r.text_diff == 2 <= config.TEXT_DIFF_THRESHOLD
    assert not r.changed


def test_thresholds_sit_between_measured_noise_and_measured_change():
    """阈值必须落在实测噪声与实测最小真实变化之间，否则标定就白做了。"""
    assert 2 < config.TEXT_DIFF_THRESHOLD < 27, "文本阈值离开了噪声2与真实变化27之间"
    assert config.AHASH_CHANGED_THRESHOLD > 0, "aHash 阈值必须高于实测噪声 0"
    assert config.STABLE_THRESHOLD >= 0


# --- 局部变化：整屏判据看不见小控件 ---
#
# 2026-09-08 实测：开关翻转两次，整屏 aHash 汉明都是 0、文本差都是 0，
# 而点击处周围的局部 MAD 是 17.0 / 16.98；点空白处三次都是 0.0。
# 噪声 0 / 信号 17，阈值取 4。
#
# 不修的后果不是「少报一次变化」：模型点了开关、工具说没变化，
# 它以为没点中就再点一次 —— 又翻回去了，熔断还把这算成无进展。

def test_local_mad_sees_a_change_the_whole_screen_judge_misses():
    from PIL import Image

    from iphone_agent import config
    from iphone_agent.perceive.change import local_mad
    a = Image.new("RGB", (600, 1200), "white")
    b = a.copy()
    for x in range(480, 520):          # 只改一小块，相当于一个开关翻转
        for y in range(690, 710):
            b.putpixel((x, y), (0, 200, 0))
    assert local_mad(a, b, 500, 700) >= config.LOCAL_CHANGED_MAD
    assert local_mad(a, b, 100, 100) == 0.0, "在别处不该看到它"


def test_local_mad_is_zero_when_nothing_changed():
    from PIL import Image

    from iphone_agent.perceive.change import local_mad
    a = Image.new("RGB", (600, 1200), "white")
    assert local_mad(a, a.copy(), 300, 600) == 0.0


def test_local_mad_reads_the_same_scene_the_same_at_any_image_size():
    """半径是比例，所以同一个场景放大一倍，读数应该基本不变。

    这正是它从绝对像素改成比例买到的性质：换一台屏更大的手机、或者只是
    把镜像窗口拉大，图像宽从 624 变成 900，写死 40px 就只罩住开关一角，
    MAD 被周围没变的像素稀释 —— 同一个开关翻转会读出不同的数。
    """
    from PIL import Image

    from iphone_agent.perceive.change import local_mad

    def scene(k):
        """一个开关大小的色块：位置和尺寸都按 k 缩放，几何关系不变。"""
        a = Image.new("RGB", (600 * k, 1200 * k), "white")
        b = a.copy()
        for x in range(480 * k, 520 * k):
            for y in range(690 * k, 710 * k):
                b.putpixel((x, y), (0, 200, 0))
        return local_mad(a, b, 500 * k, 700 * k)

    small, big = scene(1), scene(2)
    assert abs(small - big) < 1.0, f"换个尺寸读数就变了：{small:.2f} vs {big:.2f}"


# --- 状态栏必须裁掉，而且两个判据要裁同一块 ---
#
# 2026-09-08 实测（图像 624x1388）：顶部约 104px 是机身黑边，状态栏文字在
# y=104~133，折算比例 0.075~0.096。原来 STATUS_BAR_CROP_RATIO=0.05，只裁掉了黑边，
# **时钟和电量原样留在里面**。时钟每分钟变一次、OCR 还读不稳
# （'12:221' / '12:22⑦'、'17:054' / '17:06'），于是「什么都没发生」也有 2 的文本差。
# 而且 aHash 裁了、text_set 没裁 —— 同一条规矩两个入口只在一个入口执行。

def test_status_bar_ratio_actually_covers_the_status_bar():
    from iphone_agent import config
    # 实测这台：状态栏文字下沿在 y=133，图像高 1388
    assert config.STATUS_BAR_CROP_RATIO * 1388 >= 133, "裁不到状态栏底部"
    assert config.STATUS_BAR_CROP_RATIO < 0.20, "裁太多会把真内容也吃掉"


def test_text_set_excludes_the_status_bar():
    """时钟变了不该算画面变了。"""
    import time as _t

    from PIL import Image

    from iphone_agent import config
    from iphone_agent.driver.geometry import Frame, Rect
    from iphone_agent.perceive.elements import build_observation
    from iphone_agent.perceive.ocr import RawBox
    H = 1388
    f = Frame(Image.new("RGB", (624, H), "white"), 624, H, Rect(0, 0, 312, 694), _t.time(), 1)
    top = 1 - config.STATUS_BAR_CROP_RATIO / 2          # 下边原点：靠近顶部
    o = build_observation(f, [RawBox("02:55", 0.9, 0.1, top, 0.1, 0.02),
                              RawBox("设置", 0.9, 0.1, 0.5, 0.2, 0.03)], observation_id=1)
    assert "设置" in o.text_set
    assert "02:55" not in o.text_set, "时钟进了判定用的文字集合"
    assert "02:55" in o.elements_text, "但模型仍然应该看得到它"
