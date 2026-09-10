"""open_app 查表直达：布局表只给候选页和格子，当前帧负责确认（设计说明）。零打字。"""
from iphone_agent.driver.geometry import Rect, image_to_screen
from iphone_agent.harness.actions import Action, icon_y_above_label
from iphone_agent.harness.executor import Executor
from iphone_agent.perceive.elements import Element
from iphone_agent.twin import layout as L

HOME = ["设置", "照片", "日历", "备忘录", "时钟", "计算器", "天气", "相机"]     # ≥3 个主屏特征词
APP = ["通用", "关于本机", "软件更新"]


def _elements(labels):
    return [Element(i + 1, t, 0.9, (100 + (i % 4) * 138 - 40, 280 + (i // 4) * 150 - 11,
                                    100 + (i % 4) * 138 + 40, 280 + (i // 4) * 150 + 11),
                    (100 + (i % 4) * 138, 280 + (i // 4) * 150))
            for i, t in enumerate(labels)]


def _layout_with(labels, order=1):
    lay = L.Layout()
    lay.upsert_page(L.infer_page(_elements(labels), 624, 1388, order=order))
    return lay


def _open(fake_env, frames, layout, name="设置"):
    dev, per, _ = fake_env(frames)
    obs = per.observe(dev.capture())
    ex = Executor(dev, per, layout=layout)
    res, new = ex.run(Action("open_app", {"name": name}, "r", None, "c1"), obs)
    return res, new, dev


def test_known_app_is_opened_by_tapping_its_icon_without_typing(fake_env):
    """帧：起点（任意）→ 按 home 后的主屏 → 点图标后进了 App。

    Ruling 3：settle 一次 settle() 消耗几帧是机械细节（取决于 fast_timing 夹具的
    poll_ms/stable_span_ms 与真实时钟的相对误差，同一份代码在不同机器上跑出来的
    帧数会有 ±1 的抖动），简报里写的固定长度帧列表在这台机器上偶尔会在「刚翻回主屏」
    与「点完图标进了 App」这两段之间的分界处摆动，导致 cur 落在错误的一侧。
    改成跟 test_open_app_falls_back_to_the_home_screen_icon 同样的「按了哪个键就换哪一叠帧」
    —— 断言（零打字、via==layout、page==1、找到了「关于本机」）与简报完全一致，只是不再靠数帧数。
    """
    from tests.conftest import FakeDevice, frame_with_text, perceiver_for
    start = [frame_with_text(APP, i) for i in range(1, 5)]
    f_home = [frame_with_text(HOME, i) for i in range(5, 30)]
    f_in = [frame_with_text(APP, i) for i in range(30, 50)]     # 点开后的画面：含「关于本机」

    class Dev(FakeDevice):
        def key(self, n):
            super().key(n)
            if n == "home":
                self._frames, self._i = f_home, 0

        def tap(self, x, y):
            super().tap(x, y)
            if self._frames is f_home:
                self._frames, self._i = f_in, 0

    dev = Dev(start)
    per = perceiver_for(start + f_home + f_in)
    obs = per.observe(dev.capture())
    ex = Executor(dev, per, layout=_layout_with(HOME))
    res, new = ex.run(Action("open_app", {"name": "设置"}, "r", None, "c1"), obs)
    assert res.ok and res.extra["via"] == "layout" and res.extra["page"] == 1
    assert not any(c[0] == "type" for c in dev.calls), "查表直达零打字"
    assert not any(c == ("key", "spotlight") for c in dev.calls), "没走 Spotlight"
    assert any(c[0] == "tap" for c in dev.calls)
    assert "关于本机" in new.text_set


def test_label_missing_on_the_page_falls_back_to_spotlight(fake_env):
    """布局表说在第 1 页，翻过去当前帧上没有这个标签（用户挪走了）：不猜坐标，退回 Spotlight。

    ⚠ `other` 不能像原简报那样选一份「几乎是主屏、只是少了『设置』」的列表 ——
    那种列表本身就满足 looks_like_home（HOME_MARKERS_MIN=3），第三道闸（点后核身份）会
    顺带把它也拦下来，测出来的红/绿分不清是哪道闸拦的。这里换成一份跟 HOME_MARKERS
    完全不重叠的列表，才是专门测第一道闸（当前帧上找不到标签）。
    """
    other = ["通用", "关于本机", "辅助功能", "隐私与安全性"]
    res, new, dev = _open(fake_env, [APP] + [other] * 12, _layout_with(HOME))
    assert any(c == ("key", "spotlight") for c in dev.calls)
    assert res.extra.get("via") != "layout"


def test_tap_that_leaves_us_on_home_is_not_a_success(fake_env):
    """点了图标还在主屏（点空了）：不算开成，退回 Spotlight。"""
    res, new, dev = _open(fake_env, [APP] + [HOME] * 12, _layout_with(HOME))
    assert any(c == ("key", "spotlight") for c in dev.calls)
    assert res.extra.get("via") != "layout"


def test_app_on_page_two_flips_once_before_tapping(fake_env, monkeypatch):
    """起点 → 主屏第 1 页 → 翻页后第 2 页 → 点图标后进了微信。

    同上一条的 Ruling 3 说明：改成「按了哪个键/滚了哪个方向就换哪一叠帧」，不靠数帧数。
    翻页之后 executor 会 `time.sleep(config.AFTER_PAGE_FLIP_S)`（真机 2 秒，见那个常量的注释）——
    测试不能真睡这 2 秒，monkeypatch 成 0（precedent test_open_app_falls_back_to_the_home_screen_icon
    没有翻页所以没撞上这个问题；这条测试恰恰会翻页，必须处理）。
    """
    from iphone_agent import config
    from tests.conftest import FakeDevice, frame_with_text, perceiver_for
    monkeypatch.setattr(config, "AFTER_PAGE_FLIP_S", 0.0)

    page2 = ["微信", "支付宝", "淘宝", "抖音", "小红书", "美团", "京东", "拼多多"]
    lay = _layout_with(HOME, order=1)
    lay.upsert_page(L.infer_page(_elements(page2), 624, 1388, order=2))
    wechat = ["微信", "通讯录", "发现", "我"]     # 点开后的画面

    start = [frame_with_text(APP, i) for i in range(1, 5)]
    f_home = [frame_with_text(HOME, i) for i in range(5, 30)]
    f_page2 = [frame_with_text(page2, i) for i in range(30, 55)]
    f_in = [frame_with_text(wechat, i) for i in range(55, 80)]

    class Dev(FakeDevice):
        def key(self, n):
            super().key(n)
            if n == "home":
                self._frames, self._i = f_home, 0

        def scroll(self, d, a):
            super().scroll(d, a)
            if self._frames is f_home:
                self._frames, self._i = f_page2, 0

        def tap(self, x, y):
            super().tap(x, y)
            if self._frames is f_page2:
                self._frames, self._i = f_in, 0

    dev = Dev(start)
    per = perceiver_for(start + f_home + f_page2 + f_in)
    obs = per.observe(dev.capture())
    ex = Executor(dev, per, layout=lay)
    res, new = ex.run(Action("open_app", {"name": "微信"}, "r", None, "c1"), obs)
    assert res.ok and res.extra["via"] == "layout" and res.extra["page"] == 2
    assert dev.calls.count(("scroll", "right", "page")) == 1
    assert not any(c[0] == "type" for c in dev.calls)


def test_no_layout_or_unknown_app_goes_straight_to_spotlight(fake_env):
    res, new, dev = _open(fake_env, [APP] * 8, None, name="设置")
    assert any(c == ("key", "spotlight") for c in dev.calls)
    res2, new2, dev2 = _open(fake_env, [APP] * 8, _layout_with(HOME), name="微信")
    assert any(c == ("key", "spotlight") for c in dev2.calls)


def test_tap_coordinates_use_the_current_frame_not_the_start_frame(fake_env):
    """闸 2：坐标必须用**当前帧** cur 的几何算（`image_to_screen(..., self._frame_of(cur))`），
    不能沿用起点帧 obs 的几何。

    默认夹具 `frame_with_text(...)` 每一帧的 `window_rect`/`width_px`/`height_px` 都相同
    （conftest 里的默认参数），这种情况下就算实现偷懒把 `cur` 换成 `obs` 去算坐标，
    数值也逐位相同——测不出这道闸。这里让起点帧（obs，400x800，rect 从 (0,0) 起）和
    翻回主屏后观察到的帧（cur，624x1388，rect 从 (40,20) 起）用不同的窗口几何：
    缩放比例和窗口偏移都不同，「用 cur 算」和「用 obs 算」必然算出不同的屏幕坐标。

    断言：实际发出的 tap 等于**独立地、只用 cur 那一帧的几何**重新算出来的期望坐标——
    不是照抄实现内部同一行代码，而是用 `icon_y_above_label` + `image_to_screen`
    这两个已经在别处验证过的函数，从 cur 的 OCR 结果重新推一遍。

    自证见 task-5-report.md 附录：临时把 `_open_app_from_layout` 里的
    `self._frame_of(cur)` 改成 `self._frame_of(obs)`，这条测试会失败；改回来会通过。
    """
    from tests.conftest import FakeDevice, frame_with_text, perceiver_for

    rect_obs = Rect(0, 0, 200, 400)          # 起点帧的窗口几何
    rect_home = Rect(40, 20, 300, 660)       # 主屏帧的窗口几何：偏移、缩放都跟 rect_obs 不同
    start = [frame_with_text(APP, i, w=400, h=800, rect=rect_obs) for i in range(1, 5)]
    f_home = [frame_with_text(HOME, i, w=624, h=1388, rect=rect_home) for i in range(5, 30)]
    f_in = [frame_with_text(APP, i, w=624, h=1388, rect=rect_home) for i in range(30, 50)]

    class Dev(FakeDevice):
        def key(self, n):
            super().key(n)
            if n == "home":
                self._frames, self._i = f_home, 0

        def tap(self, x, y):
            super().tap(x, y)
            if self._frames is f_home:
                self._frames, self._i = f_in, 0

    dev = Dev(start)
    per = perceiver_for(start + f_home + f_in)
    obs = per.observe(dev.capture())
    ex = Executor(dev, per, layout=_layout_with(HOME))

    # 独立算出「用当前帧（cur，home 几何）」应得的期望坐标——f_home 里每一帧文字相同，
    # 元素像素位置也相同，用哪一张都行。
    cur_probe = per.observe(f_home[0][0])
    label = next(e for e in cur_probe.elements if e.text.replace(" ", "") == "设置")
    iy = icon_y_above_label(label.center[1], label.box)
    expected = image_to_screen(label.center[0], iy, ex._frame_of(cur_probe))

    res, new = ex.run(Action("open_app", {"name": "设置"}, "r", None, "c1"), obs)

    assert res.ok and res.extra["via"] == "layout"
    assert ("tap", *expected) in dev.calls, (
        f"期望用当前帧几何算出的 tap 坐标 {expected}，实际调用是 {dev.calls}")


# ---------- 终审：直达没成要看得见（I3）、坏表不变 device_error（I2）、M2/M3/M4 ----------

def _switching_dev(start, on_home, on_tap=None):
    """按了 home 就换成 on_home 那叠帧；在 on_home 上点了就换成 on_tap（None = 点了画面不变）。
    同 test_known_app_is_opened_by_tapping_its_icon_without_typing 的做法：不靠数帧数。"""
    from tests.conftest import FakeDevice

    class Dev(FakeDevice):
        def key(self, n):
            super().key(n)
            if n == "home":
                self._frames, self._i = on_home, 0

        def tap(self, x, y):
            super().tap(x, y)
            if on_tap is not None and self._frames is on_home:
                self._frames, self._i = on_tap, 0

    return Dev(start)


def _grid_perceiver(frames):
    """像 conftest.perceiver_for，但把每帧的文字排成主屏那样的 4 列网格（infer_page 认得出）。
    位置取自真机标注：列 x 105/243/381/519、行 y 从 290 起每 150 一行（624×1388），框 80×22。"""
    from iphone_agent.perceive.observe import Perceiver
    from iphone_agent.perceive.ocr import RawBox
    texts = {f.frame_id: t for f, t in frames}
    cols = (105, 243, 381, 519)

    def ocr(img):
        out = []
        for i, t in enumerate(texts.get(ocr.fid, [])):
            cx, cy = cols[i % 4], 290 + (i // 4) * 150
            out.append(RawBox(t, 0.99, (cx - 40) / 624, 1 - (cy + 11) / 1388, 80 / 624, 22 / 1388))
        return out

    ocr.fid = None
    per = Perceiver(ocr=ocr)
    orig = per.observe

    def observe(frame):
        ocr.fid = frame.frame_id
        return orig(frame)

    per.observe = observe
    return per


def _run_open(dev, per, layout, name):
    obs = per.observe(dev.capture())
    return Executor(dev, per, layout=layout).run(Action("open_app", {"name": name}, "r", None, "c1"), obs)


def test_miss_reason_no_hit_is_recorded_on_the_spotlight_result(fake_env):
    """表里没有这个 App：走 Spotlight，最终结果上记 layout_miss=no_hit。"""
    res, new, dev = _open(fake_env, [APP] + [["微信", "通讯录"]] * 12, _layout_with(HOME), name="微信")
    assert any(c == ("key", "spotlight") for c in dev.calls)
    assert res.ok and res.extra["via"] == "row"
    assert res.extra["layout_miss"] == "no_hit"
    assert '"layout_miss": "no_hit"' in res.to_json(), "extra 摊平进 steps.jsonl 的 result 顶层"


def test_no_layout_means_no_miss_reason(fake_env):
    """没表 = 没尝试，不记 layout_miss —— 否则评测里分不清「没扫过」和「扫过但没用上」。"""
    res, new, dev = _open(fake_env, [APP] + [["设置", "通用"]] * 12, None)
    assert res.ok and "layout_miss" not in res.extra


def test_miss_reason_label_missing_even_when_spotlight_also_fails(fake_env, monkeypatch):
    """翻到了、当前帧找不到标签 → label_missing；后面 Spotlight 和翻主屏都失败（app_not_found），
    这个原因照样挂在最终那个结果上。"""
    from iphone_agent import config
    monkeypatch.setattr(config, "AFTER_PAGE_FLIP_S", 0.0)
    other = ["通用", "关于本机", "辅助功能", "隐私与安全性"]
    res, new, dev = _open(fake_env, [APP] + [other] * 12, _layout_with(HOME))
    assert res.error == "app_not_found"
    assert res.extra["layout_miss"] == "label_missing"
    assert res.extra["typed"] is False, "Spotlight 那条路自己的 extra 不能被覆盖（_channel_failure 读它）"


def test_miss_reason_still_home_when_the_tap_leaves_us_on_home(fake_env):
    res, new, dev = _open(fake_env, [APP] + [HOME] * 12, _layout_with(HOME))
    assert any(c == ("key", "spotlight") for c in dev.calls)
    assert res.extra.get("via") != "layout"
    assert res.extra["layout_miss"] == "still_home"


def test_tap_on_a_third_party_page_that_changes_nothing_is_still_home():
    """M2：一页全是第三方 App（0 个主屏特征词，looks_like_home 认不出，但成得了网格），
    点完画面不变 —— 原来既不像 Spotlight 也不像主屏，报 ok=True, via=layout 假成功。
    点后再认一次：还能认出同一页主屏 = 没开成。"""
    from tests.conftest import frame_with_text
    from tests.test_twin_scan import THIRD_PARTY_LABELS
    start = [frame_with_text(APP, i) for i in range(1, 5)]
    f_page = [frame_with_text(THIRD_PARTY_LABELS, i) for i in range(5, 60)]
    dev = _switching_dev(start, f_page, on_tap=None)          # 点了画面不变
    per = _grid_perceiver(start + f_page)
    res, new = _run_open(dev, per, _layout_with(THIRD_PARTY_LABELS), "微博")
    assert any(c == ("key", "spotlight") for c in dev.calls), "没开成要退回 Spotlight"
    assert res.extra.get("via") != "layout"
    assert res.extra["layout_miss"] == "still_home"


def test_tap_on_a_third_party_page_that_opens_the_app_is_a_success():
    """M2 的另一半：同一页点完真的进了 App（认不出主屏网格）→ 照常直达成功，不误杀。"""
    from tests.conftest import frame_with_text
    from tests.test_twin_scan import THIRD_PARTY_LABELS
    start = [frame_with_text(APP, i) for i in range(1, 5)]
    f_page = [frame_with_text(THIRD_PARTY_LABELS, i) for i in range(5, 60)]
    f_in = [frame_with_text(["微博", "推荐"], i) for i in range(60, 90)]
    dev = _switching_dev(start, f_page, on_tap=f_in)
    per = _grid_perceiver(start + f_page + f_in)
    res, new = _run_open(dev, per, _layout_with(THIRD_PARTY_LABELS), "微博")
    assert res.ok and res.extra["via"] == "layout" and "layout_miss" not in res.extra
    assert not any(c == ("key", "spotlight") for c in dev.calls)


def test_broken_layout_falls_back_to_spotlight_not_device_error(fake_env):
    """I2：表坏了（find_app 会抛）→ 等于没有孪生：走 Spotlight，layout_miss=error，不是 device_error。"""
    bad = L.Layout([L.Page("p_bad", 1, ["x"], [L.Cell(1, 1, 123, None)], True, "")])   # label 不是 str
    res, new, dev = _open(fake_env, [APP] + [["设置", "通用"]] * 12, bad)
    assert res.error != "device_error", res.hint
    assert any(c == ("key", "spotlight") for c in dev.calls)
    assert res.ok and res.extra["via"] == "row" and res.extra["layout_miss"] == "error"


def test_device_errors_on_the_direct_path_still_surface(fake_env):
    """I2 只包孪生自己的逻辑：设备调用（这里是回主屏的 key home）抛的异常照旧往外抛，
    和 Spotlight 那条路一致 —— 设备坏了不该被当成「表没用上」悄悄换条路。"""
    dev, per, _ = fake_env([APP] * 12)
    dev.raise_on["key"] = RuntimeError("镜像断了")
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per, layout=_layout_with(HOME)).run(
        Action("open_app", {"name": "设置"}, "r", None, "c1"), obs)
    assert res.error == "device_error"


def test_page_order_out_of_range_is_not_flipped_to(fake_env, monkeypatch):
    """M3：表里的页码不在 1..HOME_PAGES_MAX → 不回主屏、不翻页，直接 Spotlight。"""
    from iphone_agent import config
    # 闸坏了会翻 49 页、每页硬等 AFTER_PAGE_FLIP_S —— 测试要红得快，不要卡 100 秒
    monkeypatch.setattr(config, "AFTER_PAGE_FLIP_S", 0.0)
    for order in (50, 0, config.HOME_PAGES_MAX + 1):
        res, new, dev = _open(fake_env, [APP] + [["设置", "通用"]] * 12, _layout_with(HOME, order=order))
        assert dev.calls[0] == ("key", "spotlight"), (order, dev.calls[:3])
        assert not any(c[0] == "scroll" for c in dev.calls), order
        assert res.extra["layout_miss"] == "page_out_of_range", order


def test_label_on_the_current_frame_is_matched_case_insensitively():
    """M4：表里是「App Store」，当前帧 OCR 读成「App store」—— 查表和找标签用同一个 norm_label，直达成功。"""
    from tests.conftest import frame_with_text, perceiver_for
    home = ["设置", "照片", "日历", "App store", "时钟", "计算器", "天气", "相机"]
    start = [frame_with_text(APP, i) for i in range(1, 5)]
    f_home = [frame_with_text(home, i) for i in range(5, 30)]
    f_in = [frame_with_text(["Today", "游戏", "App"], i) for i in range(30, 50)]
    dev = _switching_dev(start, f_home, on_tap=f_in)
    per = perceiver_for(start + f_home + f_in)
    lay = _layout_with(["设置", "照片", "日历", "App Store", "时钟", "计算器", "天气", "相机"])
    res, new = _run_open(dev, per, lay, "App Store")
    assert res.ok and res.extra.get("via") == "layout", res.extra
    assert not any(c == ("key", "spotlight") for c in dev.calls)


def test_direct_open_that_lands_in_another_app_falls_back_and_says_so(fake_env):
    """直达点了「设置」图标，进的却是备忘录（图标被挪过 / 标签被遮）：看图说不是 →
    记 layout_miss=wrong_app、退回 Spotlight，不报假成功（2026-09-10 同一类问题）。"""
    from tests.conftest import FakeDevice, SeesApps, frame_with_text, perceiver_for
    start = [frame_with_text(APP, i) for i in range(1, 5)]
    f_home = [frame_with_text(HOME, i) for i in range(5, 30)]
    f_note = [frame_with_text(["返回", "Wiamzusr1lp"], i) for i in range(30, 50)]

    class Dev(FakeDevice):
        def key(self, n):
            super().key(n)
            if n == "home":
                self._frames, self._i = f_home, 0

        def tap(self, x, y):
            super().tap(x, y)
            if self._frames is f_home:
                self._frames, self._i = f_note, 0

    frames = start + f_home + f_note
    dev, per = Dev(start), perceiver_for(frames)
    asker = SeesApps(frames, lambda t: "备忘录" if "Wiamzusr1lp" in t else ("设置" if "关于本机" in t else "主屏"))
    ex = Executor(dev, per, layout=_layout_with(HOME), asker=asker)
    res, _ = ex.run(Action("open_app", {"name": "设置"}, "r", None, "c1"), per.observe(dev.capture()))
    assert res.extra.get("layout_miss") == "wrong_app", res.to_json()
    assert res.extra["wrong_app"][0]["via"] == "layout" and res.extra["wrong_app"][0]["seen"] == "备忘录"
    assert any(c == ("key", "spotlight") for c in dev.calls), "直达开错了要退回 Spotlight"
    assert not (res.ok and res.extra.get("via") == "layout")
