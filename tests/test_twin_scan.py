"""扫描主屏：回第一页、一页页往右翻、写表（docs/32 §4.4）。2026-09-14 起翻遍所有主屏页。"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from iphone_agent import config
from iphone_agent.driver.geometry import Frame, Rect
from iphone_agent.perceive.elements import Element
from iphone_agent.perceive.observe import Perceiver
from iphone_agent.perceive.ocr import RawBox
from iphone_agent.twin import layout as L
from iphone_agent.twin import scan as scan_mod
from iphone_agent.twin.layout import infer_page
from iphone_agent.twin.scan import refresh_from_observation, scan_home, walk_home_pages
from tests.conftest import (
    CountingAsker,
    FakeDevice,
    PagedDevice,
    frame_with_text,
    grid_perceiver_for,
    page_frame,
)

LABELS = Path(__file__).resolve().parent.parent / "evalset" / "labels" / "00-current.json"
W, H = 624, 1388


def home_ocr():
    """按真机标注的位置吐 RawBox（归一化、左下原点），让 Perceiver 观察到一张真实布局的主屏。"""
    d = json.loads(LABELS.read_text(encoding="utf-8"))
    boxes = []
    for e in d["elements"]:
        if e.get("text_center") is None:
            continue
        cx, cy = e["text_center"]
        boxes.append(RawBox(e["label"], 0.99, (cx - 40) / W, 1 - (cy + 11) / H, 80 / W, 22 / H))
    return lambda img: list(boxes)


def home_frames(n=6):
    from PIL import Image
    return [(Frame(Image.new("RGB", (W, H), "white"), W, H, Rect(0, 0, 312, 694), 0.0, i + 1), [])
            for i in range(n)]


def test_scan_writes_the_visible_page_and_goes_home_first(tmp_path):
    dev = FakeDevice(home_frames())
    per = Perceiver(ocr=home_ocr())
    path = tmp_path / "layout.json"
    lay = scan_home(dev, per, path, log=lambda *a: None)
    assert lay is not None and path.exists()
    assert [c[0] for c in dev.calls[:3]] == ["key", "key", "key"], "先按 home 回第一页"
    assert not any(c[0] in ("tap", "type") for c in dev.calls), "扫描只看不点"
    assert lay.find_app("设置") is not None
    assert L.Layout.load(path).find_app("备忘录").row == 4


def test_scan_refuses_when_not_on_home(tmp_path):
    frames = [frame_with_text(["通用", "关于本机", "软件更新"], i + 1) for i in range(6)]
    dev = FakeDevice(frames)
    from tests.conftest import perceiver_for
    per = perceiver_for(frames)
    assert scan_home(dev, per, tmp_path / "layout.json", log=lambda *a: None) is None
    assert not (tmp_path / "layout.json").exists()


def test_scan_never_raises(tmp_path, monkeypatch):
    dev = FakeDevice(home_frames())
    dev.raise_on["capture"] = RuntimeError("镜像断了")
    assert scan_home(dev, Perceiver(ocr=home_ocr()), tmp_path / "layout.json", log=lambda *a: None) is None


def test_cli_twin_scan_and_show(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from iphone_agent.cli.commands import dispatch
    from iphone_agent.workspace import Workspace
    dev = FakeDevice(home_frames())
    session = SimpleNamespace(dev=dev, per=Perceiver(ocr=home_ocr()), workspace=Workspace(tmp_path),
                              asker=None)
    assert dispatch(session, ["twin", "scan"]) == 0
    out = capsys.readouterr().out
    # Ruling 1：scan 摘要行是 sorted(labels[:6])，真机数据前六个是 ASCII（App Store 等），
    # 不含中文「设置」——只断言这里能看到扫描结果的数字。中文断言挪到 show。
    assert "24" in out and "1 页" in out
    assert dispatch(session, ["twin", "show"]) == 0
    shown = capsys.readouterr().out
    assert "第 1 页" in shown and "设置" in shown


# --- 任务中顺手刷新：refresh_from_observation 四道闸（Task 6，控制方裁决 R-a/R-b） ---

# 8 个系统 App，≥3 个命中 looks_like_home 的 HOME_MARKERS。
HOME_LABELS = ["设置", "照片", "日历", "备忘录", "时钟", "计算器", "天气", "相机"]
# 8 个第三方 App，0 个命中 HOME_MARKERS —— 用来单独测「闸 1」。
THIRD_PARTY_LABELS = ["微博", "支付宝", "淘宝", "抖音", "小红书", "美团", "京东", "拼多多"]


def _grid_elements(labels, row_y=(290, 440)):
    """把一串标签排成 4 列 × N 行的网格（参考 tests/test_twin_layout.py 的 _page_with）。

    列 x 取 105/243/381/519（间距 138，远大于 COL_TOL·624≈37），行 y 从 290 起每 150 一行
    （远大于 ROW_TOL·1388≈56），框 (cx-40, cy-11, cx+40, cy+11)。
    """
    cols = (105, 243, 381, 519)
    els = []
    for i, t in enumerate(labels):
        cx, cy = cols[i % len(cols)], row_y[i // len(cols)]
        els.append(Element(i + 1, t, 0.99, (cx - 40, cy - 11, cx + 40, cy + 11), (cx, cy)))
    return els


def _home_obs(labels, frame_id=1, row_y=(290, 440)):
    return SimpleNamespace(elements=_grid_elements(labels, row_y), width_px=W, height_px=H, frame_id=frame_id)


def test_refresh_overwrites_a_known_page_when_an_app_moved(tmp_path):
    """已知页 + 像主屏 + 能成网格 → 写：格子更新，页的 id/order 不变，revision +1。"""
    path = tmp_path / "layout.json"
    lay = L.Layout.load(path)
    page = infer_page(_grid_elements(HOME_LABELS), W, H, order=1)
    orig_by_label = {c.label: (c.row, c.col) for c in page.cells}
    lay.upsert_page(page)
    assert lay.save(path) is True
    pid, rev1 = lay.pages[0].id, L.Layout.load(path).revision

    moved = HOME_LABELS.copy()
    moved[0], moved[-1] = moved[-1], moved[0]              # 「设置」和「相机」换了位置
    ok = refresh_from_observation(lay, _home_obs(moved), path, run_name="run-a")
    assert ok is True

    again = L.Layout.load(path)
    assert again.revision == rev1 + 1
    assert len(again.pages) == 1
    p = again.pages[0]
    assert p.id == pid and p.order == 1, "同一页：id 和顺序不变"
    new_by_label = {c.label: (c.row, c.col) for c in p.cells}
    assert new_by_label["设置"] == orig_by_label["相机"], "格子真的按新观察更新了"
    assert new_by_label["相机"] == orig_by_label["设置"]


def test_refresh_gate1_looks_like_home_blocks_a_known_non_home_page(tmp_path):
    """闸 1：表里已有一页 0 个主屏特征词（第三方 App）——过不了 looks_like_home，不写。

    这条必须是「表里有它」的场景，否则会被闸 3 先挡住，测不到闸 1。
    """
    path = tmp_path / "layout.json"
    lay = L.Layout.load(path)
    page = infer_page(_grid_elements(THIRD_PARTY_LABELS), W, H, order=1)
    assert page is not None
    lay.upsert_page(page)
    assert lay.save(path) is True
    before = L.Layout.load(path).to_dict()

    ok = refresh_from_observation(lay, _home_obs(THIRD_PARTY_LABELS), path, run_name="run-b")
    assert ok is False
    assert L.Layout.load(path).to_dict() == before, "文件不变"


def test_refresh_gate3_only_overwrites_known_pages(tmp_path):
    """闸 3：像主屏、能成网格，但表里没有这页 —— 不追加新页，不写。"""
    path = tmp_path / "layout.json"
    lay = L.Layout.load(path)                              # 空表
    ok = refresh_from_observation(lay, _home_obs(HOME_LABELS), path, run_name="run-c")
    assert ok is False
    assert lay.pages == []
    assert not path.exists()


def test_refresh_gate2_needs_a_grid(tmp_path):
    """闸 2：3 个主屏特征词排成一列（成不了 ≥3 列的网格）—— 不写。"""
    path = tmp_path / "layout.json"
    lay = L.Layout.load(path)
    els = [Element(i + 1, t, 0.99, (65, 279 + i * 150 - 11, 145, 279 + i * 150 + 11), (105, 279 + i * 150))
           for i, t in enumerate(["设置", "备忘录", "计算器"])]
    obs = SimpleNamespace(elements=els, width_px=W, height_px=H, frame_id=1)
    ok = refresh_from_observation(lay, obs, path, run_name="run-d")
    assert ok is False
    assert not path.exists()


def test_refresh_returns_false_on_stale_snapshot(tmp_path):
    """写失败（两个快照制造 RevisionConflict）→ False，不抛，磁盘不变。"""
    path = tmp_path / "layout.json"
    a = L.Layout.load(path)
    a.upsert_page(infer_page(_grid_elements(HOME_LABELS), W, H, order=1))
    a.save(path)
    b = L.Layout.load(path)                                 # 旧快照
    a.save(path)                                             # a 再写一次，磁盘 revision 变成 2
    before = L.Layout.load(path).to_dict()

    moved = HOME_LABELS.copy()
    moved[0], moved[1] = moved[1], moved[0]
    ok = refresh_from_observation(b, _home_obs(moved), path, run_name="run-e")
    assert ok is False
    assert L.Layout.load(path).to_dict() == before, "拿旧快照写回要被拒，磁盘不变"


def test_refresh_gate3_non_empty_table_does_not_touch_an_unknown_page(tmp_path):
    """闸 3（非空表）：表里有 A，观察到一个像主屏、能成网格、但与 A 不 same_page 的 C →
    False，文件字节不变。空表那条测不出「拿 C 去覆盖 A」这种错。"""
    path = tmp_path / "layout.json"
    lay = L.Layout.load(path)
    lay.upsert_page(infer_page(_grid_elements(HOME_LABELS), W, H, order=1))
    assert lay.save(path) is True
    assert not L.same_page(HOME_LABELS, OTHER_HOME_LABELS)
    before = path.read_bytes()

    ok = refresh_from_observation(lay, _home_obs(OTHER_HOME_LABELS), path, run_name="run-g")
    assert ok is False
    assert path.read_bytes() == before, "文件字节不变"
    assert len(lay.pages) == 1 and set(lay.pages[0].labels) == set(HOME_LABELS), "内存里的表也不变"


# --- scan_home：扫描是页序的唯一权威（终审 I1） ---

# 另一张像主屏（3 个主屏特征词）、却与 HOME_LABELS 对不上（Jaccard 3/13）的第 1 页。
OTHER_HOME_LABELS = ["设置", "照片", "日历", "微博", "支付宝", "淘宝", "抖音", "小红书"]


class _GridPer:
    """假感知：observe 返回预设标签排成的网格。scan_home 只读 elements/width_px/height_px/frame_id。"""
    def __init__(self, labels):
        self.labels = labels

    def observe(self, frame):
        return _home_obs(self.labels, frame_id=frame.frame_id)

    # 扫描翻页只跑 OCR（2026-09-14）：同一份假观察。state_key 是 walk 判「翻不动了」用的键。
    def observe_text(self, frame):
        o = _home_obs(self.labels, frame_id=frame.frame_id)
        o.state_key = hash(tuple(self.labels))
        o.perception = {"ocr": len(self.labels), "vision": "off", "screen": "off"}
        return o


def _scan(labels, path):
    return scan_home(FakeDevice(home_frames()), _GridPer(labels), path, log=lambda *a: None)


def test_rescan_of_a_changed_first_page_replaces_the_stale_one(tmp_path):
    """(a) 先扫 A、再扫与 A 对不上的 C → 表里只剩 C 一页、order=1。
    (b) 接着拿 A 的帧做任务中刷新 → False、表不变：被删的旧第 1 页不能被复活。"""
    path = tmp_path / "layout.json"
    assert _scan(HOME_LABELS, path) is not None
    assert _scan(OTHER_HOME_LABELS, path) is not None
    lay = L.Layout.load(path)
    assert len(lay.pages) == 1, [p.labels for p in lay.pages]
    assert lay.pages[0].order == 1 and set(lay.pages[0].labels) == set(OTHER_HOME_LABELS)
    assert lay.find_app("相机") is None, "旧第 1 页上的 App 不能再被查到"

    before = path.read_bytes()
    assert refresh_from_observation(lay, _home_obs(HOME_LABELS), path, run_name="run-h") is False
    assert path.read_bytes() == before


def test_rescan_after_moving_one_icon_keeps_the_same_page(tmp_path):
    """(c) 先扫 A、再扫 A 挪了一个图标的版本 → 仍一页、id 不变、格子按新的来。"""
    path = tmp_path / "layout.json"
    first = _scan(HOME_LABELS, path)
    pid = first.pages[0].id
    moved = HOME_LABELS.copy()
    moved[0], moved[-1] = moved[-1], moved[0]
    _scan(moved, path)
    lay = L.Layout.load(path)
    assert len(lay.pages) == 1 and lay.pages[0].id == pid and lay.pages[0].order == 1
    assert (lay.find_app("设置").row, lay.find_app("设置").col) == (2, 4)


def test_scan_refuses_a_grid_without_home_markers(tmp_path):
    """looks_like_home 闸：第三方 App 排得成 ≥3 列 ≥2 行的网格，但主屏特征词为 0 → 不扫、不写盘。
    （`test_scan_refuses_when_not_on_home` 的帧连网格都成不了，删掉这道闸它照样绿。）"""
    path = tmp_path / "layout.json"
    assert infer_page(_grid_elements(THIRD_PARTY_LABELS), W, H) is not None, "前提：它能成网格"
    assert _scan(THIRD_PARTY_LABELS, path) is None
    assert not path.exists()


def test_refresh_never_raises_when_infer_page_breaks(tmp_path, monkeypatch):
    """函数内部抛异常（这里让 infer_page 炸）→ False，不抛。"""
    path = tmp_path / "layout.json"
    lay = L.Layout.load(path)
    monkeypatch.setattr(scan_mod, "infer_page", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    ok = refresh_from_observation(lay, _home_obs(HOME_LABELS), path, run_name="run-f")
    assert ok is False
    assert not path.exists()


# --- 翻遍所有主屏页（2026-09-14）：walk_home_pages 是页序的唯一权威 ---
#
# 布局表以前只有第 1 页：scan_home 只看当前可见的那一页，refresh_from_observation 从不追加。
# 这台手机上设置 / 备忘录 / 提醒事项在第 2 页、记账本在第 3 页 —— 查表直达 23 次全是 no_hit。

P1 = HOME_LABELS
P2 = THIRD_PARTY_LABELS          # 全是第三方 App：过不了 looks_like_home，但它是从第 1 页翻过来的
P3 = ["记账本", "滴滴出行", "高德地图", "携程旅行", "饿了么", "闲鱼", "得物", "Keep"]
P4 = ["知乎", "豆瓣", "网易云音乐", "QQ音乐", "喜马拉雅", "得到", "微信读书", "番茄小说"]
APP_LIBRARY = ["App资源库", "建议", "最近添加", "社交", "效率与财务", "娱乐", "工具", "购物"]


def _walk_env(*label_lists, asker=None):
    pages = [page_frame(t, i + 1) for i, t in enumerate(label_lists)]
    return PagedDevice(pages), grid_perceiver_for(pages, asker=asker)


def _quiet(*a):
    pass


def test_walk_records_every_home_page_with_its_real_order(tmp_path):
    dev, per = _walk_env(P1, P2, P3)
    path = tmp_path / "layout.json"
    w = walk_home_pages(dev, per, path, log=_quiet)
    lay = L.Layout.load(path)
    assert [(p.order, set(p.labels)) for p in sorted(lay.pages, key=lambda p: p.order)] == \
        [(1, set(P1)), (2, set(P2)), (3, set(P3))]
    assert lay.find_app("记账本").page_order == 3
    assert lay.find_app("微博").page_order == 2, "全是第三方 App 的第 2 页也要记"
    assert w.stop == "last_page" and w.looked == 3
    assert dev.calls[:3] == [("key", "home")] * 3, "先回第一页"
    # I1，2026-09-14：第 3 次翻页复现了第 3 页，不直接采信 —— 再翻一次核实（还是第 3 页）才停，
    # 所以是 4 次翻页，不是 3 次。
    assert dev.calls.count(("scroll", "right", "page")) == 4, "复核过一次才认翻到头了"
    assert not any(c[0] in ("tap", "type") for c in dev.calls), "扫描只看不点"


@pytest.mark.parametrize("marker", ["App资源库", "App 资源库", "App Library"])
def test_walk_stops_at_the_app_library_and_does_not_record_it(tmp_path, marker):
    """App 资源库在最后一页右边。它也排得成网格，但不是主屏页 —— 认出它的字就停。"""
    lib = [marker] + APP_LIBRARY[1:]
    assert L.infer_page(_grid_elements(lib), W, H) is not None, "前提：它排得成网格"
    dev, per = _walk_env(P1, P2, lib)
    path = tmp_path / "layout.json"
    w = walk_home_pages(dev, per, path, log=_quiet)
    lay = L.Layout.load(path)
    assert sorted(p.order for p in lay.pages) == [1, 2]
    assert lay.find_app("社交") is None
    assert w.stop == "app_library" and w.looked == 2


def test_walk_stops_at_home_pages_max(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOME_PAGES_MAX", 2)
    dev, per = _walk_env(P1, P2, P3)
    path = tmp_path / "layout.json"
    w = walk_home_pages(dev, per, path, log=_quiet)
    assert sorted(p.order for p in L.Layout.load(path).pages) == [1, 2]
    assert w.stop == "max_pages"
    assert dev.calls.count(("scroll", "right", "page")) == 1, "到上限就不再翻"


def test_walk_replaces_stale_pages_by_order_and_drops_pages_past_the_last(tmp_path):
    """表里：1=P1、2=P3、3=P4（旧的）。手机上现在：P1、P2、P3。扫完：1=P1、2=P2、3=P3，
    旧的第 3 页 P4 删掉（翻到头了，知道一共几页），第 2 页上旧的 P3 挪到第 3 页。"""
    path = tmp_path / "layout.json"
    lay = L.Layout.load(path)
    for order, labels in ((1, P1), (2, P3), (3, P4)):
        lay.upsert_page(infer_page(_grid_elements(labels), W, H, order=order))
    assert lay.save(path)
    dev, per = _walk_env(P1, P2, P3)
    walk_home_pages(dev, per, path, log=_quiet)
    got = L.Layout.load(path)
    assert [(p.order, set(p.labels)) for p in sorted(got.pages, key=lambda p: p.order)] == \
        [(1, set(P1)), (2, set(P2)), (3, set(P3))]
    assert got.find_app("知乎") is None


# --- I1 终审（2026-09-14）：一次丢手势的翻页不能删掉已知页；只有肯定信号（App 资源库）才能删 ---


class _DeadScroll(PagedDevice):
    """翻页手势整个失灵：调用被记下，但画面永远不动（复现真机事故：一次丢手势的翻页把
    3 页表删到只剩第 1 页——见 reviewer probe）。"""
    def scroll(self, d, a):
        self.calls.append(("scroll", d, a))


def test_dropped_flip_gesture_does_not_wipe_known_pages(tmp_path):
    """I1 (a)：先用健康设备扫出正确的 3 页表，再用翻页整个失灵的设备重扫一次——
    表必须还是 1..3，不能因为一次翻页没生效、复现了第 1 页就把后面的页全删了。"""
    path = tmp_path / "layout.json"
    frames = [page_frame(P1, 1), page_frame(P2, 2), page_frame(P3, 3)]
    per = grid_perceiver_for(frames)
    good = PagedDevice(frames)
    w1 = walk_home_pages(good, per, path, log=_quiet)
    assert w1.saved and sorted(p.order for p in L.Layout.load(path).pages) == [1, 2, 3], (w1.stop, w1.pages)

    bad = _DeadScroll(frames)
    w2 = walk_home_pages(bad, per, path, log=_quiet)
    after = sorted(p.order for p in L.Layout.load(path).pages)
    assert after == [1, 2, 3], f"pages lost after one dropped flip: {after}"
    assert w2.stop == "last_page" and w2.looked == 1, (w2.stop, w2.looked)


def test_walk_stops_at_app_library_deletes_stale_pages_past_it(tmp_path):
    """I1 (a) 的正向一半：翻到了 App 资源库——肯定信号，确实翻到头了——才能删掉表里过时的、
    页序超过最后一页的页。"""
    path = tmp_path / "layout.json"
    lay = L.Layout.load(path)
    lay.upsert_page(infer_page(_grid_elements(P4), W, H, order=3))
    assert lay.save(path)
    dev, per = _walk_env(P1, P2, APP_LIBRARY)
    w = walk_home_pages(dev, per, path, log=_quiet)
    assert w.stop == "app_library" and w.looked == 2
    got = L.Layout.load(path)
    assert sorted(p.order for p in got.pages) == [1, 2]
    assert got.find_app("知乎") is None, "肯定信号（看到了 App 资源库）才删掉后面过时的页"


class _DropOneScroll(PagedDevice):
    """第 `drop_call_index` 次 scroll("right","page") 是丢的手势（记下调用、画面不动），其余正常。"""
    def __init__(self, pages, drop_call_index):
        super().__init__(pages)
        self._drop_call_index = drop_call_index
        self._scroll_calls = 0

    def scroll(self, d, a):
        self._scroll_calls += 1
        if d == "right" and a == "page" and self._scroll_calls == self._drop_call_index:
            self.calls.append(("scroll", d, a))     # 记下这次调用，但画面不动
            return
        super().scroll(d, a)


def test_walk_retries_once_after_a_dropped_flip_and_recovers(tmp_path):
    """I1 (b)：复现已经见过的页先别急着认定翻到头了——翻一次再核实。这里让第 2 次翻页手势
    丢失（第 3 页那次翻页画面没动，复现了第 2 页），第 3 次（复核）翻页手势正常——
    应该翻过去、继续找到第 3 页上的 App，而不是把「复现」误判成「翻到头了」。"""
    pages = [page_frame(P1, 1), page_frame(P2, 2), page_frame(P3, 3)]
    dev = _DropOneScroll(pages, drop_call_index=2)
    per = grid_perceiver_for(pages)
    w = walk_home_pages(dev, per, tmp_path / "layout.json", want="记账本", log=_quiet)
    assert w.stop == "found" and w.found is not None and w.found.order == 3, (w.stop, w.pages)
    assert [p.order for p in w.pages] == [1, 2, 3], [p.labels for p in w.pages]
    assert dev.calls.count(("scroll", "right", "page")) == 3, "1 次正常 + 1 次丢手势 + 1 次复核重翻"


def test_walk_with_want_stops_on_the_page_that_has_it(tmp_path):
    """找一个 App：找到就停，交回页序、当前 OCR 观察里的那个元素；只翻到那一页，
    后面没看过的页不动（不知道一共几页，不删）。"""
    path = tmp_path / "layout.json"
    lay = L.Layout.load(path)
    lay.upsert_page(infer_page(_grid_elements(P4), W, H, order=3))
    assert lay.save(path)
    dev, per = _walk_env(P1, P2, P3)
    w = walk_home_pages(dev, per, path, want="支付宝", log=_quiet)
    assert w.stop == "found" and w.found is not None
    assert w.found.order == 2 and w.found.element.text == "支付宝"
    assert w.found.obs.perception["vision"] == "off", "找的时候只跑 OCR"
    assert w.found.rested_at is not None, "翻过页：要点的话得等够 AFTER_PAGE_FLIP_S"
    assert dev.calls.count(("scroll", "right", "page")) == 1
    got = L.Layout.load(path)
    assert sorted(p.order for p in got.pages) == [1, 2, 3], "没翻到的第 3 页不删"


def test_walk_with_want_on_page_one_has_no_flip_to_wait_for(tmp_path):
    dev, per = _walk_env(P1, P2)
    w = walk_home_pages(dev, per, None, want="设置", log=_quiet)
    assert w.found.order == 1 and w.found.rested_at is None
    assert not any(c[0] == "scroll" for c in dev.calls)
    assert w.layout is None, "没给路径：不记表"


def test_walk_with_want_uses_the_unified_matcher(tmp_path):
    """两个标签都包含「备忘」、都不精确 → 不点（不猜），翻到头也算没找到。"""
    dev, per = _walk_env(["备忘录", "语音备忘录", "设置", "照片", "日历", "时钟", "计算器", "相机"])
    w = walk_home_pages(dev, per, None, want="备忘", log=_quiet)
    assert w.found is None and w.stop == "last_page"


def test_walk_refuses_when_page_one_does_not_look_like_home(tmp_path):
    dev, per = _walk_env(P2, P1)
    path = tmp_path / "layout.json"
    w = walk_home_pages(dev, per, path, want="设置", log=_quiet)
    assert w.found is None and w.stop == "not_home"
    assert not any(c[0] == "scroll" for c in dev.calls), "第 1 页都认不出是主屏，不往下翻"
    assert not path.exists()


def test_walk_is_ocr_only(tmp_path):
    """翻页找 App / 扫描，每页只跑 OCR：整屏解析一页 20–30 秒（2026-09-14 实测翻主屏中位 112 秒）。"""
    asker = CountingAsker()
    dev, per = _walk_env(P1, P2, P3, asker=asker)
    walk_home_pages(dev, per, tmp_path / "layout.json", want="记账本", log=_quiet)
    assert asker.calls == 0


def test_walk_does_not_hard_wait_after_page_flips(tmp_path, monkeypatch, long_sleeps):
    """AFTER_PAGE_FLIP_S 管的是「翻页后多久才能点」。只看不点，settle 就够了。"""
    monkeypatch.setattr(config, "AFTER_PAGE_FLIP_S", 5.0)
    dev, per = _walk_env(P1, P2, P3)
    walk_home_pages(dev, per, tmp_path / "layout.json", log=_quiet)
    assert long_sleeps == []


def test_walk_never_raises_when_its_own_logic_breaks(tmp_path, monkeypatch):
    """孪生坏了 = 没有孪生（docs/32 不变式 5）：认页的逻辑一直抛，两页都认不出网格，
    最后一页翻不动了（M2 之后识别出错不再让整个 walk 短路成 error，见下一条测试）。"""
    monkeypatch.setattr(scan_mod, "infer_page", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    dev, per = _walk_env(P1, P2)
    w = walk_home_pages(dev, per, tmp_path / "layout.json", log=_quiet)
    assert w.stop == "last_page" and w.found is None and w.pages == []
    assert w.error and "boom" in w.error, "识别出错要能看见，不是悄悄吞掉（CLAUDE.md §3）"


def test_recognition_crash_does_not_stop_the_want_search(tmp_path, monkeypatch):
    """M2，2026-09-14 终审：`infer_page` 在每一页上都炸——页识别和找 want 分开算，
    找 App 照样能在识别失败的那一帧上跑，能找到就还是能找到；崩掉的那几页不写表。"""
    monkeypatch.setattr(scan_mod, "infer_page", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    dev, per = _walk_env(P1, P2)
    w = walk_home_pages(dev, per, tmp_path / "layout.json", want="支付宝", log=_quiet)
    assert w.stop == "found" and w.found is not None and w.found.order == 2
    assert w.found.element.text == "支付宝"
    assert w.error and "boom" in w.error, "识别出错要能看见，不是悄悄吞掉"
    assert w.pages == [] and w.layout is None, "识别一直没成功，没有页可写"


def test_walk_lets_device_errors_through_but_scan_home_does_not(tmp_path):
    """设备的异常照旧往外抛（和 open_app 其它路一致：镜像断了不是「表没用上」）；
    scan_home 是 CLI 的锦上添花，自己兜住。"""
    dev, per = _walk_env(P1, P2)
    dev.raise_on["scroll"] = RuntimeError("镜像断了")
    with pytest.raises(RuntimeError):
        walk_home_pages(dev, per, tmp_path / "layout.json", log=_quiet)
    dev, per = _walk_env(P1, P2)
    dev.raise_on["scroll"] = RuntimeError("镜像断了")
    assert scan_home(dev, per, tmp_path / "layout.json", log=_quiet) is None


def test_walk_save_failure_is_logged_not_fatal(tmp_path, monkeypatch):
    lines = []
    monkeypatch.setattr(L.Layout, "save", lambda self, path: False)
    dev, per = _walk_env(P1, P2)
    w = walk_home_pages(dev, per, tmp_path / "layout.json", want="支付宝", log=lines.append)
    assert w.found is not None and w.found.order == 2, "写表失败不影响找 App"
    assert w.saved is False
    assert any("写入失败" in s for s in lines), lines


def test_walk_save_conflict_reloads_layout_in_place_so_the_next_save_succeeds(tmp_path):
    """M4，2026-09-14 终审：写表冲突（被另一个写者抢先）时，把调用方手里这个 Layout 对象原地
    刷新成磁盘最新版——不然它的 revision 停在冲突前的旧值上，同一个任务里后面的存盘会一直
    冲突下去（`refresh_from_observation` 同款 bug、同款修法）。"""
    path = tmp_path / "layout.json"
    seed = L.Layout()
    seed.upsert_page(infer_page(_grid_elements(P1), W, H, order=1))
    assert seed.save(path)

    caller_layout = L.Layout.load(path)             # 站在 ex.layout 的位置：此刻 revision 对得上（R1）
    other = L.Layout.load(path)
    other.upsert_page(infer_page(_grid_elements(P4), W, H, order=2))
    assert other.save(path)                          # 并发写者抢先，磁盘 revision 变成 R2

    dev, per = _walk_env(P1, P2)
    w = walk_home_pages(dev, per, path, layout=caller_layout, log=_quiet)
    assert w.saved is False, "caller_layout 还停在 R1，这次存盘应该冲突"
    assert w.layout is caller_layout, "原地刷新，不是换了个新对象——调用方手里那份要跟着变"
    assert caller_layout.revision == L.Layout.load(path).revision, "对象已经刷新成磁盘最新版"
    assert {p.order for p in caller_layout.pages} == {1, 2}, "刷新后的内容也是磁盘上的（含并发写者写的第 2 页）"

    # 下一次存盘（同一个对象）应该成功——revision 已经刷新过，对得上磁盘，不会再冲突
    caller_layout.upsert_page(infer_page(_grid_elements(P3), W, H, order=3))
    assert caller_layout.save(path) is True
    assert sorted(p.order for p in L.Layout.load(path).pages) == [1, 2, 3]


def test_scan_home_walks_all_pages_and_reports_each(tmp_path):
    lines = []
    dev, per = _walk_env(P1, P2, P3)
    lay = scan_home(dev, per, tmp_path / "layout.json", log=lines.append)
    assert lay is not None and sorted(p.order for p in lay.pages) == [1, 2, 3]
    pages = [s.strip() for s in lines if s.strip().startswith("第 ")]
    assert [s.split("：")[0] for s in pages] == ["第 1 页", "第 2 页", "第 3 页"], lines
