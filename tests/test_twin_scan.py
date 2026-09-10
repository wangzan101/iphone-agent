"""扫描主屏：回第一页、看一眼、写表（设计说明）。第一期只扫当前可见页。"""
import json
from pathlib import Path
from types import SimpleNamespace

from iphone_agent.driver.geometry import Frame, Rect
from iphone_agent.perceive.elements import Element
from iphone_agent.perceive.observe import Perceiver
from iphone_agent.perceive.ocr import RawBox
from iphone_agent.twin import layout as L
from iphone_agent.twin import scan as scan_mod
from iphone_agent.twin.layout import infer_page
from iphone_agent.twin.scan import refresh_from_observation, scan_home
from tests.conftest import FakeDevice, frame_with_text

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
