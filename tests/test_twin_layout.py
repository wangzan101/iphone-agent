"""主屏布局：从一帧观察算网格（设计说明、§2.2）。

用 evalset/labels/00-current.json 里真机标出来的标签位置当输入 ——
24 个图标 4 列 × 6 行，dock 4 个无标签，底部一个「搜索」按钮。
"""
import json
from pathlib import Path

from iphone_agent.perceive.elements import Element
from iphone_agent.twin import layout as L

LABELS = Path(__file__).resolve().parent.parent / "evalset" / "labels" / "00-current.json"


def home_elements():
    """把标注里的 text_center 变成 OCR 会读出来的元素：标签框约 80×22 像素。"""
    d = json.loads(LABELS.read_text(encoding="utf-8"))
    els = []
    for i, e in enumerate(d["elements"], start=1):
        tc = e.get("text_center")
        if tc is None:                      # dock 和「搜索」按钮：OCR 读不到标签
            if e["kind"] == "button":
                cx, cy = e["center"]
                els.append(Element(i, "搜索", 0.99, (cx - 30, cy - 11, cx + 30, cy + 11), (cx, cy)))
            continue
        cx, cy = tc
        els.append(Element(i, e["label"], 0.99, (cx - 40, cy - 11, cx + 40, cy + 11), (cx, cy)))
    return els, d["size"][0], d["size"][1]


def test_real_home_page_is_a_4_by_6_grid():
    els, w, h = home_elements()
    page = L.infer_page(els, w, h, order=1)
    assert page is not None
    assert page.complete is True
    rows = {c.row for c in page.cells}
    cols = {c.col for c in page.cells}
    assert cols == {1, 2, 3, 4} and rows == {1, 2, 3, 4, 5, 6}
    assert len(page.cells) == 24


def test_cells_land_where_the_labeler_put_them():
    els, w, h = home_elements()
    page = L.infer_page(els, w, h)
    by_label = {c.label: (c.row, c.col) for c in page.cells}
    assert by_label["设置"] == (5, 1)
    assert by_label["备忘录"] == (4, 1)
    assert by_label["日历"] == (5, 4)
    assert by_label["Telegram"] == (6, 4)
    assert page.labels == sorted(by_label)


def test_search_button_and_badges_are_not_cells():
    els, w, h = home_elements()
    els.append(Element(99, "465", 0.9, (540, 200, 560, 215), (550, 207)))    # 徽章数字
    page = L.infer_page(els, w, h)
    labels = {c.label for c in page.cells}
    assert "搜索" not in labels and "465" not in labels


def test_cell_carries_app_id_from_the_same_slug_as_skills():
    from iphone_agent.memory.screenmap import app_id_for
    els, w, h = home_elements()
    page = L.infer_page(els, w, h)
    cell = next(c for c in page.cells if c.label == "设置")
    assert cell.app == app_id_for("设置")


def test_too_few_labels_is_not_a_page():
    els = [Element(1, "设置", 0.9, (80, 880, 130, 900), (105, 893)),
           Element(2, "照片", 0.9, (220, 880, 270, 900), (243, 893))]
    assert L.infer_page(els, 624, 1388) is None


def test_page_identity_is_the_label_set():
    a = ["设置", "照片", "日历", "备忘录", "时钟"]
    assert L.same_page(a, ["设置", "照片", "日历", "备忘录", "时钟"])
    assert L.same_page(a, ["设置", "照片", "日历", "备忘录"])           # 少读一个还是它
    assert not L.same_page(a, ["微信", "支付宝", "淘宝", "抖音", "小红书"])


def test_page_round_trips_through_dict():
    els, w, h = home_elements()
    page = L.infer_page(els, w, h, order=2, evidence={"run": "r1", "frame": "frame_003.png"})
    back = L.Page.from_dict(page.to_dict())
    assert back == page
    assert page.id.startswith("p_") and page.order == 2


def test_digits_and_the_search_word_are_filtered_even_inside_the_label_band():
    """验证 _is_label 的两条过滤分支真正工作。

    这条测试**不用** evalset/labels/00-current.json 的真机标注：真机第 3 行四列
    （y≈592，x≈105/243/381/519）已经被 24 个真图标占满，没有空位。上一轮的写法
    把假的 "3" 和 "搜索" 也放在这两个坐标上，结果先到的真图标（Google 在 (105,592)、
    计算器在 (243,592)）先占了 (3,1)/(3,2) 两个格子进 seen，假元素到达时被
    `if rc in seen: continue` 这条**去重**分支丢掉了 —— 跟 _is_label 一点关系都没有。
    复审者做过变异测试：把 _is_label 换成 `bool(t.strip())`（删掉 isdigit 和
    NOT_LABELS 两条过滤），那条测试照样通过，证明它没钉住任何东西。

    这里改成自己造一组干净的元素：第 1、2 行各放 4 个正常标签占满 4 列，第 3 行
    **只放**一个纯数字徽章和一个"搜索"，不跟任何真实元素同格。这样它们能不能被
    _is_label 过滤掉，直接决定第 3 行是否存在 —— 真正测到 isdigit() 和 NOT_LABELS
    这两条分支，而不是被无关的去重逻辑掩盖。
    """
    w, h = 624, 1388
    row1_y, row2_y, row3_y = 290, 440, 592
    cols_x = [105, 243, 381, 519]
    row1_labels = ["设置", "照片", "日历", "备忘录"]
    row2_labels = ["时钟", "计算器", "天气", "相机"]

    els = []
    idx = 1
    for cx, label in zip(cols_x, row1_labels, strict=True):
        els.append(Element(idx, label, 0.99, (cx - 40, row1_y - 11, cx + 40, row1_y + 11), (cx, row1_y)))
        idx += 1
    for cx, label in zip(cols_x, row2_labels, strict=True):
        els.append(Element(idx, label, 0.99, (cx - 40, row2_y - 11, cx + 40, row2_y + 11), (cx, row2_y)))
        idx += 1
    # 第 3 行没有别的元素占位：数字徽章和"搜索"落在空格子上，不会被去重分支误吃掉
    els.append(Element(idx, "465", 0.9, (65, row3_y - 11, 145, row3_y + 11), (105, row3_y)))
    idx += 1
    els.append(Element(idx, "搜索", 0.9, (203, row3_y - 11, 283, row3_y + 11), (243, row3_y)))

    page = L.infer_page(els, w, h)
    assert page is not None
    labels = {c.label for c in page.cells}
    # 两个假元素被过滤掉之后，第 3 行整行都不存在
    assert len(page.cells) == 8
    assert {c.row for c in page.cells} == {1, 2}
    assert "465" not in labels
    assert "搜索" not in labels


def test_same_cell_keeps_first_label_when_two_elements_map_to_it():
    """验证同一格两个元素时取 elements 列表里先到的那个。

    标签有时被 OCR 拆成多行，多行落在同一个 (row, col) 格子。
    _is_label 过滤后，多个元素可能聚类到同一格；见 infer_page 的 if rc in seen: continue 分支。
    这条测试构造这个场景，验证取的是 elements 列表里**先出现**的元素的文本。
    """
    els, w, h = home_elements()
    # 在真机数据已有的"设置"（第 5 行第 1 列，中心 (105, 893)）之后，再追加两个
    # 都会聚类到 (5, 1) 的元素：这样 (5,1) 会有三个候选，去重分支取最先出现的那个。
    # 第 5 行 y ≈ 893，第 1 列 x ≈ 105
    # 构造两个都在这附近、会聚类到同一格的文本
    els.append(Element(99, "设置", 0.9, (100, 880, 110, 900), (105, 893)))      # 先到
    els.append(Element(100, "设置App", 0.9, (100, 902, 110, 920), (105, 911)))  # 后到（相邻 y，也会聚到同一行）
    page = L.infer_page(els, w, h)
    # 应该只有一个 cell 对应这个位置，且是第一个元素的标签
    cells_at_5_1 = [c for c in page.cells if c.row == 5 and c.col == 1]
    assert len(cells_at_5_1) == 1
    assert cells_at_5_1[0].label == "设置"


# ---------- Layout：读写与查表 ----------

def _page_with(labels, order=1):
    els = [Element(i + 1, t, 0.9, (100 + (i % 4) * 138 - 40, 280 + (i // 4) * 150 - 11,
                                    100 + (i % 4) * 138 + 40, 280 + (i // 4) * 150 + 11),
                   (100 + (i % 4) * 138, 280 + (i // 4) * 150))
           for i, t in enumerate(labels)]
    return L.infer_page(els, 624, 1388, order=order)


def test_load_missing_file_is_an_empty_layout(tmp_path):
    lay = L.Layout.load(tmp_path / "layout.json")
    assert lay.pages == [] and lay.revision is None
    assert lay.find_app("设置") is None


def test_find_app_matches_label_ignoring_spaces_and_only_on_complete_pages(tmp_path):
    lay = L.Layout.load(tmp_path / "layout.json")
    lay.upsert_page(_page_with(["设置", "照片", "Safari 浏览器", "日历", "备忘录", "时钟", "计算器", "天气"]))
    hit = lay.find_app("Safari浏览器")
    assert hit is not None and hit.page_order == 1 and (hit.row, hit.col) == (1, 3)
    assert lay.find_app("设置").label == "设置"
    assert lay.find_app("微信") is None
    lay.pages[0].complete = False
    assert lay.find_app("设置") is None, "没扫完整的页不能拿来直达"


def test_upsert_overwrites_the_same_page_and_keeps_its_id(tmp_path):
    lay = L.Layout.load(tmp_path / "layout.json")
    first = _page_with(["设置", "照片", "日历", "备忘录", "时钟", "计算器", "天气", "相机"])
    lay.upsert_page(first)
    pid = lay.pages[0].id
    rearranged = _page_with(["照片", "设置", "日历", "备忘录", "时钟", "计算器", "天气", "相机"], order=7)
    lay.upsert_page(rearranged)                          # 用户换了两个图标的位置
    assert len(lay.pages) == 1
    assert lay.pages[0].id == pid and lay.pages[0].order == 1, "同一页：id 和顺序不变，格子整页覆盖"
    assert lay.find_app("照片").col == 1


def test_upsert_appends_a_different_page(tmp_path):
    lay = L.Layout.load(tmp_path / "layout.json")
    lay.upsert_page(_page_with(["设置", "照片", "日历", "备忘录", "时钟", "计算器", "天气", "相机"], order=1))
    lay.upsert_page(_page_with(["微信", "支付宝", "淘宝", "抖音", "小红书", "美团", "京东", "拼多多"], order=2))
    assert [p.order for p in lay.pages] == [1, 2]
    assert lay.find_app("淘宝").page_order == 2


def test_save_and_load_round_trip_with_revision(tmp_path):
    p = tmp_path / "layout.json"
    lay = L.Layout.load(p)
    lay.upsert_page(_page_with(["设置", "照片", "日历", "备忘录", "时钟", "计算器", "天气", "相机"]))
    assert lay.save(p) is True
    again = L.Layout.load(p)
    assert again.revision == 1 and again.find_app("设置") is not None
    assert again.save(p) is True and L.Layout.load(p).revision == 2


def test_save_with_stale_snapshot_returns_false_and_keeps_disk(tmp_path):
    p = tmp_path / "layout.json"
    a = L.Layout.load(p)
    a.upsert_page(_page_with(["设置", "照片", "日历", "备忘录", "时钟", "计算器", "天气", "相机"]))
    a.save(p)
    b = L.Layout.load(p)                                   # 两个快照
    a.save(p)                                              # a 先写，revision 2
    assert b.save(p) is False, "拿旧快照写回要被拒，不抛"
    assert L.Layout.load(p).revision == 2


def test_save_failure_never_raises(tmp_path, monkeypatch):
    from iphone_agent.twin import schema
    lay = L.Layout.load(tmp_path / "layout.json")
    monkeypatch.setattr(schema, "write_json", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    assert lay.save(tmp_path / "layout.json") is False


def test_save_failure_does_not_poison_self_revision(tmp_path):
    """save 失败（RevisionConflict/OSError）之后，内存里的 self.revision 必须保持失败前的旧值。
    如果被错误地更新成别的东西，下一次 save 会拿一个假 revision 去跟磁盘核对，本该成功的写入
    也会被拒。"""
    p = tmp_path / "layout.json"
    a = L.Layout.load(p)
    a.upsert_page(_page_with(["设置", "照片", "日历", "备忘录", "时钟", "计算器", "天气", "相机"]))
    a.save(p)
    b = L.Layout.load(p)
    stale_revision = b.revision
    a.save(p)                                              # a 先写，磁盘 revision 变成 2
    assert b.save(p) is False, "拿旧快照写回要被拒，不抛"
    assert b.revision == stale_revision, "失败的 save 不能污染内存里的 revision"


def test_find_app_prefers_exact_match_on_a_later_page_over_a_loose_match_on_an_earlier_page(tmp_path):
    """精确优先、全表扫描：第 1 页只有「设置助手」（对「设置」是包含匹配），第 2 页有精确的
    「设置」。按 order 扫完全表，精确匹配（哪怕在后面的页）必须赢过前面页的包含匹配。"""
    lay = L.Layout.load(tmp_path / "layout.json")
    lay.upsert_page(_page_with(["设置助手", "照片", "日历", "备忘录", "时钟", "计算器", "天气", "相机"],
                                order=1))
    lay.upsert_page(_page_with(["设置", "微信", "支付宝", "淘宝", "抖音", "小红书", "美团", "京东"],
                                order=2))
    hit = lay.find_app("设置")
    assert hit is not None and hit.page_order == 2 and hit.label == "设置"


def test_load_with_corrupted_revision_is_an_empty_layout_that_self_heals_on_save(tmp_path):
    """磁盘上 revision 损坏（字符串），schema/pages 都合法：load 按不变式 5「读坏=空表」；
    紧接着 save 必须能把这个坏文件整页覆盖掉——不能抛（schema.write_json 内部
    (on_disk or 0) + 1 对字符串做算术会 TypeError），也不能被当成 RevisionConflict 拒绝
    （那样这个文件就永远写不进去，孪生停止增长且没有任何信号）。"""
    p = tmp_path / "layout.json"
    p.write_text(json.dumps({"schema": L.SCHEMA, "pages": [], "revision": "weird"}), encoding="utf-8")
    lay = L.Layout.load(p)
    assert lay.pages == [] and lay.revision is None
    lay.upsert_page(_page_with(["设置", "照片", "日历", "备忘录", "时钟", "计算器", "天气", "相机"]))
    assert lay.save(p) is True
    again = L.Layout.load(p)
    assert again.revision == 1 and again.find_app("设置") is not None


# ---------- I2：值类型坏了 = 读坏 = 空表（不变式 5） ----------

def _write_layout_with_cell(p, **cell_override):
    page = _page_with(["设置", "照片", "日历", "备忘录", "时钟", "计算器", "天气", "相机"]).to_dict()
    page["cells"][0].update(cell_override)
    p.write_text(json.dumps({"schema": L.SCHEMA, "pages": [page], "revision": 1}), encoding="utf-8")


def test_load_with_a_non_string_label_is_an_empty_layout(tmp_path):
    """⚠ 2026-09-10 终审：label 写成 123，load 照收，find_app 里 norm 123 抛 AttributeError，
    冒到 executor 变成 device_error，Spotlight 一次都没按。读坏必须 = 空表。"""
    p = tmp_path / "layout.json"
    _write_layout_with_cell(p, label=123)
    lay = L.Layout.load(p)
    assert lay.pages == []
    assert lay.find_app("设置") is None


def test_load_with_bad_row_col_or_order_types_is_an_empty_layout(tmp_path):
    p = tmp_path / "layout.json"
    for bad in ({"row": "1"}, {"col": 1.5}, {"row": True}, {"col": None}):
        _write_layout_with_cell(p, **bad)
        assert L.Layout.load(p).pages == [], bad
    page = _page_with(["设置", "照片", "日历", "备忘录", "时钟", "计算器", "天气", "相机"]).to_dict()
    for bad_order in (True, "1", 1.0):
        page["order"] = bad_order
        p.write_text(json.dumps({"schema": L.SCHEMA, "pages": [page]}), encoding="utf-8")
        assert L.Layout.load(p).pages == [], bad_order


def test_load_keeps_a_none_label(tmp_path):
    """label=None 是合法的（没读到的格子）：不能被新校验误杀。"""
    p = tmp_path / "layout.json"
    _write_layout_with_cell(p, label=None, unknown=True)
    lay = L.Layout.load(p)
    assert len(lay.pages) == 1 and lay.pages[0].cells[0].label is None


# ---------- M4：标签相等只有一个入口 ----------

def test_norm_label_ignores_spaces_and_case():
    assert L.norm_label("App Store") == L.norm_label("App store") == L.norm_label("appstore")
    assert L.norm_label("Safari 浏览器") == L.norm_label("Safari浏览器")


# ---------- M6：「读布局表，空或坏就 None」一个入口 ----------

def test_load_or_none_is_none_for_missing_empty_or_broken(tmp_path):
    p = tmp_path / "layout.json"
    assert L.Layout.load_or_none(p) is None                              # 不存在
    p.write_text(json.dumps({"schema": L.SCHEMA, "pages": []}), encoding="utf-8")
    assert L.Layout.load_or_none(p) is None                              # 空表
    p.write_text("{坏的", encoding="utf-8")
    assert L.Layout.load_or_none(p) is None                              # 坏 JSON
    _write_layout_with_cell(p, label=123)
    assert L.Layout.load_or_none(p) is None                              # 值类型坏


def test_load_or_none_returns_a_non_empty_layout(tmp_path):
    p = tmp_path / "layout.json"
    lay = L.Layout.load(p)
    lay.upsert_page(_page_with(["设置", "照片", "日历", "备忘录", "时钟", "计算器", "天气", "相机"]))
    assert lay.save(p) is True
    got = L.Layout.load_or_none(p)
    assert got is not None and got.find_app("设置") is not None and got.revision == 1


def test_load_or_none_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(L.Layout, "load", classmethod(lambda cls, path: (_ for _ in ()).throw(RuntimeError("x"))))
    assert L.Layout.load_or_none(tmp_path / "layout.json") is None
