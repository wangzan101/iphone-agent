"""主屏布局表：第几页第几行第几列是哪个 App（docs/32 §1.5、§2.2）。

存**行列号**不存像素：像素每次按当前帧现算。坐标是这个项目里最脆的东西（docs/14），
换机器、拉窗口、改缩放都会让它失效；行列号不会。

网格怎么算：主屏上 OCR 读得到的只有图标**下面的标签**。标签的 y 聚成几行、x 聚成几列，
行列号就是它在两组簇里的序号。图标中心在标签上方 ICON_ABOVE_LABEL_RATIO 倍文字高处，
点的时候用当前帧的标签框现算（executor 那边做），这里不碰。

dock 里的图标没有标签，OCR 读不到；负一屏、文件夹不扫，App 资源库认出来就停（docs/32 §4.4）。
2026-09-14 起翻遍所有主屏页、按真实页序写表（twin/scan.walk_home_pages）；原来只扫可见的第 1 页。
「没读到」不等于「空」：格子标 unknown，页标 complete=False。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from iphone_agent.memory.screenmap import app_id_for
from iphone_agent.twin.schema import new_id

SCHEMA = 1
# 主屏的标签只在这一带：上面是状态栏和小组件，下面是 Spotlight 搜索按钮和 dock（真机 624×1388：
# 标签 y 290–1043，搜索按钮 1158，dock 1283）。
LABEL_BAND = (0.15, 0.80)
COL_TOL = 0.06        # 同一列的标签 x 中心相差不超过屏宽的 6%（列距约 22%）
ROW_TOL = 0.04        # 同一行的标签 y 中心相差不超过屏高的 4%（行距约 11%）
MIN_COLS, MIN_ROWS = 3, 2
NOT_LABELS = {"搜索", "search"}             # Spotlight 按钮的字，不是 App
PAGE_SAME = 0.6                             # 两页标签集合的 Jaccard ≥ 这个就是同一页


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    label: str | None
    app: str | None
    unknown: bool = False


@dataclass
class Page:
    id: str
    order: int
    labels: list[str]
    cells: list[Cell]
    complete: bool
    scanned: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "order": self.order, "labels": list(self.labels),
                "cells": [asdict(c) for c in self.cells], "complete": self.complete,
                "scanned": self.scanned, "evidence": dict(self.evidence)}

    @classmethod
    def from_dict(cls, d: dict) -> Page:
        """磁盘上的值是不可信输入：类型不对就抛 TypeError，`Layout.load` 接住 → 空表。

        ⚠ 2026-09-10 终审：layout.json 里一个格子的 label 写成 `123`，这里原来照收，
        `find_app` 对它做字符串处理抛 AttributeError，冒到 executor 的 `except Exception`
        变成 device_error —— open_app 失败，Spotlight 一次都没按。孪生坏了没有「等于没有孪生」
        （不变式 5），反而把一个本来能开的 App 弄得打不开。所以在读入口一次校验干净：
        label 是 str 或 None；row/col/order 是 int（bool 是 int 的子类，单独排掉）。
        """
        order = d["order"]
        if not _is_int(order):
            raise TypeError(f"page.order 必须是 int：{order!r}")
        labels = list(d.get("labels") or [])
        if not all(isinstance(t, str) for t in labels):
            raise TypeError(f"page.labels 必须全是 str：{labels!r}")
        cells = [Cell(**c) for c in d.get("cells") or []]
        for c in cells:
            if not (_is_int(c.row) and _is_int(c.col)):
                raise TypeError(f"cell 的 row/col 必须是 int：{c!r}")
            if c.label is not None and not isinstance(c.label, str):
                raise TypeError(f"cell.label 必须是 str 或 None：{c!r}")
        return cls(id=d["id"], order=order, labels=labels,
                   cells=cells, complete=bool(d.get("complete")),
                   scanned=str(d.get("scanned") or ""), evidence=dict(d.get("evidence") or {}))


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _cluster(values: list[float], tol: float) -> list[float]:
    """一维聚类：排序后相邻差 ≤ tol 归同一簇，返回各簇中心（升序）。

    单链聚类：一串间距都小于 tol 的点会被链式并成一簇（e.g. [0,5,10,15,20,25,30]
    tol=6 → [15.0]，一簇跨 30 个单位而容差只有 6）。真机间距（列 138 / 行 150）
    远大于容差（COL_TOL·624≈37.4 / ROW_TOL·1388≈55.5），当前不会误触发。
    """
    if not values:
        return []
    vs = sorted(values)
    groups: list[list[float]] = [[vs[0]]]
    for v in vs[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _nearest(v: float, centers: list[float]) -> int:
    return min(range(len(centers)), key=lambda i: abs(centers[i] - v)) + 1


def _is_label(text: str) -> bool:
    t = text.strip()
    return bool(t) and t.casefold() not in NOT_LABELS and not t.isdigit()


def infer_page(elements, width_px: int, height_px: int, *, order: int = 1,
               evidence: dict | None = None) -> Page | None:
    """一帧主屏观察 → 一页网格。认不出网格（列少于 3、行少于 2）返回 None，不猜。"""
    y_lo, y_hi = LABEL_BAND[0] * height_px, LABEL_BAND[1] * height_px
    labels = [e for e in elements
              if _is_label(e.text) and y_lo <= e.center[1] <= y_hi]
    if not labels:
        return None
    cols = _cluster([e.center[0] for e in labels], COL_TOL * width_px)
    rows = _cluster([e.center[1] for e in labels], ROW_TOL * height_px)
    if len(cols) < MIN_COLS or len(rows) < MIN_ROWS:
        return None
    cells: list[Cell] = []
    seen: set[tuple[int, int]] = set()
    for e in labels:
        rc = (_nearest(e.center[1], rows), _nearest(e.center[0], cols))
        if rc in seen:
            # 同一格读出两段文字（标签被 OCR 拆成两行）：取先到的。
            # 先到者的顺序来自 Observation.elements 的既有顺序（Perceiver 给元素编号的顺序），不是空间顺序。
            continue
        seen.add(rc)
        label = e.text.strip()
        cells.append(Cell(row=rc[0], col=rc[1], label=label, app=app_id_for(label)))
    cells.sort(key=lambda c: (c.row, c.col))
    return Page(id=new_id("p"), order=order, labels=sorted(c.label for c in cells),
                cells=cells, complete=True,
                scanned=time.strftime("%Y-%m-%dT%H:%M:%S"), evidence=dict(evidence or {}))


def same_page(a_labels, b_labels) -> bool:
    a, b = set(a_labels), set(b_labels)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= PAGE_SAME


@dataclass(frozen=True)
class Hit:
    page_order: int
    row: int
    col: int
    label: str


def norm_label(s: str) -> str:
    """两个 App 标签算不算同一个：去空格、不分大小写。一个规则一个入口（CLAUDE.md §7）——
    查表（find_app）和 executor 在当前帧上找标签都用它。原来 executor 只去空格不转小写，
    OCR 把「App Store」读成「App store」时查表命中、当前帧却找不到，白白退回 Spotlight。"""
    return s.replace(" ", "").lower()


def match_label(name: str, candidates, key=None):
    """在一组候选里找「就是 name 这个 App」的那一个；没有就 None。key 把候选变成它的文字（默认候选本身）。

    一个 App 名匹配规则一个入口（CLAUDE.md §7）：查表（Layout.find_app）、查表直达在当前帧上找标签、
    翻主屏找 App（twin/scan.walk_home_pages）都调这里。

    1. norm_label 相等（去空格、不分大小写）的，取第一个 —— 不管前面有没有包含匹配的；
    2. 没有精确的：包含 name 的候选**只有一个**（按归一化后的文字算，同一个标签读到两次算一个），
       而且 name 归一化后至少 2 个字，才取它。两个以上都包含 = 说不清是哪个，不猜。

    ⚠ 2026-09-14：原来三处三个规矩 —— 查表「精确优先、否则第一个包含的」，直达「只认精确」，
      翻主屏「`name in e.text` 的第一个」。主屏第 1 页真实读到的小组件文字（「今天无日程」「大部晴朗无云」
      「示例城区1」）排在图标标签上面，按元素顺序取第一个包含的就先撞上它们；单字名（「M」）包含在一堆
      标签里。命中是肯定结论可以直接用（CLAUDE.md §2），所以包含匹配只在唯一时才算命中；
      不加任何「像不像小组件」的猜测。
    """
    want = norm_label(name or "")
    if not want:
        return None
    key = key or (lambda c: c)
    loose: dict[str, object] = {}
    for c in candidates:
        text = key(c)
        if text is None:
            continue
        t = norm_label(text)
        if t == want:
            return c
        if want in t:
            loose.setdefault(t, c)
    if len(want) >= 2 and len(loose) == 1:
        return next(iter(loose.values()))
    return None


class Layout:
    """布局表本体。load 永不抛（读坏 = 空表），save 永不抛（写坏 = False）：不变式 5。"""

    def __init__(self, pages: list[Page] | None = None, revision: int | None = None):
        self.pages: list[Page] = list(pages or [])
        self.revision = revision

    @classmethod
    def load(cls, path) -> Layout:
        from iphone_agent.twin.schema import read_json
        d = read_json(path)
        if not d or d.get("schema") not in (None, SCHEMA):
            return cls()
        try:
            pages = [Page.from_dict(p) for p in d.get("pages") or []]
        except (KeyError, TypeError, ValueError):
            return cls()
        revision = d.get("revision")
        # revision 类型不对（bool 是 int 子类要单独排掉）：按不变式 5「读坏=空表」处理，
        # 不能把它原样收进内存——下次 save 会拿着这个坏值去跟 schema.write_json 对比，
        # 要么再次触发算术错误，要么和净化后的磁盘值不等被当成 RevisionConflict 永远写不进去。
        if revision is not None and (isinstance(revision, bool) or not isinstance(revision, int)):
            return cls()
        return cls(pages, revision)

    @classmethod
    def load_or_none(cls, path) -> Layout | None:
        """「读布局表，空或坏就 None」的唯一入口：主循环和 CLI 构造 Executor 前都调这里。
        永不抛 —— 孪生任何一处坏了都等于没有孪生（不变式 5）。"""
        try:
            lay = cls.load(path)
        except Exception:       # noqa: BLE001 —— load 本身已不抛，这层兜的是以后改坏了它
            return None
        return lay if lay.pages else None

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "pages": [p.to_dict() for p in sorted(self.pages, key=lambda p: p.order)]}

    def save(self, path) -> bool:
        from iphone_agent.twin import schema
        try:
            self.revision = schema.write_json(path, self.to_dict(), expect_revision=self.revision)
            return True
        except (schema.RevisionConflict, OSError):
            return False

    def upsert_page(self, page: Page) -> None:
        """标签集合对得上就整页覆盖（保留 id 和 order），对不上就追加。
        布局是直接观察到的事实，不是转移的结果 —— 不需要两次确认（docs/32 §4.1）。"""
        for i, old in enumerate(self.pages):
            if same_page(old.labels, page.labels):
                self.pages[i] = Page(id=old.id, order=old.order, labels=page.labels, cells=page.cells,
                                     complete=page.complete, scanned=page.scanned, evidence=page.evidence)
                return
        self.pages.append(page)

    def find_app(self, name: str) -> Hit | None:
        """表里哪一格是这个 App。规则就是 match_label：按页序排好的全表里，精确的（哪怕在后面的页）
        赢过前面页的包含匹配；包含匹配只在唯一时才算。没扫完整的页不拿来直达。"""
        cells = [(p, c) for p in sorted(self.pages, key=lambda p: p.order) if p.complete
                 for c in p.cells if c.label is not None]
        got = match_label(name, cells, key=lambda pc: pc[1].label)
        if got is None:
            return None
        p, c = got
        return Hit(p.order, c.row, c.col, c.label)
