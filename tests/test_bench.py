"""评测框架自己的测试：指标算得对不对、diff 报得准不准。

评测集是用来抓 agent 退化的尺子，尺子自己不能刻错。每个指标各有一条「该抓住」和一条「不该误报」。
"""
import json

import pytest
from PIL import Image

from iphone_agent.eval import bench as B

_N = [0]


def _frame(root, rel, color="white"):
    """每帧像素都不同（用一个递增的像素做指纹），FakePerceiver 才能靠像素认出是哪一帧。"""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (100, 200), color)
    _N[0] += 1
    img.putpixel((0, 0), (_N[0] % 256, (_N[0] // 256) % 256, 7))
    img.save(p)
    return p


class FakePerceiver:
    """按帧路径返回预设元素。bench 真跑 observe，测试就得给它一个会 observe 的东西。"""

    def __init__(self):
        self.by_path: dict[str, list] = {}
        self.asker = None

    def observe(self, frame):
        from types import SimpleNamespace
        # bench 通过 Frame 传图；我们用图的像素找回是哪一帧
        key = frame.image.tobytes()
        els = self.by_path.get(key, [])
        return SimpleNamespace(elements=[
            SimpleNamespace(id=e["id"], text=e["text"], center=tuple(e["center"]), box=tuple(e["box"]),
                            kind=e.get("kind"), state=e.get("state"), source=e.get("source", "ocr"))
            for e in els])


_PER = FakePerceiver()


def _cache(root, frame_path, elements):
    from PIL import Image as _I
    _PER.by_path[_I.open(frame_path).convert("RGB").tobytes()] = elements


def _el(text, cx, cy, kind=None, state=None):
    return {"id": 1, "text": text, "center": [cx, cy], "box": [cx - 10, cy - 5, cx + 10, cy + 5],
            "kind": kind, "state": state, "source": "ocr"}


@pytest.fixture
def evalset(tmp_path):
    root = tmp_path / "evalset"
    (root / "explore" / "app").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "frames").mkdir()
    return root


def _taps(root, rows):
    (root / "explore" / "app" / "taps.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))


# ---------- tap_coverage：正样本位置要有元素 ----------

def test_tap_coverage_counts_a_covered_positive(evalset):
    fp = _frame(evalset, "explore/app/frames/001.png", "red")
    _cache(evalset, fp, [_el("按钮", 50, 100)])
    _taps(evalset, [{"before": "001.png", "after": "002.png", "text": "按钮", "x": 52, "y": 98,
                     "changed": True, "hamming": 9, "local_mad": 30.0}])
    r = B.bench(evalset, _PER)["metrics"]["tap_coverage"]
    assert (r["hits"], r["total"]) == (1, 1)


def test_tap_coverage_catches_a_missed_positive(evalset):
    """点了有反应的位置，agent 一个元素都没给 —— 这就是「认不出来」。"""
    fp = _frame(evalset, "explore/app/frames/001.png", "red")
    _cache(evalset, fp, [_el("别处", 10, 10)])
    _taps(evalset, [{"before": "001.png", "after": "002.png", "text": "图标", "x": 50, "y": 100,
                     "changed": True, "hamming": 9, "local_mad": 30.0}])
    r = B.bench(evalset, _PER)["metrics"]["tap_coverage"]
    assert (r["hits"], r["total"]) == (0, 1)
    assert "图标" in r["misses"][0]


# ---------- dead_tap_precision：负样本位置不该被标成可点 ----------

def test_dead_tap_flags_a_dead_spot_labelled_clickable(evalset):
    """时钟里的「睡眠|起床闹钟」：点了没反应，agent 却标成 cell。模型会去点它。"""
    fp = _frame(evalset, "explore/app/frames/001.png", "blue")
    _cache(evalset, fp, [_el("睡眠", 50, 100, kind="cell")])
    _taps(evalset, [{"before": "001.png", "after": "001.png", "text": "睡眠", "x": 50, "y": 100,
                     "changed": False, "hamming": 0, "local_mad": 0.0}])
    r = B.bench(evalset, _PER)["metrics"]["dead_tap_precision"]
    assert (r["hits"], r["total"]) == (0, 1)
    assert "cell" in r["misses"][0]


def test_dead_tap_does_not_punish_plain_text(evalset):
    """点了没反应、agent 也只当它是文字 —— 没误导，不算错。"""
    fp = _frame(evalset, "explore/app/frames/001.png", "blue")
    _cache(evalset, fp, [_el("说明文字", 50, 100, kind=None)])
    _taps(evalset, [{"before": "001.png", "after": "001.png", "text": "说明文字", "x": 50, "y": 100,
                     "changed": False, "hamming": 0, "local_mad": 0.0}])
    r = B.bench(evalset, _PER)["metrics"]["dead_tap_precision"]
    assert (r["hits"], r["total"]) == (1, 1)


# ---------- 标注驱动的三个 ----------

def _label(evalset, frame_name, elements):
    fp = _frame(evalset, f"frames/{frame_name}", "green")
    (evalset / "labels" / f"{frame_name[:-4]}.json").write_text(
        json.dumps({"frame": frame_name, "elements": elements}, ensure_ascii=False), encoding="utf-8")
    return fp


def test_switch_state_is_scored_against_the_label(evalset):
    fp = _label(evalset, "s.png", [{"label": "飞行模式 开关", "kind": "switch", "center": [80, 100],
                                   "state": "off", "clickable": True}])
    _cache(evalset, fp, [_el("开关", 80, 100, kind="switch", state=None)])
    r = B.bench(evalset, _PER)["metrics"]["switch_state_acc"]
    assert (r["hits"], r["total"]) == (0, 1), "state=None 就是没报，该算错"
    _cache(evalset, fp, [_el("开关", 80, 100, kind="switch", state="off")])
    assert B.bench(evalset, _PER)["metrics"]["switch_state_acc"]["hits"] == 1


def test_clickable_precision_flags_info_rows_labelled_as_cells(evalset):
    """「型号名称 iPhone 16e」这种信息行，agent 标成 cell 就是在骗模型去点。"""
    fp = _label(evalset, "c.png", [{"label": "型号名称", "kind": "cell", "center": [50, 100],
                                   "clickable": False}])
    _cache(evalset, fp, [_el("型号名称", 50, 100, kind="cell")])
    r = B.bench(evalset, _PER)["metrics"]["clickable_precision"]
    assert (r["hits"], r["total"]) == (0, 1)


def test_clickable_precision_ignores_titles_and_sections(evalset):
    """标题、分组名本来就没人会点，agent 没标它们也不该加分。"""
    fp = _label(evalset, "t.png", [{"label": "视觉", "kind": "section", "center": [50, 100],
                                   "clickable": False}])
    _cache(evalset, fp, [])
    assert B.bench(evalset, _PER)["metrics"]["clickable_precision"]["total"] == 0


def test_text_readback_checks_the_row_not_the_whole_screen(evalset):
    """计算器显示区那个「7」：要在**那一行**读到，别处读到不算。"""
    fp = _label(evalset, "r.png", [{"label": "结果", "kind": "text", "center": [50, 100],
                                   "value": "7", "clickable": False}])
    _cache(evalset, fp, [_el("7", 90, 180)])              # 别处的 7
    assert B.bench(evalset, _PER)["metrics"]["text_readback"]["hits"] == 0
    _cache(evalset, fp, [_el("7", 60, 102)])              # 同一行
    assert B.bench(evalset, _PER)["metrics"]["text_readback"]["hits"] == 1


def test_redacted_values_are_skipped(evalset):
    fp = _label(evalset, "p.png", [{"label": "序列号", "kind": "cell", "center": [50, 100],
                                   "value": "<redact>", "clickable": False}])
    _cache(evalset, fp, [])
    assert B.bench(evalset, _PER)["metrics"]["text_readback"]["total"] == 0


# ---------- 同一帧只感知一次 ----------

def test_each_frame_is_perceived_once_per_bench(evalset):
    """一帧被点了五次，感知只该跑一次 —— 视觉命中缓存也还有 OCR 的几百毫秒。"""
    fp = _frame(evalset, "explore/app/frames/001.png", "red")
    _cache(evalset, fp, [_el("a", 50, 100)])
    _taps(evalset, [{"before": "001.png", "after": "001.png", "text": "a", "x": 50, "y": 100,
                     "changed": True, "hamming": 9, "local_mad": 9.0}] * 5)
    calls = []
    orig = _PER.observe
    _PER.observe = lambda f: (calls.append(1), orig(f))[1]
    try:
        B.bench(evalset, _PER)
    finally:
        _PER.observe = orig
    assert len(calls) == 1


# ---------- diff ----------

def test_diff_lists_newly_broken_and_newly_fixed():
    a = {"ts": "a", "metrics": {"tap_coverage": {"value": 0.5, "misses": ["x", "y"]}}}
    b = {"ts": "b", "metrics": {"tap_coverage": {"value": 0.5, "misses": ["y", "z"]}}}
    out = "\n".join(B.diff(a, b))
    assert "新错: z" in out and "修好: x" in out and "y" not in out.split("修好")[1].split("新错")[0]


# ---------- affordance：把「眼睛」和「下一步」接上 ----------

def test_affordance_label_splits_the_four_reactions():
    """标签从执行验证记录自动来。toggle 是整屏哈希看不见、只有局部灰度差看得见的那种。"""
    assert B.affordance_label({"changed": False, "hamming": 0, "local_mad": 0.1}) == "none"
    assert B.affordance_label({"changed": True, "hamming": 0, "local_mad": 33.0}) == "toggle"
    assert B.affordance_label({"changed": True, "hamming": 28, "local_mad": 34.0}) == "navigate"
    assert B.affordance_label({"changed": True, "hamming": 40, "local_mad": 50.0, "left_app": True}) == "leave"


def test_affordance_scores_kind_against_actual_reaction(evalset):
    """开关被标成 cell：agent 以为点了会跳转，实际是原地翻转。这就是「看到了但不理解」。"""
    fp = _frame(evalset, "explore/app/frames/001.png", "red")
    _cache(evalset, fp, [_el("飞行模式", 50, 100, kind="cell")])
    _taps(evalset, [{"before": "001.png", "after": "002.png", "text": "飞行模式", "x": 50, "y": 100,
                     "changed": True, "hamming": 0, "local_mad": 33.0}])
    r = B.bench(evalset, _PER)
    a = r["metrics"]["affordance_acc"]
    assert (a["hits"], a["total"]) == (0, 1)
    assert "实际 toggle，预测 navigate" in a["misses"][0]
    assert r["affordance_confusion"] == {"toggle": {"navigate": 1}}


def test_affordance_leave_is_not_in_the_denominator(evalset):
    """v1 从 kind 推不出「会跳出 App」，算它错是冤枉的；只记进混淆矩阵。"""
    fp = _frame(evalset, "explore/app/frames/001.png", "red")
    _cache(evalset, fp, [_el("前往设置", 50, 100, kind="button")])
    _taps(evalset, [{"before": "001.png", "after": "002.png", "text": "前往设置", "x": 50, "y": 100,
                     "changed": True, "hamming": 40, "local_mad": 50.0, "left_app": True}])
    r = B.bench(evalset, _PER)
    assert r["metrics"]["affordance_acc"]["total"] == 0
    assert r["affordance_confusion"]["leave"] == {"navigate": 1}


def test_clickable_precision_matches_cells_by_row_not_distance(evalset):
    """设置里的信息行：标注 center 在行中央，agent 元素在行两端。40px 内没东西
    不等于「没标成可点」—— 基线第一版 100% 就是这么白得的。"""
    fp = _label(evalset, "row.png", [{"label": "型号名称", "kind": "cell", "center": [312, 418],
                                     "clickable": False}])
    _cache(evalset, fp, [_el("型号名称", 100, 418, kind="cell"), _el("iPhone 16e", 500, 418)])
    r = B.bench(evalset, _PER)["metrics"]["clickable_precision"]
    assert (r["hits"], r["total"]) == (0, 1), "行首那个 cell 就在同一行上，该算错"


def test_text_readback_ignores_punctuation_width():
    assert B._norm("到期：2026/11/28") == B._norm("到期:2026/11/28")
    assert B._norm("正在载入…") == B._norm("正在载入.")
    assert "18.3.1" in B._norm("iOS版本 18.3.1〉")
