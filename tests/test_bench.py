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

    def observe_text(self, frame):
        from types import SimpleNamespace
        full = self.observe(frame)
        return SimpleNamespace(elements=[e for e in full.elements if e.source != "vision"])

    def task_scope(self, identity, mode):
        import contextlib
        self.modes = getattr(self, "modes", []) + [mode]
        return contextlib.nullcontext(self)


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
    """「型号名称 iPhone 15」这种信息行，agent 标成 cell 就是在骗模型去点。"""
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
    # Task 10：bench 现在跑两列（整屏 + OCR），各自独立记忆；OCR 列在这个 FakePerceiver 里也经过 observe()
    # （observe_text 委托给 observe，见上面的 FakePerceiver.observe_text）。每列各摊一次，不是 5 次。
    assert len(calls) == 2


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
    _cache(evalset, fp, [_el("型号名称", 100, 418, kind="cell"), _el("iPhone 15", 500, 418)])
    r = B.bench(evalset, _PER)["metrics"]["clickable_precision"]
    assert (r["hits"], r["total"]) == (0, 1), "行首那个 cell 就在同一行上，该算错"


def test_text_readback_ignores_punctuation_width():
    assert B._norm("到期：2026/11/28") == B._norm("到期:2026/11/28")
    assert B._norm("正在载入…") == B._norm("正在载入.")
    assert "18.3.1" in B._norm("iOS版本 18.3.1〉")


# ---------- Task 10：整屏列不变（always 里跑）、新增 OCR 列、label_agree、twin_fingerprint ----------

def test_ocr_column_is_scored_from_observe_text_and_the_full_column_runs_in_always(evalset):
    """spec 2026-09-14 §10.2：整屏那一列不变（在 always 里跑）；OCR 那一列量 on_demand 默认元素表的盲区。"""
    fp = _frame(evalset, "explore/app/frames/001.png", "red")
    _cache(evalset, fp, [_el("按钮", 50, 100), {**_el("图标", 80, 150, kind="icon"), "source": "vision"}])
    _taps(evalset, [{"before": "001.png", "after": "002.png", "text": "图标", "x": 80, "y": 150,
                     "changed": True, "hamming": 9, "local_mad": 30.0}])
    _PER.modes = []
    r = B.bench(evalset, _PER)
    assert r["metrics"]["tap_coverage"]["hits"] == 1 and r["ocr"]["metrics"]["tap_coverage"]["hits"] == 0
    assert _PER.modes == ["always"]
    assert "[OCR 列]" in "\n".join(B.diff(r, r)) and "OCR 列" in "\n".join(B.summary(r))


def test_label_agree_compares_both_prompts_on_the_same_real_candidates(evalset):
    from iphone_agent.perceive.observe import Perceiver
    from iphone_agent.perceive.ocr import RawBox
    from iphone_agent.perceive.screen import PROMPT_LABEL, Candidate
    _frame(evalset, "explore/app/frames/001.png", "red")
    row = {"before": "001.png", "after": "002.png", "text": "a", "x": 1, "y": 1,
           "changed": True, "hamming": 9, "local_mad": 30.0}
    _taps(evalset, [row, row])                       # 同一帧两行：只比一次

    class Ctx:
        def candidates(self, texts):
            return [Candidate(1, "she-zhi", "设置", "s_1", "通用")]

    class Asker:
        prompts: list = []

        def ask_json(self, prompt, images, max_tokens=None):
            Asker.prompts.append(prompt)
            short = prompt.startswith(PROMPT_LABEL)
            return {"screen": {"app": "设置", "name": "通用" if short else " 通用 ", "same_as": 1,
                               "anchors": ["通用", "关于本机"] if short else ["关于本机"]}}
    per = Perceiver(ocr=lambda im: [RawBox("通用", 0.99, 0.1, 0.8, 0.3, 0.04)], asker=Asker())
    r = B.label_agree(evalset, per, Ctx(), "rev-1")
    assert r["frames"] == 1 and r["both_ok"] == 1 and len(Asker.prompts) == 2
    assert all("1. 设置 / 通用" in p for p in Asker.prompts), "两个提示词拿到同一份候选"
    assert r["agree"] == {"app": 1.0, "name": 1.0, "same_as": 1.0, "anchors": 0.0}
    assert r["mismatches"][0]["fields"] == ["anchors"] and r["twin_revision"] == "rev-1"


def test_twin_fingerprint_changes_with_the_twin(tmp_path):
    d = tmp_path / "apps"
    assert B.twin_fingerprint(d) == "0:empty"
    (d / "she-zhi" / "screens").mkdir(parents=True)
    (d / "she-zhi" / "screens" / "s_1.json").write_text("{}", encoding="utf-8")
    one = B.twin_fingerprint(d)
    (d / "she-zhi" / "screens" / "s_1.json").write_text('{"x": 1}', encoding="utf-8")
    assert one.startswith("1:") and B.twin_fingerprint(d) != one


# ---------- 终审 I3：--label-agree-limit 的取样器 —— 按 App 分层轮询，确定性 ----------

def _frames(root, spec):
    """spec: {app_name: n}，生成 root/explore/<app>/frames/NNN.png 路径列表（不用真的写文件）。"""
    out = []
    for app, n in spec.items():
        for i in range(n):
            out.append(root / "explore" / app / "frames" / f"{i:03d}.png")
    return out


def test_sampler_caps_at_the_limit_and_is_deterministic(tmp_path):
    frames = _frames(tmp_path, {"a": 5, "b": 5, "c": 5})
    r1 = B.sample_label_agree_frames(tmp_path, frames, 6)
    r2 = B.sample_label_agree_frames(tmp_path, frames, 6)
    assert len(r1) == 6
    assert r1 == r2, "同样的输入必须得到同样的样本——没有随机数"


def test_sampler_represents_every_app_when_n_covers_all_of_them(tmp_path):
    frames = _frames(tmp_path, {"a": 1, "b": 4, "c": 2})
    r = B.sample_label_agree_frames(tmp_path, frames, 3)
    apps = {B._frame_app(tmp_path, fp) for fp in r}
    assert apps == {"a", "b", "c"}, "N ≥ App 数时每个 App 都要有代表"


def test_sampler_round_robins_not_one_app_first(tmp_path):
    """轮询：不是先把 a 拿满再拿 b，是每一轮每个 App 各拿一个。"""
    frames = _frames(tmp_path, {"a": 5, "b": 5})
    r = B.sample_label_agree_frames(tmp_path, frames, 4)
    apps = [B._frame_app(tmp_path, fp) for fp in r]
    assert apps == ["a", "b", "a", "b"]


def test_sampler_returns_everything_when_limit_covers_all_frames(tmp_path):
    frames = _frames(tmp_path, {"a": 2, "b": 2})
    assert B.sample_label_agree_frames(tmp_path, frames, 100) == frames
    assert B.sample_label_agree_frames(tmp_path, frames, None) == frames
