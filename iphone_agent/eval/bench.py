"""评测：把 evalset/ 里的三种标签和 agent 的感知结果对上，算出几个能抓住盲区的数。

## 三种标签，三个来源

· **执行验证**（evalset/explore/<app>/taps.jsonl）—— 自动探索时每点一下记一条：
  这个位置点了，画面变没变。零人工，手机自己说的。正样本 = 变了，负样本 = 没变。
· **直接看图的标注**（evalset/labels/*.json）—— 我看图标的 kind / state / clickable / value。
  数量少，但给出 OCR 和执行验证都给不出的东西（开关是开是关、这行能不能进）。
· **历史运行**（runs/）—— 模型用编号点了且生效的目标。见 replay.py。

## 指标：每一个都对着试点里量出来的一个盲区

    tap_coverage        执行验证的正样本位置，agent 给了编号吗            —— 抓「认不出来」
    dead_tap_precision  负样本位置，agent 有没有标成可点的 kind          —— 抓「让模型去点没用的东西」
    switch_state_acc    标注的开关，agent 报的 state 对吗                —— 抓「认出开关不知道开关」
    clickable_precision 标注 clickable=False 的，agent 标成可点的比例    —— 抓「可点/不可点不区分」
    text_readback       标注的 value（显示区/输入框的值），agent 读到了吗 —— 抓「单个字符 OCR 丢」
    affordance_acc      「点了会怎样」预测得对吗                          —— 抓「看到了但不理解」

## affordance：把「眼睛」和「下一步」接上

前五个测的都是「有没有看到」。但今天撞到的失败全是「看到了、不知道点了会怎样」：
计算器的 '−' 和「添加账户 +」、信息行「型号名称」、跳出 App 的「前往设置」。

标签是免费的：执行验证每条记录的 after 帧就是「点了会怎样」的答案，只是以前只当
「变没变」的布尔用。从 (hamming, local_mad, left_app) 自动分四类：

    none      hamming≤2 且 local_mad<4        信息行、说明文字
    toggle    hamming≤2 但 local_mad≥4        开关、选中态 —— 整屏哈希看不见的那种
    navigate  hamming>2                        跳转、弹窗（v1 不区分）
    leave     跳出了 App

agent 侧 v1 从 kind 推：switch→toggle，其他可点类型→navigate，text/无→none。
要真正预测得给视觉解析加 effect 字段（改 prompt，缓存全失效），那是 v2。

## 缓存

缓存在 VisionAsker 那一层（evalset/vision-cache/），缓存的是**模型的原始回复**。
bench 每次都真跑 Perceiver.observe：OCR 本地几百毫秒，视觉调用命中缓存，融合现算。
所以改融合、改坐标映射、改 OCR 后处理——全部不花钱，而且**改完立刻反映在分数上**；
只有改 prompt 或换模型才真调。一度把缓存做在融合之后，改一行 fuse() 就得全部重调。

## 用法

    iphone eval bench                 跑全部，存 evalset/results/<ts>.json，打印摘要
    iphone eval bench --label-agree   另外比短标注和整屏解析的标注（第一次要真调视觉，见 spec 2026-09-14 §10.2）
    iphone eval diff <a.json> <b.json>  两次结果对比，列出翻转的样本
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

CLICKABLE_KINDS = {"button", "icon", "input", "switch", "cell", "tab", "folder", "link"}
NEAR_PX = 40          # 点击位置和元素中心距离在这以内就算「覆盖到了」


def affordance_label(rec: dict) -> str:
    """执行验证一条记录 → 它实际的反应类型。阈值和 explore.py 判「变没变」的同一套。"""
    if rec.get("left_app"):
        return "leave"
    if not rec.get("changed"):
        return "none"
    if rec.get("hamming", 0) <= 2 and rec.get("local_mad", 0) >= 4.0:
        return "toggle"
    return "navigate"


def affordance_guess(elements: list[dict]) -> str:
    """agent 对「点这儿会怎样」的预测（v1：从 kind 推）。elements 是点击位置附近的元素。"""
    kinds = {(e.get("kind") or "") for e in elements}
    if "switch" in kinds:
        return "toggle"
    if kinds & CLICKABLE_KINDS:
        return "navigate"
    return "none"


class Perception:
    """对一张帧跑 agent 的感知，同一次 bench 里同一帧只跑一次。

    ocr_only=False：整屏那一列。在 task_scope(None, "always") 里跑 observe —— 测的就是看全屏交回的东西；
      ⚠ 2026-09-14（spec 按需看图 §10.2）：改动前后这一列的指标必须完全不变（按中心和 kind 匹配，不看编号）。
    ocr_only=True：observe_text，也就是 on_demand 默认的元素表。不调视觉、零成本，只报告不设门槛。
    """

    def __init__(self, perceiver, ocr_only: bool = False):
        self.per = perceiver
        self.ocr_only = ocr_only
        self._memo: dict[str, list[dict]] = {}

    def elements(self, frame_path: str) -> list[dict]:
        if frame_path in self._memo:
            return self._memo[frame_path]
        from PIL import Image

        from iphone_agent.driver.geometry import Frame, Rect
        img = Image.open(frame_path).convert("RGB")
        W, H = img.size
        frame = Frame(img, W, H, Rect(0, 0, W, H), 0.0, 0)
        if self.ocr_only:
            obs = self.per.observe_text(frame)
        else:
            scope = getattr(self.per, "task_scope", None)
            with scope(None, "always") if scope is not None else contextlib.nullcontext():
                obs = self.per.observe(frame)
        els = [{"id": e.id, "text": e.text, "center": list(e.center), "box": list(e.box),
                "kind": e.kind, "state": e.state, "source": e.source} for e in obs.elements]
        self._memo[frame_path] = els
        return els


def _near(el: dict, x: int, y: int) -> bool:
    cx, cy = el["center"]
    if math.dist((cx, cy), (x, y)) <= NEAR_PX:
        return True
    x1, y1, x2, y2 = el.get("box", (cx, cy, cx, cy))
    return x1 <= x <= x2 and y1 <= y <= y2


def _same_row(el: dict, y: int, tol: int = 22) -> bool:
    """整行一体的元素（设置里的 cell）：标注的 center 在行中央，agent 的元素在行两端
    （左边文字、右边值/›）。按 y 匹配，不按距离 —— 否则 40px 内没东西，
    「没标成可点」就白得一分。基线第一版 clickable_precision 100% 就是这么来的。"""
    return abs(el["center"][1] - y) <= tol


_PUNCT = str.maketrans({"：": ":", "，": ",", "…": ".", "。": ".", "（": "(", "）": ")",
                        "＞": ">", "〉": ">", "›": ">", " ": ""})


def _norm(t: str) -> str:
    """比文字前把全角/半角标点、省略号、空格抹平。OCR 对这些的读法随机，
    '到期:' vs '到期：' 不是感知错误。"""
    return t.translate(_PUNCT).rstrip(".")


@dataclass
class Metric:
    name: str
    hits: int = 0
    total: int = 0
    misses: list = field(default_factory=list)      # 明细：错在哪

    @property
    def value(self) -> float | None:
        return self.hits / self.total if self.total else None

    def add(self, ok: bool, detail=None):
        self.total += 1
        if ok:
            self.hits += 1
        elif detail is not None:
            self.misses.append(detail)


METRICS = ("tap_coverage", "dead_tap_precision", "switch_state_acc",
           "clickable_precision", "text_readback", "affordance_acc")


def _score(root: Path, per: Perception) -> tuple[dict, dict, int]:
    """六个指标 + affordance 混淆矩阵 + 看过的帧数（按样本行计）。逻辑与 2026-09-14 之前的 bench() 逐行相同。"""
    m = {k: Metric(k) for k in METRICS}
    confusion: dict[str, dict[str, int]] = {}      # 真值 → {预测: 次数}
    frames_seen = 0

    # ---- 执行验证 ----
    for taps in sorted(glob.glob(str(root / "explore" / "*" / "taps.jsonl"))):
        app_dir = Path(taps).parent
        for line in Path(taps).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            fp = str(app_dir / "frames" / r["before"])
            if not os.path.exists(fp):
                continue
            els = per.elements(fp)
            frames_seen += 1
            near = [e for e in els if _near(e, r["x"], r["y"])]
            key = f"{app_dir.name}/{r['before']} {r['text']!r}@({r['x']},{r['y']})"
            if r["changed"]:
                m["tap_coverage"].add(bool(near), key)
            else:
                # 没反应的位置：agent 若把它标成可点的 kind，就是在误导模型
                bad = [e for e in near if (e.get("kind") or "") in CLICKABLE_KINDS]
                m["dead_tap_precision"].add(not bad,
                                            key + f" ← 标成了 {[e.get('kind') for e in bad]}")
            # leave 类 agent 没法从 kind 推出来（v1），不计入分母，只记进混淆矩阵
            truth = affordance_label(r)
            guess = affordance_guess(near)
            confusion.setdefault(truth, {}).setdefault(guess, 0)
            confusion[truth][guess] += 1
            if truth != "leave":
                m["affordance_acc"].add(guess == truth, key + f" ← 实际 {truth}，预测 {guess}")

    # ---- 直接看图的标注 ----
    for lp in sorted(glob.glob(str(root / "labels" / "*.json"))):
        gt = json.loads(Path(lp).read_text(encoding="utf-8"))
        fp = str(root / "frames" / gt["frame"])
        if not os.path.exists(fp):
            continue
        els = per.elements(fp)
        frames_seen += 1
        tag = Path(lp).stem
        for g in gt["elements"]:
            cx, cy = g["center"]
            # 行类元素按 y 找同一行的全部；点状元素（开关、图标、按钮）按距离
            near = ([e for e in els if _same_row(e, cy)] if g["kind"] == "cell"
                    else [e for e in els if _near(e, cx, cy)])
            if g["kind"] == "switch":
                sw = [e for e in near if e.get("kind") == "switch"]
                got = sw[0].get("state") if sw else None
                m["switch_state_acc"].add(got == g.get("state"),
                                          f"{tag} {g['label']} 真值 {g.get('state')} 报 {got}")
            if g.get("clickable") is False and g["kind"] not in ("title", "section", "header", "text"):
                # 明确标了不可点的行/区域（title/section 这类本来就没人会点，不算）
                bad = [e for e in near if (e.get("kind") or "") in CLICKABLE_KINDS]
                m["clickable_precision"].add(not bad,
                                             f"{tag} {g['label']} ← 标成了 {[e.get('kind') for e in bad]}")
            if g.get("value") and not str(g["value"]).startswith("<redact"):
                want = _norm(str(g["value"]))
                rowtexts = _norm("".join(e["text"] for e in els if abs(e["center"][1] - cy) <= 25))
                m["text_readback"].add(want in rowtexts, f"{tag} {g['label']} 要 {want!r} 行内读到 {rowtexts[:40]!r}")

    return m, confusion, frames_seen


def _dump(m: dict[str, Metric]) -> dict:
    return {k: {"value": v.value, "hits": v.hits, "total": v.total, "misses": v.misses} for k, v in m.items()}


def bench(root: Path, perceiver) -> dict:
    """perceiver 是 Perceiver（或长得像它的东西）。它的 asker 该带 cache_dir，不然每帧真调模型。"""
    t0 = time.time()
    m, confusion, frames_seen = _score(root, Perception(perceiver))
    asker = getattr(perceiver, "asker", None)
    vision = asker.stats() if asker is not None and hasattr(asker, "stats") else {}
    m_ocr, confusion_ocr, _ = _score(root, Perception(perceiver, ocr_only=True))
    return {
        "ts": time.strftime("%Y%m%d-%H%M%S"),
        "frames": frames_seen,
        "seconds": round(time.time() - t0, 1),
        "vision": vision,
        "metrics": _dump(m),
        "affordance_confusion": confusion,
        "ocr": {"metrics": _dump(m_ocr), "affordance_confusion": confusion_ocr},
    }


def _metric_lines(metrics: dict, indent: str = "  ") -> list[str]:
    lines = []
    for k, v in metrics.items():
        if v["total"] == 0:
            lines.append(f"{indent}{k:<20} —        （没有样本）")
            continue
        lines.append(f"{indent}{k:<20} {v['value'] * 100:5.1f}%   {v['hits']}/{v['total']}")
    return lines


def summary(res: dict) -> list[str]:
    v = res.get("vision") or {}
    lines = [f"# {res['frames']} 帧 · {res.get('seconds', 0)}s · 视觉调用 {v.get('calls', 0)} 次"
             f"（缓存命中 {v.get('cache_hits', 0)}）"]
    lines += _metric_lines(res["metrics"])
    conf = res.get("affordance_confusion") or {}
    if conf:
        lines.append("  affordance 混淆（行=实际，列=预测）：")
        cols = ("none", "toggle", "navigate", "leave")
        lines.append("    " + " " * 9 + "".join(f"{c:>9}" for c in cols))
        for t in cols:
            if t in conf:
                lines.append(f"    {t:>9}" + "".join(f"{conf[t].get(c, 0):>9}" for c in cols))
    if "ocr" in res:
        lines.append("  OCR 列（observe_text = on_demand 的默认元素表；只报告，不设门槛）：")
        lines += _metric_lines(res["ocr"]["metrics"], indent="    ")
    la = res.get("label_agree")
    if la:
        lines.append(f"  label_agree（孪生 {la['twin_revision']}；{la['both_ok']}/{la['frames']} 帧两边都标成，"
                     f"候选为空 {la['candidates_empty']}）：")
        for k, val in la["agree"].items():
            lines.append(f"    {k:<8} " + ("—" if val is None else f"{val * 100:5.1f}%"))
        lines.append(f"    不一致 {len(la['mismatches'])} 帧、标不成 {len(la['unavailable'])} 帧（明细在结果文件里）")
    return lines


def _diff_metrics(ma: dict, mb: dict) -> list[str]:
    lines = []
    for k in ma:
        va, vb = ma[k]["value"], mb.get(k, {}).get("value")
        if va is None or vb is None:
            continue
        arrow = "↑" if vb > va + 1e-9 else ("↓" if vb < va - 1e-9 else "=")
        lines.append(f"  {k:<20} {va * 100:5.1f}% → {vb * 100:5.1f}%  {arrow}")
        a_miss, b_miss = set(ma[k]["misses"]), set(mb[k]["misses"])
        for x in sorted(b_miss - a_miss)[:8]:
            lines.append(f"      新错: {x}")
        for x in sorted(a_miss - b_miss)[:8]:
            lines.append(f"      修好: {x}")
    return lines


def diff(a: dict, b: dict) -> list[str]:
    lines = [f"# {a['ts']} → {b['ts']}"] + _diff_metrics(a["metrics"], b["metrics"])
    if "ocr" in a and "ocr" in b:        # 老结果没有 OCR 列：只比整屏那一列
        lines.append("  [OCR 列]")
        lines += _diff_metrics(a["ocr"]["metrics"], b["ocr"]["metrics"])
    return lines


LABEL_FIELDS = ("app", "name", "same_as", "anchors")


def _bench_frames(root: Path) -> list[Path]:
    """bench 用到的全部帧，去重、保持首次出现的顺序（与 _score 的遍历顺序一致）。"""
    seen: dict[str, None] = {}
    for taps in sorted(glob.glob(str(root / "explore" / "*" / "taps.jsonl"))):
        app_dir = Path(taps).parent
        for line in Path(taps).read_text(encoding="utf-8").splitlines():
            if line.strip():
                fp = app_dir / "frames" / json.loads(line)["before"]
                if fp.exists():
                    seen.setdefault(str(fp))
    for lp in sorted(glob.glob(str(root / "labels" / "*.json"))):
        fp = root / "frames" / json.loads(Path(lp).read_text(encoding="utf-8"))["frame"]
        if fp.exists():
            seen.setdefault(str(fp))
    return [Path(p) for p in seen]


def twin_fingerprint(apps_dir: Path) -> str:
    """孪生的版本记号：屏文件数 + 全部内容的哈希。label_agree 的候选取自当时的孪生，结果里要写明是哪一份。"""
    apps_dir = Path(apps_dir)
    files = sorted(p for p in apps_dir.rglob("*.json") if ".corrupt" not in p.parts) if apps_dir.exists() else []
    if not files:
        return "0:empty"
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(apps_dir).as_posix().encode("utf-8"))
        h.update(p.read_bytes())
    return f"{len(files)}:{h.hexdigest()[:16]}"


def _frame_app(root: Path, fp: Path) -> str:
    """label_agree 分层抽样用的『App』分组：explore 帧按 App 目录名分组（root/explore/<app>/frames/…）；
    直接看图标注的帧（root/frames/…）没有 App 目录，自成一组。"""
    try:
        rel = fp.relative_to(root).parts
    except ValueError:
        rel = fp.parts
    if len(rel) >= 2 and rel[0] == "explore":
        return rel[1]
    return "_labels"


def sample_label_agree_frames(root: Path, frames: list[Path], limit: int) -> list[Path]:
    """终审 I3：--label-agree-limit 的取样器。按帧所属 App 分层、轮询取，确定性、不用随机数——
    这样 N=60 的一次跑是公平、可复现的样本；N ≥ App 数时每个 App 至少有一帧代表。"""
    if limit is None or limit <= 0 or len(frames) <= limit:
        return list(frames)
    groups: dict[str, list[Path]] = {}
    for fp in frames:
        groups.setdefault(_frame_app(root, fp), []).append(fp)
    apps = sorted(groups)
    idx = dict.fromkeys(apps, 0)
    out: list[Path] = []
    while len(out) < limit:
        progressed = False
        for a in apps:
            if len(out) >= limit:
                break
            i = idx[a]
            if i < len(groups[a]):
                out.append(groups[a][i])
                idx[a] = i + 1
                progressed = True
        if not progressed:
            break
    return out


def label_agree(root: Path, perceiver, ctx, twin_revision: str, limit: int | None = None) -> dict:
    """同一批帧、同一份真实候选，PROMPT_LABEL 与 PROMPT_SCREEN 的完整标注逐字段比（spec 2026-09-14 §10.2）。

    候选按每帧的 OCR 文字经 ctx.candidates 挑（和实时一样）。缓存键包含提示词、所以也包含候选。
    这是离线参考，不是闸门：错合、错归的零容忍闸门放在真机 A/B 上人工核对（§10.3）。
    limit：终审 I3，--label-agree-limit。None（默认）不抽样，比全部帧。
    """
    from PIL import Image

    from iphone_agent.driver.geometry import Frame, Rect
    from iphone_agent.perceive.screen import label_screen, parse_screen_full
    from iphone_agent.twin.identify import norm_text, plain_app_id
    asker = perceiver.asker
    agree = dict.fromkeys(LABEL_FIELDS, 0)
    both, empty, unavailable, mismatches = 0, 0, [], []
    all_frames = _bench_frames(root)
    frames = sample_label_agree_frames(root, all_frames, limit) if limit is not None else all_frames
    for fp in frames:
        rel = fp.relative_to(root).as_posix()
        img = Image.open(fp).convert("RGB")
        W, H = img.size
        obs = perceiver.observe_text(Frame(img, W, H, Rect(0, 0, W, H), 0.0, 0))
        cands = list(ctx.candidates([e.text for e in obs.elements])) if ctx is not None else []
        empty += not cands
        short, short_status = label_screen(img, asker, cands)
        full = parse_screen_full(img, asker, cands)
        if short is None or full.label is None:
            unavailable.append({"frame": rel, "label": short_status, "screen": full.label_status})
            continue
        both += 1
        same = {"app": plain_app_id(short.app) == plain_app_id(full.label.app),
                "name": norm_text(short.name) == norm_text(full.label.name),
                "same_as": short.same_as == full.label.same_as,
                "anchors": {norm_text(a) for a in short.anchors} == {norm_text(a) for a in full.label.anchors}}
        for k, ok in same.items():
            agree[k] += ok
        if not all(same.values()):
            mismatches.append({"frame": rel, "fields": [k for k, ok in same.items() if not ok],
                               "label": short.to_json(), "screen": full.label.to_json(),
                               "candidates": [c.to_json() for c in cands]})
    return {"twin_revision": twin_revision, "frames": len(frames), "both_ok": both, "candidates_empty": empty,
            "agree": {k: (agree[k] / both if both else None) for k in LABEL_FIELDS},
            "unavailable": unavailable, "mismatches": mismatches}


def save(res: dict, root: Path = Path("evalset")) -> Path:
    d = root / "results"
    d.mkdir(exist_ok=True)
    p = d / f"{res['ts']}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return p
