"""离线回放评测：拿历史运行当尺子，量一次改动到底好了多少。

`runs/` 里躺着 93 次真机运行、1016 步，每步都存了动作前后的截图和当时的判定结果。
在这之前它们只是日志；这里把它们当**评测集**用 —— 改完感知或判定，先在这 1000 步上
量一遍，而不是跑两个任务凭感觉说「好像顺了」。

## 能量什么，不能量什么

能量的两件事，数据是完整的：

· **elements** —— 同一张历史截图，纯 OCR 认出几个元素，加上屏幕解析之后认出几个。
  差值就是「以前根本没有抓手的东西」有多少。这是「识别不出来」的直接刻度。
· **effect** —— 历史上判成「没有变化」的那 172 步，前后两帧都还在。让看图的那一方
  重判一次，看它推翻多少。每推翻一次，就是当时白记的一次无进展 —— 那些是要喂给
  熔断器（NO_PROGRESS_STOP=6）的。

**不能**量的一件事，得说在前头：**候选栏修复没法用历史数据回放。**
中文输入失败时，执行层会把刚打的拼音退格删掉（不删的话下次输入会叠在残留上）。
于是 after 帧里候选栏已经没了 —— 当时的失败路径把证据自己擦掉了。
那条只能靠真机重跑，和 tests/test_vision_layer.py 里那组回归钉着。

## 花销

elements 和 effect 都要真调模型，一步一次。先用 --limit 跑小样本看看单步花多少，
再决定要不要铺满。`--no-vision` 完全不调模型，只验证纯 OCR 那一路没被改坏。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from iphone_agent.harness import judge
from iphone_agent.perceive.elements import build_observation
from iphone_agent.perceive.ocr import run_vision_ocr
from iphone_agent.perceive.screen import parse_screen


@dataclass
class Step:
    run: Path
    n: int
    raw: dict

    @property
    def before_file(self) -> Path | None:
        f = (self.raw.get("observation") or {}).get("frame_file")
        return self.run / f if f else None

    @property
    def after_file(self) -> Path | None:
        f = self.raw.get("after_frame_file")
        return self.run / f if f else None

    @property
    def result(self) -> dict:
        return self.raw.get("result") or {}

    @property
    def action(self) -> dict:
        return self.raw.get("action") or {}


def iter_steps(runs_root: Path, limit: int | None = None, *, need_after: bool = False):
    """按运行目录名（也就是时间）顺序产出步。缺帧的步直接跳过 —— 没图没法评。"""
    out: list[Step] = []
    for jsonl in sorted(runs_root.glob("*/steps.jsonl")):
        for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            s = Step(run=jsonl.parent, n=raw.get("step", 0), raw=raw)
            if s.before_file is None or not s.before_file.exists():
                continue
            if need_after and (s.after_file is None or not s.after_file.exists()):
                continue
            out.append(s)
            if limit and len(out) >= limit:
                return out
    return out


def _load(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


class _Frame:
    """build_observation 只用到这几个字段，不必把 driver.Frame 整个搬过来。"""

    def __init__(self, image):
        self.image = image
        self.width_px, self.height_px = image.size
        self.frame_id = 0
        from iphone_agent.driver.geometry import Rect
        self.window_rect = Rect(0, 0, *image.size)


@dataclass
class ElementsReport:
    steps: int = 0
    ocr_total: int = 0
    fused_total: int = 0
    vision_only: int = 0            # 视觉看到、OCR 完全没读到的 —— 以前没有抓手的那些
    kinds: dict = field(default_factory=dict)
    vision_failed: int = 0

    def lines(self) -> list[str]:
        if not self.steps:
            return ["没有可评的步"]
        out = [f"元素覆盖（{self.steps} 步）",
               f"  纯 OCR      每屏 {self.ocr_total / self.steps:.1f} 个",
               f"  加上屏幕解析 每屏 {self.fused_total / self.steps:.1f} 个",
               f"  其中 OCR 完全读不到的：每屏 {self.vision_only / self.steps:.1f} 个"
               f"（共 {self.vision_only} 个）"]
        if self.kinds:
            top = sorted(self.kinds.items(), key=lambda kv: -kv[1])[:8]
            out.append("  类型分布：" + "  ".join(f"{k}×{v}" for k, v in top))
        if self.vision_failed:
            out.append(f"  ⚠ 视觉调用失败 {self.vision_failed} 步（这些步只算了 OCR）")
        return out


def eval_elements(steps, asker) -> ElementsReport:
    """同一张历史截图，两条路各认出多少元素。"""
    rep = ElementsReport()
    for s in steps:
        img = _load(s.before_file)
        frame = _Frame(img)
        boxes = run_vision_ocr(img)
        items = parse_screen(img, asker)
        if asker is not None and not items:
            rep.vision_failed += 1
        obs = build_observation(frame, boxes, observation_id=0, screen_items=items)
        rep.steps += 1
        rep.ocr_total += len(boxes)
        rep.fused_total += len(obs.elements)
        for e in obs.elements:
            if e.source == "vision":
                rep.vision_only += 1
            if e.kind:
                rep.kinds[e.kind] = rep.kinds.get(e.kind, 0) + 1
    return rep


@dataclass
class EffectReport:
    steps: int = 0
    overturned: int = 0             # 当时判「没变化」，看图说其实生效了
    agreed: int = 0
    no_answer: int = 0
    examples: list = field(default_factory=list)

    def lines(self) -> list[str]:
        if not self.steps:
            return ["没有可评的步"]
        pct = 100 * self.overturned / self.steps
        out = [f"「这一步成没成」复核（{self.steps} 步，都是当时判成没变化的）",
               f"  看图说其实生效了：{self.overturned}（{pct:.0f}%）← 当时白记的无进展",
               f"  看图也说没生效：  {self.agreed}",
               f"  没问成：          {self.no_answer}"]
        for run, n, act, why in self.examples[:5]:
            out.append(f"    · {run} step{n} {act} —— {why}")
        return out


def eval_effect(steps, asker) -> EffectReport:
    """当时判成「没变化」的步，让看图的那一方重判一次。

    ⚠ 这里没有 ground truth，模型的话不是真理。它的价值在于**分歧的量级**：
      如果它在 172 步里推翻了 3 步，那说明像素判据基本够用；推翻了 60 步，
      说明熔断器一直在数错数。要确认某一条到底谁对，只能人去看那两张图 ——
      examples 就是给这个用的。
    """
    rep = EffectReport()
    for s in steps:
        eff = judge.did_action_work(
            _load(s.before_file), _load(s.after_file),
            f"{s.action.get('name')}({s.action.get('args')})",
            (s.raw.get("model") or {}).get("expect"), asker)
        rep.steps += 1
        if eff is None:
            rep.no_answer += 1
        elif eff.worked:
            rep.overturned += 1
            rep.examples.append((s.run.name, s.n, s.action.get("name"), eff.why))
        else:
            rep.agreed += 1
    return rep


def pick_no_change(steps) -> list[Step]:
    """挑出「动作报成功、但判定说没变化」的步 —— 那是 changed=False 的全部来源。"""
    return [s for s in steps if s.result.get("ok") and s.result.get("changed") is False]
