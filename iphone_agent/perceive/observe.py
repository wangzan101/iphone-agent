from __future__ import annotations

import copy
import hashlib
import time
from contextlib import contextmanager

from PIL import Image

from iphone_agent import config, timing
from iphone_agent.driver.geometry import Frame
from iphone_agent.model.vision import OUTCOME_KINDS
from iphone_agent.perceive import policy
from iphone_agent.perceive.elements import (
    Observation,
    build_observation,
    build_zoom_observation,
    zoom_size,
)
from iphone_agent.perceive.ocr import run_vision_ocr
from iphone_agent.perceive.screen import ScreenParse, label_screen, parse_screen, parse_screen_full

LABEL_PURPOSES = ("adopt", "identity")
LABEL_STATUSES = ("ok", "missing", "invalid", "failed", "off")


class ObservationFrozen(RuntimeError):
    """Observation 已经推给模型（finalize 过），又有人想补写感知字段 —— 程序 bug（spec 2026-09-14 §6.3）。"""


def _fresh_stats(mode: str) -> dict:
    return {"mode": mode, "observations": 0, "observe_text": 0,
            "parse_by": {"always": 0, "model": 0, "fallback": 0},
            "label": {p: dict.fromkeys(LABEL_STATUSES, 0) for p in LABEL_PURPOSES},
            "ms": {"parse": 0, "label": 0},
            "vision_failures": dict.fromkeys(OUTCOME_KINDS, 0),
            "candidates_empty": 0, "label_dup_exact": 0}


def _pixels_digest(img) -> str:
    """裁掉状态栏后的像素指纹（裁剪比例只走 config.STATUS_BAR_CROP_RATIO 一个入口）。只量精确重复，不复用（spec §8.3）。"""
    top = int(img.height * config.STATUS_BAR_CROP_RATIO)
    body = img.crop((0, top, img.width, img.height))
    return hashlib.sha256(body.tobytes() + str(body.size).encode()).hexdigest()


class Perceiver:
    """observe(frame)：OCR +（按模式）屏幕解析 → 融合 → 元素 → 标记图 → 哈希。observation_id 递增。

    coord_mode 是当前模型的坐标约定，由 Session 注入。perceive 层收到的只是一个格式化参数，不认识模型。

    asker（VisionAsker）是**可选**的第二路感知。给 None 就退回纯 OCR —— 这条降级路必须一直留着：
    视觉那一路会超时、会抽风、会被用户关掉。

    ⚠ 2026-09-14（spec 按需看图 §3.3、§6.3）：Perceiver 是**唯一**能写 obs.screen / screen_candidates /
      perception 的地方。构造之后唯一的补写方法是 ensure_label；finalize 之后再补写抛 ObservationFrozen。
    """

    def __init__(self, ocr=run_vision_ocr, coord_mode: str = "norm1000", asker=None):
        self._ocr = ocr
        self._next_id = 1
        self.coord_mode = coord_mode
        self.asker = asker
        self._identity = None
        self._mode: str | None = None
        self._stats = _fresh_stats(policy.DEFAULT_MODE)
        self._labeled_px: set[str] = set()

    @property
    def mode(self) -> str:
        """本任务的整屏解析模式：task_scope 挂上的那个；不在任何 scope 里时是 policy.DEFAULT_MODE。"""
        return self._mode or policy.DEFAULT_MODE

    @contextmanager
    def task_scope(self, identity, mode: str):
        """任务期间挂上认屏上下文（可以是 None）和本次的整屏解析模式，并清零本次的 stats；
        finally 里一定解除上下文和模式（spec 2026-09-14 §3.3）。

        ⚠ Perceiver 是会话级的，上下文和模式是任务级的：不能在共享对象上临时挂一个值然后忘了摘 ——
          下一个任务会拿着上一个任务的孪生挑候选、按上一个任务的模式解析（2026-09-12 codex 评审第 4 条）。
        ⚠ 2026-09-14：原来的 bind_identity 只在孪生初始化成功时才挂，孪生一坏，模式覆盖和 stats 清零就一起
          不生效。现在 loop 每个任务都无条件进这里。stats 退出时**不**恢复（计划 R14）：loop 关掉 scope 之后
          才写 run.json。
        """
        policy.check_mode(mode)
        prev = (self._identity, self._mode)
        self._identity, self._mode = identity, mode
        self._stats, self._labeled_px = _fresh_stats(mode), set()
        try:
            yield self
        finally:
            self._identity, self._mode = prev

    def stats(self) -> dict:
        """本次任务的感知统计（spec §8.3）。成功失败都记（CLAUDE.md §3）。"""
        return copy.deepcopy(self._stats)

    def _candidates(self, texts: list[str]) -> list:
        ctx = self._identity
        if ctx is None:
            return []
        try:
            return list(ctx.candidates(texts))
        except Exception:       # noqa: BLE001 —— 孪生坏了 = 没有候选（docs/32 不变式 5），观察照常
            return []

    def _vision_call(self, phase: str, fn):
        """跑一次视觉调用，返回 (结果, 毫秒, 本次的失败原因或 None)。

        失败原因在调用返回后**立刻**从 asker.last_outcome 抄出来 —— 它下一次调用就会被清空。
        phase 是计时段名（parse / label），按运行阶段计时的唯一挂点（spec 2026-09-14 §8.2）。
        """
        t0 = time.perf_counter()
        with timing.phase(phase):
            out = fn()
        ms = int((time.perf_counter() - t0) * 1000)
        oc = getattr(self.asker, "last_outcome", None)
        err = None
        if isinstance(oc, dict) and oc.get("kind"):
            err = {"kind": str(oc["kind"]), "detail": str(oc.get("detail") or "")[:400]}
            fails = self._stats["vision_failures"]
            fails[err["kind"]] = fails.get(err["kind"], 0) + 1
        return out, ms, err

    def observe(self, frame: Frame, *, requested: bool = False, fallback: bool = False) -> Observation:
        """OCR，再按 policy.parse_trigger 决定跑不跑整屏解析（spec 2026-09-14 §3.3）。

        requested=True：模型调了 observe（看全屏）；fallback=True：程序兜底（§3.5）。
        解析时带上孪生候选，一次拿到元素和标注。
        ⚠ 两路都失败不能等于观察失败：parse_screen_full 自己吞掉所有异常返回空列表，OCR 才是命脉。
        ⚠ 模式只管**这里**，不管 zoom：zoom 是模型自己开口要看清一小块，5 秒、每个元素都是它要的。
          一度把开关做在 VisionAsker.enabled 上，结果关掉整屏解析连 zoom 一起废了。
        """
        t0 = time.perf_counter()
        with timing.phase("ocr"):
            boxes = self._ocr(frame.image)
        ocr_ms = int((time.perf_counter() - t0) * 1000)
        has_asker = self.asker is not None
        why = policy.parse_trigger(self.mode, requested, fallback, has_asker)
        parse_ms, parse_err = 0, None
        if why is None:
            cands, sp = [], ScreenParse([], "off", None, "off")
        else:
            cands = self._candidates([b.text for b in boxes])
            sp, parse_ms, parse_err = self._vision_call(
                "parse", lambda: parse_screen_full(frame.image, self.asker, cands))
            self._stats["parse_by"][why] += 1
            timing.note_parse(why)
            self._stats["ms"]["parse"] += parse_ms
        obs = build_observation(frame, boxes, observation_id=self._next_id,
                                coord_mode=self.coord_mode, screen_items=sp.items,
                                full_screen=sp.vision in ("ok", "empty"))
        # parse 说这一帧为什么解析、为什么没解析；vision=off 配 parse=None 是「按策略没做」，不是「视觉说没有」。
        obs.perception = {"ocr": len(boxes), "vision": sp.vision, "screen": sp.label_status,
                          "mode": self.mode, "parse": why if has_asker else "no_asker",
                          "label_by": "parse" if why is not None else None,
                          "parse_error": parse_err, "label_error": None,
                          "ms": {"ocr": ocr_ms, "parse": parse_ms, "label": 0}}
        obs.screen = sp.label
        obs.screen_candidates = cands
        self._stats["observations"] += 1
        self._next_id += 1
        return obs

    def observe_text(self, frame: Frame) -> Observation:
        """只跑 OCR 的观察：不做整屏解析、不挑认屏候选。observation_id 和 observe 共用一个计数。

        ⚠ 2026-09-14：翻主屏找 App 原来每页都走 observe（OCR + 整屏视觉解析，一页 20–30 秒），
          还为了比 aHash 再完整观察一次 —— 翻主屏开 App 中位 112 秒（n=23），比要打字的 Spotlight
          （44–53 秒）还慢。翻页找 App 要的只是图标下面的标签，OCR 就读得到。
          perception 如实写 off：这一帧没做视觉，不是「视觉说没有」（CLAUDE.md §3）。
        """
        t0 = time.perf_counter()
        with timing.phase("ocr"):
            boxes = self._ocr(frame.image)
        ocr_ms = int((time.perf_counter() - t0) * 1000)
        obs = build_observation(frame, boxes, observation_id=self._next_id, coord_mode=self.coord_mode)
        obs.perception = {"ocr": len(boxes), "vision": "off", "screen": "off",
                          "mode": self.mode, "parse": None, "label_by": None,
                          "parse_error": None, "label_error": None,
                          "ms": {"ocr": ocr_ms, "parse": 0, "label": 0}}
        self._stats["observe_text"] += 1
        self._next_id += 1
        return obs

    def ensure_label(self, obs, purpose: str) -> None:
        """需要时给这一帧补一次短标注（spec 2026-09-14 §5.2）。幂等：同一个 obs 只调一次。

        purpose：adopt = 进入 loop 的观察；identity = executor._identity 的身份核验。两类分开计数。
        要不要标只问 policy.wants_label。候选由这帧的 OCR 文字经 twin/context.py 的规则挑出。
        标注失败时这一帧就是 unlabeled，不猜；原因写进本帧的 label_error（§9）。
        """
        if purpose not in LABEL_PURPOSES:
            raise ValueError(f"purpose 只能是 {LABEL_PURPOSES}，收到 {purpose!r}")
        if getattr(obs, "finalized", False):
            raise ObservationFrozen(f"observation #{obs.observation_id} 已推给模型，不能再补写标注")
        if not policy.wants_label(self.mode, obs, self.asker is not None):
            return
        cands = self._candidates([e.text for e in obs.elements if e.source != "vision"])
        if not cands:
            self._stats["candidates_empty"] += 1
        (label, status), ms, err = self._vision_call(
            "label", lambda: label_screen(obs.image, self.asker, cands))
        obs.screen, obs.screen_candidates = label, cands
        obs.perception.update({"screen": status, "label_by": "label", "label_error": err})
        obs.perception.setdefault("ms", {})["label"] = ms
        counts = self._stats["label"][purpose]
        counts[status] = counts.get(status, 0) + 1
        self._stats["ms"]["label"] += ms
        if purpose == "adopt":
            digest = _pixels_digest(obs.image)
            if digest in self._labeled_px:
                self._stats["label_dup_exact"] += 1
            self._labeled_px.add(digest)

    def finalize(self, obs) -> None:
        """push_obs 推给模型前调（spec §6.3）：从此这一帧的感知字段不再改动。"""
        obs.finalized = True

    def zoom(self, base: Observation, box: tuple[int, int, int, int]) -> Observation:
        """把 base 的某一块放大重看。不碰设备 —— 用的是它当时那张图。

        为什么重新跑一遍 OCR：**放大之后它明显更准**。小字、图标旁边的标签、
        候选栏里挤在一起的词，在 624x1388 的整图上常常读错或读不到，
        放大到长边 1400 再读就出来了。这正是 zoom 的价值所在，不是顺手做的。

        局部屏幕解析也在这儿：区域小、元素少，输出天然短 ——
        不会像整屏解析那样撞上 max_tokens 然后整屏归零。所以这里**不看整屏解析模式**：
        模式管的是「每次观察都全屏解析」那件事，而这里是模型自己开口要看清楚。
        zoom 观察的 perception 是空的：它不进孪生（重放跳过 zoom 之后那一帧），也不标注（spec §5.2）。
        """
        with timing.phase("zoom"):
            crop = base.image.crop(box)
            zoomed = crop.resize(zoom_size(crop.width, crop.height), Image.LANCZOS)
            boxes = self._ocr(zoomed)
            items = parse_screen(zoomed, self.asker) if self.asker is not None else []
            obs = build_zoom_observation(base, box, zoomed, boxes,
                                         observation_id=self._next_id, screen_items=items)
            # 兜底按被放大的那一帧判（spec §3.1）；从 zoom 帧再 zoom 就沿用它记下的那一帧。
            obs.zoom_base_vision = (base.perception.get("vision") if base.perception
                                    else base.zoom_base_vision)
            self._next_id += 1
            return obs
