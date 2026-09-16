"""自动探索一个 App，产出评测集的原料：帧 + 点击 + 「变没变」。

## 为什么是这个形状

评测集最硬的标签不是人（或我）看图标出来的，是**执行验证**：点了这个位置，
画面变没变。它零人工、不依赖任何模型、而且是手机自己说的。试点里 9 次全中。

所以大规模评测集的主体就是「随机游走 + 记录每一步」：

    打开 App → 抓帧 → 纯 OCR 找元素 → 挑一个没点过的 → 点 → 抓帧
      → 记 (前帧, 目标, 后帧, hamming, local_mad, changed) → 继续

每一步都是一条样本：变了 = 那个位置可点（正样本），没变 = 不可点（负样本）。
帧留着给 agent 的感知离线跑，我只需要抽几帧看 kind/state 之类 OCR 给不出的东西。

## 安全

这是在真手机上真点。三道闸：
· 文字含危险词的元素一律不点（发送/删除/购买……见 DANGER）
· 每个 App 有点击上限和深度上限
· 每次点完检查有没有跳出 App（回到主屏 / 进了别的 App），跳出就 home 重开

`--shallow` 给社交和商店类：只在首屏点，点完就想办法回首屏。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from iphone_agent.driver.geometry import image_to_screen
from iphone_agent.driver.timing import IOS_TIMING
from iphone_agent.harness.safety import DANGER as _DANGER
from iphone_agent.harness.settle import settle
from iphone_agent.perceive.change import local_mad
from iphone_agent.perceive.hashing import ahash, hamming, state_key
from iphone_agent.perceive.ocr import run_vision_ocr, vision_box_to_pixels

# 含这些字的元素不点。宁可漏掉一些正样本，也不能在别人手机上发出一条消息。
# 词表本体在 harness/safety.py —— 主循环的安全闸用的是同一份（一个规则一个入口，CLAUDE.md §7）。
DANGER = _DANGER
STATUS_BAR = 0.11


@dataclass
class Tap:
    step: int
    before: str                 # 帧文件名
    after: str
    text: str                   # 点的元素文字（OCR）
    x: int
    y: int
    hamming: int
    local_mad: float
    changed: bool               # 综合判定
    left_app: bool = False
    note: str = ""


class Explorer:
    def __init__(self, session, out: Path, taps: int = 15, depth: int = 3,
                 shallow: bool = False, log=print):
        self.s = session
        self.dev = session.dev
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "frames").mkdir(exist_ok=True)
        self.max_taps = taps
        self.max_depth = depth
        self.shallow = shallow
        self.log = log
        self.tried: set[tuple[int, str]] = set()     # (画面 state_key, 元素文字)：键盘按过一位还能按下一位
        self.records: list[Tap] = []
        self._fid = 0

    # ---- 帧 ----
    def _save(self, frame, tag: str) -> str:
        self._fid += 1
        name = f"{self._fid:03d}-{tag}.png"
        frame.image.save(self.out / "frames" / name)
        return name

    def _elements(self, frame):
        W, H = frame.width_px, frame.height_px
        out = []
        for b in run_vision_ocr(frame.image):
            x1, y1, x2, y2 = vision_box_to_pixels(b, W, H)
            cy = (y1 + y2) // 2
            if cy < H * STATUS_BAR:                  # 状态栏
                continue
            t = b.text.strip()
            if not t or DANGER.search(t):
                continue
            out.append((t, (x1 + x2) // 2, cy))
        return out

    def _left_app(self, frame) -> bool:
        from types import SimpleNamespace

        from iphone_agent.harness.executor import looks_like_home
        return looks_like_home(SimpleNamespace(elements=[SimpleNamespace(text=t) for t, _, _ in self._elements(frame)]))

    # ---- 打开 / 回家 ----
    def open(self, app: str) -> bool:
        from iphone_agent.harness.actions import Action
        from iphone_agent.harness.executor import Executor
        from iphone_agent.perceive.observe import Perceiver
        # ⚠ 开 App 用**纯 OCR** 的 perceiver。_open_app 内部要 observe 五到八次
        #   （Spotlight 前后各一次、主屏每翻一页一次），带整屏解析的话每次 20~34s，
        #   一次失败的 open_app 要花四五分钟。探索只需要读到 App 名，OCR 够了。
        per = Perceiver(coord_mode=self.s.per.coord_mode, asker=None)
        ex = Executor(self.dev, per, asker=None)
        self._reset_to_home()
        obs = per.observe(self.dev.capture())
        res, new = ex.run(Action("open_app", {"name": app}, "explore", None, "x"), obs)
        ok = bool(res.ok)
        self.log(f"  open_app({app!r}) → {'ok' if ok else res.error}")
        return ok

    def _reset_to_home(self):
        """回到主屏第一页，**确认**回到了才算。

        ⚠ 一次 home 不够。2026-09-09 批跑：上一个 App 留下的模态弹层、全屏视频、
          系统设置的教学浮窗，一次 home 退不干净，下一个 App 的 open_app 就在
          污染过的画面上跑 —— 34 个 App 里 16 个「打不开」，单独跑却都能开。
          第一次 home 退出前台 App，第二次回主屏第一页，然后用 OCR 确认。
        """
        for _attempt in range(3):
            self.dev.key("home"); time.sleep(0.7)
            self.dev.key("home"); time.sleep(0.7)
            if self._left_app(self.dev.capture()):
                return
        self.log("  ⚠ 按了六次 home 还没回到主屏，画面上可能有关不掉的东西")

    def _go_back(self, frame) -> bool:
        """试着点左上角的返回。找不到返回 False。"""
        for t, x, y in self._elements(frame):
            if y < frame.height_px * 0.17 and x < frame.width_px * 0.35 and \
                    (t.startswith("<") or t.startswith("＜") or "返回" in t or t in ("取消", "完成", "关闭")):
                sx, sy = image_to_screen(x, y, frame)
                self.dev.tap(sx, sy)
                return True
        return False

    # ---- 主循环 ----
    def run(self, app: str) -> list[Tap]:
        if not self.open(app):
            return []
        (self.out / "taps.jsonl").unlink(missing_ok=True)
        time.sleep(1.0)
        frame = self.dev.capture()
        # ⚠ open_app 报了 ok 不等于真到了那个 App。2026-09-09 第一轮探索：
        #   「设置」的 15 条记录全是时钟的界面，「X」是 YouTube，「计算器」停在 Spotlight ——
        #   当时 _match_row 的假成功还没修，探索照单全收，评测集里就有了三个 App 的脏数据。
        #   这里不信它，自己看一眼：还在 Spotlight 或主屏就是没开成。
        from iphone_agent.harness.executor import SPOTLIGHT_MARKERS
        texts = {t for t, _, _ in self._elements(frame)}
        if any(m in t for t in texts for m in SPOTLIGHT_MARKERS) or self._left_app(frame):
            self.log(f"  open_app({app!r}) 报 ok 但画面还在 Spotlight/主屏，不算开成，跳过")
            return []
        root_name = self._save(frame, "root")
        depth = 0
        for step in range(1, self.max_taps + 1):
            els = self._elements(frame)
            fh = ahash(frame.image, STATUS_BAR)
            sk = state_key(fh, (t for t, _, _ in els))
            cand = [(t, x, y) for t, x, y in els if (sk, t) not in self.tried]
            if not cand:
                # 这一屏点遍了：回一层，或者结束
                if depth > 0 and self._go_back(frame):
                    frame, _ = settle(self.dev, frame, IOS_TIMING["tap"], ahash)
                    depth -= 1
                    continue
                self.log("  这一屏没有没点过的元素了，停")
                break
            t, x, y = cand[0]
            self.tried.add((sk, t))
            before_name = self._save(frame, "before") if step > 1 else root_name
            sx, sy = image_to_screen(x, y, frame)
            self.dev.tap(sx, sy)
            after, settled = settle(self.dev, frame, IOS_TIMING["tap"], ahash)
            self.dev.release_all()
            hd = hamming(fh, ahash(after.image, STATUS_BAR))
            lm = local_mad(frame.image, after.image, x, y)
            changed = hd > 2 or lm >= 4.0
            left = changed and self._left_app(after)
            after_name = self._save(after, "after") if changed else before_name
            rec = Tap(step, before_name, after_name, t, x, y, hd, round(lm, 1), changed, left)
            self.records.append(rec)
            self._append(rec)       # 每步落盘：中途崩一次不能把前面的记录一起带走
            self.log(f"  {step:>2}. tap {t[:14]!r:<18} ({x},{y})  hamming={hd:<3} mad={lm:<5.1f} "
                     f"{'变了' if changed else '没变'}{'  ← 跳出 App 了' if left else ''}")
            if left:
                rec.note = "left app; reopened"
                if not self.open(app):
                    break
                frame = self.dev.capture()
                depth = 0
                continue
            if changed:
                depth += 1
                if self.shallow or depth >= self.max_depth:
                    # 回到根：先试返回，不行就重开
                    if self._go_back(after):
                        after, _ = settle(self.dev, after, IOS_TIMING["tap"], ahash)
                        depth -= 1
                    else:
                        self.open(app); after = self.dev.capture(); depth = 0
            frame = after
        self._write()
        return self.records

    def _append(self, rec: Tap):
        with open(self.out / "taps.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def _write(self):
        # 记录已经逐步 append 过了；这里只汇报。left_app 时 note 是事后改的，重写一遍保证一致。
        with open(self.out / "taps.jsonl", "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        n = len(self.records)
        pos = sum(1 for r in self.records if r.changed)
        self.log(f"  → {n} 次点击：{pos} 正样本 / {n - pos} 负样本，帧 {self._fid} 张，写入 {self.out}")
