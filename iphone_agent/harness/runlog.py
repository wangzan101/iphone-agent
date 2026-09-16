"""runs/<时间戳-shortid>/：run.json、steps.jsonl、frame_NNN.png（spec §7）。"""
from __future__ import annotations

import json
import secrets
import time
from datetime import datetime
from pathlib import Path

from iphone_agent.driver.geometry import Frame, Rect


def frame_name(frame_id: int) -> str:
    """帧文件名的唯一入口：落盘（save_frame）和孪生的屏 id（twin 按 run:frame_file 建屏）共用 ——
    两边各写一遍，实时与重放算出的屏 id 就会分叉（spec 2026-09-12 §4.1，codex 评审第 1 条）。"""
    return f"frame_{frame_id:03d}.png"


def observation_snapshot(obs, frame_file: str) -> dict:
    r = obs.window_rect
    snap = {
        "frame_file": frame_file,
        "width_px": obs.width_px,
        "height_px": obs.height_px,
        "window_rect": {"x": r.x, "y": r.y, "w": r.w, "h": r.h},
        "ahash": obs.ahash,
        "elements": [_element_row(e) for e in obs.elements],
    }
    # 感知状态：视觉那一路失败要在日志里看得见（spec 2026-09-11 §3.1）。没有就不写，老调用方不变。
    perception = getattr(obs, "perception", None)
    if perception:
        snap["perception"] = dict(perception)
    label = getattr(obs, "screen", None)
    if label is not None:
        snap["screen"] = label.to_json()
    cands = getattr(obs, "screen_candidates", None)
    if cands:
        snap["screen_candidates"] = [c.to_json() for c in cands]
    return snap


def _element_row(e) -> dict:
    """一个元素落盘的样子。

    ⚠ source **总是写**，哪怕它是默认的 'ocr'。2026-09-09 加视觉那一路时这里漏了，
      后果是日志里完全看不出哪些元素来自屏幕解析 —— 排查一次真机失败时，我因此
      误判「视觉层没生效」，绕了一圈才从 exec_ms 反推出它其实在跑。
      诊断字段省不得：省下的那点字节，换的是下次看日志时的一个错误结论。
      kind/state 则是多数元素本来就没有，空着不写。
    """
    row = {"id": e.id, "text": e.text, "confidence": e.confidence,
           "box": list(e.box), "center": list(e.center), "source": e.source}
    if e.kind:
        row["kind"] = e.kind
    if e.state:
        row["state"] = e.state
    return row


def frame_stub(obs) -> Frame:
    """Observation 持有图与 frame_id，足够落盘。"""
    return Frame(obs.image, obs.width_px, obs.height_px, Rect(0, 0, 1, 1), 0.0, obs.frame_id)


class RunLog:
    def __init__(self, root: Path, task: str, model: str, config_snapshot: dict, prompt_hash: str):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / f"{stamp}-{secrets.token_hex(2)}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self._run = {
            "task": task, "model": model, "model_version": "",
            "started_at": time.time(), "ended_at": None, "end_reason": None,
            "steps": 0, "done": None, "usage": {}, "prompt_hash": prompt_hash,
            "schema_version": 1, "config": config_snapshot,
            "memory": None,
            "knowledge": None,
        }
        self._write_run()
        self._saved: set[int] = set()

    def _write_run(self):
        (self.dir / "run.json").write_text(json.dumps(self._run, ensure_ascii=False, indent=2))

    def save_frame(self, frame: Frame) -> str:
        name = frame_name(frame.frame_id)
        if frame.frame_id not in self._saved:
            frame.image.save(self.dir / name)
            self._saved.add(frame.frame_id)
        return name

    def step(self, record: dict) -> None:
        with (self.dir / "steps.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def procedure_step(self, record: dict) -> None:
        """剧本内部动作的子记录。kind 字段让屏幕图和提取器能区分它，旧日志没有 kind 就是普通 step。"""
        self.step({"kind": "procedure_step", **record})

    def set_knowledge(self, knowledge: dict) -> None:
        """路由结果、注入原文、工具列表、剧本执行情况、提取结果。与 memory 并列，写在 finish 之后。"""
        self._run["knowledge"] = knowledge
        self._write_run()

    def finish(self, reason: str, steps: int, done_status, done_result, model_version: str, usage_total: dict,
               budget: dict | None = None):
        self._run.update({"ended_at": time.time(), "end_reason": reason, "steps": steps,
                          "done": None if done_status is None else {"status": done_status, "result": done_result},
                          "model_version": model_version, "usage": usage_total})
        # budget 是可选的（老调用方不传）：run 级 token 账汇总，来自内存里的
        # budgets_by_call，不是这里读 jsonl 现算的 —— 见 loop.py finally 里的注释。
        if budget is not None:
            self._run["budget"] = budget
        self._write_run()

    def set_audit(self, note: dict) -> None:
        """事后审计的留档。**只是提请注意，不改任务终态** ——
        它当前的覆盖率很低（见 harness/audit.py），把它变成判定会误伤正确的运行。"""
        self._run["audit"] = note
        self._write_run()

    def set_twin(self, summary: dict) -> None:
        """孪生这次的账：实时认屏计数、收尾记账统计、错误。成功失败都写 —— 能失败的东西要看得见。"""
        self._run["twin"] = summary
        self._write_run()

    def set_section(self, key: str, value) -> None:
        """run.json 顶层的一节（perception / taps / startup_timing，spec 2026-09-14 §8）。
        只改这一项再整体重写，排在 finish() 之后调用也不会抹掉终态。"""
        self._run[key] = value
        self._write_run()

    def set_memory(self, injected: str | None, recalled: list[str],
                   written: list[dict], runs_skipped: int = 0,
                   shadow: dict | None = None) -> None:
        """留档注入原文与写入结果。

        记忆后来会被覆盖，不留原文就再也还原不出当时模型看见了什么，
        验收第 A 条（召回可观测）也就无从判定。

        runs_skipped 是拼摘要时读不出来的运行记录数。它是「摘要代码自己坏了」
        与「确实没有历史运行」的唯一区分信号 —— 两种情况在 injected 里长得
        一模一样（都没有「最近的运行」那一段），不单独记就再也分不开。

        shadow 是「转正/淘汰如果真做了会发生什么」的记账（used / outcome /
        audit_available / would_verify / would_trash）。这一轮**只记不动**：
        判据对不对要靠攒够真实运行来验，先让它在日志里跑一段时间，别急着让代码
        照着它删记忆。默认 None 是为了让不关心它的调用方（老测试、手工补记）照常用。

        只改 self._run 里 memory 这一项再整体重写文件，所以排在 finish() 之后
        调用（写记忆要等设备释放）也不会把 finish 写的终态抹掉。
        """
        self._run["memory"] = {"injected": injected, "recalled": recalled,
                               "written": written, "runs_skipped": runs_skipped,
                               "shadow": shadow}
        self._write_run()

    @staticmethod
    def read_steps(run_dir: Path) -> list[dict]:
        p = Path(run_dir) / "steps.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
