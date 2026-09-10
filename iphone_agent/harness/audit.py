"""事后审计：检查成功报告是否有观察证据支持。

报告里的数值出现过，不代表它来自正确页面。例如，更新提示中的版本号
不能当成当前系统版本。这里结合观察过的文字与屏幕图检查报告。

屏幕图尚未识别的目的地不能判为「未到达」。检查覆盖率取决于已有
观察，不能把未发现问题当成成功证明。回归测试使用合成轨迹。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# 版本号、金额、时间、容量这类**具体值**。纯粹的短数字（一位）不算，噪声太大。
_VALUE = re.compile(r"[0-9]+(?:[.:][0-9]+)+|[0-9]{2,}")
_NAMED = re.compile(r"[「『]([^」』]{2,12})[」』]")


@dataclass
class RunFacts:
    run_dir: str
    task: str
    result: str
    screen_texts: set[str]


def load_run(run_dir: Path) -> RunFacts | None:
    """一次成功运行的任务、报告、和它见过的所有屏幕文字。失败的运行返回 None。"""
    try:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("end_reason") != "done_success":
        return None
    texts: set[str] = set()
    try:
        with (run_dir / "steps.jsonl").open(encoding="utf-8") as f:
            for line in f:
                try:
                    st = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for e in (st.get("observation") or {}).get("elements") or []:
                    t = e.get("text", "").strip()
                    if t:
                        texts.add(t)
    except OSError:
        return None
    return RunFacts(run_dir.name, meta.get("task") or "",
                    str((meta.get("done") or {}).get("result", "")), texts)


def fabricated_values(facts: RunFacts) -> list[str]:
    """报告里出现、但**任何一屏都没有过**的具体值。

    精确性由构造保证：屏幕上从没出现过的数字，模型只能是编的。
    代价是召回低 —— 读错地方（值确实在屏上，只是来自错的屏）这一类抓不到，
    而那恰恰是实测中唯一发生过的一类。
    """
    blob = " ".join(facts.screen_texts)
    return sorted({v for v in _VALUE.findall(facts.result) if v not in blob})


def named_targets(task: str) -> list[str]:
    """任务里用「」点名的东西。

    ⚠ 只认书名号里的。不加这一条的话，「不要修改任何设置」里的「设置」
    会被当成目标 —— 实测这一个误判就能把审计结果整个淹掉。
    """
    return _NAMED.findall(task)


def never_reached(m, facts: RunFacts, steps: list[dict]) -> list[str]:
    """任务点名的屏里，这次运行一次都没到过的那些。

    ⚠ 当前的图太稀，误报率高得没法用 —— 见模块开头。只作参考，别当判定。
    """
    labels: dict[str, tuple[int, int]] = {}
    for e in m.edges.values():
        if not e.target or e.src == e.dst:
            continue
        label = e.target.split("@")[0].strip()
        if len(label) >= 2 and (label not in labels or e.count > labels[label][1]):
            labels[label] = (e.dst, e.count)
    # ⚠ 目的地本身**认不出来**的，不能拿来审计 —— 那时「没到过」其实是「认不出来」。
    #   第一版漏了这一条，于是把「天气」「iPhone 储存空间」这种只到过 1 次的
    #   目的地全判成可疑，把 5 次正确的运行标成了假成功。
    identifiable = {n.key for n in m.stable_nodes()}
    named = [w for w in named_targets(facts.task)
             if w in labels and labels[w][0] in identifiable]
    if not named:
        return []
    seen = set()
    for st in steps:
        els = (st.get("observation") or {}).get("elements") or []
        ts = {e["text"].strip() for e in els if e.get("text", "").strip()}
        if ts:
            n = m.identify(ts)
            if n is not None:
                seen.add(n.key)
    return [w for w in named if labels[w][0] not in seen]
