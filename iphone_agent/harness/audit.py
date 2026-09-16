"""事后审计：模型报了成功，这个成功可信吗？

`done(success)` 现在是不可验证的 —— 只能信模型。2026-09-08 一晚上就出了两次假成功，
**两次都是我人肉去看截图才发现的**。日常用不可能每次都看。

## 两次假成功长什么样

1. Spotlight 残留着上一轮的「记账本」，模型**一个字没输**就报了成功。
   它没撒谎 —— 屏幕上确实写着那四个字，是环境不干净让任务在错误的前提上开始。
2. 模型在通用页找不到「关于本机」，改去设置首页，把「**更新到** iOS 26.6.1」
   这条**可更新版本**当成当前版本报了成功。真值 18.3.1。

## 实测：模型不编数字

`fabricated_values()` 把报告里的数值型 token 拿去和这次运行**所有观察过的屏幕文字**
比对。2026-09-08 跑遍当时全部成功运行：**零命中**。

这是个有意义的结果，它把问题缩窄了：**假成功不是幻觉，是读错了地方**。
报出来的值确实在屏幕上，只是来自错的那一屏。

所以真正的防线是「这个值是不是来自该来的那一屏」，而不是「这个值存不存在」。

## 「有没有到过该到的屏」—— 机制是对的，覆盖率还不够

`never_reached()`：任务里 `「」` 点名的目标，去屏幕图上找对应的屏，
再看这次运行有没有到过。

**用两条真实轨迹验证过**（只把任务文案改成带书名号的形式，轨迹原样）：

    ⚠ 标为可疑   runs/20260908-022551-5204   报了 iOS 26.6.1（假的）  没到过「关于本机」
      放行       runs/20260908-024924-eca3   报了 18.3.1（真的）

**第一版误报率高得没法用**：10 次点名的成功运行标出 7 次可疑，其中至少 5 次是
**正确的运行**。原因不是阈值，是漏了一条 —— **目的地本身认不出来的不能拿来审计**，
那时「没到过」其实是「认不出来」。「天气」「iPhone 储存空间」这种只到过 1 次的
目的地根本不在可识别节点里。补上这一条之后，32 次成功运行里 **0 误报**。

**现在的局限是覆盖率，不是正确性**：要能审，任务得用「」点名，且那个目的地
在图上到过 ≥2 次。当前 30 个节点里 19 个只到过 1 次，所以实际能审的很少。
需要的是**更密的图**，不是更松的阈值 —— 对审计来说误报比漏报更致命：
一个吵闹的审计器只会训练人忽略它。
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
