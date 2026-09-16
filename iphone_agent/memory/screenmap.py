"""从运行日志里把屏幕图拼出来。

每一步日志都记了完整的元素列表 + 动作 + 结果，所以
**(屏幕A, 动作) → 屏幕B 这张图一直在生成，只是从来没人把它拼起来**。

节点 = 一屏；节点的**指纹** = 同一屏多次访问、文字集合的交集
（内容会变：数字、列表、时间；骨架不变：标题、按钮、tab 栏）。

指纹稳不稳是可证伪的，已经量过（`scripts/map_experiment.py`，2026-09-08）：
222 次观察 / 26 个 run，指纹词数中位数 18，最小 7，**0 个空指纹，1/17 撞车**。
及格线定在「中位数 ≥ 4 且不撞车」。

⚠ 边只记**语义目标**，不记坐标。坐标是这个项目里最脆的东西 ——
坐标约定一改整套动作全废（2026-09-08 真机，连拒五次）；窗口一拉大局部判据就瞎。
"""
from __future__ import annotations

import json
import os
import re as _re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pypinyin import lazy_pinyin

JACCARD = 0.5          # 判「同一屏」的相似度门槛
MIN_VISITS = 2         # 至少访问过这么多次才谈得上指纹（交集要有意义）

SYSTEM = "system"      # 主屏、Spotlight、小组件页
UNKNOWN = "unknown"    # app_switcher 之后到下一次 open_app / home 之前：归属不明，不提草稿


def label_like(t: str) -> bool:
    """像个界面标签吗？（原 harness/whereami._label_like，挪到这里让 skills 层也能用，
    skills 不许 import harness。）"""
    t = t.strip()
    if not (2 <= len(t) <= 12):
        return False
    if any(c in t for c in ":/·•><＞＜"):
        return False
    return any("一" <= c <= "鿿" for c in t) or (t.isascii() and t[0].isalpha())


def slugify(text: str, max_len: int = 48) -> str:
    """任意文字 → 合法 id 片段：中文转拼音，只留 [a-z0-9-]，合并连字符。空则 'x'。"""
    parts = lazy_pinyin(text) if not text.isascii() else [text]
    s = "-".join(parts).lower()
    s = _re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    s = _re.sub(r"-{2,}", "-", s)[:max_len].strip("-")
    if not s or not s[0].isalnum():
        s = "x" + s
    return s


def app_id_for(open_name: str) -> str:
    """open_app 的参数 → App id。ASCII 直接小写；中文转拼音（与 executor._open_app 同一函数）。"""
    return slugify(open_name)


def _action_of(st: dict) -> tuple[str | None, dict, dict]:
    a = st.get("action") or {}
    return a.get("name"), (a.get("args") or a.get("args_raw") or {}), (st.get("result") or {})


def _icon_tap_label(st: dict, owner: str, name: str | None, args: dict, res: dict) -> str | None:
    """这条记录上的动作是不是「在主屏上点开了一个 App 图标」？是就把标签还原出来，不是就 None。

    真机数据的判别式（见文件头引用的 dry-run）：owner 还是 SYSTEM（主屏/Spotlight/小组件）、
    target 是 icon_above、画面真的变了、id 在**这一步自己的观察**里能还原出文字——
    四条都满足才算「进了一个 App」。少一条都不算：
      - owner 不是 SYSTEM：已经在 App 里了，那多半是在点底部 tab 栏，不是又进了一次
      - target 不是 icon_above（比如 text）：点的是别的东西，Spotlight 的搜索框就是这么被排除的
      - changed 是假：点了但画面没变，没有真的进去
      - id 还原不出文字：不知道点的是哪个图标，宁可不认，也不要瞎猜一个 App
    """
    if owner != SYSTEM or name != "tap" or args.get("target") != "icon_above" or not res.get("changed"):
        return None
    eid = args.get("id")
    if eid is None:
        return None
    return _id_text_in(st, eid)


def _id_text_in(step: dict, eid) -> str | None:
    """元素编号只在**当次观察**里有效，所以还原永远用这一步自己的元素表。"""
    for e in (step.get("observation") or {}).get("elements") or []:
        if e.get("id") == eid:
            t = (e.get("text") or "").strip()
            return t or None
    return None


def ownership(steps: list[dict]) -> list[str]:
    """每条记录**观察时**的归属（spec §3.1 的事件表）。动作发生在这一屏上，改变的是之后的归属。

    进 App 有两种信号：成功的 open_app；或者在主屏上点开一个图标（`_icon_tap_label`）——
    后者是真机常见的开 App 方式，工具描述里也明确写了「主屏上开 App 用 icon_above」。
    """
    owner = SYSTEM
    out = []
    for st in steps:
        out.append(owner)
        name, args, res = _action_of(st)
        if name == "open_app":
            if res.get("ok") and res.get("changed"):
                owner = app_id_for(str(args.get("name", "")))
        elif name == "key":
            k = args.get("name")
            if k in ("home", "spotlight"):
                owner = SYSTEM
            elif k == "app_switcher":
                owner = UNKNOWN
        else:
            label = _icon_tap_label(st, owner, name, args, res)
            if label is not None:
                owner = app_id_for(label)
    return out


def apps_seen(steps: list[dict]) -> dict[str, str]:
    """跟 ownership 同一套信号，走的是同一张归属表——不重新扫一遍「现在在哪」。"""
    seen: dict[str, str] = {}
    for st, owner in zip(steps, ownership(steps), strict=True):
        name, args, res = _action_of(st)
        if name == "open_app" and res.get("ok") and res.get("changed"):
            n = str(args.get("name", ""))
            seen.setdefault(app_id_for(n), n)
        else:
            label = _icon_tap_label(st, owner, name, args, res)
            if label is not None:
                seen.setdefault(app_id_for(label), label)
    return seen


# 认屏时指纹要被当前屏覆盖到多少。**不能要求 100%** —— 指纹里的词只要有一个
# 被 OCR 读得略有出入（'＜设置' / '<设置'），整条就废。
#
# 2026-09-08 用 29 个 run 的 270 次观察量的：
#   阈值 1.00 → 认对 270 认错 0；留一法（指纹用前 n-1 次算、测第 n 次）能认出 21/24
#   阈值 0.90 → 认对 269 认错 1；留一法 22/24
#   阈值 0.75 → 认对 263 认错 7
# 取 0.9：**误认比认不出更糟** —— 误认会把模型指到错的地方，认不出只是没帮上忙 ——
# 所以偏严，但要能容忍一次 OCR 抖动。
COVERAGE = 0.9


@dataclass
class Node:
    """一屏。"""
    key: int
    fingerprint: set[str] = field(default_factory=set)
    visits: int = 0
    samples: list[str] = field(default_factory=list)      # 出处：run/step
    owners: Counter = field(default_factory=Counter)       # 这一屏各次访问时的归属，多数者为准
    # 首次访问时那一屏的原始文字集合，**认屏**（jaccard）用它。
    # 不能用 fingerprint 认屏：指纹会被交集越削越小，越削越容易跟别的屏撞上。
    representative: set[str] = field(default_factory=set)

    def coverage(self, texts: set[str]) -> float:
        """指纹有多大比例出现在当前屏上。"""
        if not self.fingerprint:
            return 0.0
        return len(self.fingerprint & texts) / len(self.fingerprint)

    def matches(self, texts: set[str]) -> bool:
        return self.coverage(texts) >= COVERAGE


@dataclass
class Edge:
    """从一屏到另一屏的一个动作。**只存语义目标，不存坐标。**"""
    src: int
    dst: int
    action: str
    target: str | None
    ok: bool
    count: int = 1


def _texts(step: dict) -> set[str]:
    els = (step.get("observation") or {}).get("elements") or []
    return {e["text"].strip() for e in els if e.get("text", "").strip()}


def _semantic_target(action: dict, id_to_text: dict[int, str] | None = None) -> str | None:
    """动作里可以长期复用的那部分：**文字**、App 名、方向。坐标和元素编号一律丢掉。

    ⚠ 元素编号（id）只在**当次观察**里有效，跨运行毫无意义 —— 所以 tap 必须用
    动作发生时那一屏的元素表把 id 还原成文字。还原不出来就返回 None，
    宁可这条边没有目标，也不要存一个下次会指向别的东西的数字。
    """
    a = action.get("args") or action.get("args_raw") or {}
    name = action.get("name")
    if name == "tap":
        eid = a.get("id")
        text = (id_to_text or {}).get(eid) if eid is not None else None
        if text is None:
            return None
        t = a.get("target")
        return f"{text}@{t}" if t and t != "text" else text
    if name in ("scroll", "scroll_until", "collect"):
        return " ".join(str(a.get(k)) for k in ("direction", "text", "amount") if a.get(k))
    if name in ("open_app", "key", "type"):
        return str(a.get("name") or a.get("text") or "")
    return None


def _jaccard(a: set[str], b: set[str]) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


class ScreenMap:
    """节点 + 边。从 run 日志喂进来，越用越全。"""

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.edges: dict[tuple[int, int, str, str | None], Edge] = {}
        self.runs: set[str] = set()
        self.app_names: dict[str, str] = {}       # App id → open_app 原文，给自动建 APP.md 用
        # run_id → 那个 run 的 steps.jsonl 的 mtime。喂过的就不再喂第二遍。
        self.ingested: dict[str, float] = {}

    # ---- 建图 ----
    def _place(self, texts: set[str], where: str, owner: str = SYSTEM) -> int:
        best, best_s = None, 0.0
        for n in self.nodes:
            s = _jaccard(texts, n.representative)
            if s > best_s:
                best, best_s = n, s
        if best is not None and best_s >= JACCARD:
            best.visits += 1
            if len(best.samples) < 5:
                best.samples.append(where)
            # 增量交集：和「把历史上所有集合重新求交」结果完全相同，但不必留着它们
            # （docs/20 C2/C4：留着就是单节点 O(v²) 的重复求交 + 无界内存）。
            best.fingerprint &= texts
            best.owners[owner] += 1
            return best.key
        node = Node(key=len(self.nodes), fingerprint=set(texts), visits=1,
                    samples=[where], representative=set(texts))
        node.owners[owner] += 1
        self.nodes.append(node)
        return node.key

    def ingest_run(self, steps: Iterable[dict], run_id: str) -> None:
        steps = list(steps)
        self.runs.add(run_id)
        self.app_names.update({k: v for k, v in apps_seen(steps).items() if k not in self.app_names})
        owners = ownership(steps)
        prev_key: int | None = None
        prev_action: dict | None = None
        prev_ids: dict[int, str] = {}
        for st, owner in zip(steps, owners, strict=True):
            texts = _texts(st)
            if not texts:
                continue
            key = self._place(texts, f"{run_id}#{st.get('step')}", owner)
            if prev_key is not None and prev_action:
                self._add_edge(prev_key, key, prev_action, prev_ids,
                               ok=bool((st.get("result") or {}).get("ok", True)))
            prev_key, prev_action = key, st.get("action") or None
            # 动作发生在**这一屏**上，所以 id→文字的对照表要用这一步的观察
            prev_ids = {e["id"]: e["text"].strip()
                        for e in ((st.get("observation") or {}).get("elements") or [])
                        if e.get("text", "").strip()}

    def _add_edge(self, src: int, dst: int, action: dict,
                  id_to_text: dict[int, str], ok: bool) -> None:
        name = action.get("name") or "?"
        tgt = _semantic_target(action, id_to_text)
        k = (src, dst, name, tgt)
        if k in self.edges:
            self.edges[k].count += 1
        else:
            self.edges[k] = Edge(src, dst, name, tgt, ok)

    # ---- 用图 ----
    def identify(self, texts: set[str]) -> Node | None:
        """当前这屏是图上的哪个节点？多个命中时取指纹最长的（最具体）。"""
        hits = [n for n in self.nodes if n.visits >= MIN_VISITS and n.matches(texts)]
        # 多个命中时取**重合词最多**的那个，不是指纹最长的 —— 长指纹只说明那屏内容多。
        return max(hits, key=lambda n: len(n.fingerprint & texts)) if hits else None

    def out_edges(self, key: int) -> list[Edge]:
        return [e for e in self.edges.values() if e.src == key]

    def stable_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.visits >= MIN_VISITS and n.fingerprint]

    def node_app(self, key: int) -> str | None:
        n = next((x for x in self.nodes if x.key == key), None)
        if n is None or not n.owners:
            return None
        return n.owners.most_common(1)[0][0]

    def subgraph(self, app: str) -> ScreenMap:
        """只含归属这个 App 的稳定节点，只保留两端都在的边。键不重排，whereami 按 key 找节点。"""
        sub = ScreenMap()
        keep = {n.key for n in self.stable_nodes() if self.node_app(n.key) == app}
        sub.nodes = [n for n in self.nodes if n.key in keep]
        sub.edges = {k: e for k, e in self.edges.items() if e.src in keep and e.dst in keep}
        sub.runs = set(self.runs)
        sub.app_names = {app: self.app_names.get(app, app)}
        return sub

    def to_json(self) -> str:
        """导出**全部**节点，不只是稳定的那些。

        ⚠ 这是缓存，不是报表。只导出 stable 的话，visits==1 的节点重载后就没了，
        下次再见到同一屏又是一个新的 visits==1 —— 它永远到不了 2，指纹永远长不出来。
        谁算「稳定」由 `stable_nodes()` 在读的时候判。
        """
        return json.dumps({
            # version 2 = 加了 owners / runs / app_names（归属那一层）。
            # ⚠ 故意不兼容 version 1：老缓存里没有归属，读进来每个节点的 owners 都是空的，
            #   subgraph() 会当场返回空图 —— 而 mtime 没变就不会重喂，这个空会一直空下去。
            #   宁可整张图重建一次（几秒），也不要一个安静失效的 App 子图。
            "version": 2,
            "ingested": self.ingested,
            "runs": sorted(self.runs),
            "app_names": self.app_names,
            "nodes": [{"key": n.key, "visits": n.visits,
                       "fingerprint": sorted(n.fingerprint),
                       "representative": sorted(n.representative),
                       # Counter 不是 JSON 原生类型，落盘成普通 dict[str, int]，读回来再包成 Counter。
                       "owners": dict(n.owners),
                       "samples": n.samples}
                      for n in self.nodes],
            "edges": [{"src": e.src, "dst": e.dst, "action": e.action,
                       "target": e.target, "ok": e.ok, "count": e.count}
                      for e in self.edges.values()],
        }, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, s: str) -> ScreenMap:
        """`to_json` 的逆。格式不对就抛，让调用方决定是不是从头建。"""
        d = json.loads(s)
        if d.get("version") != 2:
            raise ValueError("screenmap cache version mismatch")
        m = cls()
        for n in d["nodes"]:
            m.nodes.append(Node(key=n["key"], fingerprint=set(n["fingerprint"]),
                                visits=n["visits"], samples=list(n.get("samples", [])),
                                owners=Counter({str(k): int(v) for k, v in (n.get("owners") or {}).items()}),
                                representative=set(n["representative"])))
        for e in d["edges"]:
            m.edges[(e["src"], e["dst"], e["action"], e["target"])] = Edge(
                e["src"], e["dst"], e["action"], e["target"], e["ok"], e["count"])
        m.ingested = {k: float(v) for k, v in d.get("ingested", {}).items()}
        m.runs = set(d.get("runs", []))
        m.app_names = dict(d.get("app_names", {}))
        return m

def appmap_json(sub: ScreenMap, app: str, built_from: list[str]) -> str:
    """apps/<id>/map.json 的格式（spec §2.2）。与 to_json 不同：带 schema_version/app/built_from，边拆出 how，无悬空边。"""
    keys = {n.key for n in sub.nodes}
    edges = []
    for e in sub.edges.values():
        if e.src not in keys or e.dst not in keys:
            continue
        target, how = (e.target.split("@", 1) + ["text"])[:2] if e.target and "@" in e.target else (e.target, "text")
        edges.append({"src": e.src, "dst": e.dst, "action": e.action, "target": target, "how": how,
                      "ok": e.ok, "count": e.count})
    return json.dumps({
        "schema_version": 1, "app": app, "built_from": sorted(built_from),
        "nodes": [{"key": n.key, "visits": n.visits, "fingerprint": sorted(n.fingerprint), "samples": n.samples}
                  for n in sub.nodes],
        "edges": edges,
    }, ensure_ascii=False, indent=1) + "\n"


def _read_steps(f: Path) -> list[dict] | None:
    """读一个 steps.jsonl。坏行跳过；整个文件读不了返回 None。"""
    steps: list[dict] = []
    try:
        # ⚠ 必须 with —— 第一版写的 `for line in f.open(...)`，句柄一直不关，
        #   runs/ 一多就漏文件描述符（pytest 的 ResourceWarning 抓到的）。
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return None
    return steps


def _run_finished(run_dir: Path) -> bool:
    """这个 run 跑完了没有？

    ⚠ 没跑完的 run 的 steps.jsonl 还在被追加。要是把它记进 ingested，
    下次 mtime 一变就会**整个重喂一遍**，节点的 visits 全部重复计数。
    所以：只有 run.json 里有 end_reason 的才记，没结束的每次都重喂（图本身仍然对）。
    """
    try:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(isinstance(meta, dict) and meta.get("end_reason"))


def build_from_runs(runs_root: Path, cache: Path | None = None) -> ScreenMap:
    """把 runs/ 下所有 steps.jsonl 喂进去。坏文件跳过，不让一个坏 run 顶掉整张图。

    给了 `cache` 就先从它加载，只喂 mtime 变过（或没喂过）的 run，喂完写回。
    这条路挡在任务启动的关键路径上（docs/20 C1），runs/ 一多全量重建就是纯浪费。
    `cache=None` 时不读也不写缓存，全量重建，不碰磁盘——但喂入顺序跟有缓存时一样：
    没结束的 run 仍然放在最后喂（见 `_run_finished`），不是「跟以前完全一样」。
    """
    m = ScreenMap()
    if cache is not None and cache.exists():
        try:
            m = ScreenMap.from_json(cache.read_text(encoding="utf-8"))
        except Exception:      # noqa: BLE001 —— 坏缓存当没有，从头建；它只是加速，不是真相
            m = ScreenMap()
    changed = False
    in_flight: list[tuple[str, list[dict]]] = []
    for f in sorted(Path(runs_root).glob("*/steps.jsonl")):
        run_id = f.parent.name
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        if m.ingested.get(run_id) == mtime:
            continue
        steps = _read_steps(f)
        if steps is None:
            continue
        if not _run_finished(f.parent):
            # 还在跑：这一趟照喂（图该是最新的），但**不进缓存也不记 ingested**，
            # 留到它结束以后再落盘。
            in_flight.append((run_id, steps))
            continue
        if steps:
            m.ingest_run(steps, run_id)
        m.ingested[run_id] = mtime
        changed = True
    if cache is not None and changed:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            # 先写 .tmp 再 replace：中途挂掉也不会留下半张读不出来的图。
            # 文件名带 pid：同一进程里跑两个 workspace、或两个进程共享同一 cache 路径时，
            # 两边的 .tmp 不会互相覆盖对方还没 replace 完的那一份（Minor 5）。
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
            tmp.write_text(m.to_json(), encoding="utf-8")
            tmp.replace(cache)
        except OSError:
            pass               # 写不了缓存不影响本次使用
    # 落盘之后才喂没跑完的 run —— 顺序就是这里唯一在做的事
    for run_id, steps in in_flight:
        if steps:
            m.ingest_run(steps, run_id)
    return m


def find_nodes(m: ScreenMap, want: str) -> list[Node]:
    """指纹里含某段文字的节点，按访问次数从多到少。"""
    hit = [n for n in m.stable_nodes() if any(want in t for t in n.fingerprint)]
    return sorted(hit, key=lambda n: -n.visits)


def route(m: ScreenMap, src: int, dst: int, max_hops: int = 4) -> list[Edge] | None:
    """从 src 到 dst 的一条路。广度优先，所以给出的是**最短**的那条。

    ⚠ 只走**有语义目标**的边。没有目标的边（tap 的 id 还原不出文字、
    done 之类）没法复述给模型，路上有这么一步就等于断了。

    ⚠ 同一对节点之间可能有多条边，取走过次数最多的那条 —— 走得多的更可靠。
    """
    if src == dst:
        return []
    by_src: dict[int, list[Edge]] = {}
    for e in m.edges.values():
        if e.target and e.src != e.dst:
            by_src.setdefault(e.src, []).append(e)
    for v in by_src.values():
        v.sort(key=lambda e: -e.count)

    seen = {src}
    frontier: list[tuple[int, list[Edge]]] = [(src, [])]
    for _ in range(max_hops):
        nxt = []
        for node, path in frontier:
            for e in by_src.get(node, []):
                if e.dst in seen:
                    continue
                if e.dst == dst:
                    return [*path, e]
                seen.add(e.dst)
                nxt.append((e.dst, [*path, e]))
        if not nxt:
            return None
        frontier = nxt
    return None
