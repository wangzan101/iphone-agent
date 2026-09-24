"""从运行日志里提剧本草稿；按步骤 hash 合并；sync。

只从校验通过的运行里长（docs/06 的硬约束）：终态成功、审计没标可疑、没调过剧本、
路径上每个动作都真的成功、长度合理。任何一条不满足就不提 —— 宁可少一条草稿，
也不要一条自信的错误路径进库。
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

from iphone_agent import config
from iphone_agent import fingerprint as F
from iphone_agent.appids import resolve_app_id
from iphone_agent.memory.screenmap import (
    SYSTEM,
    UNKNOWN,
    appmap_json,
    apps_seen,
    build_from_runs,
    label_like,
    ownership,
)
from iphone_agent.skills import model as M
from iphone_agent.skills.store import Catalog, SkillStore

SKIP = {"observe", "wait", "recall", "recall_runs", "use_skill"}
# ⚠ 手抄自 harness/actions.py 的 NON_COUNTING（skills 不许 import harness，没法共享一份）。
# 目前两边一致；那边加/删元动作时记得回来同步，否则这道「任一动作失败就整条不提」
# 的闸门会悄悄漏掉新的元动作类型。
EXPECT_WORDS = 8      # 每步 expect 最多取几个词
WEAK_EXPECT = F.WEAK_EXPECT   # 锚点少于这么多的步骤标「指纹弱」；定义只有 fingerprint 一份


def records_of(run_dir: Path) -> list[dict]:
    """顶层、带观察的记录。procedure_step 子记录不参与提取（调过剧本的运行本来就不提）。
    自己读 steps.jsonl 而不用 RunLog.read_steps：skills 不许 import harness。"""
    p = Path(run_dir) / "steps.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        if s.get("kind") != "procedure_step" and isinstance(s.get("observation"), dict):
            out.append(s)
    return out


def _expect_of(record: dict | None) -> tuple[str, ...]:
    """这一步走完之后「该到的屏」长什么样。

    ⚠ 2026-09-16：进指纹的只能是**稳定证据**（fingerprint.stable_texts：OCR 那一路、状态栏以下）。
      原来这里照单全收，于是状态栏时钟（`今 ④5`）和整屏解析给图标起的名字（`电池图标（绿色电池）`）
      都进了指纹，而 on_demand 下它们一个都不会再出现 —— 那之前学到的剧本全废在第一步。
    """
    if not record:
        return ()
    o = record.get("observation") or {}
    out: list[str] = []
    for t in F.stable_texts(o.get("elements"), int(o.get("height_px") or 0)):
        if label_like(t):
            out.append(t)
        if len(out) >= EXPECT_WORDS:
            break
    return tuple(out)


def _tap_effect(res: dict) -> str:
    """这次 tap 在源运行里**做成了什么**（M.EFFECTS）。只认肯定信号，其余一律 unknown。

    - `local_mad` 在结果里 = 整屏判据说没变、只有点击处那一小块变了 —— 开关翻转那一类
      （executor._after 的 ⚠ 注释：开关翻两次，汉明和文本差都是 0，局部 MAD 17.0），不是导航；
    - 有新文字出现（`appeared`，transition 的集合差，一个入口）才算换了一屏。
    没有 appeared 的成功点击（纯图标页、OCR 没读出新字、看图复核翻过来的 changed）都判不了，
    判不了就是 unknown，风险按写算。孪生那份 outcome_of 更准，但它要认屏结果，提取这里没有。"""
    if "local_mad" in res:
        return "state_changed"
    return M.NAVIGATED if res.get("appeared") else "unknown"


def _id_text(record: dict, eid) -> str | None:
    for e in (record.get("observation") or {}).get("elements") or []:
        if e.get("id") == eid:
            t = (e.get("text") or "").strip()
            return t or None
    return None


def candidate_steps(records: list[dict]) -> tuple[str, list[M.Step]] | None:
    """最后一次进 App 到 done 之间的动作 → 步骤。任一动作不满足 spec §3.2 第 4 条 → None。"""
    owners = ownership(records)
    # 「进了一个 App」只在 ownership() 里定义一次:这里只找 owners 从 system/unknown
    # 变成某个具体 App 的最后一次转折,不重新判定"这算不算进 App"——这正是上一版的 bug
    # (一个规则两个入口,只改了一个:ownership() 学会了认图标点击,这里的字面量扫描
    # 却没跟着改,结果 App 认对了、起点还是没跟上)。
    start = None
    for i in range(len(records) - 1):
        if owners[i] in (SYSTEM, UNKNOWN) and owners[i + 1] not in (SYSTEM, UNKNOWN):
            start = i
    if start is None:
        return None
    app = owners[start + 1]
    steps: list[M.Step] = []
    pending_find: dict | None = None
    for i in range(start, len(records)):
        st = records[i]
        a = st.get("action") or {}
        name = a.get("name")
        args = a.get("args") or a.get("args_raw") or {}
        res = st.get("result") or {}
        nxt = records[i + 1] if i + 1 < len(records) else None
        if not name:
            continue                      # 没有动作的记录（纯观察），不是第 4 条要管的对象
        # ⚠ 顺序即不变量：任何带动作的记录，先判「这一步是否真的成功」——
        # 包括 SKIP 类元动作（recall/use_skill/... 真的会 ok=False）和 done 本身——
        # 再谈要不要 break/continue。不能有任何短路排在这条检查前面，
        # 否则一个被拒绝/失败的动作会被悄悄放过而不是让整条草稿作废（review 发现 1）。
        if st.get("validation") or not res.get("ok"):
            return None
        if name == "done":
            if pending_find:
                return None              # 悬空的滚动没被 tap 折进去，done 也救不了（review 发现 2）
            break
        if name in SKIP:
            continue
        expect = _expect_of(nxt)
        if i == start:
            # 剧本是给未来执行的，进入方式要用最稳的那条，而不是历史上碰巧用的那条：
            # open_app 走 Spotlight，不挑主屏翻到了哪一页；历史上的真实动作可能只是
            # 一次图标点击，重放时手机不一定还停在同一页。所以不管当年是 open_app
            # 还是图标点击，第一步一律归一化成 open_app，只有 expect 仍取当年这一步的那份。
            display = apps_seen(records).get(app)
            if display is None and name == "tap":
                display = _id_text(st, args.get("id")) if args.get("id") is not None else None
            if display is None:
                return None          # 两条来源都拿不到名字，宁可不提，也不编一个 App 名字
            steps.append(M.Step(do="open_app", target=display, expect=expect))
            pending_find = None
            continue
        if name == "scroll":
            d = args.get("direction")
            if d not in ("up", "down"):
                return None
            if pending_find and pending_find["direction"] == d:
                pending_find["max_screens"] += 1
            else:
                pending_find = {"direction": d, "max_screens": 1}
            continue
        if name == "tap":
            if not res.get("changed"):
                return None
            text = _id_text(st, args.get("id")) if args.get("id") is not None else None
            if text is None:
                return None          # 坐标点击还原不出文字，不进长期知识
            steps.append(M.Step(do="tap", target=text, how=args.get("target") or "text",
                                find=pending_find, expect=expect, effect=_tap_effect(res)))
            pending_find = None
            continue
        if pending_find:
            return None              # 滚动后面不是 tap，折不进去
        if name == "type":
            if res.get("verified") != "true" and not res.get("picked"):
                return None
            steps.append(M.Step(do="type", text=str(args.get("text", "")), expect=expect))
        elif name == "scroll_until":
            if not res.get("found") or args.get("direction") not in ("up", "down", "left", "right"):
                return None
            steps.append(M.Step(do="scroll_until", direction=args["direction"], target=str(args.get("text", "")),
                                expect=expect))
        elif name == "open_app":
            if not res.get("changed"):
                return None
            steps.append(M.Step(do="open_app", target=str(args.get("name", "")), expect=expect))
        else:
            return None
    if not (2 <= len(steps) <= config.PROC_MAX_ACTIONS):
        return None
    return app, steps


def draft_procedure(run_dir: Path, catalog: Catalog, records: list[dict] | None = None) -> M.Procedure | None:
    try:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    records = records if records is not None else records_of(run_dir)
    cand = candidate_steps(records)
    if cand is None:
        return None
    app, steps = cand
    task = str(meta.get("task") or "")
    h = M.step_hash(steps)
    macos = ((meta.get("config") or {}).get("platform") or {}).get("macos")
    return M.Procedure(
        name=M.auto_name(task or "proc", h), app=app, description=task or "（无任务文本）",
        params={}, returns=[], risk=M.procedure_risk(steps, catalog.app(app)),
        status="draft", expect_source="single_run",
        provenance=M.Provenance(runs=[run_dir.name], verified_count=1, last_ok=date.today().isoformat(),
                                macos=macos),
        # ⚠ 2026-09-16：指纹弱原来只在合并时才标 —— 单次运行提出来的草稿哪怕一步只剩两个锚点，
        #   盘上也写着「不弱」，CLI 跟着说假话。过滤之后锚点够不够，建的时候就知道。
        steps=steps, weak_steps=M.weak_of(steps))


def _one_line(s: str, fallback: str) -> str:
    """display / open 直接来自模型给的 open_app name（actions 只卡 50 字，不卡换行），
    换行能在 APP.md 里伪造 status: verified —— 人批准 App 是自动验证的前置闸门（spec §4.3），
    伪造掉它，提取出来的草稿就全都自动进工具列表了。

    这里**取首行**而不是拒绝：拒绝意味着这次提取一个 App 都不建、还什么都不说，
    人只会看见「怎么什么都没提取出来」。首行就是真正的 App 名，后面几行是负载。"""
    head = s.splitlines()[0].strip() if s else ""
    return head if head and not head.startswith("---") else fallback


def ensure_app(store: SkillStore, catalog: Catalog, app_id: str, display: str) -> M.AppProfile:
    """自动切出一个 App 的空壳。

    ⚠ 2026-09-17：id 在这里再过一次 `resolve_app_id`。调用方给的可能是旧拼音 id
    （老留档里记着 `she-zhi`），也可能是英文机上的显示名 —— 两边都得落到同一个 `settings`，
    否则同一个 App 会在盘上切出两份知识，而且谁也不知道另一份在哪（appids.py 的模块注释）。
    自动切出来的 App 把**当时那个显示名**记进 aliases：下次它以另一种叫法出现时认得回来。"""
    app_id = resolve_app_id(app_id, catalog)
    a = catalog.app(app_id)
    if a is not None:
        return a
    display = _one_line(display, app_id)
    aliases = (display,) if display and display != app_id else ()
    a = M.AppProfile(name=app_id, display=display, open=display, risk="write", status="draft",
                     updated=date.today().isoformat(), body="", aliases=aliases)
    store.write_app(a)
    catalog.apps[app_id] = a
    catalog.procedures.setdefault(app_id, {})
    catalog.origin[f"app:{app_id}"] = "personal"
    return a


def _merge_effect(old: str | None, new: str | None) -> str | None:
    """两次运行对同一步的效果记录。None = 这一栏根本没记过（2026-09-16 之前提的老草稿），
    不是「记成了不导航」—— 所以拿新的来补。两边都记了又不一致时取更保守的那个：
    只有两次都说 navigated 才算 navigated。"""
    if old is None:
        return new
    if new is None or old == new:
        return old
    return "unknown" if M.NAVIGATED in (old, new) else old


def _merge_expect(old: M.Procedure, new: M.Procedure) -> None:
    """指纹取交集；顺带把新运行记下的效果合进老步骤（风险由调用方按合并后的步骤重算）。

    ⚠ 2026-09-16：新那一份是过过滤的（只剩稳定 OCR 行），所以取交集**顺带把老文件里的
      状态栏乱码和视觉图标名冲掉** —— 老剧本只要再被同一条路径提取一次，指纹就自愈了。
      交集太少时保留旧 expect（那是仅有的一份），但这一步要标「指纹弱」。"""
    weak: set[int] = set()
    merged: list[M.Step] = []
    for k, (o, n) in enumerate(zip(old.steps, new.steps, strict=True), start=1):
        o = replace(o, effect=_merge_effect(o.effect, n.effect))
        if o.do == "read":
            merged.append(o)
            continue
        inter = tuple(w for w in o.expect if w in set(n.expect))
        if len(inter) < WEAK_EXPECT:
            weak.add(k)              # 两次运行对不上这么多词：这一步的指纹本身就不可靠
            merged.append(o)
        else:
            merged.append(replace(o, expect=inter))
    old.steps = merged
    old.weak_steps = sorted(weak | set(M.weak_of(merged)))
    old.expect_source = "single_run" if old.weak_steps else "intersection"


def _queue_pending(store: SkillStore, draft: M.Procedure) -> str:
    """canonical 文件名被别的 hash 占了：排入 pending，绝不覆盖已经排队的、内容不同的草稿。
    自动名只取 hash 前 4 位，撞名不算罕见；先看 pending 里占着这个名字的是不是同一条
    （是就当已经排过队，no-op），不是就把 hash 后缀拉长直到不再冲突。"""
    h = draft.hash()
    name = draft.name
    for n in (4, 6, 8, 10, 12):
        if n != 4:
            name = M.auto_name(draft.description, h, n)
        path = store.proc_path(draft.app, name, pending=True)
        if not path.exists():
            store.write_procedure(replace(draft, name=name), pending=True)
            return "conflict"
        try:
            existing = M.procedure_from_json(path.read_text(encoding="utf-8"), draft.app)
        except (OSError, UnicodeDecodeError, M.SkillError):
            existing = None
        if existing is not None and existing.hash() == h:
            return "conflict"          # 同一条草稿已经排过队了，no-op
    raise M.SkillError("pending_collision",
                       f"{draft.app} 的 pending 目录里名字冲突，hash 后缀拉到最长（12 位）仍旧撞车：{draft.name}")


def ingest_draft(store: SkillStore, catalog: Catalog, draft: M.Procedure, run_id: str) -> str:
    same = next((p for p in catalog.procedures_of(draft.app) if p.hash() == draft.hash()), None)
    if same is None:
        if store.skill_dir(draft.name).exists():        # 名字被占（哪怕是人写的无剧本技能）就排队
            return _queue_pending(store, draft)
        store.write_procedure(draft)
        catalog.procedures.setdefault(draft.app, {})[draft.name] = draft
        return "new"
    if run_id in same.provenance.runs:
        return "seen"
    same.provenance.runs.append(run_id)
    same.provenance.verified_count = len(same.provenance.runs)
    same.provenance.last_ok = date.today().isoformat()
    _merge_expect(same, draft)
    status = "merged"
    if same.status == "draft":
        # 草稿期间风险每次都按**当前这份步骤**重算（合并可能补上了效果记录）；verified 之后不再动。
        same.risk = M.procedure_risk(same.steps, catalog.app(same.app))
    if same.status == "draft" and same.provenance.verified_count >= 2 and same.risk == "read":
        same.status = "verified"
        status = "verified"
    store.write_procedure(same, overwrite=True)
    return status


def extract_from_run(store: SkillStore, run_dir: Path, *, end_reason: str, audit_miss: list,
                     procedures_used: list, catalog: Catalog | None = None) -> list[dict]:
    """任务结束时调用（loop 的 finally，在审计之后）。spec §3.2 条件 1、2 由参数给出。"""
    if end_reason != "done_success" or audit_miss or procedures_used:
        return []
    catalog = catalog or store.load()
    records = records_of(run_dir)
    for app_id, display in apps_seen(records).items():
        ensure_app(store, catalog, app_id, display)
    draft = draft_procedure(run_dir, catalog, records)
    if draft is None:
        return []
    status = ingest_draft(store, catalog, draft, run_dir.name)
    return [{"app": draft.app, "name": draft.name, "status": status}]


def sync(store: SkillStore, runs_root: Path) -> dict:
    """全量重扫：重建每个 App 的 map.json、补建 APP.md、重跑提取、重算草稿风险。幂等。"""
    catalog = store.load()
    m = build_from_runs(runs_root)
    for app_id, display in m.app_names.items():
        ensure_app(store, catalog, app_id, display)
    maps = 0
    for app_id in m.app_names:
        store.write_map(app_id, appmap_json(m.subgraph(app_id), app_id, sorted(m.runs)))
        maps += 1
    drafts = []
    for f in sorted(Path(runs_root).glob("*/run.json")):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("end_reason") != "done_success":
            continue
        if (meta.get("audit") or {}).get("never_reached"):
            continue
        if (meta.get("knowledge") or {}).get("procedures"):
            continue
        draft = draft_procedure(f.parent, catalog)
        if draft is None:
            continue
        drafts.append({"app": draft.app, "name": draft.name,
                       "status": ingest_draft(store, catalog, draft, f.parent.name)})
    # 草稿的风险每次 sync 都按**它自己的步骤**重算（M.procedure_risk），重算之后可能够格自动 verified。
    # ⚠ 2026-09-16：重算是唯一的降档路径 —— 盘上写着 write 的老草稿只有在它自己的步骤
    #   全都算得出 read 时才会变 read（老文件没有 effect 那一栏，算出来仍是 write，
    #   要等同一条路径再被提取一次、把效果补进来）。verified 的不碰。
    for app_id, procs in list(catalog.procedures.items()):
        for p in list(procs.values()):
            if p.status != "draft":
                continue
            r = M.procedure_risk(p.steps, catalog.app(app_id))
            changed = r != p.risk
            p.risk = r
            if p.provenance.verified_count >= 2 and r == "read":
                p.status = "verified"
                changed = True
            if changed:
                store.write_procedure(p, overwrite=True)
    # 场景风险只能往高走：声明值、引用剧本、正文关键词三者取最高（spec §2.4）
    for s in list(catalog.scenarios.values()):
        if catalog.origin.get(f"skill:{s.name}") != "personal":
            continue
        referenced = [p.risk for a in s.apps for p in catalog.procedures_of(a)]
        computed = M.scenario_risk(s.risk, referenced, s.body)
        if computed != s.risk:
            store.write_scenario(M.Scenario(**{**s.__dict__, "risk": computed}), overwrite=True)
    return {"apps": sorted(m.app_names), "maps": maps, "drafts": drafts}
