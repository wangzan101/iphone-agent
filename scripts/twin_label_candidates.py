"""为孪生闸门挑标注候选（spec §8.1）。输出 JSON 到 stdout，人（Claude）逐对看帧后填 label。

pairs：① 被认成同一屏的对（错合只可能出在这里，占大头）；② 正文很像但认成不同屏的对（难负例）；
      ③ 同名分叉：同一 App 里落到不同屏 id、但 Screen.name 相同的对（重名风险，spec §4.2）。
      每个 App 分层抽，优先跨 run。
owners：open_app、主屏点图标、app_switcher、home、handover 之后的第一个事件，加随机的 App 内事件。
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from iphone_agent.twin.events import read_run_events
from iphone_agent.twin.identify import ocr_texts
from iphone_agent.twin.record import finished_runs, simulate
from iphone_agent.workspace import Workspace

random.seed(20260911)
ws = Workspace(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())
sim = simulate(ws)
events, frame_of = {}, {}
for d in finished_runs(ws.runs):
    lines = (d / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    prev = None
    for ev in read_run_events(d):
        events[ev.event_id] = (ev, prev)
        n = int(ev.event_id.rsplit(":", 1)[1])
        frame_of[ev.event_id] = str(d / json.loads(lines[n - 1])["observation"]["frame_file"])
        prev = ev

by_screen, by_app = defaultdict(list), defaultdict(list)
for eid, (owner, _state, sid) in sim.by_event.items():
    if sid:
        by_screen[(owner, sid)].append(eid)
        by_app[owner].append(eid)

same = []
for (_app, _sid), eids in by_screen.items():
    cross = [(a, b) for a in eids for b in eids if a < b and a.split(":")[0] != b.split(":")[0]]
    same += random.sample(cross, min(2, len(cross)))
hard = []
for _app, eids in by_app.items():
    sample = random.sample(eids, min(40, len(eids)))
    feats = {e: ocr_texts(events[e][0].before.elements) for e in sample}
    for a in sample:
        for b in sample:
            if a < b and sim.by_event[a][2] != sim.by_event[b][2]:
                ta, tb = feats[a], feats[b]
                if ta and tb and len(ta & tb) / len(ta | tb) >= 0.5:
                    hard.append((a, b))
# ③ 同名分叉：同一 App 里，不同屏 id 却顶着同一个 Screen.name（重名风险，spec §4.2）。
same_name = []
for app, eids in by_app.items():
    by_sid: dict[str, list[str]] = defaultdict(list)
    for eid in eids:
        sid = sim.by_event[eid][2]
        if sid:
            by_sid[sid].append(eid)
    by_name: dict[str, list[str]] = defaultdict(list)
    for sid, _es in by_sid.items():
        scr = sim.state.get(app, sid)
        if scr is not None:
            by_name[scr.name].append(sid)
    for _name, sids in by_name.items():
        if len(sids) < 2:
            continue
        for i, sid_a in enumerate(sids):
            for sid_b in sids[i + 1:]:
                a = by_sid[sid_a][0]
                b = by_sid[sid_b][0]
                same_name.append((a, b) if a < b else (b, a))
pairs = (random.sample(same, min(55, len(same))) + random.sample(hard, min(30, len(hard)))
         + random.sample(same_name, min(15, len(same_name))))
pairs = list(dict.fromkeys(pairs))      # ① 和 ③ 可能抽到同一对：去重，不拿重复凑数

owners = []
for eid, (_ev, prev) in events.items():
    if prev is not None and prev.name in ("open_app", "key", "handover", "tap") and random.random() < 0.5:
        owners.append(eid)
inapp = [e for e, v in sim.by_event.items() if v[0] not in ("system", "unknown")]
owners = random.sample(owners, min(45, len(owners))) + random.sample(inapp, min(15, len(inapp)))

print(json.dumps({
    "pairs": [{"a": a, "b": b, "frame_a": frame_of[a], "frame_b": frame_of[b],
               "predicted": "same" if sim.by_event[a][2] == sim.by_event[b][2] else "different"} for a, b in pairs],
    "owners": [{"event": e, "frame": frame_of[e], "predicted": sim.by_event[e][0]} for e in owners],
}, ensure_ascii=False, indent=1))
