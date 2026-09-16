"""合成留档：给孪生的记账 / 重建 / 渲染 / 闸门测试共用。坐标按真机图像 624x1388。"""
import json
from pathlib import Path

from iphone_agent.twin.screenfile import screen_id

W, H = 624, 1388


def el(text, fx, fy, eid, *, source="ocr", conf=0.9):
    cx, cy = fx * W, fy * H
    return {"id": eid, "text": text, "confidence": conf, "box": [cx - 20, cy - 10, cx + 20, cy + 10],
            "center": [cx, cy], "source": source}


SET = "she-zhi"


def label(app, name, anchors=(), same_as=None):
    return {"app": app, "name": name, "same_as": same_as, "anchors": list(anchors)}


def cand(index, app_id, sid, name, display=None):
    return {"index": index, "app_id": app_id, "app_display": display or app_id, "screen_id": sid, "name": name}


def sid_of(app, run_id, frame):
    """按 spec §4.1 算出某个 run 在某帧建出的屏 id：测试里的候选就是这样静态写出来的。"""
    return screen_id(app, f"{run_id}:{frame}")


def screen_obs(frame, title=None, back=None, rows=("甲", "乙", "丙"), extra=(), *,
               app="设置", same_as=None, candidates=(), labeled=True):
    """一屏 App 页：左上返回、正中标题、竖排正文行，外加视觉标注（锚点 = 标题 + 第一行）。"""
    els, i = [], 1
    if back:
        els.append(el(f"<{back}", 0.1, 0.13, i))
        i += 1
    if title:
        els.append(el(title, 0.5, 0.13, i))
        i += 1
    for k, t in enumerate(rows):
        els.append(el(t, 0.3, 0.3 + 0.06 * k, i))
        i += 1
    for e in extra:
        els.append(dict(e, id=i))
        i += 1
    obs = {"frame_file": frame, "width_px": W, "height_px": H, "elements": els}
    if labeled and title:
        obs["screen"] = label(app, title, [title, *rows[:1]], same_as)
        obs["perception"] = {"ocr": len(els), "vision": "ok", "screen": "ok"}
        if candidates:
            obs["screen_candidates"] = list(candidates)
    return obs


def home_obs(frame, labels=("设置", "备忘录", "日历")):
    return {"frame_file": frame, "width_px": W, "height_px": H,
            "elements": [el(t, 0.2 + 0.2 * k, 0.3, k + 1) for k, t in enumerate(labels)],
            "screen": label("系统", "主屏幕", labels[:2]),
            "perception": {"ocr": len(labels), "vision": "ok", "screen": "ok"}}


def id_of(obs, text):
    return next(e["id"] for e in obs["elements"] if e["text"] == text)


def step(n, obs, name, args=None, result=None, after=None, **extra):
    r = {"step": n, "ts": 1_000_000.0 + n, "observation": obs,
         "action": {"name": name, "args_raw": dict(args or {}), "args": dict(args or {})},
         "result": result if result is not None else {"ok": True, "changed": True}}
    if after:
        r["after_frame_file"] = after
    r.update(extra)
    return r


VERIFIED = {"ok": True, "changed": True, "identity": {"verified": True}}


def settings_run(t0=0.0, first=None):
    """主屏 → open_app 设置（核过身份）→ 点「通用」→ 点「关于本机」→ done。
    first = 这几屏最初被建出来的那个 run：给了就让每屏带上指向它的候选、same_as=1（「以前来过」）。"""
    def page(frame, title, back, rows):
        if first is None:
            return screen_obs(frame, title, back, rows)
        c = (cand(1, SET, sid_of(SET, first, frame), title, "设置"),)
        return screen_obs(frame, title, back, rows, same_as=1, candidates=c)
    f1 = home_obs("f1.png")
    f2 = page("f2.png", "设置", None, ("通用", "隐私", "电池"))
    f3 = page("f3.png", "通用", "设置", ("关于本机", "软件更新", "存储"))
    f4 = page("f4.png", "关于本机", "通用", ("iOS版本", "型号", "容量"))
    recs = [step(1, f1, "open_app", {"name": "设置"}, VERIFIED, after="f2.png"),
            step(2, f2, "tap", {"id": id_of(f2, "通用")}, after="f3.png"),
            step(3, f3, "tap", {"id": id_of(f3, "关于本机")}, after="f4.png"),
            step(4, f4, "done", {"status": "success", "result": "18.3.1"})]
    for r in recs:
        r["ts"] += t0
    return recs


def write_run(root: Path, run_id: str, records, finished=True) -> Path:
    d = Path(root) / "runs" / run_id
    d.mkdir(parents=True)
    (d / "steps.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                                   encoding="utf-8")
    (d / "run.json").write_text(json.dumps({"end_reason": "done_success" if finished else None}), encoding="utf-8")
    return d


def tree_bytes(apps_dir: Path) -> dict[str, bytes]:
    return {str(p.relative_to(apps_dir)): p.read_bytes() for p in sorted(Path(apps_dir).rglob("*.json"))}
