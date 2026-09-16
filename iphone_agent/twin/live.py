"""任务进行中的孪生（spec §6.1）：只读磁盘，内存里用与重放同一个 RunReplayer 边跑边记。

⚠ 这里不写盘。落盘只有 record.record_run 一个入口（任务结束后按留档重放）。
  实时新屏的 id 与重放一样由 `run:frame_file` 算出（spec §4.1），两边是同一个 id；
  `run:liveN` 只是快照没有帧文件名时的退路。
"""
from __future__ import annotations

from pathlib import Path

from iphone_agent.twin import render
from iphone_agent.twin.events import EVENT_ACTIONS, Event, Snapshot
from iphone_agent.twin.identify import Recognition
from iphone_agent.twin.record import RunReplayer, TwinState


class LiveTwin:
    def __init__(self, apps_dir: Path | None, run_id: str, known_apps: frozenset[str] = frozenset()):
        self.run_id = run_id
        self.state = TwinState(apps_dir, known_apps=known_apps)      # writable=False：flush 是空操作
        self.replayer = RunReplayer(self.state, run_id)
        self._n = 0
        self._cur: tuple | None = None
        self._seen: tuple[str, Recognition] | None = None

    def observe(self, snap: Snapshot) -> dict:
        owner, rec = self.replayer.recognize(snap)
        self._cur = (snap, owner, rec)
        self._seen = (owner, rec)
        return {"owner": owner, "state": rec.state, "screen_id": rec.screen_id, "name": rec.name}

    def after_action(self, action: dict, result: dict, after: Snapshot | None, ts: float) -> dict | None:
        if self._cur is None:
            return None
        snap, owner, rec = self._cur
        self._cur = None
        name = str(action.get("name") or "")
        if name not in EVENT_ACTIONS:
            # ⚠ 剧本：内部动作（可能开了别的 App）这里看不见，归属不再可信。重放能看到子记录，会记对。
            self.replayer.tracker.mark_unknown()
        else:
            self._n += 1
            # 事件 id 只在快照没有帧文件名时当首次出处的退路；主循环总会给帧文件名（spec §4.1）。
            ev = Event(f"{self.run_id}:live{self._n}", float(ts), snap,
                       {"name": name, "args": dict(action.get("args") or {})}, dict(result or {}), after,
                       f"{self.run_id}:live{self._n + 1}" if after is not None else None)
            self.replayer.apply(ev, owner, rec)
        if after is None:
            # ⚠ 没新画面的动作（recall/recall_runs/search_memory/use_skill，以及失败动作
            #   activate_failed/out_of_window/device_error）之后，下一个动作其实还是在
            #   这同一帧上做的：不重新认屏，_cur 就一直是 None，下一次 after_action
            #   会在最上面直接 return（_cur is None）——那个动作的归属和转移全被跳过，
            #   与重放不一致（重放里下一个事件的 before 帧正是这一帧，照样认一次）。
            #   这里原地对同一帧重新认屏、重新武装 _cur，行为与重放对齐（2026-09-11 评审）。
            return self.observe(snap)
        return None

    def position(self) -> str | None:
        if self._seen is None:
            return None
        owner, rec = self._seen
        return render.position(self.state, owner, rec, self.state.display(owner))

    def route(self, task: str) -> str | None:
        if self._seen is None:
            return None
        owner, rec = self._seen
        return render.route(self.state, owner, rec, task)

    def summary(self) -> dict:
        return self.state.stats.to_dict()
