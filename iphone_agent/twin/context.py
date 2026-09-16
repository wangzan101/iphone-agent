"""本次任务的认屏上下文（spec 2026-09-12 §3.3）：给 Perceiver 挑候选，给 Executor 换算 App 名、学别名。

只在一个任务里存在：run_task 建、用 Perceiver.task_scope 挂上、finally 解除（codex 评审第 4 条）。
数据全来自实时孪生（LiveTwin.state）：开跑时读盘，运行中在内存里长 —— 与重放同一份 TwinState 逻辑。
"""
from __future__ import annotations

from iphone_agent.perceive.screen import Candidate
from iphone_agent.twin.identify import norm_text

# 保险丝，不是按现在的规模定的值（memory：不要拿当前测试规模限制设计）。候选只来自与这一帧文字
# 沾边的屏，孪生长大提示词也不会被整个塞满；这个上限只防意外。
SCREEN_CANDIDATES_MAX = 20


class ScreenIdentityContext:
    def __init__(self, live):
        self.live = live

    def candidates(self, ocr_texts) -> list[Candidate]:
        texts = {n for n in (norm_text(t) for t in ocr_texts) if n}
        st = self.live.state
        scored: list[tuple[int, str, str, str]] = []
        for app in st.all_apps():
            for s in st.screens_of(app):
                hits = sum(1 for a in s.anchors if a in texts)
                if hits:
                    scored.append((-hits, app, s.id, s.name))
        scored.sort()          # (-相同锚点数, app_id, screen_id)：同分时次序固定（codex 评审第 13 条）
        return [Candidate(i, app, st.display(app), sid, name)
                for i, (_, app, sid, name) in enumerate(scored[:SCREEN_CANDIDATES_MAX], start=1)]

    def resolve_app(self, label_app: str) -> str | None:
        return self.live.state.resolve_app(label_app)

    def learn_app_alias(self, label_app: str, app_id: str) -> None:
        self.live.state.learn_alias(label_app, app_id)
