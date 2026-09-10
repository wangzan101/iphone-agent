"""扫主屏：回第一页、看一眼、写表（设计说明）。

第一期只扫**当前可见的那一页**和它上面读得到标签的格子。翻页扫全部、文件夹、App 资源库、负一屏
都推后 —— 只有首页和 dock 已经能验证「开 App 少打字」这件事值不值（codex 评审第 7 点）。
只看不点，零风险。用户点一下才扫（CLI / 网页），任务中经过主屏时的刷新在 loop 里顺手做。
"""
from __future__ import annotations

from iphone_agent.harness.executor import looks_like_home, return_to_first_home_page
from iphone_agent.twin.layout import Layout, Page, infer_page, same_page

_WHY_NOT = {"not_home": "现在不在主屏（认不出主屏特征词），没扫",
            "no_grid": "主屏上认不出网格（标签太少或排不成行列），没扫"}


def _home_page_from(obs, evidence: dict) -> tuple[Page | None, str | None]:
    """一帧观察 → 主屏的一页。扫描和任务中刷新共用这一道闸，判定只写这一处（项目开发约定）。

    返回 (page, None)；或 (None, 原因)，原因是 `not_home`（looks_like_home 不成立）
    或 `no_grid`（infer_page 认不出 ≥3 列 ≥2 行的网格）。页的 order 固定为 1：
    扫描时它就是第 1 页；刷新时它只会覆盖表里已有的页，`upsert_page` 保留旧 order。
    """
    if not looks_like_home(obs):
        return None, "not_home"
    page = infer_page(obs.elements, obs.width_px, obs.height_px, order=1, evidence=evidence)
    if page is None:
        return None, "no_grid"
    return page, None


def scan_home(dev, per, path, *, log=print) -> Layout | None:
    """回主屏第一页 → 观察 → 算网格 → 整页覆盖写。任何一步不对返回 None，绝不抛。

    回主屏第一页的按键次数和等待规则跟 open_app 共用同一个入口
    （`executor.return_to_first_home_page`），一个规则一个入口。
    """
    try:
        return_to_first_home_page(dev, dev.capture())
        obs = per.observe(dev.capture())
        page, why = _home_page_from(obs, {"frame_id": obs.frame_id})
        if page is None:
            log(f"  {_WHY_NOT[why]}")
            return None
        lay = Layout.load(path)
        # ⚠ 2026-09-10 终审：这里原来直接 upsert_page —— 标签集合对不上就**追加**。第 1 页大改后
        #   重扫，表里出现两页 order=1，旧页永远删不掉，find_app 按 order 排序仍先命中过时的那页
        #   （翻到第 1 页找不到标签，每次都白回一趟主屏再退回 Spotlight）；而且那页还活在表里，
        #   任务中经过它时刷新会把它以 order=1 重新「确认」一遍，过时的表永远不会自愈。
        # 扫描是页序唯一权威：第 1 页就是现在看到的这页，表里其它自称第 1 页却对不上的都是过时的，
        # 删掉。刷新从不追加（refresh_from_observation 的闸 3），所以删掉的页不会被复活。
        lay.pages = [p for p in lay.pages if p.order != 1 or same_page(p.labels, page.labels)]
        lay.upsert_page(page)
        if not lay.save(path):
            log("  布局表写入失败（被别人先写了或磁盘错误），这次没保存")
            return None
        log(f"  第 1 页：{len(page.cells)} 个格子，{', '.join(page.labels[:6])}…")
        return lay
    except Exception as e:      # noqa: BLE001 —— 扫描是锦上添花，出错只报不抛
        log(f"  扫描出错：{type(e).__name__}: {e}")
        return None


def refresh_from_observation(layout: Layout, obs, path, run_name: str) -> bool:
    """任务中顺手经过表里已有的主屏页时，用这一帧把那一页整页覆盖（设计说明）。

    只覆盖，绝不追加新页。任务中经过的主屏页不知道自己是第几页（模型会左右翻主屏），
    而 `upsert_page` 命中后保留旧 order —— 如果把认不出来的新页按
    `order=len(layout.pages)+1` 追加进去，第 2 页可能被错记成第 1 页，`open_app` 从此
    翻错页（有「当前帧找不到标签就退回 Spotlight」兜着，不会点错，但表永远不会自愈）。
    spec §2.2 明写「页的顺序在扫描时定」，所以新页只能由 `iphone twin scan` 写入
    （它先回第一页，知道顺序）。代价：用户从没扫描过时，任务中的刷新什么也不做 ——
    第 0 天先扫一次，这就是 spec §4.4 的「用户点一下才扫」。

    另外：`looks_like_home` 的特征词全是系统 App，一页全是第三方 App 的主屏页过不了
    第一道闸，第一期不会被刷新（第一期范围本来就是第 1 页）。

    四道闸，任一不过就返回 False、不写盘：
    1. `looks_like_home(obs)` 不成立
    2. `infer_page(...)` 认不出网格（列少于 3、行少于 2）
    （1、2 两道和 scan_home 共用 `_home_page_from`，判定只写一处）
    3. 表里没有任何一页与它 `same_page`（只覆盖已知页）
    4. `Layout.save` 写不进去（revision 冲突 / 磁盘错误）
    整个函数体再包一层 try/except：孪生任何一处坏了都不影响任务。
    """
    try:
        page, _ = _home_page_from(obs, {"run": run_name, "frame_id": obs.frame_id})
        if page is None:
            return False
        if not any(same_page(p.labels, page.labels) for p in layout.pages):
            return False
        layout.upsert_page(page)
        return layout.save(path)
    except Exception:      # noqa: BLE001 —— 孪生任何一处坏了都不影响任务
        return False
