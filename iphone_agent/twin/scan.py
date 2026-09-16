"""扫主屏：回第一页、一页页往右翻、看、写表（docs/32 §4.4）。

`walk_home_pages` 是唯一一个翻主屏的例程，两个调用方：
· `iphone twin scan`（scan_home）—— 不找什么，翻到头，把每一页按真实页序写进布局表；
· open_app 查表直达没开成时（executor._open_app_from_home）—— 边翻边找那个 App，找到就停，
  翻过的页顺手写表。
只看不点（点是调用方的事），每页只跑 OCR（Perceiver.observe_text）。任务中经过主屏时的刷新在
loop 里顺手做（refresh_from_observation，只覆盖不追加）。

⚠ 2026-09-14：原来只扫当前可见的那一页（第一期范围），refresh_from_observation 又从不追加，
  布局表永远只有第 1 页。这台手机上设置 / 备忘录 / 提醒事项在第 2 页、记账本在第 3 页 ——
  查表直达 layout_miss=no_hit 23/23 次。翻页是唯一知道页序的时刻，所以新页只在这里写。
文件夹、负一屏不扫；App 资源库认出来就停，不当主屏页。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from iphone_agent import config
from iphone_agent.driver.timing import IOS_TIMING
from iphone_agent.harness.executor import looks_like_home, return_to_first_home_page
from iphone_agent.harness.settle import settle
from iphone_agent.perceive.hashing import ahash
from iphone_agent.twin.layout import Layout, Page, infer_page, match_label, norm_label, same_page

_WHY_NOT = {"not_home": "现在不在主屏（认不出主屏特征词），没扫",
            "no_grid": "主屏上认不出网格（标签太少或排不成行列），没扫"}

# App 资源库的字（按 norm_label 比：去空格、不分大小写）。它在最后一页主屏的右边，也排得成网格，
# 但不是主屏页 —— 认出这几个字就停（肯定信号，CLAUDE.md §2）。
APP_LIBRARY_MARKERS = ("app资源库", "applibrary")

# walk 停下来的原因，固定就这几个（见 walk_home_pages 的 docstring）
WALK_STOPS = ("found", "last_page", "app_library", "max_pages", "not_home", "error")


def is_app_library(obs) -> bool:
    return any(m in norm_label(e.text) for e in obs.elements for m in APP_LIBRARY_MARKERS)


def _home_page_from(obs, evidence: dict) -> tuple[Page | None, str | None]:
    """一帧观察 → 主屏的一页。扫描和任务中刷新共用这一道闸，判定只写这一处（CLAUDE.md §7）。

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


@dataclass
class Found:
    """找到了：第几页、当前那一帧 OCR 观察里的标签元素、那一帧的观察、那一页翻停的时刻。"""
    order: int
    element: object
    obs: object
    rested_at: float | None      # time.monotonic()；None = 第 1 页，没翻过页（点之前不用等）


@dataclass
class Walk:
    stop: str                                  # WALK_STOPS 之一
    looked: int = 0                            # 看过几页主屏（App 资源库、重复的那页不算）
    found: Found | None = None
    pages: list[Page] = field(default_factory=list)   # 认出网格的页，按页序
    layout: Layout | None = None               # 写过的表；没给路径 / 没有页可写 = None
    saved: bool | None = None                  # None = 没写；False = 写失败（被别人先写了或磁盘错误）
    error: str | None = None


def _search_want(obs, want: str | None):
    """在这一帧上找 want。跟页识别分开算、自己兜一层 try（M2, 2026-09-14）：

    找 App 靠的是当前帧上的文字，跟这一页认不认得出网格是两回事——页识别（infer_page /
    same_page）坏了不该连累找 App。`want` 类型不对（理论上不会到这里，executor 那边已经
    校过）也只当没找到，不把整个 walk 炸掉。
    """
    if not want:
        return None
    try:
        return match_label(want, obs.elements, key=lambda e: e.text)
    except Exception:      # noqa: BLE001 —— 找不到就是没找到，不连累别的
        return None


def _look(obs, order: int, seen_keys: list, pages: list[Page], want: str | None):
    """看一页：是不是主屏的第 order 页、要不要停、want 在不在上面。纯孪生逻辑，不碰设备。

    识别（是不是主屏、认不认得出网格、是不是 App 资源库）和找 want 分开算 ——
    识别炸了这页就当认不出网格，不写表，但不终止 walk、不连累找 App（M2, 2026-09-14 终审）。
    停下来的两种情况（`not_home` / `last_page` / `app_library`）不在那一帧上找 want，
    跟原来一样：这几页本来就不是「认出来的新主屏页」。

    返回 (停的原因 | None, 这一页 Page | None, want 的元素 | None, 识别出错时的说明 | None)。
    """
    evidence = {"frame_id": obs.frame_id}
    page, rec_error = None, None
    try:
        if order == 1:
            page, why = _home_page_from(obs, evidence)
            if why == "not_home":
                return "not_home", None, None, None
        else:
            # 「这是不是同一个画面」只认 state_key（aHash + OCR 文字集合，hashing.state_key 的注释）；
            # OCR 噪声会让同一页偶尔出两个键，再用网格标签兜一道 —— 同一页记成两个页序比少翻一页坏。
            if obs.state_key in seen_keys:
                return "last_page", None, None, None
            if is_app_library(obs):
                return "app_library", None, None, None
            page = infer_page(obs.elements, obs.width_px, obs.height_px, order=order, evidence=evidence)
            if page is not None and any(same_page(page.labels, p.labels) for p in pages):
                return "last_page", None, None, None
    except Exception as e:      # noqa: BLE001 —— M2, 2026-09-14：识别炸了当认不出网格，不连累找 App
        page, rec_error = None, f"{type(e).__name__}: {e}"
    return None, page, _search_want(obs, want), rec_error


def _reload_in_place(lay: Layout, path) -> None:
    """写盘失败后，把调用方手里这个对象刷新成磁盘上最新的版本（原地改，不换对象）。

    ⚠ 2026-09-14（M4 终审）：`lay.save` 冲突（RevisionConflict）时它自己不抛，只回 False，
    `lay.revision` 停在那个过时的值上不会跟着变。调用方（executor 的 `self.layout` /
    loop 的 `ex.layout`）手里还攥着同一个对象，如果不把它顶成磁盘最新版，这一整个任务里
    后面任何一次存盘都会拿着同一个过时 revision 去比，永远冲突下去。
    `refresh_from_observation` 是同一个模式（同款 bug、同款修法）。
    """
    fresh = Layout.load(path)
    lay.pages = fresh.pages
    lay.revision = fresh.revision


def _record(lay: Layout, pages: list[Page], known_last: int | None) -> None:
    """把翻到的页按真实页序写进表。翻页是页序唯一的权威：

    · 表里自称第 k 页、却和现在的第 k 页对不上的，是过时的，删掉；
    · 现在的第 k 页在表里记在别的页序上（用户挪过整页），那条也删掉，以现在看到的为准；
    · 翻到了头（known_last = 一共几页）时，页序超过最后一页的也删。只找到半路就停的不删 —— 后面没看过。
    认不出网格的页不写也不删（没看清不等于没有，CLAUDE.md §2）。

    ⚠ 2026-09-14（I1 终审）：`known_last` 只能来自**肯定信号**——翻到了 App 资源库
      （`app_library`）。原来 `last_page`（翻页后画面没变 / 网格跟见过的页撞了）也算「翻到头」，
      而这本身是个代理指标，一次翻页手势没生效（镜像抖动、点漏了）就会让下一帧复现上一页的
      `state_key`，`looked` 停在很小的数上，known_last 一小，`_record` 就把表里所有更大页序
      的页全删了——真机复现：3 页表 + 一次丢手势的翻页 → `[1,2,3]` 变成 `[1]`。
      代理指标的否定结论必须复核，不能直接拿来删状态（CLAUDE.md §2）：现在只有看见了
      App 资源库这个肯定证据，才确认「翻到头了、后面没有页了」去删；`last_page` 只表示
      「这次没能再往下看」，不代表后面真的没有页，不删。见 `walk_home_pages` 里翻页复核的那段。
    """
    # ⚠ 2026-09-10 终审：这里原来直接 upsert_page —— 标签集合对不上就**追加**。第 1 页大改后
    #   重扫，表里出现两页 order=1，旧页永远删不掉，find_app 按 order 排序仍先命中过时的那页
    #   （翻到第 1 页找不到标签，每次都白回一趟主屏再退回 Spotlight）；而且那页还活在表里，
    #   任务中经过它时刷新会把它以 order=1 重新「确认」一遍，过时的表永远不会自愈。
    # 扫描是页序唯一权威：第 1 页就是现在看到的这页，表里其它自称第 1 页却对不上的都是过时的，
    # 删掉。刷新从不追加（refresh_from_observation 的闸 3），所以删掉的页不会被复活。
    # 2026-09-14：同一条规矩推广到每个页序（翻页数着翻，第 k 页也是权威）。
    for page in pages:
        keep = []
        for p in lay.pages:
            same = same_page(p.labels, page.labels)
            if p.order == page.order and not same:
                continue          # 这个页序上过时的页
            if p.order != page.order and same:
                continue          # 这一页原来记在别的页序上
            keep.append(p)
        lay.pages = keep
        lay.upsert_page(page)     # 命中的只剩同页序那一条：整页覆盖、保留 id；没有就按 page.order 追加
    if known_last is not None:
        lay.pages = [p for p in lay.pages if p.order <= known_last]


def walk_home_pages(dev, per, path, *, want: str | None = None, layout: Layout | None = None,
                    start=None, log=print) -> Walk:
    """回第一页 → 一页页往右翻 → 每页只跑 OCR → （有 want 时）找到就停 → 把认出来的页按页序写表。

    停下来的原因（Walk.stop）：
      found        want 在这一页上（match_label 命中）
      last_page    翻过去还是见过的那一页：翻到头了
      app_library  翻到了 App 资源库（认它的字）：不当主屏页，停
      max_pages    看了 config.HOME_PAGES_MAX 页
      not_home     第 1 页认不出主屏特征词（looks_like_home）：可能根本没回到主屏，不往下翻、不找
      error        孪生自己的逻辑抛了异常（坏了 = 没有孪生，docs/32 不变式 5）

    页怎么认：第 1 页过 `_home_page_from`（looks_like_home + 网格，和任务中刷新同一道闸）；
    第 2 页起是从第 1 页往右翻过来的，只要 infer_page 认得出网格 —— 一页全是第三方 App 的主屏页
    过不了 looks_like_home（特征词全是系统 App），这正是原来只有第 1 页进得了表的原因之一。
    认不出网格的页照样在上面找 want，只是不写表。

    写表：path 不是 None、又认出了页才写（layout 给了就在它上面改 —— 调用方手里那份跟着变，
    revision 对得上；没给就 Layout.load(path)），用 Layout.save（revision 核对）；写失败只 log，不影响找 App。
    ⚠ 2026-09-14（M4）：写失败（被别人先写了）时把这个对象原地刷新成磁盘最新版（`_reload_in_place`），
      不然它手里那个过时 revision 会让调用方（`self.layout` / `ex.layout`）接下来在同一个任务里
      每次存盘都冲突。

    异常：孪生自己的逻辑（认页、匹配、改表）坏了交回 stop=error，不抛；**设备和感知的调用
    （key / scroll / capture / settle / observe_text）不包**，异常照旧往外抛 —— 和 open_app 查表直达
    同一个规矩（executor._open_app_from_layout 的 ⚠ I2）：镜像断了不是「表没用上」，吞掉它再去走
    下一条路只会在一个死通道上再撞一次。scan_home 是 CLI 的锦上添花，它自己再兜一层。
    ⚠ 2026-09-14（M2）：识别（认页）自己的异常在 `_look` 内部就近接住，不再让整个 walk 交回
      stop=error —— 那一页就当认不出网格（不写表），但**找 want 照样在这一帧上跑**：一页
      识别炸了不该连累找 App。这里剩下的 try/except 是兜底（`_look` 万一还有漏网的）。

    等待：翻页后只 settle，不硬等 AFTER_PAGE_FLIP_S —— 那条管的是「翻页后多久才能点」，这里只看不点
    （2026-09-14）。找到的那页若是翻过来的，Found.rested_at 记着它翻停的时刻，调用方点之前等满
    （executor.wait_out_page_flip）。

    ⚠ 2026-09-14（I1 终审）：翻页后如果下一帧复现了已经见过的画面（`state_key` 撞上 `seen_keys`
      或网格跟见过的页 `same_page`），先别急着认定「翻到头了」—— 这是代理指标的否定结论，
      必须复核（CLAUDE.md §2）：翻一次再看，还是复现才真算数。真机复现过一次丢手势的翻页
      （`dev.scroll` 没让画面变化）：下一帧原样复现上一页，`looked` 停在很小的数上，
      如果这就被当成「翻到头了」去删表里更大页序的页，一次抖动能删光整张表。
      复核之后即便确认了，也只是 `stop="last_page"`——`_record` 现在只信 `app_library`
      这一个肯定信号来删页（见 `_record` 的 ⚠），`last_page` 从不删已知页。
    """
    return_to_first_home_page(dev, start if start is not None else dev.capture())
    frame = dev.capture()
    walk = Walk(stop="max_pages")
    seen_keys: list = []
    rested_at: float | None = None
    order = 1

    def _look_at(obs):
        try:
            stop, page, hit, rec_error = _look(obs, order, seen_keys, walk.pages, want)
        except Exception as e:      # noqa: BLE001 —— _look 自己兜了识别和找 App，这里兜的是漏网的
            return "error", None, None, f"{type(e).__name__}: {e}", True
        if rec_error is not None:
            log(f"  第 {order} 页：识别出错（{rec_error}），这页没记")
        return stop, page, hit, rec_error, False

    while True:
        obs = per.observe_text(frame)
        stop, page, hit, rec_error, fatal = _look_at(obs)
        if fatal:
            walk.stop, walk.error = stop, rec_error
            break
        walk.error = walk.error or rec_error
        if stop == "last_page":
            # 复现已经见过的一页：先翻一次再核实，不直接采信这个否定结论（见上面的 ⚠）。
            dev.scroll("right", "page")
            frame, _ = settle(dev, frame, IOS_TIMING["scroll"], ahash)
            rested_at = time.monotonic()
            obs = per.observe_text(frame)
            stop, page, hit, rec_error, fatal = _look_at(obs)
            if fatal:
                walk.stop, walk.error = stop, rec_error
                break
            walk.error = walk.error or rec_error
        if stop is not None:
            walk.stop = stop
            if stop == "not_home":
                log(f"  {_WHY_NOT['not_home']}")
            break
        walk.looked = order
        if page is not None:
            walk.pages.append(page)
            log(f"  第 {order} 页：{len(page.cells)} 个格子，{', '.join(page.labels[:6])}…")
        elif rec_error is None:
            log(f"  第 {order} 页：认不出网格（标签太少或排不成行列），这页没记")
        if hit is not None:
            walk.found = Found(order, hit, obs, rested_at)
            walk.stop = "found"
            break
        if order >= config.HOME_PAGES_MAX:
            break
        seen_keys.append(obs.state_key)
        dev.scroll("right", "page")
        frame, _ = settle(dev, frame, IOS_TIMING["scroll"], ahash)
        rested_at = time.monotonic()
        order += 1
    if path is not None and walk.pages:
        # I1 终审：known_last 只信 app_library 这一个肯定信号（见 _record 的 ⚠），last_page 不删页。
        known_last = walk.looked if walk.stop == "app_library" else None
        try:
            lay = layout if layout is not None else Layout.load(path)
            _record(lay, walk.pages, known_last)
            walk.layout = lay
            walk.saved = lay.save(path)
            if not walk.saved:
                _reload_in_place(lay, path)      # M4：别让调用方手里的对象停在过时 revision 上
        except Exception as e:      # noqa: BLE001 —— 孪生坏了 = 没有孪生，找 App 照旧
            walk.saved = False
            walk.error = walk.error or f"{type(e).__name__}: {e}"
        if not walk.saved:
            log("  布局表写入失败（被别人先写了或磁盘错误），这次没保存")
    return walk


def scan_home(dev, per, path, *, log=print) -> Layout | None:
    """`iphone twin scan`：回主屏第一页、翻到头、按页序整页写表。任何一步不对返回 None，绝不抛。

    回主屏第一页的按键次数和等待规则跟 open_app 共用同一个入口
    （`executor.return_to_first_home_page`），翻页、认页、写表跟 open_app 翻主屏共用 walk_home_pages
    —— 一个规则一个入口。
    """
    try:
        w = walk_home_pages(dev, per, path, log=log)
    except Exception as e:      # noqa: BLE001 —— 扫描是锦上添花，出错只报不抛
        log(f"  扫描出错：{type(e).__name__}: {e}")
        return None
    if w.stop == "error":
        log(f"  扫描出错：{w.error}")
    elif w.stop == "last_page":
        # I1，2026-09-14：这是复核过的否定结论（翻了一次再确认过还是复现），不是「肯定翻到头了」——
        # 没看见 App 资源库就没有肯定证据，措辞不能说成确认过的事实。
        log(f"  翻页翻不动了（复核过，不一定是最后一页）：这次看到 {w.looked} 页")
    elif w.stop == "app_library":
        log(f"  翻到 App 资源库：主屏一共 {w.looked} 页")
    elif w.stop == "max_pages":
        log(f"  看了 {w.looked} 页（上限 HOME_PAGES_MAX），后面没翻")
    if not w.pages:
        if w.stop != "not_home":
            log("  一页网格都没认出来，没写表")
        return None
    return w.layout if w.saved else None


def refresh_from_observation(layout: Layout, obs, path, run_name: str) -> bool:
    """任务中顺手经过表里已有的主屏页时，用这一帧把那一页整页覆盖（docs/32 §4.1）。

    只覆盖，绝不追加新页。任务中经过的主屏页不知道自己是第几页（模型会左右翻主屏），
    而 `upsert_page` 命中后保留旧 order —— 如果把认不出来的新页按
    `order=len(layout.pages)+1` 追加进去，第 2 页可能被错记成第 1 页，`open_app` 从此
    翻错页（有「当前帧找不到标签就退回 Spotlight」兜着，不会点错，但表永远不会自愈）。
    spec §2.2 明写「页的顺序在扫描时定」，所以新页只能由翻主屏写入 —— `walk_home_pages`
    （`iphone twin scan`，以及 open_app 查表没开成时翻主屏找 App）：它先回第一页、数着翻，知道顺序。
    代价：没翻过主屏时，任务中的刷新什么也不做。

    另外：`looks_like_home` 的特征词全是系统 App，一页全是第三方 App 的主屏页过不了
    第一道闸，任务中不会被刷新；它只在翻主屏时按页序重写（walk 认第 2 页起的页只要网格，2026-09-14）。

    四道闸，任一不过就返回 False、不写盘：
    1. `looks_like_home(obs)` 不成立
    2. `infer_page(...)` 认不出网格（列少于 3、行少于 2）
    （1、2 两道和 scan_home 共用 `_home_page_from`，判定只写一处）
    3. 表里没有任何一页与它 `same_page`（只覆盖已知页）
    4. `Layout.save` 写不进去（revision 冲突 / 磁盘错误）—— 写不进去时把 `layout` 原地刷新成磁盘
       最新版（`_reload_in_place`，M4，2026-09-14），不然调用方手里这个对象（`ex.layout`）停在
       过时 revision 上，这一整个任务里后面的存盘会一直冲突下去（和 `walk_home_pages` 同一个坑）。
    整个函数体再包一层 try/except：孪生任何一处坏了都不影响任务。
    """
    try:
        page, _ = _home_page_from(obs, {"run": run_name, "frame_id": obs.frame_id})
        if page is None:
            return False
        if not any(same_page(p.labels, page.labels) for p in layout.pages):
            return False
        layout.upsert_page(page)
        ok = layout.save(path)
        if not ok:
            _reload_in_place(layout, path)
        return ok
    except Exception:      # noqa: BLE001 —— 孪生任何一处坏了都不影响任务
        return False
