import json

import pytest

from iphone_agent.driver.geometry import OutOfWindow
from iphone_agent.driver.injector import ActivateFailed
from iphone_agent.harness.actions import Action
from iphone_agent.harness.executor import Executor


def act(action_name, **args):
    return Action(action_name, args, "r", None, "c1")


def test_tap_changes_screen(fake_env):
    dev, per, frames = fake_env([["通用"], ["关于本机"], ["关于本机"], ["关于本机"]])
    obs = per.observe(dev.capture())
    ex = Executor(dev, per)
    res, new_obs = ex.run(act("tap", id=1, x=obs.elements[0].center[0], y=obs.elements[0].center[1]), obs)
    assert res.ok and res.changed is True and dev.calls[0][0] == "tap"
    assert new_obs is not None and "关于本机" in new_obs.text_set


def test_tap_unchanged_gives_hint(fake_env):
    dev, per, frames = fake_env([["通用"], ["通用"], ["通用"], ["通用"]])
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("tap", x=10, y=10), obs)
    assert res.ok and res.changed is False and "未检测到明显变化" in res.hint


def test_scroll_unchanged_hint_says_bottom(fake_env):
    dev, per, frames = fake_env([["a"], ["a"], ["a"], ["a"]])
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("scroll", direction="down", amount="page"), obs)
    assert "到底" in res.hint


def test_window_moved_rejected_without_injection(fake_env):
    dev, per, frames = fake_env([["a"], ["a"]])
    obs = per.observe(dev.capture())
    from iphone_agent.driver.geometry import Rect
    dev.rect = Rect(50, 50, 200, 400)
    res, new_obs = Executor(dev, per).run(act("tap", x=10, y=10), obs)
    assert not res.ok and res.error == "window_moved" and dev.calls == [] and new_obs is not None


def test_activate_failed_reported(fake_env):
    dev, per, frames = fake_env([["a"], ["a"]])
    obs = per.observe(dev.capture())
    dev.raise_on["tap"] = ActivateFailed("x")
    res, _ = Executor(dev, per).run(act("tap", x=10, y=10), obs)
    assert res.error == "activate_failed"


def test_activate_failed_returns_none_observation(fake_env):
    """裁决二：三类设备错误不应重推同一个 observation，避免消息流里出现重复观察块。"""
    dev, per, frames = fake_env([["a"], ["a"]])
    obs = per.observe(dev.capture())
    dev.raise_on["tap"] = ActivateFailed("x")
    res, new_obs = Executor(dev, per).run(act("tap", x=10, y=10), obs)
    assert res.error == "activate_failed" and new_obs is None


def test_out_of_window_returns_none_observation(fake_env):
    dev, per, frames = fake_env([["a"], ["a"]])
    obs = per.observe(dev.capture())
    dev.raise_on["tap"] = OutOfWindow("x")
    res, new_obs = Executor(dev, per).run(act("tap", x=10, y=10), obs)
    assert res.error == "out_of_window" and new_obs is None


def test_device_error_returns_none_observation(fake_env):
    dev, per, frames = fake_env([["a"], ["a"]])
    obs = per.observe(dev.capture())
    dev.raise_on["tap"] = RuntimeError("boom")
    res, new_obs = Executor(dev, per).run(act("tap", x=10, y=10), obs)
    assert res.error == "device_error" and new_obs is None


def test_type_verified_true_when_new_text_appears_near_last_tap(fake_env):
    # ⚠ 目标文本用 ASCII（"safari"）而不是中文：非 ASCII 现在走输入法编排
    #    （发拼音 → 看候选 → 点选），那条路自己出结果，没有 verified 字段。
    #    这条测的是 ASCII 直打之后的落屏校验，两件事。
    # 裁决三：重新设计帧序列，让「type 之前的基线」确定不含目标文本。
    #
    # fast_timing fixture 把 IOS_TIMING 的每一项都换成 Timing(0, 10, 20, 1000)：
    # poll_ms=10、stable_span_ms=20，比例 2:1，与生产环境 IOS_TIMING 各项的比例一致
    # （例如 tap: poll_ms=150, stable_span_ms=400 ≈ 2.67；这里取整数 2 简化）。
    # 在这个比例下，settle() 判定「稳定」需要连续看到 4 帧内容不变（1 次 last + 3 次轮询），
    # 因为 stable_since 在第一次相邻帧相同时才置位，之后还要再等满 stable_span_ms（=2 个
    # poll 间隔）才会返回。
    #
    # 于是把帧序列分两段、每段 4 帧一模一样：
    #   [0]            —— tap 之前的初始观察（"搜索"）
    #   [1..4] "搜索"   —— tap 动作后的 settle 消耗这 4 帧，稳定后返回 [4]（仍是干净的"搜索"，
    #                      不含 "safari"，所以 type 的基线一定干净）
    #   [5..8] "搜索","设置" —— type 动作后的 settle 接着消耗这 4 帧，稳定后返回 [8]
    #                      （新增了目标文本 "safari"）
    # FakeDevice 的帧索引只增不减，一旦某段落用完当前 4 帧也会正好在该段落末尾稳定下来，
    # 不会提前越界读到下一段——这样基线（tap 后）与回读（type 后）确定分别是"干净"与"命中"。
    dev, per, frames = fake_env([
        ["搜索"],
        ["搜索"], ["搜索"], ["搜索"], ["搜索"],
        ["搜索", "safari"], ["搜索", "safari"], ["搜索", "safari"], ["搜索", "safari"],
    ])
    obs = per.observe(dev.capture())
    ex = Executor(dev, per)
    _, obs = ex.run(act("tap", x=100, y=65), obs)
    res, _ = ex.run(act("type", text="safari"), obs)
    assert res.extra["verified"] == "true"
    assert dev.calls[-1] == ("type", "safari")


def test_open_app_not_found(fake_env):
    dev, per, frames = fake_env([["主屏"], ["搜索框"], ["搜索框"], ["搜索框"], ["搜索框"], ["搜索框"]])
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("open_app", name="设置"), obs)
    assert res.error == "app_not_found"
    assert ("enter",) not in dev.calls


def test_done_executes_nothing(fake_env):
    dev, per, frames = fake_env([["a"]])
    obs = per.observe(dev.capture())
    res, new_obs = Executor(dev, per).run(act("done", status="success", result="ok"), obs)
    assert res.ok and new_obs is None and dev.calls == []


# --- open_app：输入落屏校验 + 点匹配行，不盲按回车（2026-09-07 真机暴露） ---

def test_query_landed_rejects_text_that_was_already_on_screen():
    """Spotlight 一打开就列常用 App，目标本来就在画面上 ——
    那不是「输入成功」的证据。只认输入后才出现的文字。"""
    from types import SimpleNamespace

    from iphone_agent.harness.executor import Executor

    def o(texts):
        return SimpleNamespace(elements=[SimpleNamespace(text=t) for t in texts])

    # 输入前就有「设置」（常用 App 列表）→ 不能算输入成功
    assert not Executor._query_landed(o(["搜索", "设置", "微信"]), o(["搜索", "设置", "微信"]), "设置")
    # 输入前没有、输入后才出现 → 才算成功
    assert Executor._query_landed(o(["搜索", "微信"]), o(["搜索", "设置", "微信"]), "设置")


def test_match_row_skips_the_search_box_band_at_the_bottom():
    """输入没落屏时，屏幕**底部**那一带（搜索框 + 输入法候选栏）里含目标字样的东西
    不是结果行。

    ⚠ 这条原来叫 skips_the_query_echo_at_the_top，排的是顶部 25% —— 那是旧 Spotlight
      的布局。iOS 18 搜索框在底部（实测 y≈0.92H），候选栏在它上面（≈0.88H）；
      而「最佳搜索结果」的 App 标签在顶部 y≈0.246H，刚好被 25% 排掉，差 6 像素。
      2026-09-09 评测集跑出来的：open_app('Safari 浏览器') 因此点到「提示」区一条长句。
    """
    from types import SimpleNamespace

    from iphone_agent.harness.executor import Executor

    app_row = SimpleNamespace(text="设置", center=(100, 341))          # 最佳搜索结果，顶部
    cand_bar = SimpleNamespace(text="1 设置", center=(100, 1218))      # 输入法候选栏，底部
    obs = SimpleNamespace(height_px=1388, elements=[app_row, cand_bar])

    # 精确等于 App 名的行不看位置
    assert Executor._match_row(obs, "设置", exclude_top=True) is app_row
    # 只有候选栏时：没确认落屏就不能拿它当结果；确认了才退回包含匹配
    obs.elements = [cand_bar]
    assert Executor._match_row(obs, "设置", exclude_top=True) is None
    assert Executor._match_row(obs, "设置", exclude_top=False) is cand_bar


def test_match_row_returns_none_when_nothing_matches():
    from types import SimpleNamespace

    from iphone_agent.harness.executor import Executor
    obs = SimpleNamespace(height_px=1400,
                          elements=[SimpleNamespace(text="微信", center=(100, 600))])
    assert Executor._match_row(obs, "设置", exclude_top=True) is None


# --- 高阶滚动：scroll_until / collect（把 N 次滚动压成一步） ---
#
# 帧序列的算法：fast_timing 下每次 settle 恰好吃掉 4 帧（见上面 type 用例的推导），
# 所以「一屏」= 连续 4 张一模一样的帧，初始观察另外吃掉 1 张。
#
# ⚠ 屏与屏之间必须真的「变了」，否则循环会当场判到底 —— 而 conftest 造的假帧里
# 文本判据永远够不到阈值 8（每屏才两三行），全靠 aHash。下面这几组是实测挑出来的：
# 相邻两组的 aHash 汉明距离 14–17，远在阈值 2 之上。改文字前先量一下再改。
S1 = ["通用", "蜂窝网络"]
S2 = ["隐私与安全", "屏幕使用时间"]
S3 = ["关于本机", "软件更新"]
L1 = ["一", "二", "三"]
L2 = ["三", "四", "五"]          # 与 L1 在「三」上重叠，模拟相邻屏的重叠
L3 = ["五", "六", "七"]          # 再下一屏，与 L2 在「五」上重叠


def screens(*groups, initial):
    """[初始屏] + 每屏 4 张相同的帧。"""
    return [initial] + [g for g in groups for _ in range(4)]


def test_scroll_until_scrolls_several_screens_within_one_call(fake_env):
    """全部价值在这里：一次 run() 里滚了两屏，对外只是一个动作。"""
    dev, per, _ = fake_env(screens(S2, S3, initial=S1))
    obs = per.observe(dev.capture())
    res, new_obs = Executor(dev, per).run(
        act("scroll_until", direction="down", text="关于本机"), obs)
    assert res.ok and res.extra["found"] is True
    assert res.extra["screens"] == 2
    assert [c for c in dev.calls if c[0] == "scroll"] == [("scroll", "down", "page")] * 2
    # 命中的那一屏就是交回去的新观察，模型下一步能直接 tap 它
    assert new_obs is not None and "关于本机" in new_obs.text_set
    assert new_obs.element(res.extra["matched"]["id"]).text == "关于本机"


def test_scroll_until_stops_at_the_bottom_using_the_same_criterion_as_scroll(fake_env):
    """「滚不动了」的判据就是单步 scroll 判 changed 的那个 did_change，不另立一套。

    ⚠ 要**连续两屏**没变化才收工，所以这里给了三屏 S2：第一屏是变化（S1→S2），
    后两屏才是连续两次没变化。改成两屏就是在测「一次没变化就收工」那个旧行为。
    """
    dev, per, _ = fake_env(screens(S2, S2, S2, initial=S1))
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(
        act("scroll_until", direction="down", text="找不到的东西"), obs)
    assert res.ok and res.extra["found"] is False
    assert res.extra["reached_end"] is True and res.extra["screens"] == 3
    assert "滚不动了" in res.hint


def test_scroll_until_does_not_scroll_when_the_text_is_already_on_screen(fake_env):
    dev, per, _ = fake_env(screens(S2, initial=S3))
    obs = per.observe(dev.capture())
    res, new_obs = Executor(dev, per).run(
        act("scroll_until", direction="down", text="关于本机"), obs)
    assert res.extra["found"] is True and res.extra["screens"] == 0
    assert [c for c in dev.calls if c[0] == "scroll"] == []
    assert new_obs is None, "一屏没滚却推了个新观察，模型会以为画面翻页了"


def test_scroll_until_stops_at_the_screen_cap_and_says_it_is_not_the_bottom(fake_env, monkeypatch):
    """上限是防死循环的兜底。撞上它必须说清「还没到底」——
    否则模型会把「没找到」当成「列表里没有这一项」。"""
    from iphone_agent import config
    monkeypatch.setattr(config, "SCROLL_MAX_SCREENS", 2)
    dev, per, _ = fake_env(screens(S1, S2, S3, initial=L1))
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(
        act("scroll_until", direction="down", text="永远不出现"), obs)
    assert res.extra["screens"] == 2 and res.extra["reached_end"] is False
    assert res.extra["found"] is False and "还没到底" in res.hint


def test_collect_dedupes_overlapping_screens_and_keeps_first_seen_order(fake_env):
    """相邻两屏必然重叠，不去重汇总里全是重复行；顺序就是页面的阅读顺序，丢了没法读。"""
    dev, per, _ = fake_env(screens(L2, L2, L2, initial=L1))
    obs = per.observe(dev.capture())
    res, new_obs = Executor(dev, per).run(act("collect", direction="down"), obs)
    assert res.extra["lines"] == ["一", "二", "三", "四", "五"]
    assert res.extra["count"] == 5
    assert res.extra["reached_end"] is True and res.extra["screens"] == 3
    assert new_obs is not None


def test_collect_stops_at_the_screen_cap_and_says_it_is_not_the_bottom(fake_env, monkeypatch):
    from iphone_agent import config
    monkeypatch.setattr(config, "SCROLL_MAX_SCREENS", 1)
    dev, per, _ = fake_env(screens(L2, S1, initial=L1))
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("collect", direction="down"), obs)
    assert res.extra["screens"] == 1 and res.extra["reached_end"] is False
    assert res.extra["lines"] == ["一", "二", "三", "四", "五"]
    assert "还没到底" in res.hint


def test_collect_truncates_instead_of_overflowing_the_message_window(fake_env, monkeypatch):
    """汇总必须留在滑窗给最新工具结果的额度之内 ——
    被从中间切断的 JSON 比「只收了一部分」更坏。"""
    from iphone_agent import config
    monkeypatch.setattr(config, "COLLECT_MAX_CHARS", 3)
    dev, per, _ = fake_env(screens(L2, L2, L2, initial=L1))
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("collect", direction="down"), obs)
    assert res.extra["truncated"] is True
    assert res.extra["lines"] == ["一", "二", "三"]
    assert json.loads(res.to_json())["lines"] == ["一", "二", "三"]


def test_scroll_until_reports_the_window_move_like_any_other_action(fake_env):
    """走的是 run() 里同一套设备错误处理，不另开一条路径。"""
    from iphone_agent.driver.geometry import Rect
    dev, per, _ = fake_env(screens(S2, initial=S1))
    obs = per.observe(dev.capture())
    dev.rect = Rect(50, 50, 200, 400)
    res, _ = Executor(dev, per).run(act("scroll_until", direction="down", text="x"), obs)
    assert not res.ok and res.error == "window_moved"
    assert [c for c in dev.calls if c[0] == "scroll"] == []


# --- 手工入口：iphone scroll_until / collect ---
# 高阶滚动的价值全在 Executor 内部那个循环里，所以手工验证也必须走 Executor，
# 不能绕过去直接调 dev.scroll —— 那验的就不是同一个东西了。

def _session(dev, per):
    from types import SimpleNamespace
    # asker=None：假 Session 得长得像真 Session。手动命令现在也把它递给 Executor，
    # 而 None 正是「没配 key / 关掉了视觉」那条降级路，跟这些用例要验的东西无关。
    return SimpleNamespace(dev=dev, per=per, asker=None)


def test_cli_scroll_until_goes_through_the_executor(fake_env, capsys):
    from iphone_agent.cli.commands import dispatch
    dev, per, _ = fake_env(screens(S2, S3, initial=S1))
    assert dispatch(_session(dev, per), ["scroll_until", "down", "关于本机"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["found"] is True and out["screens"] == 2
    assert dev.released, "手工跑完没释放设备，按住的键会留在手机上"


def test_cli_collect_goes_through_the_executor(fake_env, capsys):
    from iphone_agent.cli.commands import dispatch
    dev, per, _ = fake_env(screens(L2, L2, L2, initial=L1))
    assert dispatch(_session(dev, per), ["collect", "down"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["lines"] == ["一", "二", "三", "四", "五"]


def test_cli_rejects_a_bad_direction_before_touching_the_device(fake_env, capsys):
    from iphone_agent.cli.commands import dispatch
    dev, per, _ = fake_env(screens(S2, initial=S1))
    assert dispatch(_session(dev, per), ["collect", "sideways"]) == 2
    assert dev.calls == []


def test_scroll_until_hands_back_the_last_observation_when_the_device_fails_midway(fake_env):
    """滚到一半设备炸了，也要把最后一次成功的观察交回去。

    已经滚过的屏把画面挪走了，模型手里那份观察就过期了 —— 只报错不给新观察，
    它下一步会点在一个不存在的画面上（这个项目为这一类错栽过）。
    """
    dev, per, _ = fake_env(screens(S2, S3, initial=S1))
    obs = per.observe(dev.capture())
    ex = Executor(dev, per)
    calls = {"n": 0}
    real_scroll = dev.scroll

    def flaky(d, a):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("滚轮没发出去")
        return real_scroll(d, a)

    dev.scroll = flaky
    res, new_obs = ex.run(act("scroll_until", direction="down", text="关于本机"), obs)
    assert not res.ok and res.error == "device_error"
    assert res.extra["screens"] == 1
    assert new_obs is not None and new_obs is not obs


def test_one_lost_scroll_does_not_end_the_collect_early(fake_env):
    """这次改动要防的就是这个：中间丢一次滚动，不能把半个列表当成全部交出去。

    「画面没变」有歧义 —— 既可能是真到底，也可能是这一次滚动没生效。
    屏幕序列是 L1 → L2 → L2（丢了一次）→ L3：如果只看一次没变化就收工，
    汇总里会缺掉 L3 那一屏的内容，而调用方无从分辨。
    """
    dev, per, _ = fake_env(screens(L2, L2, L3, L3, L3, initial=L1))
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("collect", direction="down"), obs)
    assert "六" in res.extra["lines"], "丢一次滚动就提前收工了，L3 那一屏没被收进来"
    assert res.extra["reached_end"] is True


# --- 中文输入：驱动 iOS 自己的输入法（发拼音 → 看候选 → 点选）---
#
# 为什么必须这样：镜像只转发 keycode、不读事件里的 unicode 载荷（设计说明），
# 汉字没有 keycode 打不出来；粘贴这条路在本环境实测不通（设计说明）。
#
# ⚠ 假 OCR 把元素从上往下排、每个 +60px（高 800）。候选带的判据是 y > 640 且
#   不是最底下那一行（最底下是输入框），所以候选必须排在倒数第二、第三位。
#   这里用 _PAD 把前面填满 —— 真机上候选和输入框都挤在最底部 10% 里。

_PAD = [f"占位{i}" for i in range(10)]       # 把候选顶到 y > 640 的带里


def _ime_env(fake_env, *candidates, after_pick=None):
    typing = _PAD + list(candidates) + ["Q 搜索"]      # 最后一个是输入框（最底下）
    picked = after_pick if after_pick is not None else ["设置", "飞行模式"]
    return fake_env([["Q 搜索"]] + [typing] * 4 + [picked] * 4)


def test_chinese_goes_through_the_ime_not_raw_keystrokes(fake_env):
    """打进去的必须是拼音，不是汉字 —— 汉字发不出 keycode。"""
    dev, per, _ = _ime_env(fake_env, "1设置", "2 摄制")
    obs = per.observe(dev.capture())
    res, new = Executor(dev, per).run(act("type", text="设置"), obs)
    assert ("type", "shezhi") in dev.calls, f"没有发拼音，实际发了 {dev.calls}"
    assert not any(c[0] == "type" and c[1] == "设置" for c in dev.calls), "把汉字直接发出去了"
    assert res.ok, f"没选中候选：{res.hint}"
    assert res.extra["pinyin"] == "shezhi" and res.extra["picked"] == "1设置"
    assert new is not None


def test_ime_only_picks_an_exact_candidate(fake_env):
    """只认「剥掉编号后正好等于目标」的候选。

    '设置' 和 '摄制' 只差一个音，包含匹配会点错 —— 而点错没有回头路，字已经上屏。
    """
    dev, per, _ = _ime_env(fake_env, "1 摄制", "2 摄制组")
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("type", text="设置"), obs)
    assert res.ok is False and res.error == "candidate_not_found"
    assert res.extra["candidates"] == ["1 摄制", "2 摄制组"], "没把看到的候选交回去"
    assert "没上屏" in res.hint, "得说清楚字还没上屏，模型才知道可以重试"


def test_ime_does_not_tap_when_nothing_matches(fake_env):
    """找不到候选时绝对不能瞎点一个 —— 点错了字就上屏了，没有撤销。"""
    dev, per, _ = _ime_env(fake_env, "1 摄制")
    obs = per.observe(dev.capture())
    Executor(dev, per).run(act("type", text="设置"), obs)
    assert not any(c[0] == "tap" for c in dev.calls), f"没匹配上却点了：{dev.calls}"


def test_ascii_still_takes_the_direct_path(fake_env):
    """ASCII 不该绕输入法：它本来就能直接打，绕一圈只会更慢更容易错。"""
    dev, per, _ = fake_env([["Q 搜索"]] * 9)
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("type", text="safari"), obs)
    assert ("type", "safari") in dev.calls
    assert "pinyin" not in res.extra


def test_candidate_not_found_reports_that_the_screen_changed(fake_env):
    """打拼音让候选栏冒出来了，屏幕**真的变了**，必须如实报。

    以前这里不设 changed，默认 None；guard 里 `if changed and ...` 对 None 是假，
    于是这一步被记成无进展。逐字输入三个字就自己撞上 NO_PROGRESS_WARN。
    """
    dev, per, _ = _ime_env(fake_env, "1 摄制", "2 摄制组")
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("type", text="设置"), obs)
    assert res.error == "candidate_not_found"
    assert res.changed is True, "候选栏都出来了还报没变化"
    assert res.hamming is not None and res.text_diff is not None


def test_candidate_not_found_tells_the_model_it_can_tap_the_candidate(fake_env):
    """候选栏就在下一张截图上，模型能直接点。以前的提示只说「换更短的词再试」，
    把它往错的方向引 —— 而 OCR 读错单字（「一」读成「—」）时，只有眼睛能救。"""
    dev, per, _ = _ime_env(fake_env, "1 摄制", "2 摄制组")
    obs = per.observe(dev.capture())
    res, new = Executor(dev, per).run(act("type", text="设置"), obs)
    assert "tap" in res.hint, "没告诉模型可以直接点候选"
    assert "截图" in res.hint
    assert new is not None, "得把带候选栏的观察交回去，否则模型下一轮看不到它"


def test_open_app_types_pinyin_not_chinese(fake_env):
    """中文 App 名走拼音字母，**不走输入法**。

    2026-09-08 实测：Spotlight 自己按拼音匹配 App 名 —— 干净的搜索框里打
    "yimujizhang"，「一木记账」直接出现在「最佳搜索结果」（空框里它不在，对照过）。

    不能走输入法：「一木记账」不在词库里，候选是「以募集章」「一目几张」，
    永远选不中；而 dev.type() 拿到中文会走粘贴，粘贴在本环境不通（设计说明）。
    """
    dev, per, _ = fake_env([["Q 搜索"]] * 4 + [["Q 搜索", "一木记账"]] * 6)
    obs = per.observe(dev.capture())
    Executor(dev, per).run(act("open_app", name="一木记账"), obs)
    typed = [c[1] for c in dev.calls if c[0] == "type"]
    assert typed == ["yimujizhang"], f"没打拼音，实际打了 {typed}"
    assert not any("一木" in t for t in typed), "把中文直接发出去了（会走粘贴，本环境不通）"


def test_open_app_still_types_ascii_names_verbatim(fake_env):
    """英文名不该被转换 —— 它本来就能直接打。"""
    dev, per, _ = fake_env([["Q 搜索"]] * 4 + [["Q 搜索", "Safari"]] * 6)
    obs = per.observe(dev.capture())
    Executor(dev, per).run(act("open_app", name="Safari"), obs)
    assert [c[1] for c in dev.calls if c[0] == "type"] == ["Safari"]


# --- 「一次都没动过」不是「滚到底了」 ---
#
# 2026-09-08 真机（runs/example-run）：在通用页 scroll_until 找「关于本机」，
# 滚了两次画面纹丝不动，工具报「滚不动了（到底/到顶）」。模型于是相信关于本机不在
# 通用里，改去设置首页，把「更新到 iOS 26.6.1」这条**可更新版本**当成当前版本
# 报了 done(success) —— 设备版本与报告不一致。误导性的诊断直接造出了一次假成功。

def test_scroll_until_says_it_never_moved_instead_of_claiming_the_end(fake_env):
    dev, per, _ = fake_env([["甲", "乙"]] * 12)      # 每一屏都一样 = 根本没滚动
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("scroll_until", direction="down", text="关于本机"), obs)
    assert res.extra["never_moved"] is True
    assert "一次都没变" in res.hint, res.hint
    assert "到底" not in res.hint.replace("不是到底了", ""), f"还在说到底：{res.hint}"


def test_real_end_of_list_still_says_reached_end(fake_env):
    """真的滚动过、然后停下来 —— 那才是到底，说法不能跟着改。"""
    dev, per, _ = fake_env([["甲"], ["乙"], ["丙"]] + [["丁"]] * 9)
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("scroll_until", direction="down", text="找不到的"), obs)
    assert res.extra["reached_end"] is True
    assert not res.extra.get("never_moved"), "明明滚动过"
    assert "到底" in res.hint, res.hint


def test_collect_also_distinguishes_the_two(fake_env):
    dev, per, _ = fake_env([["甲", "乙"]] * 12)
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("collect", direction="down"), obs)
    assert res.extra["never_moved"] is True
    assert "一次都没变" in res.hint, res.hint


# --- 输入法的两种病，说法必须分开 ---
#
# 2026-09-08 实测：
#   · 候选栏**是空的** = iOS 键盘不在中文模式，拼音字母直接上屏。换词再试没有用。
#   · 候选栏有东西但没有目标 = 词库里没这个词。换更短的、或者自己看着点。
# 而且 select_all_and_delete（Cmd+A + Delete）**清不掉待上屏的拼音缓冲区** ——
# 打完 'shezhi' 调它，搜索框只从 'Q shezhi' 变成 'Q shezh'，掉了一个字符。
# 不撤销的话下一次输入会叠在残留上：连着两次 'shezhi' 的候选是「则设置设置」。

def test_empty_candidate_bar_is_reported_as_wrong_ime_mode(fake_env):
    # 候选栏空、**但拼音字母出现在屏幕上** = 按键进去了，只是输入法在英文模式
    dev, per, _ = fake_env([_PAD + ["Q shezhi"]] * 10)
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("type", text="设置"), obs)
    assert res.error == "ime_not_chinese", res.error
    assert "不在中文模式" in res.hint
    assert "换个词再试没有用" in res.hint, "得说清楚这条路走不通，别让它继续试"


def test_it_undoes_exactly_what_it_typed_and_no_more(fake_env):
    """⚠ 绝不能盲目连按 delete —— 在备忘录、聊天框里会删掉本来就有的内容。

    候选栏为空时会先切一次输入法重试，所以退格发生两次：一次为了重试、
    一次为了放弃。**每一次都必须正好是自己打进去的那么多个字符。**
    """
    dev, per, _ = _ime_env(fake_env)
    obs = per.observe(dev.capture())
    Executor(dev, per).run(act("type", text="设置"), obs)
    backs = [c for c in dev.calls if c[0] == "backspace"]
    assert backs, "一次都没退"
    assert all(c == ("backspace", len("shezhi")) for c in backs), f"退多了或退少了：{backs}"


def test_an_empty_candidate_bar_triggers_one_ime_switch_and_retry(fake_env):
    """候选栏空 = 键盘不在中文模式。切一次再试是模型自己也会做的事，
    没必要为此浪费一步和一次模型调用。但**只切一次** —— 切完还是空就如实报告。"""
    dev, per, _ = fake_env([_PAD + ["Q shezhi"]] * 10)
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("type", text="设置"), obs)
    assert [c for c in dev.calls if c[0] == "toggle_ime"] == [("toggle_ime",)], dev.calls
    assert res.error == "ime_not_chinese"


def test_candidate_not_found_also_undoes_the_pinyin(fake_env):
    """否则下一次输入会叠在残留上。"""
    dev, per, _ = _ime_env(fake_env, "1 摄制", "2 摄制组")
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("type", text="设置"), obs)
    assert res.error == "candidate_not_found"
    assert ("backspace", len("shezhi")) in dev.calls, f"没退掉：{dev.calls}"


def test_ascii_typed_in_chinese_mode_is_corrected(fake_env):
    """⚠ 中文模式下打 ASCII，字母会变成**拼音待上屏**，英文根本打不出来 ——
    而且是静默失败：动作报成功、屏幕也变了（候选栏冒出来了）。
    混合输入必须管这一侧，不能只管中文那一侧。"""
    # 打完 ASCII 之后屏幕上出现候选栏 = 当时在中文模式
    dev, per, _ = fake_env([_PAD + ["1 萨法瑞", "2 洒法"] + ["Q 搜索"]] * 10)
    obs = per.observe(dev.capture())
    Executor(dev, per).run(act("type", text="safari"), obs)
    assert ("toggle_ime",) in dev.calls, f"没切输入法：{dev.calls}"
    assert ("backspace", len("safari")) in dev.calls, "没退掉打错的那几个字母"
    typed = [c[1] for c in dev.calls if c[0] == "type"]
    assert typed == ["safari", "safari"], f"应当切完重打一次：{typed}"


def test_ascii_in_english_mode_is_left_alone(fake_env):
    """没有候选栏就说明本来就是英文模式 —— 不该白切一次。"""
    dev, per, _ = fake_env([["Q 搜索", "safari"]] * 10)
    obs = per.observe(dev.capture())
    Executor(dev, per).run(act("type", text="safari"), obs)
    assert not any(c[0] == "toggle_ime" for c in dev.calls), f"白切了：{dev.calls}"
    assert [c[1] for c in dev.calls if c[0] == "type"] == ["safari"]


def test_switch_ime_forgets_the_believed_mode(fake_env):
    """⚠ 切完是什么模式必须重新验证，不能想当然地翻转 ——
    这是信念不是事实，没有 API 能查。"""
    dev, per, _ = fake_env([["Q 搜索"]] * 8)
    obs = per.observe(dev.capture())
    ex = Executor(dev, per)
    ex._ime_chinese = True
    ex.run(act("switch_ime"), obs)
    assert ("toggle_ime",) in dev.calls
    assert ex._ime_chinese is None, "翻转了而不是置为未知"


def test_a_known_english_mode_does_not_block_switching_to_chinese(fake_env):
    """⚠ 第一版把重试条件写成「信念不是英文才重试」，结果恰恰在**明知道在英文模式**时
    跳过了切换 —— 逻辑正好反了。真机表现：打完 safari 再打中文必然失败。
    决定要不要切的是刚拿到的**证据**（候选栏空不空），不是旧信念。"""
    dev, per, _ = _ime_env(fake_env, "1设置", "2 摄制")
    obs = per.observe(dev.capture())
    ex = Executor(dev, per)
    ex._ime_chinese = False               # 上一步打过英文
    res, _ = ex.run(act("type", text="设置"), obs)
    assert ("toggle_ime",) in dev.calls, f"明知道在英文模式却没切：{dev.calls}"
    assert res.ok, res.hint



def test_letters_that_never_reached_the_screen_are_a_different_diagnosis(fake_env):
    """⚠ 候选栏空**而且拼音也没出现在屏幕上** = 按键根本没进去，
    不是「输入法模式不对」。这两种说法差很远：模式不对可以切一下，
    按键进不去时切几次都没用。

    2026-09-08 批跑踩到：34 次 Spotlight 打字全部没落进去，而报的是
    「字母是直接上屏的」—— 屏幕上一个字母都没有。
    """
    dev, per, _ = _ime_env(fake_env)          # 屏上既没候选也没拼音
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("type", text="设置"), obs)
    assert res.error == "keystrokes_not_landing", res.error
    assert not any(c[0] == "toggle_ime" for c in dev.calls), "字母都没进去，切输入法没意义"


def test_typing_failure_states_the_fact_and_does_not_coach_the_model(fake_env):
    """⚠ 以前这里劝模型：第一次「先 tap 输入框聚焦」，第三次「别再试了」。两种都是把设备故障
    交给模型 —— 09-08 批跑 8 个任务白烧几十步，09-10 改了说法模型照样换 20 种方式试。
    现在只陈述事实；真正的处理在恢复阶梯（recovery.py），有阶梯时模型根本看不到这句。"""
    dev, per, _ = _ime_env(fake_env)
    res, _ = Executor(dev, per).run(act("type", text="设置"), obs=per.observe(dev.capture()))
    assert res.error == "keystrokes_not_landing"
    assert "通道" in res.hint
    for coaching in ("先 tap 输入框", "没有用", "done(failed)"):
        assert coaching not in res.hint, res.hint


# --- 通道失效 → 恢复阶梯 → 重做，对模型透明（harness/recovery.py）---

def test_typing_that_lands_after_the_cheap_rung_is_reported_as_plain_success(fake_env):
    """第一次按键没进去，阶梯第 ① 级（release_all + 重新解析）之后重打就进去了：
    模型只看到一次成功的 type，没有 hint、没有错误；重启一次都没做。
    假设备：release_all 之前只给「没候选没拼音」的帧，之后才给正常的帧 —— 确定性，不靠数帧。"""
    from iphone_agent.harness.recovery import Recovery
    from tests.conftest import FakeDevice, frame_with_text, perceiver_for
    dead = _PAD + ["Q 搜索"]
    alive = _PAD + ["1设置", "2 摄制"] + ["Q 搜索"]
    dead_frames = [frame_with_text(dead, i) for i in range(1, 20)]
    alive_frames = [frame_with_text(alive, i) for i in range(20, 40)] + \
                   [frame_with_text(["设置", "飞行模式"], i) for i in range(40, 60)]

    class DeviceThatWakesOnRelease(FakeDevice):
        def capture(self):
            if self.released and self._frames is not alive_frames:
                self._frames, self._i = alive_frames, 0
            return super().capture()

    dev = DeviceThatWakesOnRelease(dead_frames)
    per = perceiver_for(dead_frames + alive_frames)
    restarts = []
    rec = Recovery(dev, per, restart=lambda: restarts.append(1) or 0.0,
                   ensure=lambda d, p, obs=None: (True, "连接正常", None))
    ex = Executor(dev, per, recovery=rec)
    res, new = ex.run(act("type", text="设置"), per.observe(dev.capture()))
    assert res.ok and res.error is None, (res.error, res.hint)
    assert res.extra.get("picked") == "1设置"
    assert restarts == [], "第 ① 级就救回来了，不该重启镜像"
    assert [e["rung"] for e in rec.events] == ["resolve"]


def test_typing_dead_for_good_climbs_to_restart_then_raises(fake_env):
    """一直不落：resolve → connect → restart 都试过、重启不超过 RECOVERY_MAX_RESTARTS，
    最后抛 DeviceChannelDead —— loop 把它记成 device_error 结束任务，不再让模型烧步数。"""
    from iphone_agent import config
    from iphone_agent.harness.recovery import DeviceChannelDead, Recovery
    dev, per, _ = _ime_env(fake_env)
    restarts = []
    rec = Recovery(dev, per, restart=lambda: restarts.append(1) or 0.5,
                   ensure=lambda d, p, obs=None: (True, "连接正常", None))
    ex = Executor(dev, per, recovery=rec)
    with pytest.raises(DeviceChannelDead) as e:
        ex.run(act("type", text="设置"), per.observe(dev.capture()))
    assert e.value.channel == "keyboard"
    assert [t["rung"] for t in e.value.tried] == ["resolve", "connect", "restart"]
    assert len(restarts) == 1 <= config.RECOVERY_MAX_RESTARTS
    assert "恢复阶梯" in str(e.value) and "镜像" in str(e.value)


def test_open_app_that_could_not_type_is_a_keyboard_channel_failure(fake_env):
    """open_app 打不出字（typed=False）和 type 打不出字是同一个通道的同一件事，
    进同一个阶梯 —— 不是「找不到 App」。"""
    from iphone_agent.harness.recovery import DeviceChannelDead, Recovery
    spotlight = ["Q 搜索", "Siri建议"]
    dev, per, _ = fake_env([spotlight] * 40)
    rec = Recovery(dev, per, restart=lambda: 0.0, ensure=lambda d, p, obs=None: (True, "ok", None),
                   max_restarts=1)
    with pytest.raises(DeviceChannelDead) as e:
        Executor(dev, per, recovery=rec).run(act("open_app", name="设置"), per.observe(dev.capture()))
    assert e.value.channel == "keyboard"


# --- 打字通道死掉时，open_app 还有一条退路 ---
#
# 2026-09-08 批跑实测：打字整条通道死掉（39/40 次失败），而**同一时间**点击和滚动
# 完全正常（日历 tap 6/6、音乐 9/10）。那时候 open_app 只有 Spotlight 一条路，
# 于是整批任务卡死在"打不开 App"上。
# 退路必须走**另一条通道**才有意义：回主屏、翻页、点图标，全程不打字。

def test_open_app_falls_back_to_the_home_screen_icon(fake_env):
    # Spotlight 那一屏没有目标；回主屏之后第一页上就有
    spotlight = ["Q 搜索", "Siri建议"]
    home = ["设置", "微信", "相机"]
    # Spotlight 帧给足：settle 会多抓几帧，给少了打字后的观察会漏到 home 帧，
    # 那时候「设置」在屏幕上，精确匹配直接中，就走不到退路了。
    # 点了图标之后画面得**真的进 App**（2026-09-10 起开没开成看身份：还在主屏就不算开成）。
    from tests.conftest import FakeDevice, frame_with_text, perceiver_for
    f_spot = [frame_with_text(spotlight, i) for i in range(1, 11)]
    f_home = [frame_with_text(home, i) for i in range(11, 30)]
    f_in = [frame_with_text(["通用", "关于本机", "辅助功能"], i) for i in range(30, 45)]

    class Dev(FakeDevice):
        def key(self, n):
            super().key(n)
            if n == "home":
                self._frames, self._i = f_home, 0
        def tap(self, x, y):
            super().tap(x, y)
            if self._frames is f_home:
                self._frames, self._i = f_in, 0

    dev = Dev(f_spot)
    per = perceiver_for(f_spot + f_home + f_in)
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("open_app", name="设置"), obs)
    assert res.ok, res.hint
    assert res.extra.get("via") == "home_icon", res.extra
    assert "主屏" in (res.hint or "")


def test_the_fallback_never_types(fake_env):
    """⚠ 这是退路存在的**全部理由** —— 它必须走另一条通道。
    哪天有人在这条路上加了打字，它就白做了。"""
    dev, per, _ = fake_env([["Q 搜索"]] * 4 + [["设置", "微信"]] * 8)
    obs = per.observe(dev.capture())
    ex = Executor(dev, per)
    before = len([c for c in dev.calls if c[0] == "type"])
    ex._open_app_from_home("设置", obs, typed=False, query="shezhi")
    assert len([c for c in dev.calls if c[0] == "type"]) == before, "退路里打字了"


def test_it_uses_icon_above_because_tapping_the_label_does_nothing(fake_env):
    """主屏幕上 OCR 只读得到图标**下面**的标签，点标签本身打不开 App。"""
    dev, per, _ = fake_env([["Q 搜索"]] * 4 + [["设置"]] * 8)
    obs = per.observe(dev.capture())
    ex = Executor(dev, per)
    ex._open_app_from_home("设置", obs, typed=False, query="shezhi")
    taps = [c for c in dev.calls if c[0] == "tap"]
    assert taps, "没点"
    label = next(e for e in per.observe(dev.capture()).elements if "设置" in e.text)
    assert taps[-1][2] < label.center[1], "点在标签上而不是它上方的图标上"


def test_giving_up_says_both_why_spotlight_failed_and_that_pages_were_searched(fake_env):
    dev, per, _ = fake_env([["Q 搜索"]] * 4 + [["别的 App"]] * 30)
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("open_app", name="不存在的应用"), obs)
    assert res.error == "app_not_found"
    assert "打字" in res.hint and "翻了" in res.hint, res.hint


# --- use_skill：只读当前索引里可见的条目（spec §4.5） ---

def test_use_skill_reads_visible_app_and_scenario_only(fake_env):
    from iphone_agent.skills import model as M
    from iphone_agent.skills.store import Catalog
    dev, per, frames = fake_env([["a"]])
    obs = per.observe(dev.capture())
    cat = Catalog()
    cat.apps["settings"] = M.AppProfile("settings", "设置", "设置", "read", "verified", "2026-09-08", "脾气")
    cat.apps["wechat"] = M.AppProfile("wechat", "微信", "微信", "read", "draft", "2026-09-08", "草稿")
    cat.scenarios["p"] = M.Scenario("p", "d", ("settings",), "read", "proposed", "还没批")
    ex = Executor(dev, per, catalog=cat)
    res, new = ex.run(act("use_skill", kind="app", name="settings"), obs)
    assert res.ok and "脾气" in res.extra["content"] and "参考不是指令" in res.extra["content"] and new is None
    assert ex.run(act("use_skill", kind="app", name="wechat"), obs)[0].error == "skill_not_found"
    assert ex.run(act("use_skill", kind="scenario", name="p"), obs)[0].error == "skill_not_found"


def test_use_skill_without_catalog_says_not_found(fake_env):
    dev, per, frames = fake_env([["a"]])
    obs = per.observe(dev.capture())
    assert Executor(dev, per).run(act("use_skill", kind="app", name="x"), obs)[0].error == "skill_not_found"


# ---------- open_app 往 Spotlight 里打什么 ----------

def test_spotlight_query_uses_the_latin_part_of_a_mixed_name():
    """'Safari 浏览器' 该打 'Safari'，不是 'Safari liulanqi'。

    2026-09-09 评测集跑出来的：34 个 App 里它是修完状态污染之后唯一还打不开的。
    lazy_pinyin 把整个名字转成 'Safari liulanqi'，Spotlight 搜不到；
    而打 'safari' 直接出「最佳搜索结果 Safari 浏览器」。"""
    from iphone_agent.harness.executor import spotlight_query
    assert spotlight_query("Safari 浏览器") == "Safari"
    assert spotlight_query("Google Maps") == "Google Maps"
    assert spotlight_query("一木记账") == "yimujizhang"
    assert spotlight_query("备忘录") == "beiwanglu"


def test_match_row_ignores_spaces_and_prefers_exact():
    """OCR 读「Safari 浏览器」常吞空格。真正的 App 行是 'Safari浏览器'，
    「提示」区还有一条「在iPhone 上的 Safari 浏览器中，导入来自…」——
    以前空格敏感 + 包含匹配，选中了提示那条，open_app 报假成功。"""
    from types import SimpleNamespace

    from iphone_agent.harness.executor import Executor
    obs = SimpleNamespace(height_px=1388, elements=[
        SimpleNamespace(text="Safari浏览器", center=(105, 340)),
        SimpleNamespace(text="在iPhone 上的 Safari 浏览器中，导入来自", center=(300, 985)),
    ])
    hit = Executor._match_row(obs, "Safari 浏览器", exclude_top=False)
    assert hit.text == "Safari浏览器"
    # 精确匹配不受位置限制：exclude_top=True 时（旧逻辑排顶部 25%）也要选中 y=340 那行
    assert Executor._match_row(obs, "Safari 浏览器", exclude_top=True).text == "Safari浏览器"
    # 只有长句时也要能退回包含匹配
    obs.elements = obs.elements[1:]
    assert Executor._match_row(obs, "Safari 浏览器", exclude_top=False).text.startswith("在iPhone")


# --- open_app 开没开成，看「还在不在 Spotlight」，不看画面变没变（2026-09-10 真机）---
#
# 「最佳搜索结果」是一排图标 + 下面的标签，和主屏一样 OCR 只读得到标签，点标签打不开。
# open_app 点了「备忘录」标签报 ok：changed 拿的是按 Spotlight 之前的主屏做基线，主屏→Spotlight
# 当然「变了」。模型对着 Spotlight 又点两下才进去，第一下还是死点。

def test_open_app_retries_the_icon_above_when_the_label_tap_leaves_spotlight_open(fake_env):
    from tests.conftest import FakeDevice, frame_with_text, perceiver_for
    home = ["设置", "微信", "相机"]
    spot = ["最佳搜索结果", "备忘录", "beiwanglu", "在App中搜索"]
    inside = ["备忘录", "你好", "新建备忘录"]
    frames_home = [frame_with_text(home, i) for i in range(1, 6)]
    frames_spot = [frame_with_text(spot, i) for i in range(6, 40)]
    frames_in = [frame_with_text(inside, i) for i in range(40, 60)]

    class Dev(FakeDevice):
        """第一次 tap（标签）画面还是 Spotlight；第二次 tap（图标）才进 App。"""
        def tap(self, x, y):
            super().tap(x, y)
            if sum(1 for c in self.calls if c[0] == "tap") == 2:
                self._frames, self._i = frames_in, 0
        def type(self, t):
            super().type(t)
            self._frames, self._i = frames_spot, 0

    dev = Dev(frames_home)
    per = perceiver_for(frames_home + frames_spot + frames_in)
    obs = per.observe(dev.capture())
    res, new = Executor(dev, per).run(act("open_app", name="备忘录"), obs)
    assert res.ok and res.extra["via"] == "icon_above", res.to_json()
    taps = [c for c in dev.calls if c[0] == "tap"]
    assert len(taps) == 2 and taps[1][2] < taps[0][2], "第二下该点在标签上方的图标位置"
    assert not any("在App中搜索" in e.text for e in new.elements)


def test_open_app_that_never_leaves_spotlight_is_not_a_success(fake_env):
    spot = ["最佳搜索结果", "备忘录", "beiwanglu", "在App中搜索"]
    dev, per, _ = fake_env([["设置", "微信"]] * 3 + [spot] * 60)
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("open_app", name="备忘录"), obs)
    assert not res.ok, "画面从头到尾在 Spotlight，不能报开成了"
    assert res.error == "app_not_found"


# --- 没 tap 过也要验打字：整屏「输入前没有、输入后才有」（2026-09-10）---

def test_type_without_a_prior_tap_is_verified_on_the_whole_screen(fake_env):
    """Spotlight 里打字前面没有 tap（键盘是 key(spotlight) 唤出来的）。以前一律 unknown，
    通道死了阶梯也触发不了。"""
    dev, per, _ = fake_env([["Q 搜索"]] * 3 + [["Q abc", "最佳搜索结果"]] * 8)
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("type", text="abc"), obs)
    assert res.extra["verified"] == "true", res.to_json()


def test_type_that_changes_nothing_anywhere_is_verified_false_without_a_tap(fake_env):
    dev, per, _ = fake_env([["Q 搜索", "Siri建议"]] * 12)
    obs = per.observe(dev.capture())
    res, _ = Executor(dev, per).run(act("type", text="abc"), obs)
    assert res.extra["verified"] == "false", res.to_json()
    assert Executor._channel_failure(act("type", text="abc"), res) == "keyboard"


# --- open_app 进的是不是目标 App：问看图的那一方（2026-09-10 真机假成功）---
#
# 打字通道死了，Spotlight 搜索框里还留着上一个任务打的 beiwanglu。旧结果列表底部恰好有个
# 分组标题「设置」。open_app("设置") 精确匹配中了这个标题：点它没反应（还在 Spotlight）→
# 点它上方的「图标」→ 实际点进了一条备忘录 → 离开了 Spotlight → 报 ok via=icon_above。
# 模型以为进了设置，此后十几步都在收拾这个局面（runs/example-run）。
# 「离开 Spotlight / 离开主屏」只说明有东西被打开了，说明不了打开的是谁。

def _app_of(texts):
    if "Wiamzusr1lp" in texts or "新建备忘录" in texts:
        return "备忘录"
    if "关于本机" in texts:
        return "设置"
    return "主屏"


def test_open_app_that_lands_in_the_wrong_app_is_not_a_success(fake_env):
    """真机那一次的复现：打字没落屏 → 精确匹配中分组标题「设置」→ 点上方进了一条备忘录。
    看图说不是设置 → 不报成功，走不打字的退路（回主屏点图标）→ 真进了设置。"""
    from tests.conftest import FakeDevice, SeesApps, frame_with_text, perceiver_for
    start = ["设置", "微信", "相机", "备忘录"]
    stale = ["最佳搜索结果", "备忘录", "语音备忘录", "beiwanglu", "新备忘录", "导入的备忘录", "设置", "在App中搜索"]
    note = ["返回", "Wiamzusr1lp"]
    inside = ["通用", "关于本机", "辅助功能"]
    f_start = [frame_with_text(start, i) for i in range(1, 5)]
    f_stale = [frame_with_text(stale, i) for i in range(5, 40)]
    f_note = [frame_with_text(note, i) for i in range(40, 60)]
    f_home = [frame_with_text(start, i) for i in range(60, 90)]
    f_in = [frame_with_text(inside, i) for i in range(90, 110)]

    class Dev(FakeDevice):
        def key(self, n):
            super().key(n)
            if n == "spotlight":
                self._frames, self._i = f_stale, 0
            elif n == "home":
                self._frames, self._i = f_home, 0

        def tap(self, x, y):
            super().tap(x, y)
            if self._frames is f_stale and sum(c[0] == "tap" for c in self.calls) == 2:
                self._frames, self._i = f_note, 0       # 第二下「图标」点进了一条备忘录
            elif self._frames is f_home:
                self._frames, self._i = f_in, 0

    frames = f_start + f_stale + f_note + f_home + f_in
    dev, per, asker = Dev(f_start), perceiver_for(frames), SeesApps(frames, _app_of)
    res, new = Executor(dev, per, asker=asker).run(act("open_app", name="设置"), per.observe(dev.capture()))
    assert res.ok and res.extra["via"] == "home_icon", res.to_json()
    assert res.extra["identity"]["verified"] is True
    wrong = res.extra["wrong_app"]
    assert wrong[0]["via"] == "icon_above" and wrong[0]["seen"] == "备忘录", wrong
    assert "关于本机" in new.text_set


def _spotlight_opens_notes_env():
    """Spotlight 路：打字落屏 → 点标签还在 Spotlight → 点上方图标真进了备忘录。"""
    from tests.conftest import FakeDevice, frame_with_text, perceiver_for
    home = ["设置", "微信", "相机"]
    spot = ["最佳搜索结果", "备忘录", "beiwanglu", "在App中搜索"]
    inside = ["备忘录", "你好", "新建备忘录"]
    f_home = [frame_with_text(home, i) for i in range(1, 6)]
    f_spot = [frame_with_text(spot, i) for i in range(6, 40)]
    f_in = [frame_with_text(inside, i) for i in range(40, 60)]

    class Dev(FakeDevice):
        def tap(self, x, y):
            super().tap(x, y)
            if sum(1 for c in self.calls if c[0] == "tap") == 2:
                self._frames, self._i = f_in, 0

        def type(self, t):
            super().type(t)
            self._frames, self._i = f_spot, 0

    frames = f_home + f_spot + f_in
    return Dev(f_home), perceiver_for(frames), frames


def test_open_app_says_whether_the_app_was_verified(fake_env):
    """看图说是 → verified=True；没有看图的那一方 → 照旧报 ok，但 verified=None 看得见（§3）。"""
    from tests.conftest import SeesApps
    dev, per, frames = _spotlight_opens_notes_env()
    res, _ = Executor(dev, per, asker=SeesApps(frames, _app_of)).run(
        act("open_app", name="备忘录"), per.observe(dev.capture()))
    assert res.ok and res.extra["identity"]["verified"] is True, res.to_json()
    assert "wrong_app" not in res.extra

    dev, per, frames = _spotlight_opens_notes_env()
    res, _ = Executor(dev, per).run(act("open_app", name="备忘录"), per.observe(dev.capture()))
    assert res.ok and res.extra["identity"] == {"verified": None}, res.to_json()


def test_home_icon_that_opens_another_app_is_reported_honestly(fake_env):
    """退路在主屏上点了带「设置」字样的东西，进的却是备忘录：不报 ok，
    hint 说清点开的是什么 —— 不能再说「翻了几页也没找到这个图标」，那是另一种失败。"""
    from tests.conftest import FakeDevice, SeesApps, frame_with_text, perceiver_for
    f_spot = [frame_with_text(["Q 搜索", "Siri建议"], i) for i in range(1, 11)]
    f_home = [frame_with_text(["设置", "微信", "相机"], i) for i in range(11, 30)]
    f_note = [frame_with_text(["返回", "Wiamzusr1lp"], i) for i in range(30, 45)]

    class Dev(FakeDevice):
        def key(self, n):
            super().key(n)
            if n == "home":
                self._frames, self._i = f_home, 0

        def tap(self, x, y):
            super().tap(x, y)
            if self._frames is f_home:
                self._frames, self._i = f_note, 0

    frames = f_spot + f_home + f_note
    dev, per = Dev(f_spot), perceiver_for(frames)
    res, _ = Executor(dev, per, asker=SeesApps(frames, _app_of)).run(
        act("open_app", name="设置"), per.observe(dev.capture()))
    assert not res.ok and res.error == "app_not_found", res.to_json()
    assert "备忘录" in res.hint and "翻了" not in res.hint, res.hint
    assert res.extra["wrong_app"][-1]["via"] == "home_icon"
    assert res.extra["typed"] is False, "打字通道失效的判据要保住：阶梯靠它触发"
