from iphone_agent import config
from iphone_agent.harness.actions import Action
from iphone_agent.harness.guard import ActionGuard, signature


def act(name, **args):
    return Action(name, args, "r", None, "c")


def test_signature_buckets_coords():
    # 分桶是向下取整（floor），不是就近取整：相邻两点仍可能跨桶边界落进不同桶
    # （例如 100 和 120，桶宽 40 时分别落在 [80,120) 与 [120,160)）。这是可接受的
    # 漏拦（安全方向——本该判定重复的动作没被拦下，动作仍会执行，无进展计数兜底），
    # 而不是误拦。这里选同一桶内的两个点（100 与 110，桶宽 40 时都在 [80,120)）
    # 来验证分桶本身生效。
    s1 = signature(act("tap", x=100, y=100), 7, 800, 1700)
    s2 = signature(act("tap", x=110, y=110), 7, 800, 1700)   # 同一 5% 桶（40x85）
    s3 = signature(act("tap", x=300, y=100), 7, 800, 1700)
    assert s1 == s2 and s1 != s3
    assert signature(act("tap", x=100, y=100), 8, 800, 1700) != s1  # 不同画面


def test_repeat_detected_only_after_executed():
    g = ActionGuard()
    a = act("scroll", direction="down", amount="page")
    assert not g.check_repeat(a, 1, 800, 1700)
    g.record_executed(a, 1, 800, 1700)
    assert g.check_repeat(a, 1, 800, 1700)
    assert not g.check_repeat(a, 2, 800, 1700)  # 画面变了，同动作允许


def test_no_progress_warn_then_stop_and_reset():
    g = ActionGuard()
    a = act("tap", x=1, y=1)
    assert [g.record_outcome(a, False) for _ in range(2)] == [None, None]
    assert g.record_outcome(a, False) == "warn"
    assert g.record_outcome(a, False) is None
    assert g.record_outcome(a, False) is None
    assert g.record_outcome(a, False) == "stop"
    assert g.record_outcome(a, True) is None and g.no_progress == 0


def test_observe_and_wait_do_not_count_or_reset():
    g = ActionGuard()
    a = act("tap", x=1, y=1)
    g.record_outcome(a, False); g.record_outcome(a, False)
    g.record_outcome(act("wait", seconds=1), True)
    g.record_outcome(act("observe"), False)
    assert g.no_progress == 2


# --- 二状态震荡（2026-09-07 验收第 5 次）---
# 主屏 App 页 ⇄ 小组件页来回七步，aHash 逐步是 A B A B A B A。
# 两道熔断都没拦住：每一步画面都真的变了，于是 record_outcome 每步都清零。
# 「画面变了」不等于「有进展」——回到一个见过的画面，是原地打转。

def test_returning_to_a_seen_screen_is_not_progress():
    g = ActionGuard()
    a = act("scroll", direction="left", amount="page")
    g.record_screen(0xA)                       # 起点 A
    assert g.record_outcome(a, True, 0xB) is None   # A→B，B 是新画面，算进展
    assert g.no_progress == 0
    g.record_outcome(a, True, 0xA)             # B→A，A 见过
    g.record_outcome(a, True, 0xB)             # A→B，B 见过
    assert g.no_progress == 2


def test_two_state_oscillation_eventually_stops():
    """验收第 5 次那七步：到第 6 次原地打转必须叫停。"""
    g = ActionGuard()
    a = act("scroll", direction="left", amount="page")
    g.record_screen(0xA)
    g.record_outcome(a, True, 0xB)             # 唯一一次真进展
    verdicts = [g.record_outcome(a, True, 0xA if i % 2 == 0 else 0xB) for i in range(6)]
    assert verdicts[config.NO_PROGRESS_WARN - 1] == "warn"
    assert verdicts[config.NO_PROGRESS_STOP - 1] == "stop"


def test_new_screen_still_resets_progress():
    """走进没见过的画面仍然是进展——修复不能把正常前进也判成打转。"""
    g = ActionGuard()
    a = act("tap", x=1, y=1)
    g.record_screen(0xA)
    g.record_outcome(a, True, 0xB)
    g.record_outcome(a, True, 0xA)             # 返回上一级，见过
    assert g.no_progress == 1
    assert g.record_outcome(a, True, 0xC) is None   # 进了另一个新条目
    assert g.no_progress == 0


# --- 中文输入的多步过程不该吃掉无进展预算 ---
#
# 输一个 App 名要六到八步（打拼音 → 看候选 → 点候选 → 打下一段），全在同一片
# 区域里折腾。留在无进展计数里，三步就到 WARN，六步就被终止 —— 而任务还没开始做。

def test_type_does_not_burn_the_no_progress_budget():
    g = ActionGuard()
    for _ in range(config.NO_PROGRESS_STOP + 2):
        assert g.record_outcome(act("type", text="yi"), changed=False) is None
    assert g.no_progress == 0


def test_type_is_still_caught_by_same_screen_dedup():
    """豁免的只是无进展那一道。往一个没聚焦的输入框反复打同样的字 ——
    正是同屏去重该拦的，不能一起放掉。"""
    g = ActionGuard()
    a = act("type", text="yi")
    assert not g.check_repeat(a, 1, 800, 1700)
    g.record_executed(a, 1, 800, 1700)
    assert g.check_repeat(a, 1, 800, 1700), "同一画面上重复打同样的字，没拦住"
    assert not g.check_repeat(a, 2, 800, 1700), "画面变了就该放行"


def test_other_actions_still_count():
    """别把豁免开太宽 —— tap 无进展照样得计。"""
    g = ActionGuard()
    for _ in range(config.NO_PROGRESS_WARN):
        r = g.record_outcome(act("tap", x=1, y=1), changed=False)
    assert r == "warn" and g.no_progress == config.NO_PROGRESS_WARN


# --- 两种「无进展」必须分开说 ---
#
# 2026-09-08 真机踩到（runs/example-run）：模型在底部 tab 之间来回点，
# 每一步 changed=True，画面真的变了，熔断判「无进展」判**对**了 —— 回到见过的
# 画面就是原地打转。但给它的说法是「连续 6 步没有检测到画面变化」。
# 于是模型以为自己**没点中**，换了三个不同的图标继续试，在错误的假设上白烧六步。

def test_oscillating_between_seen_screens_says_revisited():
    from iphone_agent.harness.guard import no_progress_hint
    g = ActionGuard()
    g.record_screen(1)
    g.record_screen(2)
    g.record_outcome(act("tap", x=1, y=1), changed=True, new_screen_hash=2)
    assert g.last_reason == "revisited"
    says = no_progress_hint(g.last_reason)
    assert "到过的画面" in says and "生效" in says, says
    assert "没点中" not in says, "画面明明变了，不能说没点中"


def test_nothing_happening_says_unchanged():
    from iphone_agent.harness.guard import no_progress_hint
    g = ActionGuard()
    g.record_outcome(act("tap", x=1, y=1), changed=False, new_screen_hash=9)
    assert g.last_reason == "unchanged"
    assert "没点中" in no_progress_hint(g.last_reason)


def test_real_progress_clears_the_reason():
    g = ActionGuard()
    g.record_outcome(act("tap", x=1, y=1), changed=False, new_screen_hash=1)
    assert g.last_reason == "unchanged"
    g.record_outcome(act("tap", x=2, y=2), changed=True, new_screen_hash=7)
    assert g.last_reason is None and g.no_progress == 0


# --- 点击整个不生效，要说出来 ---
#
# 2026-09-08 撞到的真实状态：镜像窗口滚轮还灵、key 还灵，**点击整个不生效**。
# SkyLight 后台点、CGEvent 前台点、HID 全局点全试了；指针位置对、按键没卡、
# 权限齐、镜像 App 重启过 —— 都没用。
# 这时模型收到的是「没检测到变化，可能没点中」，于是它换目标、换方式、
# 一路试到熔断，20 步全废在一件它根本改变不了的事情上。

def test_three_dead_taps_are_counted():
    g = ActionGuard()
    for i in range(3):
        g.record_outcome(act("tap", x=i, y=i), changed=False, new_screen_hash=1)
    assert g.dead_taps == 3 >= config.DEAD_TAPS_ALERT


def test_a_tap_that_works_resets_the_count():
    g = ActionGuard()
    g.record_outcome(act("tap", x=1, y=1), changed=False, new_screen_hash=1)
    g.record_outcome(act("tap", x=2, y=2), changed=True, new_screen_hash=2)
    assert g.dead_taps == 0


def test_non_tap_actions_do_not_count():
    """滚动没反应是另一回事（可能真到底了），不能混进来。"""
    g = ActionGuard()
    for _ in range(4):
        g.record_outcome(act("scroll", direction="down"), changed=False, new_screen_hash=1)
    assert g.dead_taps == 0


def test_dead_taps_are_counted_for_the_recovery_ladder_not_for_a_hint():
    """攒到 DEAD_TAPS_ALERT 的后果是 loop 走恢复阶梯（recovery.py），不是一句劝模型的话 ——
    guard 里不该再有那段文案。"""
    import iphone_agent.harness.guard as guard_mod
    assert not hasattr(guard_mod, "dead_taps_hint")
    g = ActionGuard()
    for i in range(config.DEAD_TAPS_ALERT):
        g.record_outcome(act("tap", x=i, y=i), changed=False, new_screen_hash=1)
    assert g.dead_taps >= config.DEAD_TAPS_ALERT


def test_change_below_fingerprint_resolution_is_not_a_revisit():
    """开关翻转：汉明 0、文字差 0，只有局部 MAD 看得见。指纹分不清两个状态，就没资格说「见过」。"""
    from types import SimpleNamespace
    g = ActionGuard()
    obs = SimpleNamespace(state_key=1)
    g.record_screen(1)
    res = SimpleNamespace(ok=True, changed=True, extra={"local_mad": 17.0})
    assert g.record_result(act("tap", x=1, y=1), res, obs) is None
    assert g.no_progress == 0 and g.last_reason is None
    res = SimpleNamespace(ok=True, changed=True, extra={"judged": {"worked": True}})
    g.record_result(act("tap", x=1, y=1), res, obs)
    assert g.no_progress == 0


def test_record_result_still_counts_plain_revisits_and_no_change():
    from types import SimpleNamespace
    g = ActionGuard()
    g.record_screen(1)
    g.record_result(act("tap", x=1, y=1), SimpleNamespace(ok=True, changed=True, extra={}), SimpleNamespace(state_key=1))
    assert g.no_progress == 1 and g.last_reason == "revisited"
    g.record_result(act("tap", x=2, y=2), SimpleNamespace(ok=True, changed=False, extra={}), SimpleNamespace(state_key=1))
    assert g.no_progress == 2 and g.last_reason == "unchanged"
    assert g.record_result(act("tap", x=3, y=3), SimpleNamespace(ok=False, changed=None, extra={}), None) is None
    assert g.no_progress == 2, "失败的动作不记账"
