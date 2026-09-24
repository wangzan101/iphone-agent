"""剧本执行。用「屏幕状态机」而不是帧序列当假设备：每个动作把屏幕推进到下一屏，
抓帧永远返回当前屏 —— settle 抓几帧都一样，不用数帧。"""
import pytest

from iphone_agent import config
from iphone_agent.driver.geometry import Rect
from iphone_agent.harness.executor import Executor
from iphone_agent.harness.guard import ActionGuard
from iphone_agent.harness.procedure import COUNTED_FAILURES, ProcedureRunner
from iphone_agent.harness.runlog import RunLog
from iphone_agent.perceive.observe import Perceiver
from iphone_agent.perceive.ocr import RawBox
from iphone_agent.skills import model as M
from tests.conftest import frame_with_text


class Sim:
    def __init__(self, screens):
        self.screens = list(screens)
        self.i = 0
        self.fid = 0
        self.calls = []
        self.rect = Rect(0, 0, 200, 400)
        self.texts = {}
        self.released = False
        self.raise_on = {}

    def capture(self):
        self.fid += 1
        f, t = frame_with_text(self.screens[min(self.i, len(self.screens) - 1)], self.fid)
        self.texts[self.fid] = t
        return f

    def window_rect(self):
        return self.rect

    def ensure_frame_valid(self, frame):
        pass

    def _act(self, name, *a):
        if name in self.raise_on:
            raise self.raise_on[name]
        self.calls.append((name, *a))
        self.i += 1

    def tap(self, x, y): self._act("tap", x, y)
    def scroll(self, d, a): self._act("scroll", d, a)
    def type(self, t): self._act("type", t)
    def key(self, n): self._act("key", n)
    def toggle_ime(self): self._act("toggle_ime")
    def backspace(self, n): self._act("backspace", n)
    def select_all_and_delete(self): self._act("select_all_and_delete")
    def enter(self): self._act("enter")
    def release_all(self): self.released = True


def sim_perceiver(sim):
    """文字竖着堆（与 conftest.perceiver_for 同一套几何）；含 '|' 的文字拆成同一行的几个元素。"""
    def ocr(img):
        out = []
        for i, t in enumerate(sim.texts.get(ocr._fid, [])):
            parts = t.split("|")
            for j, p in enumerate(parts):
                out.append(RawBox(p, 0.99, 0.1 + j * 0.3, 0.85 - i * 0.055, 0.25, 0.04))
        return out
    per = Perceiver(ocr=ocr)
    orig = per.observe

    def observe(frame, **kw):
        ocr._fid = frame.frame_id
        return orig(frame, **kw)
    per.observe = observe
    return per


def proc(steps, params=None, returns=None):
    return M.Procedure(name="p", app="settings", description="d", params=params or {}, returns=returns or [],
                       risk="read", status="verified", expect_source="intersection",
                       provenance=M.Provenance(), steps=steps)


@pytest.fixture
def harness(tmp_path):
    def make(screens, deadline_offset=600):
        import time
        sim = Sim(screens)
        per = sim_perceiver(sim)
        ex = Executor(sim, per)
        guard = ActionGuard()
        log = RunLog(tmp_path, task="t", model="m", config_snapshot={}, prompt_hash="h")
        runner = ProcedureRunner(ex, guard, log, deadline=time.time() + deadline_offset)
        obs = per.observe(sim.capture())
        guard.record_screen(obs.ahash)
        return sim, runner, obs, log
    return make


def test_happy_path_returns_read_value_and_logs_substeps(harness):
    sim, runner, obs, log = harness([["设置", "通用"], ["关于本机", "软件更新"], ["iOS版本|18.3.1", "型号名称"]])
    p = proc([M.Step(do="tap", target="通用", expect=("关于本机",)),
              M.Step(do="tap", target="关于本机", expect=("iOS版本",)),
              M.Step(do="read", row="iOS版本", as_="version")], returns=["version"])
    res, new = runner.run(p, {}, obs, "c1", "settings__p")
    assert res.ok and res.extra["returns"] == {"version": "18.3.1"}
    assert res.extra["steps_done"] == 3 and res.extra["actions"] == 2
    assert new is not None and "iOS版本" in new.text_set
    subs = [s for s in RunLog.read_steps(log.dir) if s.get("kind") == "procedure_step"]
    assert [s["step"] for s in subs] == [1, 2] and all(s["parent_call_id"] == "c1" for s in subs)
    assert subs[0]["expect_ok"] is True and subs[0]["action"]["name"] == "tap"
    # ⚠ 2026-09-11 final review：子记录没写 ts，twin/events.py 兜底成 0.0，
    #   剧本碰过的屏在孪生里全带 1970 年戳。子记录必须自带真实时间戳。
    assert all(isinstance(s.get("ts"), (int, float)) and s["ts"] > 0 for s in subs)


def test_expect_mismatch_hands_back_current_screen(harness):
    sim, runner, obs, log = harness([["设置", "通用"], ["别的屏", "x"]])
    p = proc([M.Step(do="tap", target="通用", expect=("关于本机", "软件更新"))])
    res, new = runner.run(p, {}, obs, "c1", "settings__p")
    assert not res.ok and res.error == "expect_mismatch" and res.error in COUNTED_FAILURES
    assert res.extra["step"] == 1 and res.extra["steps_done"] == 0 and res.extra["expected"] == ["关于本机", "软件更新"]
    assert "从这儿接着手动做" in res.hint and new is not None and "别的屏" in new.text_set


# 2026-09-07 提取、2026-09-16 真机上一跑就废的那条指纹（一次真机运行的第 1 步）：
# 4 个真锚点 + 状态栏时钟 + 3 个只有整屏解析才有的图标名。
LEGACY = ("今 ④5", "设置", "蜂窝网络", "向右箭头", "个人热点",
          "电池图标（绿色电池）", "电池", "VPN图标（蓝色地球仪）")


def test_legacy_fingerprint_ignores_status_bar_and_vision_only_junk(harness):
    """老剧本里那些永远对不上的词不该再让整条剧本中止：真锚点还在，就算到了该到的屏。"""
    sim, runner, obs, log = harness([["设置", "通用"], ["设置|蜂窝网络", "个人热点|电池"]])
    res, _ = runner.run(proc([M.Step(do="tap", target="通用", expect=LEGACY)]), {}, obs, "c1", "settings__p")
    assert res.ok and res.extra["steps_done"] == 1


def test_a_really_different_screen_still_fails(harness):
    """过滤掉对不上的词不等于不再核对：真锚点一个都不在，照旧 expect_mismatch。"""
    sim, runner, obs, log = harness([["设置", "通用"], ["微信|通讯录", "发现|我"]])
    res, _ = runner.run(proc([M.Step(do="tap", target="通用", expect=LEGACY)]), {}, obs, "c1", "settings__p")
    assert not res.ok and res.error == "expect_mismatch"


def test_mismatch_hint_shows_expected_and_actual(harness):
    """失败要能被看见（CLAUDE.md §3）：只说「期望什么」，下一个人还得自己去翻留档才知道当时在哪。"""
    sim, runner, obs, log = harness([["设置", "通用"], ["微信|通讯录", "发现|我"]])
    res, _ = runner.run(proc([M.Step(do="tap", target="通用", expect=("关于本机", "软件更新"))]),
                        {}, obs, "c1", "settings__p")
    assert "关于本机" in res.hint and "软件更新" in res.hint
    assert "微信" in res.hint and "通讯录" in res.hint
    assert res.extra["expected"] == ["关于本机", "软件更新"] and res.extra["matched"] == []
    assert "微信" in res.extra["actual"]


def test_target_not_found_without_find_does_not_scroll(harness):
    sim, runner, obs, log = harness([["设置"]])
    res, new = runner.run(proc([M.Step(do="tap", target="通用", expect=())]), {}, obs, "c1", "settings__p")
    assert res.error == "target_not_found" and sim.calls == [] and new is None


def test_find_scrolls_until_target_appears(harness):
    sim, runner, obs, log = harness([["设置", "a"], ["设置", "b"], ["设置", "通用"], ["关于本机"]])
    p = proc([M.Step(do="tap", target="通用", find={"direction": "down", "max_screens": 3}, expect=("关于本机",))])
    res, _ = runner.run(p, {}, obs, "c1", "settings__p")
    assert res.ok and [c[0] for c in sim.calls] == ["scroll", "scroll", "tap"] and res.extra["actions"] == 3


def test_find_exhausted_reports_moved_screens(harness):
    sim, runner, obs, log = harness([["设置", "a"], ["设置", "b"], ["设置", "c"]])
    p = proc([M.Step(do="tap", target="通用", find={"direction": "down", "max_screens": 1}, expect=())])
    res, new = runner.run(p, {}, obs, "c1", "settings__p")
    assert res.error == "target_not_found" and res.extra["moved_screens"] == 1 and new is not None


def test_ambiguous_target_fails_instead_of_picking_first(harness):
    sim, runner, obs, log = harness([["通用设置", "通用"]])
    res, _ = runner.run(proc([M.Step(do="tap", target="通用", expect=())]), {}, obs, "c1", "settings__p")
    assert res.error == "ambiguous_target" and sim.calls == []


def test_action_cap_stops_the_procedure(harness, monkeypatch):
    monkeypatch.setattr("iphone_agent.harness.procedure.config.PROC_MAX_ACTIONS", 1)
    sim, runner, obs, log = harness([["a", "b"], ["c", "d"], ["e", "f"]])
    p = proc([M.Step(do="tap", target="a", expect=()), M.Step(do="tap", target="c", expect=())])
    res, _ = runner.run(p, {}, obs, "c1", "settings__p")
    assert res.error == "procedure_limit" and res.extra["step"] == 2 and res.extra["steps_done"] == 1


def test_repeated_action_on_unchanged_screen_is_refused(harness):
    sim, runner, obs, log = harness([["设置", "通用"], ["设置", "通用"], ["设置", "通用"]])
    p = proc([M.Step(do="tap", target="通用", expect=()), M.Step(do="tap", target="通用", expect=())])
    res, _ = runner.run(p, {}, obs, "c1", "settings__p")
    assert res.error == "repeated_action" and len([c for c in sim.calls if c[0] == "tap"]) == 1


def test_device_error_is_not_a_counted_failure(harness):
    sim, runner, obs, log = harness([["设置", "通用"]])
    sim.raise_on["tap"] = RuntimeError("boom")
    res, _ = runner.run(proc([M.Step(do="tap", target="通用", expect=())]), {}, obs, "c1", "settings__p")
    assert res.error == "device_error" and res.error not in COUNTED_FAILURES


def test_params_are_substituted_into_targets(harness):
    sim, runner, obs, log = harness([["设置", "通用"], ["关于本机"]])
    p = proc([M.Step(do="tap", target="{项}", expect=("关于本机",))], params={"项": {"type": "string"}})
    res, _ = runner.run(p, {"项": "通用"}, obs, "c1", "settings__p")
    assert res.ok


def test_read_requires_unique_label_and_takes_right_side_only(harness):
    """两种失败必须分开报：命中多个是「不敢猜」，一个没命中是「这屏没有这一行」——
    两个错误码都计入剧本的 fail_streak，报错报反了就是在冤枉另一件事。"""
    from iphone_agent.harness.procedure import _StepFailed, read_row
    sim, runner, obs, log = harness([["iOS版本|18.3.1", "iOS版本号|x"]])
    with pytest.raises(_StepFailed) as many:
        read_row(obs, "iOS版本")          # 两个元素都含「iOS版本」
    assert many.value.error == "ambiguous_target" and many.value.error in COUNTED_FAILURES
    with pytest.raises(_StepFailed) as none:
        read_row(obs, "电池健康")          # 这屏一个都没有
    assert none.value.error == "target_not_found" and none.value.error in COUNTED_FAILURES
    assert read_row(obs, "iOS版本号") == "x"


def test_no_progress_stop_is_handed_to_the_caller(harness):
    """熔断的判决是一次性的（record_outcome 只在计数**正好等于**阈值那一下报）。
    剧本要是自己吃掉这一枪，计数器继续往上爬，主循环就再也收不到 stop —— 熔断被拆了。"""
    sim, runner, obs, log = harness([["设置", "通用"]])
    runner.guard.no_progress = config.NO_PROGRESS_STOP - 1
    res, _ = runner.run(proc([M.Step(do="tap", target="通用", expect=())]), {}, obs, "c1", "settings__p")
    assert res.error == "no_progress" and res.extra["guard_verdict"] == "stop"


def test_no_progress_warn_is_not_swallowed(harness):
    """第 3 次无进展的 warn 落在剧本里，模型就永远收不到这条提醒了。"""
    sim, runner, obs, log = harness([["设置", "通用"]])
    runner.guard.no_progress = config.NO_PROGRESS_WARN - 1
    res, _ = runner.run(proc([M.Step(do="tap", target="通用", expect=())]), {}, obs, "c1", "settings__p")
    assert res.ok and res.extra["guard_verdict"] == "warn"


def test_device_error_after_find_hands_back_the_post_scroll_screen(harness):
    """交回去的必须是**失败那一刻**的屏：滚了两屏才出错，给的还是滚之前那张，
    模型照着「当前屏幕已交给你」接着做就是在照一张过期的图操作。"""
    sim, runner, obs, log = harness([["设置", "a"], ["设置", "b"], ["设置", "通用"]])
    sim.raise_on["tap"] = RuntimeError("boom")
    p = proc([M.Step(do="tap", target="通用", find={"direction": "down", "max_screens": 3}, expect=())])
    res, new = runner.run(p, {}, obs, "c1", "settings__p")
    assert res.error == "device_error"
    assert new is not None and "通用" in new.text_set and "a" not in new.text_set


def test_failed_scroll_still_counts_as_a_moved_screen(harness):
    sim, runner, obs, log = harness([["设置", "a"], ["设置", "b"]])
    sim.raise_on["scroll"] = RuntimeError("boom")
    p = proc([M.Step(do="tap", target="通用", find={"direction": "down", "max_screens": 3}, expect=())])
    res, _ = runner.run(p, {}, obs, "c1", "settings__p")
    assert res.error == "device_error" and res.extra["moved_screens"] == 1


def test_a_brace_in_the_executor_hint_still_returns_a_result(harness):
    """执行器的 hint 里嵌着模型/剧本给的字符串，含 { } 很正常。
    拿它去 .format() 会在失败处理里再炸一次，调用方连 ToolResult 都拿不到。"""
    sim, runner, obs, log = harness([["设置", "通用"]])
    res, _ = runner.run(proc([M.Step(do="type", text="「{x}」", expect=())]), {}, obs, "c1", "settings__p")
    assert not res.ok and res.error == "pinyin_failed" and "{x}" in res.hint


def test_read_step_also_checks_the_deadline(harness):
    sim, runner, obs, log = harness([["iOS版本|18.3.1"]], deadline_offset=-1)
    p = proc([M.Step(do="read", row="iOS版本", as_="version")], returns=["version"])
    res, _ = runner.run(p, {}, obs, "c1", "settings__p")
    assert res.error == "procedure_limit" and res.extra["step"] == 1


# ⚠ 2026-09-18（Task 7 评审发现 1、2）：剧本内部动作的 tap expect 恒为 None（procedure.py 造 Action 那处），
#   结论于是恒是 unknown —— 但剧本每一步都用指纹自己核对过，说它「没人核实」是错的。
#   不排除的话：进剧本时的 no_progress 会原样带着走完整条剧本（本来第一个到新画面的动作就该清零），
#   而且新的 "unverified" 会盖掉 guard_verdict 里的 warn（那几行是按 warn → stop 单调升级写的）。

def test_a_replay_that_reaches_new_screens_clears_the_no_progress_count(harness):
    sim, runner, obs, log = harness([["设置", "通用"], ["关于本机", "软件更新"], ["iOS版本|18.3.1", "型号名称"]])
    runner.guard.no_progress = 2
    p = proc([M.Step(do="tap", target="通用", expect=("关于本机",)),
              M.Step(do="tap", target="关于本机", expect=("iOS版本",))])
    res, _ = runner.run(p, {}, obs, "c1", "settings__p")
    assert res.ok and res.extra["steps_done"] == 2
    assert runner.guard.no_progress == 0 and runner.guard.unverified_streak == 0


def test_a_replay_never_hands_back_an_unverified_verdict(harness):
    sim, runner, obs, log = harness([["设置", "通用"], ["关于本机", "软件更新"],
                                     ["iOS版本|18.3.1", "型号名称"], ["电池|健康", "充电"]])
    p = proc([M.Step(do="tap", target="通用", expect=("关于本机",)),
              M.Step(do="tap", target="关于本机", expect=("iOS版本",)),
              M.Step(do="tap", target="型号名称", expect=("电池",))])
    res, _ = runner.run(p, {}, obs, "c1", "settings__p")
    assert res.ok and res.extra.get("guard_verdict") is None


def test_a_warn_inside_a_replay_survives_the_steps_that_follow(harness):
    """warn 之后又走了几步到新画面：那几步不该把 warn 顶掉（它们是「自己核对过」的，不是 unknown）。
    ⚠ 单调升级那几行（procedure.py:217-220）的前提就是判决只有 warn / stop 两级。"""
    same = ["设置", "通用|关于本机"]
    sim, runner, obs, log = harness([same, same, ["一", "二"], ["三", "四"], ["五", "六"]])
    runner.guard.no_progress = config.NO_PROGRESS_WARN - 1
    p = proc([M.Step(do="tap", target="通用", expect=("设置",)),        # 画面没变 → 第 3 次无进展 → warn
              # 后面三步都到新画面（换目标绕开同屏去重）。排除之前，第三步会报 "unverified" 把 warn 顶掉。
              M.Step(do="tap", target="关于本机", expect=("一",)),
              M.Step(do="tap", target="一", expect=("三",)),
              M.Step(do="tap", target="三", expect=("五",))])
    res, _ = runner.run(p, {}, obs, "c1", "settings__p")
    assert res.ok and res.extra["steps_done"] == 4
    assert res.extra.get("guard_verdict") in (None, "warn"), "不能是 unverified"
