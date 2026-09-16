"""handover：把「这一步我不该自己做」做成一个一等动作，交给人。

五个参考项目都有它（docs/31 §3.3：Open-AutoGLM Take_over / Interact、Mobile-Agent interact、
UI-Venus CallUser、OpenGUI call_user、PhoneAgent HUMAN_INPUT）。我们以前只有 done ——
登录、验证码、支付确认、Face ID、拿不准的二选一，要么撞熔断要么 done(failed)。

语义：模型调 handover(need)，主循环把 need 交给人（网页横幅 / 系统对话框），**阻塞**到人做完点继续，
然后重新观察、把新画面交回模型接着做。没有人可交（回调为 None，评测里就是这样）→ 任务以
end_reason="handover" 结束，need 原样带出来。等人的时间不计时限。
"""
import json
import threading
import time

from iphone_agent.harness import safety
from iphone_agent.harness.actions import Action, ValidationError, validate_action
from iphone_agent.harness.guard import ActionGuard
from iphone_agent.web.server import Chat
from tests.test_loop import run

DONE = [("done", {"status": "success", "result": "x"})]
HAND = [("handover", {"need": "请在手机上完成登录，然后点继续"})]


def _tool_results(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


# ---------- 校验 ----------

def test_handover_requires_a_need():
    for bad in ({}, {"need": ""}, {"need": 3}, {"need": "x" * 400}):
        try:
            validate_action(Action("handover", bad, "r", None, "c"), None)
        except ValidationError as e:
            assert e.code == "invalid_args"
        else:
            raise AssertionError(f"{bad} 不该通过")
    a = validate_action(Action("handover", {"need": "登录"}, "r", None, "c"), None)
    assert a.args == {"need": "登录"}


# ---------- 主循环 ----------

def test_without_anyone_to_hand_over_to_the_run_ends_as_handover(fake_env, tmp_path):
    r, dev, m = run(fake_env, [["登录"]] * 4, [HAND, DONE], tmp_path)
    assert r.end_reason == "handover"
    assert "登录" in (r.error or "")
    assert len(m.seen) == 1, "没人可交就该停，不该再问模型"


def test_after_the_human_is_done_the_loop_reobserves_and_continues(fake_env, tmp_path):
    asked = []

    box = {}

    def on_handover(need, reason):
        asked.append((need, reason))
        # 人在手机上操作了：让假设备从此吐出最后那帧（首页）。不靠数初始 settle 吃掉几帧。
        box["dev"]._i = len(box["dev"]._frames) - 1
        return "已经登录好了"

    from iphone_agent.harness.loop import run_task
    from iphone_agent.workspace import Workspace
    from tests.test_loop import INITIAL_SETTLE_FRAMES, ScriptedModel, isolated_store
    specs = [["登录"], ["首页"]]
    dev, per, _ = fake_env([specs[0]] * INITIAL_SETTLE_FRAMES + specs)
    box["dev"] = dev
    m = ScriptedModel([HAND, DONE])
    r = run_task("t", dev, per, m, tmp_path, store=isolated_store(tmp_path),
                 workspace=Workspace(tmp_path), on_handover=on_handover)
    assert r.end_reason == "done_success", r.error
    assert asked and "登录" in asked[0][0]
    results = _tool_results(m.seen[1])
    assert any("已经登录好了" in t for t in results), results
    # 人做完之后画面变了，模型要看到新画面（首页），不能拿着登录页接着想
    texts = " ".join(p.get("text", "") for msg in m.seen[1] if msg["role"] == "user"
                     and isinstance(msg["content"], list) for p in msg["content"])
    assert "首页" in texts


def test_human_declining_ends_the_run_as_handover(fake_env, tmp_path):
    r, dev, m = run(fake_env, [["登录"]] * 4, [HAND, DONE], tmp_path, on_handover=lambda n, r_: None)
    assert r.end_reason == "handover" and len(m.seen) == 1


def test_time_waiting_for_the_human_does_not_count_toward_timeout(fake_env, tmp_path):
    def slow(need, reason):
        time.sleep(0.6)
        return ""

    r, dev, m = run(fake_env, [["登录"]] * 6, [HAND, DONE], tmp_path, on_handover=slow, timeout_s=0.5)
    assert r.end_reason == "done_success", r.error


def test_handover_is_recorded_with_the_need(fake_env, tmp_path):
    r, dev, m = run(fake_env, [["登录"]] * 6, [HAND, DONE], tmp_path, on_handover=lambda n, r_: "")
    recs = [json.loads(line) for line in (r.run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    h = next(x for x in recs if (x.get("action") or {}).get("name") == "handover")
    assert h["result"]["ok"] is True and h["result"].get("resumed") is True


def test_handover_does_not_burn_the_no_progress_budget():
    g = ActionGuard()
    g.record_outcome(Action("handover", {"need": "x"}, "r", None, "c"), changed=False, new_screen_hash=1)
    assert g.no_progress == 0


# ---------- Chat（网页）----------

def test_chat_handover_pauses_and_returns_the_note_on_resume():
    c = Chat()
    c.busy.acquire()
    q = c.subscribe()
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("v", c.handover("请登录", "需要账号")), daemon=True)
    t.start()
    time.sleep(0.05)
    assert t.is_alive() and c.paused
    c.resume("登好了")
    t.join(2)
    assert got["v"] == "登好了"
    ev = q.get_nowait()
    assert ev["kind"] == "handover" and "请登录" in ev["need"]


def test_chat_handover_resumed_without_a_note_returns_empty_string_not_none():
    c = Chat()
    c.busy.acquire()
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("v", c.handover("请登录", "")), daemon=True)
    t.start()
    time.sleep(0.05)
    c.resume("")
    t.join(2)
    assert got["v"] == ""


def test_chat_stop_during_handover_returns_none():
    c = Chat()
    c.busy.acquire()
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("v", c.handover("请登录", "")), daemon=True)
    t.start()
    time.sleep(0.05)
    c.stop()
    t.join(2)
    assert got["v"] is None


# ---------- 命令行：系统对话框 ----------

def test_osascript_handover_passes_need_via_argv_and_maps_buttons(monkeypatch):
    seen = {}

    class R:
        def __init__(self, code, out):
            self.returncode, self.stdout = code, out

    def fake(code, out):
        def run(argv, **kw):
            seen["argv"] = argv
            return R(code, out)
        return run

    monkeypatch.setattr(safety.subprocess, "run", fake(0, "ok"))
    assert safety.osascript_handover('请登录 "x"') == ""
    assert seen["argv"][-1] == '请登录 "x"' and "请登录" not in " ".join(seen["argv"][:-1])
    monkeypatch.setattr(safety.subprocess, "run", fake(1, ""))
    assert safety.osascript_handover("请登录") is None
