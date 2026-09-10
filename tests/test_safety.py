"""安全硬闸：写类动作在执行器侧拦下来问人，没人就拒绝；不可逆的一律拒绝。

为什么必须在程序侧而不是只靠提示词：UI-Venus-2 技术报告 Table 6（设计说明），
「良性指令 + 屏幕里潜藏攻击」下所有通用 VLM 的攻击成功率 79–94%，专门训过的仍 48%。
我们的产品形态恰恰是「用户接一个未经安全训练的通用模型」，提示词层的上限已经被测出来了。
三档照 PhoneHarness（设计说明）：read 直接做 / write 有人确认才做 / never 谁说都不做。
设计说明 的「无人只读、有人可写」由此落地：confirm 回调为 None 就是无人。
"""
import json
import threading
import time

from iphone_agent.harness import safety
from iphone_agent.harness.actions import Action
from iphone_agent.web.server import Chat
from tests.test_loop import run

DONE = [("done", {"status": "success", "result": "x"})]


def act(action_name, **args):
    return Action(action_name, args, "r", None, "c")


class _Obs:
    def __init__(self, *texts):
        from iphone_agent.perceive.elements import Element
        self.elements = [Element(i + 1, t, 0.9, (0, 0, 10, 10), (5, 5)) for i, t in enumerate(texts)]

    def element(self, eid):
        return next((e for e in self.elements if e.id == eid), None)


# ---------- 分级 ----------

def test_tap_on_send_is_write():
    assert safety.classify(act("tap", id=1), _Obs("发送")).level == "write"


def test_tap_on_transfer_is_never():
    assert safety.classify(act("tap", id=1), _Obs("转账")).level == "never"


def test_tap_on_plain_text_is_read():
    assert safety.classify(act("tap", id=1), _Obs("通用")).level == "read"


def test_row_target_uses_the_row_text():
    assert safety.classify(act("tap", id=1, target="row_right"), _Obs("删除账户")).level == "write"


def test_non_tap_actions_are_read():
    obs = _Obs("发送")
    for a in (act("type", text="发送"), act("scroll", direction="down", amount="page"),
              act("open_app", name="微信"), act("done", status="success", result="x")):
        assert safety.classify(a, obs).level == "read", a.name


def test_coordinate_tap_cannot_be_classified_and_says_so():
    """没有文字就没法判 —— 这是已知盲点，要显形（level=read 但 target 标明 unknown）。"""
    d = safety.classify(act("tap", x=10, y=10), _Obs("发送"))
    assert d.level == "read" and d.target == "(x,y)"


# ---------- 主循环 ----------

def _safety_records(r):
    return [json.loads(line) for line in (r.run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
            if '"safety"' in line]


def test_unattended_blocks_a_write(fake_env, tmp_path):
    r, dev, m = run(fake_env, [["发送"]] * 4, [[("tap", {"id": 1})], DONE], tmp_path)
    assert not any(c[0] == "tap" for c in dev.calls), dev.calls
    recs = _safety_records(r)
    assert recs and recs[0]["safety"] == {"level": "write", "decision": "blocked_unattended", "target": "发送"}
    assert recs[0]["validation"] == "blocked_unattended"


def test_attended_and_denied_does_not_execute(fake_env, tmp_path):
    asked = []

    def confirm(text):
        asked.append(text)
        return False

    r, dev, m = run(fake_env, [["发送"]] * 4, [[("tap", {"id": 1})], DONE], tmp_path, confirm=confirm)
    assert asked and "发送" in asked[0]
    assert not any(c[0] == "tap" for c in dev.calls)
    assert _safety_records(r)[0]["safety"]["decision"] == "denied"


def test_attended_and_allowed_executes_and_is_recorded(fake_env, tmp_path):
    r, dev, m = run(fake_env, [["发送"], ["已发送"], ["已发送"], ["已发送"]],
                    [[("tap", {"id": 1})], DONE], tmp_path, confirm=lambda t: True)
    assert any(c[0] == "tap" for c in dev.calls)
    rec = _safety_records(r)[0]
    assert rec["safety"]["decision"] == "confirmed" and "validation" not in rec


def test_never_is_blocked_even_when_someone_is_there(fake_env, tmp_path):
    asked = []
    r, dev, m = run(fake_env, [["转账"]] * 4, [[("tap", {"id": 1})], DONE], tmp_path,
                    confirm=lambda t: asked.append(t) or True)
    assert not asked, "不可逆的动作不该拿去问，问了等于把责任推给人"
    assert not any(c[0] == "tap" for c in dev.calls)
    assert _safety_records(r)[0]["safety"]["decision"] == "blocked_never"


def test_read_actions_leave_no_safety_record(fake_env, tmp_path):
    r, dev, m = run(fake_env, [["通用"], ["关于本机"], ["关于本机"], ["关于本机"]],
                    [[("tap", {"id": 1})], DONE], tmp_path)
    assert _safety_records(r) == []


# ---------- macOS 原生确认框 ----------

def test_osascript_confirm_passes_text_as_argv_not_in_the_script(monkeypatch):
    seen = {}

    class R:
        returncode = 0
        stdout = "ok"

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return R()

    monkeypatch.setattr(safety.subprocess, "run", fake_run)
    assert safety.osascript_confirm('点 "发送" ; do shell script "rm"') is True
    argv = seen["argv"]
    assert argv[0] == "osascript"
    assert argv[-1] == '点 "发送" ; do shell script "rm"', "文本只能走 argv，绝不能拼进脚本"
    script = " ".join(argv[:-1])
    assert "rm" not in script and "发送" not in script


def test_osascript_confirm_denies_on_cancel_or_timeout(monkeypatch):
    class R:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(safety.subprocess, "run", lambda argv, **kw: R())
    assert safety.osascript_confirm("x") is False


# ---------- Chat ----------

def test_chat_confirm_blocks_until_answered():
    c = Chat()
    q = c.subscribe()
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("ok", c.confirm("要点「发送」")), daemon=True)
    t.start()
    time.sleep(0.05)
    assert t.is_alive()
    assert c.answer(True) is True
    t.join(2)
    assert got["ok"] is True
    ev = q.get_nowait()
    assert ev["kind"] == "confirm" and "发送" in ev["text"]


def test_chat_stop_denies_a_pending_confirm():
    c = Chat()
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("ok", c.confirm("x")), daemon=True)
    t.start()
    time.sleep(0.05)
    c.stop()
    t.join(2)
    assert got["ok"] is False


def test_chat_answer_without_a_pending_question_is_a_no_op():
    assert Chat().answer(True) is False
