"""安全硬闸：写类动作在执行器侧拦下来问人，没人就拒绝；不可逆的一律拒绝。

为什么必须在程序侧而不是只靠提示词：UI-Venus-2 技术报告 Table 6（docs/28 §8.5），
「良性指令 + 屏幕里潜藏攻击」下所有通用 VLM 的攻击成功率 79–94%，专门训过的仍 48%。
我们的产品形态恰恰是「用户接一个未经安全训练的通用模型」，提示词层的上限已经被测出来了。
三档照 PhoneHarness（docs/30 §3）：read 直接做 / write 有人确认才做 / never 谁说都不做。
docs/04 的「无人只读、有人可写」由此落地：confirm 回调为 None 就是无人。
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


class _Boxes:
    """带自己框的元素：(文字, (x1, y1, x2, y2))。"""
    def __init__(self, *items):
        from iphone_agent.perceive.elements import Element
        self.elements = [Element(i + 1, t, 0.9, b, ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2))
                         for i, (t, b) in enumerate(items)]
        self.observation_id, self.width_px, self.height_px, self.coord_mode = 7, 100, 100, "pixel"

    def element(self, eid):
        return next((e for e in self.elements if e.id == eid), None)


def test_coordinate_tap_inside_a_danger_box_is_a_write():
    """spec 2026-09-14 §6.2：落点在某个框内，就和按 id 点击同等对待 —— 这是正证据，不是猜语义。"""
    d = safety.classify(act("tap", x=10, y=10), _Obs("发送"))
    assert (d.level, d.target) == ("write", "发送")


def test_coordinate_tap_inside_a_never_box_is_never():
    assert safety.classify(act("tap", x=5, y=5), _Obs("转账")).level == "never"


def test_nested_boxes_take_the_strictest_level():
    obs = _Boxes(("账单详情", (0, 0, 100, 100)), ("删除", (40, 40, 60, 60)), ("付款", (45, 45, 55, 55)))
    assert (lambda d: (d.level, d.target))(safety.classify(act("tap", x=50, y=50), obs)) == ("never", "付款")
    assert (lambda d: (d.level, d.target))(safety.classify(act("tap", x=42, y=42), obs)) == ("write", "删除")


def test_coordinate_tap_outside_every_box_is_read_and_marked_unclassified():
    """框外（纯图标）：维持现状记 read，但如实标出来（spec §6.2、§11）。"""
    obs = _Obs("发送")
    d = safety.classify(act("tap", x=50, y=50), obs)
    assert (d.level, d.target) == ("read", "(x,y)")
    tt = safety.tap_target(act("tap", x=50, y=50), obs)
    assert tt.by == "coord" and tt.hits == () and tt.unclassified is True


def test_coordinate_tap_inside_a_blank_text_box_is_unclassified():
    """落点在框里，但那个框的文字是空白（纯图标，比如视觉标注出来的一个图标框）——
    hits 非空，词表判不了，仍要标 unclassified（review 修：unclassified 曾用 `not hits` 判，
    这种情况 hits 非空就被记成「判过了」，coord_unclassified 漏计）。"""
    obs = _Boxes(("  ", (0, 0, 20, 20)))
    d = safety.classify(act("tap", x=10, y=10), obs)
    assert (d.level, d.target) == ("read", "(x,y)")
    tt = safety.tap_target(act("tap", x=10, y=10), obs)
    assert tt.by == "coord" and tt.hits != () and tt.unclassified is True
    assert tt.to_json() == {"by": "coord", "px": [10, 10], "hits": [], "unclassified": True}


def test_tap_target_json_and_echo():
    obs = _Boxes(("通用", (0, 0, 20, 20)))
    tt = safety.tap_target(act("tap", id=1, x=10, y=10), obs)
    assert tt.to_json() == {"by": "id", "px": [10, 10], "hits": ["通用"], "unclassified": False}
    assert tt.echo(obs) == {"id": 1, "text": "通用", "source": "ocr", "observation_id": 7}
    tc = safety.tap_target(act("tap", x=10, y=10), obs)
    assert tc.echo(obs) == {"x": 10, "y": 10, "hits": ["通用"], "observation_id": 7}
    assert safety.tap_target(act("scroll", direction="down"), obs) is None


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


def test_a_coordinate_tap_on_a_danger_word_is_gated_like_an_id_tap(fake_env, tmp_path):
    # 假 OCR 的第一行框在 (40,88)-(200,120)（400x800 的图），(120,104) 落在「发送」上
    r, dev, m = run(fake_env, [["发送"]] * 4, [[("tap", {"x": 120, "y": 104})], DONE], tmp_path)
    assert not any(c[0] == "tap" for c in dev.calls)
    rec = _safety_records(r)[0]
    assert rec["safety"] == {"level": "write", "decision": "blocked_unattended", "target": "发送"}
    assert rec["tap_target"] == {"by": "coord", "px": [120, 104], "hits": ["发送"], "unclassified": False}


def test_taps_are_counted_and_echoed(fake_env, tmp_path):
    """m10：不读 m.seen 里的 tool 消息 —— windowed() 会把旧的 tool 结果截到 500 字，`tapped`
    又是最后追加的字段，这个断言会很脆。改从 steps.jsonl 的 result.tapped / tap_target 读。"""
    r, dev, m = run(fake_env, [["通用"], ["关于本机"], ["关于本机"], ["关于本机"], ["关于本机"]],
                    [[("tap", {"id": 1})], [("tap", {"x": 390, "y": 790})], DONE], tmp_path)
    meta = json.loads((r.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["taps"] == {"by_id": 1, "by_coord": 1, "coord_unclassified": 1}
    recs = [json.loads(line) for line in (r.run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    tap_recs = [rec for rec in recs if rec.get("action", {}).get("name") == "tap"]
    assert tap_recs[0]["result"]["tapped"]["id"] == 1 and tap_recs[0]["result"]["tapped"]["text"] == "通用"
    assert tap_recs[0]["tap_target"] == {"by": "id", "px": [120, 104], "hits": ["通用"], "unclassified": False}
    assert tap_recs[1]["result"]["tapped"]["hits"] == []
    assert tap_recs[1]["tap_target"]["unclassified"] is True


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
