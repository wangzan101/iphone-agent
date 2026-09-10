"""网页的数据面。

最重要的一条压在最前面：**密钥一个字都不许发出去**。
这些 dict 会直接变成 JSON 走 HTTP 到浏览器里，一旦漏了就是漏在日志、
截图、聊天记录里，全都收不回来。

第二条：**没配好模型不是异常，是「还没配」**。填密钥的地方就在这个界面里，
所以每个端点在没有 key 的时候都必须仍然答得出话，而不是抛异常或者 500。
"""
import json
from types import SimpleNamespace

import pytest

from iphone_agent.model.configwrite import set_api_key
from iphone_agent.web import api
from iphone_agent.web.server import Chat
from iphone_agent.workspace import Workspace

SECRET = "sk-never-cross-the-wire-13572468"


@pytest.fixture
def clean(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for k in ("DASHSCOPE_API_KEY", "IPHONE_USE_MODEL", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.fixture
def chat(clean, tmp_path):
    return Chat(workspace=Workspace(tmp_path / "ws"))


def payload(pair) -> str:
    """端点的返回值真正会变成的那串 JSON —— 泄漏要在这一层查。"""
    code, data = pair
    return json.dumps(data, ensure_ascii=False)


# ── 密钥不过线 ──────────────────────────────────────────────────────────

def test_get_config_never_ships_the_key(chat, clean):
    set_api_key("alibaba", SECRET)
    s = payload(api.get_config(chat))
    assert SECRET not in s
    _, data = api.get_config(chat)
    assert data["providers"]["alibaba"]["has_key"] is True, "只说有没有，不说是什么"


def test_state_never_ships_the_key(chat, clean):
    set_api_key("alibaba", SECRET)
    assert SECRET not in payload(api.state(chat))


def test_put_config_response_never_echoes_the_key(chat, clean):
    out = api.put_config(chat, {"provider": "alibaba", "api_key": SECRET})
    assert SECRET not in payload(out)


def test_models_listing_carries_no_secrets(chat, clean):
    set_api_key("alibaba", SECRET)
    assert SECRET not in payload(api.models(chat))


# ── 没配好也要答得出话 ──────────────────────────────────────────────────

def test_state_says_not_ready_instead_of_blowing_up(chat, clean):
    code, data = api.state(chat)
    assert code == 200
    assert data["model"]["ready"] is False
    assert data["model"]["problem"], "得告诉界面缺什么，它才知道往哪引导"


def test_models_still_lists_choices_without_a_key(chat, clean):
    code, data = api.models(chat)
    assert code == 200
    assert data["models"] and data["providers"]
    assert data["current"] is None


def test_get_config_survives_a_missing_key(chat, clean):
    code, data = api.get_config(chat)
    assert code == 200
    assert data["current"]["spec"] is None


def test_serve_starts_even_without_a_key(clean, capsys):
    """以前没 key 就 return 2 —— 而填密钥的地方就在这个界面里，起不来就永远进不去。"""
    from iphone_agent.web import server

    started = {}

    class FakeServer:
        def __init__(self, addr, handler):
            started["addr"] = addr

        def serve_forever(self):
            raise KeyboardInterrupt          # 起来了就够了，立刻收工

        def server_close(self):
            pass

    clean.setattr(server, "_Server", FakeServer)
    rc = server.serve(port=0)
    assert rc == 0, "没配好模型不该让服务起不来"
    assert started["addr"] == ("127.0.0.1", 0), "而且必须还是只监听本机"
    assert "设置" in capsys.readouterr().err, "得在终端上告诉用户去界面里配"


# ── 结构 ────────────────────────────────────────────────────────────────

def test_models_endpoint_reflects_the_asymmetric_builtin_table(chat, clean):
    """7 个 provider、1 个内置模型 —— 设置页据此把 model id 做成输入框而不是下拉。"""
    _, data = api.models(chat)
    assert len(data["providers"]) > len(data["models"])
    assert data["default"] in [m["spec"] for m in data["models"]]


def test_doctor_returns_checks_and_a_verdict(chat, monkeypatch):
    monkeypatch.setattr(Chat, "session", lambda self: SimpleNamespace())
    code, data = api.doctor(chat)
    assert code == 200
    assert isinstance(data["ok"], bool)
    assert {c["id"] for c in data["checks"]} >= {"perm.accessibility", "model"}
    json.dumps(data)


def test_state_reports_running_and_stopping(chat, clean):
    _, a = api.state(chat)
    assert a["running"] is False and a["stopping"] is False
    chat.busy.acquire()
    chat.stop()
    try:
        _, b = api.state(chat)
        assert b["running"] is True and b["stopping"] is True
    finally:
        chat.busy.release()


def test_state_exposes_the_current_step(chat, clean):
    chat.last_step = {"n": 7, "name": "tap", "error": None}
    _, data = api.state(chat)
    assert data["step"]["n"] == 7


# ── 写入 ────────────────────────────────────────────────────────────────

def test_put_config_saves_a_key_that_resolve_then_picks_up(chat, clean):
    code, data = api.put_config(chat, {"provider": "alibaba", "api_key": SECRET})
    assert code == 200 and data["ok"] and "api_key" in data["changed"]
    _, cfg = api.get_config(chat)
    assert cfg["providers"]["alibaba"]["has_key"] is True
    assert cfg["current"]["spec"] == "alibaba:qwen3.7-plus"


def test_put_config_refuses_while_a_task_is_running(chat, clean):
    """跑到一半换模型，这一轮的前后半段会用两个不同的坐标约定。"""
    chat.busy.acquire()
    try:
        code, data = api.put_config(chat, {"model": "deepseek:deepseek-v3"})
    finally:
        chat.busy.release()
    assert code == 400 and "正在跑" in data["error"]


def test_put_config_rejects_a_key_without_a_provider(chat, clean):
    code, data = api.put_config(chat, {"api_key": SECRET})
    assert code == 400 and "服务商" in data["error"]


# ── 自定义端点 ──────────────────────────────────────────────────────────
# 后端一直支持（resolve 对不在内置表里的 provider 只要求 base_url），
# 缺的只是写入的口子和列表里的可见性。

def test_a_custom_endpoint_can_be_saved_and_then_resolves(chat, clean):
    code, data = api.put_config(chat, {
        "provider": "myvllm", "base_url": "https://my.endpoint.example/v1", "api_key": SECRET})
    assert code == 200 and set(data["changed"]) >= {"api_key", "base_url"}

    code, data = api.put_config(chat, {"model": "myvllm:qwen2.5-vl-72b"})
    assert code == 200 and data["ready"] is True
    cur = data["current"]
    assert cur["spec"] == "myvllm:qwen2.5-vl-72b"
    assert cur["base_url"] == "https://my.endpoint.example/v1"
    # 自定义端点必然没标定过坐标 —— 界面要把这件事说出来，否则用户会觉得「它变笨了」
    assert cur["calibrated"] is False and cur["allow_coord_tap"] is False


def test_custom_providers_show_up_in_the_listing(chat, clean):
    api.put_config(chat, {"provider": "myvllm", "base_url": "https://x.example/v1"})
    _, data = api.models(chat)
    by = {p["name"]: p for p in data["providers"]}
    assert by["myvllm"]["custom"] is True
    assert by["myvllm"]["base_url"] == "https://x.example/v1"
    assert by["myvllm"]["env_vars"] == ["MYVLLM_API_KEY"]     # resolve 里现推的那个名字
    assert by["alibaba"]["custom"] is False


def test_a_custom_endpoint_without_a_base_url_is_reported_not_ready(chat, clean):
    """resolve 对不在内置表里的 provider **要求 base_url**。
    只填密钥就换过去，配置存得下但用不了 —— 得如实说。"""
    api.put_config(chat, {"provider": "myvllm", "api_key": SECRET})
    code, data = api.put_config(chat, {"model": "myvllm:whatever"})
    assert code == 200 and data["ready"] is False
    assert "base_url" in data["problem"]


@pytest.mark.parametrize("bad,want", [
    ("MyVLLM:x", "服务商名字"),
    ("my vllm", "服务商名字"),
    ("", "服务商"),
])
def test_bad_provider_names_are_refused(chat, clean, bad, want):
    code, data = api.put_config(chat, {"provider": bad, "base_url": "https://x.example/v1"})
    assert code == 400 and want in data["error"]


@pytest.mark.parametrize("bad", ["my.endpoint/v1", "ftp://x/v1", "https://x .com/v1"])
def test_bad_base_urls_are_refused(chat, clean, bad):
    code, data = api.put_config(chat, {"provider": "myvllm", "base_url": bad})
    assert code == 400 and "接口地址" in data["error"]


def test_clearing_a_custom_endpoint_removes_it(chat, clean):
    api.put_config(chat, {"provider": "myvllm", "base_url": "https://x.example/v1"})
    api.put_config(chat, {"provider": "myvllm", "base_url": "", "api_key": ""})
    _, data = api.models(chat)
    assert "myvllm" not in {p["name"] for p in data["providers"]}


def test_put_config_reports_an_unusable_model_without_failing_the_save(chat, clean):
    """换到一个没有 key 的 provider：配置确实存下去了，但要如实说它还不能用。"""
    code, data = api.put_config(chat, {"model": "deepseek:deepseek-v3"})
    assert code == 200 and data["ok"] is True
    assert data["ready"] is False and data["problem"]


def test_put_config_rejects_a_bad_body(chat, clean):
    code, _ = api.put_config(chat, ["not", "an", "object"])
    assert code == 400


# ── 测试连接 ────────────────────────────────────────────────────────────

def test_test_model_without_a_key_is_not_ready(chat, clean):
    code, data = api.test_model(chat)
    assert code == 200 and data["ok"] is False and data["problem"]


def test_test_model_turns_a_401_into_a_sentence(chat, clean, monkeypatch):
    """填错的 key 和没填的 key 对用户是两件完全不同的事，不能都报成一句「不可用」。"""
    set_api_key("alibaba", SECRET)

    class Boom:
        def decide(self, *a, **kw):
            raise RuntimeError("Error code: 401 - invalid api key")

    monkeypatch.setattr(Chat, "session",
                        lambda self: SimpleNamespace(
                            resolved=SimpleNamespace(spec="alibaba:qwen3.7-plus"), model=Boom()))
    code, data = api.test_model(chat)
    assert code == 200 and data["ok"] is False
    assert "密钥不对" in data["problem"]


# ── 历史运行 / 回放 ─────────────────────────────────────────────────────

def _make_run(root, rid, task="做点什么", end="done_success", steps=2):
    d = root / rid
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps(
        {"task": task, "end_reason": end, "steps": steps, "started_at": 1757000000.0,
         "model": "alibaba:qwen3.7-plus"}, ensure_ascii=False), encoding="utf-8")
    lines = []
    for i in range(1, steps + 1):
        lines.append(json.dumps({
            "step": i, "action": {"name": "tap", "args": {"id": i}},
            "result": {"changed": True}, "model": {"reason": "理由", "latency_ms": 3000},
            "after_frame_file": f"frame_{i:03d}.png", "exec_ms": 1500}, ensure_ascii=False))
    (d / "steps.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def test_runs_is_empty_before_anything_has_run(chat, clean):
    code, data = api.runs(chat)
    assert code == 200 and data["runs"] == []


def test_runs_lists_newest_first(chat, clean):
    root = chat.workspace.runs
    _make_run(root, "20000101-005908-0518", "早的")
    _make_run(root, "20000101-010530-6551", "晚的")
    _, data = api.runs(chat)
    assert [r["task"] for r in data["runs"]] == ["晚的", "早的"]


def test_runs_distinguishes_failed_from_unfinished(chat, clean):
    """跑崩了没写完 end_reason 的运行，界面要能跟「跑完了但没做成」分开显示。"""
    root = chat.workspace.runs
    _make_run(root, "20000101-010530-6551", "没做成", end="no_progress")
    _make_run(root, "20000101-010531-6552", "没跑完", end=None)
    by = {r["task"]: r for r in api.runs(chat)[1]["runs"]}
    assert by["没做成"]["ok"] is False and by["没做成"]["finished"] is True
    assert by["没跑完"]["ok"] is False and by["没跑完"]["finished"] is False


def test_a_broken_run_json_is_skipped_not_fatal(chat, clean):
    """跑到一半崩掉留下半个 run.json 是常态。它不能让整个列表打不开。"""
    root = chat.workspace.runs
    _make_run(root, "20000101-010530-6551", "好的")
    bad = root / "20000101-010531-6552"
    bad.mkdir(parents=True)
    (bad / "run.json").write_text("{ 这不是 json", encoding="utf-8")
    _, data = api.runs(chat)
    assert [r["task"] for r in data["runs"]] == ["好的"]


def test_run_detail_prefixes_frames_with_the_run_id(chat, clean):
    """⚠ steps.jsonl 里的帧名不带目录（frame_001.png），而 /api/frame/ 是相对 runs/ 解析的。
    不补前缀就是一路 404 —— 点某一步什么都不显示。这条测试就是为那个 bug 立的。"""
    _make_run(chat.workspace.runs, "20000101-010530-6551", steps=2)
    code, data = api.run_detail(chat, "20000101-010530-6551")
    assert code == 200
    assert [s["frame"] for s in data["steps"]] == [
        "20000101-010530-6551/frame_001.png", "20000101-010530-6551/frame_002.png"]


def test_run_detail_carries_the_timings_replay_needs(chat, clean):
    """回放的节奏靠这两个。没有它们就只能固定节拍，看不出「哪一步卡了很久」。"""
    _make_run(chat.workspace.runs, "20000101-010530-6551", steps=1)
    _, data = api.run_detail(chat, "20000101-010530-6551")
    s = data["steps"][0]
    assert s["exec_ms"] == 1500 and s["latency_ms"] == 3000


@pytest.mark.parametrize("bad", [
    "../../../etc/passwd", "..", ".", "", "notarunid",
    "20000101-010530-6551/../..", "20000101-010530-655",
])
def test_run_detail_refuses_anything_that_is_not_a_run_id(chat, clean, bad):
    """这个 id 会被拼进文件路径 —— 形状校验是安全边界，不是格式洁癖。"""
    code, _ = api.run_detail(chat, bad)
    assert code == 404


def test_run_detail_survives_a_missing_steps_file(chat, clean):
    d = chat.workspace.runs / "20000101-010530-6551"
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"task": "x", "end_reason": "done_success"}), encoding="utf-8")
    code, data = api.run_detail(chat, "20000101-010530-6551")
    assert code == 200 and data["steps"] == []


def test_run_detail_skips_a_corrupt_line_instead_of_dying(chat, clean):
    d = _make_run(chat.workspace.runs, "20000101-010530-6551", steps=1)
    with (d / "steps.jsonl").open("a", encoding="utf-8") as f:
        f.write("{ 半行坏数据\n")
    code, data = api.run_detail(chat, "20000101-010530-6551")
    assert code == 200 and len(data["steps"]) == 1


# ── 学到的 ──────────────────────────────────────────────────────────────

def test_learned_answers_on_a_fresh_install(chat, clean):
    """什么都还没跑过的时候要给出空态，而不是报错 —— 这是新用户看到的第一眼。"""
    code, data = api.learned(chat)
    assert code == 200
    assert data["apps"] == [] and data["memories"] == [] and data["pending"] == 0
    json.dumps(data)


def test_learned_survives_a_broken_skills_directory(chat, clean, monkeypatch):
    """技能目录是人手可改的。它坏了也得能打开这一页 —— 否则用户连去修的入口都没有。"""
    class Boom:
        def load(self):
            raise RuntimeError("目录被人改坏了")

    monkeypatch.setattr("iphone_agent.skills.store.SkillStore", lambda *a, **k: Boom())
    code, data = api.learned(chat)
    assert code == 200 and data["apps"] == []
    assert "改坏了" in data["skills_error"], "得说清是哪一半坏了"


def test_learned_counts_what_is_waiting_for_approval(chat, clean, monkeypatch):
    """写类剧本必须人点头模型才敢调用 —— 待批准的条数要能显示在侧边栏角标上。"""
    from types import SimpleNamespace as NS

    proc = NS(name="转账", description="", risk="write", status="pending",
              steps=[1, 2], provenance=NS(verified_count=0))
    app = NS(name="bank", display="某银行", risk="write", status="active", updated="")
    monkeypatch.setattr("iphone_agent.skills.store.SkillStore",
                        lambda *a, **k: NS(load=lambda: NS(
                            visible_apps=lambda: [app],
                            procedures_of=lambda n: [proc],
                            visible_scenarios=lambda: [])))
    _, data = api.learned(chat)
    assert data["pending"] == 1
    assert data["apps"][0]["procedures"][0]["needs_approval"] is True


def test_read_only_procedures_do_not_need_approval(chat, clean, monkeypatch):
    from types import SimpleNamespace as NS

    proc = NS(name="看余额", description="", risk="read", status="pending",
              steps=[1], provenance=NS(verified_count=3))
    app = NS(name="bank", display="某银行", risk="read", status="active", updated="")
    monkeypatch.setattr("iphone_agent.skills.store.SkillStore",
                        lambda *a, **k: NS(load=lambda: NS(
                            visible_apps=lambda: [app],
                            procedures_of=lambda n: [proc],
                            visible_scenarios=lambda: [])))
    _, data = api.learned(chat)
    assert data["pending"] == 0
    assert data["apps"][0]["procedures"][0]["needs_approval"] is False


@pytest.mark.parametrize("msg,want", [
    ("Error code: 429 too many requests", "限流"),
    ("Request timed out", "超时"),
    ("Error code: 404 model not found", "端点或模型名"),
])
def test_common_failures_get_actionable_wording(msg, want):
    assert want in api._explain(RuntimeError(msg))
