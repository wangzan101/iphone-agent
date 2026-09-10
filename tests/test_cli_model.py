"""Session 选模型这条路：--model / IPHONE_USE_MODEL / config.toml / 内置默认。

关键是**原子替换**：模型换了，Perceiver 也得跟着换。它俩共用一个坐标约定。
"""
import sys

import pytest

from iphone_agent.cli.commands import Session, dispatch
from iphone_agent.harness.prompt import system_prompt
from iphone_agent.harness.tools import tool_defs
from iphone_agent.model.errors import ConfigError

ENV = ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "IPHONE_USE_MODEL", "IPHONE_USE_COORD_MODE")


class _FakePer:
    def __init__(self, coord_mode):
        self.coord_mode = coord_mode


class _FakeDev:
    """只为让命令走到读配置那一步 —— 这些用例都在没有镜像窗口的机器上跑。"""
    def capture(self):
        return None


@pytest.fixture
def clean(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                       # 没有 .iphone/config.toml
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    # Perceiver 默认会起 Vision OCR；这里只关心 coord_mode，换成空 OCR
    import iphone_agent.cli.commands as cmds
    # asker 也收下：没 key 的 Session 会传 None 进来，这正是下面那条
    # 「没 key 也要能观察」要保的行为。
    monkeypatch.setattr(cmds, "_make_perceiver",
                        lambda coord_mode, asker=None: _FakePer(coord_mode))
    return monkeypatch


def test_default_session_is_builtin_qwen(clean):
    s = Session()
    assert s.resolved.spec == "alibaba:qwen3.7-plus"
    assert s.per.coord_mode == "norm1000" and s.model.resolved is s.resolved


def test_env_model_spec_is_honoured(clean):
    clean.setenv("IPHONE_USE_MODEL", "alibaba:qwen-vl-max")
    assert Session().resolved.spec == "alibaba:qwen-vl-max"


def test_constructor_spec_beats_env(clean):
    clean.setenv("IPHONE_USE_MODEL", "alibaba:qwen-vl-max")
    assert Session(model_spec="alibaba:qwen3.7-plus").resolved.spec == "alibaba:qwen3.7-plus"


def test_select_model_swaps_everything_together(clean):
    """Codex 审出的坑：只换模型不换 Perceiver，观察头部和桥就各说各的。"""
    clean.setenv("IPHONE_USE_COORD_MODE", "")          # 确保没 legacy
    s = Session()
    per1, model1 = s.per, s.model
    assert per1.coord_mode == "norm1000"
    s.select_model("alibaba:qwen-vl-max")
    assert s.per is not per1 and s.model is not model1
    r = s.resolved
    assert r.model.allow_coord_tap is False
    assert s.model.request_kwargs([])["tools"] == tool_defs(False)
    assert "tap(x, y)" not in system_prompt(r.model.allow_coord_tap)
    s.select_model("qwen3.7-plus")                     # 裸名 = alibaba
    assert s.per.coord_mode == "norm1000" and s.model.request_kwargs([])["tools"] == tool_defs(True)


def test_per_does_not_need_a_key_but_model_does(clean):
    """screen / tap / doctor 不碰模型，没 key 也要能观察；真要调模型时才拦。"""
    clean.delenv("DASHSCOPE_API_KEY")
    s = Session()
    assert s.per.coord_mode == "norm1000"
    with pytest.raises(ConfigError):
        _ = s.model


def test_notices_printed_once(clean, capsys):
    s = Session(model_spec="alibaba:qwen-vl-max")
    _ = s.resolved; _ = s.resolved; _ = s.model
    err = capsys.readouterr().err
    assert err.count("未标定") == 1 and "alibaba:qwen-vl-max" in err


def test_model_command_switches_and_reports(clean, capsys):
    s = Session()
    assert dispatch(s, ["model", "alibaba:qwen-vl-max"]) == 0
    assert s.resolved.spec == "alibaba:qwen-vl-max"
    assert dispatch(s, ["model"]) == 0
    out = capsys.readouterr().out
    assert "alibaba:qwen-vl-max" in out and "坐标点击" in out


def test_model_command_bad_spec_reports_not_raises(clean, capsys):
    """失败的切换必须什么都不动：已选的模型、已建的 per / model 都还是原来那些对象。

    只断言 `_spec` 没被写坏太弱 —— 那只证明「抛之前没先赋值」，不证明一个**已经生效**
    的选择能扛住一次失败的切换。
    """
    s = Session()
    s.select_model("alibaba:qwen-vl-max")
    per1, model1 = s.per, s.model
    assert dispatch(s, ["model", "nope:x"]) == 2
    assert "nope" in capsys.readouterr().err
    assert s.per is per1 and s.model is model1
    assert s.resolved.spec == "alibaba:qwen-vl-max"    # 失败不改变当前选择


def test_per_and_model_never_split_across_two_resolutions(clean):
    """审出来的坑：per 走的是 _profiles() 那次**不缓存**的解析（T1），model 走的是
    resolved 那次带 key 的解析（T2）。两次之间配置变了，观察头部写的坐标约定就跟桥
    换算用的不是一个 —— 模型给的是 norm1000，桥当 pixel 乘，点到界外，连拒 5 次收工。
    REPL 里这个窗口是无限长的，工具自己还在提示人去改 config.toml。

    真实调用点是 commands.py 里那一行：

        run_task(ns.task, session.dev, session.per, session.model, RUNS_ROOT, ...)

    Python 从左往右求值：`session.per` 的**值**在 `session.model` 开跑之前就已经交出去了。
    所以「访问 model 时把 _per 置空」救不了这一行 —— 得让 per 自己把那次解析做掉。
    """
    s = Session()
    per1 = s.per                                       # 先热身：/screen 之类先跑过
    assert per1.coord_mode == "norm1000"
    clean.setenv("IPHONE_USE_COORD_MODE", "pixel")     # 人按提示去改了配置
    per2, m = s.per, s.model                           # 同一个表达式里，顺序同调用点
    assert per2.coord_mode == m.resolved.model.coord_mode, "观察头部与桥必须同一套坐标约定"
    assert per2 is per1, "一个 Session 只解析一次，热身建的 Perceiver 应当一直有效"


def test_dispatch_turns_config_errors_into_a_message(clean, capsys):
    """没 key 时 `iphone run` 该给出那句写好的中文，而不是把它裹在一坨堆栈里。"""
    clean.delenv("DASHSCOPE_API_KEY")
    clean.setattr(Session, "dev", property(lambda self: _FakeDev()))
    assert dispatch(Session(), ["run", "随便什么"]) == 2
    assert "API key" in capsys.readouterr().err


def test_dispatch_reports_a_bad_config_file_for_keyless_commands(clean, capsys, tmp_path):
    """screen / tap 以前只建个裸 Perceiver，怎么都不会因为配置炸；现在它要读 coord_mode 了。

    644 的 config.toml、不认识的 provider、写错的 [models.*] 键都从这条路出来 ——
    一个只想看看屏幕的人不该收到堆栈。
    """
    (tmp_path / ".iphone").mkdir()
    f = tmp_path / ".iphone" / "config.toml"
    f.write_text("model = 'nope:x'\n", encoding="utf-8")
    f.chmod(0o600)
    clean.setattr(Session, "dev", property(lambda self: _FakeDev()))
    assert dispatch(Session(), ["screen"]) == 2
    assert "nope" in capsys.readouterr().err


def test_serve_hands_its_startup_resolution_to_the_chat(clean, capsys):
    """serve 启动时已经解析并打过 notices 了；对话再解析一遍就是同样的话打两遍。"""
    from iphone_agent.model.config import resolve
    from iphone_agent.web.server import Chat
    r = resolve("alibaba:qwen-vl-max")
    for n in r.notices:                                # serve 启动时打的那一遍
        print(n, file=sys.stderr)
    s = Chat("alibaba:qwen-vl-max", resolved=r).session()
    assert s.resolved is r and s.per.coord_mode == r.model.coord_mode
    assert capsys.readouterr().err.count("未标定") == 1


def test_supports_tools_false_refused_at_resolve(clean, tmp_path):
    (tmp_path / ".iphone").mkdir()
    f = tmp_path / ".iphone" / "config.toml"
    f.write_text('[models."alibaba:x"]\nsupports_tools = false\n', encoding="utf-8")
    f.chmod(0o600)
    with pytest.raises(ConfigError) as ei:
        _ = Session(model_spec="alibaba:x").resolved
    assert "工具调用" in str(ei.value)


def test_serve_inherits_the_session_model_when_no_flag(clean, monkeypatch):
    """/model 换完再 /serve，网页却起在默认模型上还什么都不说 —— 就是静默分叉。"""
    seen = {}
    import iphone_agent.web as web
    monkeypatch.setattr(web, "serve", lambda port, model_spec=None: seen.update(spec=model_spec) or 0)
    s = Session()
    s.select_model("alibaba:qwen-vl-max")
    assert dispatch(s, ["serve"]) == 0
    assert seen["spec"] == "alibaba:qwen-vl-max"
    assert dispatch(s, ["serve", "--model", "alibaba:qwen3.7-plus"]) == 0
    assert seen["spec"] == "alibaba:qwen3.7-plus"      # 显式 --model 仍然压过会话


def test_serve_without_a_selection_passes_none(clean, monkeypatch):
    """没显式选过就交 None，让 resolve 自己走 IPHONE_USE_MODEL / config.toml / 默认那条路。"""
    seen = {}
    import iphone_agent.web as web
    monkeypatch.setattr(web, "serve", lambda port, model_spec=None: seen.update(spec=model_spec) or 0)
    assert dispatch(Session(), ["serve"]) == 0
    assert seen["spec"] is None
