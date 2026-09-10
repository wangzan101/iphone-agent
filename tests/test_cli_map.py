"""`iphone map` —— 让屏幕图看得见。

这张图现在会影响模型每一步的判断（每轮观察后面跟着的【位置】【路线】就是它），
所以必须能查它到底知道些什么。看得见是这个项目一贯的兑现方式（记忆也是文件）。
"""
from iphone_agent.cli.commands import COMMANDS, Session
from iphone_agent.workspace import Workspace


def test_map_is_a_registered_command():
    assert "map" in COMMANDS


def test_bad_subcommand_is_rejected_not_crashed(capsys, tmp_path):
    # 给一个 tmp 工作区：原来走模块级 RUNS_ROOT（相对 cwd），
    # 跑一次测试就会往开发机的 .iphone/screenmap.json 里写东西。
    rc = COMMANDS["map"](Session(Workspace(tmp_path)), ["没这个子命令"])
    assert rc == 2
    assert "用法" in capsys.readouterr().err


def test_show_needs_a_number(capsys, tmp_path):
    rc = COMMANDS["map"](Session(Workspace(tmp_path)), ["show", "不是数字"])
    assert rc == 2
