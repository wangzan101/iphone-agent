"""REPL：底部输入行，上方滚动日志。进程常驻，窗口/Vision/模型只初始化一次。模型调用不流式。

这里只剩「读一行、分派」。对话本身（接上文、踩刹车、把过程收成一行）在
`commands.chat_turn` 里，和 `iphone run` 共用 —— 两个入口说的话必须是同一句。
"""
from __future__ import annotations

import shlex
import subprocess

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from iphone_agent.cli.commands import REPL_HELP, Session, dispatch

console = Console()


def repl(session: Session) -> int:
    console.print("[bold]iphone[/] 控制台。")
    # ⚠ 帮助文本里有 [page|half]、[provider:model] 这种方括号，rich 会把它们当成
    #   样式标签然后炸掉（MissingStyle）。这段文字不需要着色，直接 print。
    print(REPL_HELP)
    ps = PromptSession("iphone> ")
    while True:
        try:
            with patch_stdout():
                line = ps.prompt()
        except EOFError:
            return 0
        except KeyboardInterrupt:
            # 空闲时按 Ctrl-C 只是清掉这一行，不退出 —— 退出是 Ctrl-D。
            # （跑任务时的 Ctrl-C 是另一回事，见 commands.stop_on_sigint。）
            continue
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            parts = shlex.split(line[1:])
            if not parts:
                continue
            if parts[0] == "quit":
                return 0
            if parts[0] == "help":
                print(REPL_HELP); continue
            if parts[0] == "show":
                subprocess.run(["open", "screen_marked.png"]); continue
            try:
                dispatch(session, parts, in_repl=True)
            except SystemExit as e:
                console.print(f"[red]{e}[/]")
            except KeyboardInterrupt:
                console.print("[yellow]已中断[/]")
        else:
            # 整行 = 一个任务。走 dispatch 而不是直接调 chat_turn：配置错误
            # （没 key、config.toml 权限太松）的那句中文只在 dispatch 里说一次。
            try:
                dispatch(session, ["run", line], in_repl=True)
            except KeyboardInterrupt:
                console.print("[yellow]任务已中断 —— 手机可能停在一个中间态，"
                              "下一条任务前先看一眼 /screen[/]")
