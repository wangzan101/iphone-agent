"""工作区（路径）与单次运行配置（模式、预算）。

为什么要有这两个东西：原来 `runs/`、`.iphone/memory`、`.iphone/config.toml` 是
四个模块各自的模块级常量，全相对进程 cwd；运行模式 `CONTEXT_MODE` 在 config.py
import 的那一刻读一次 env。结果是三件事都做不了（设计说明 D3/D7/D8）：

- 同一进程里同时管两台设备/两个会话（路径写死在 cwd 下）；
- 同一进程里跑 window / state 的 2×2 对照实验（模式是进程级的）；
- 在测试里换掉这些路径（只能靠 monkeypatch 模块属性，改一个漏一个）。

Workspace 把「东西放哪儿」收成一个显式对象；RunConfig 把「这一次怎么跑」
收成一个参数。默认行为不变：不设 env、不传参数时，仍是 cwd + window。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from iphone_agent import config


@dataclass(frozen=True)
class Workspace:
    """一份工作区的全部落盘位置。所有路径都挂在 root 下面，不再各自相对 cwd。"""

    root: Path

    def __post_init__(self) -> None:
        # 允许传 str：调用方常常从 env 或命令行拿到字符串。
        object.__setattr__(self, "root", Path(self.root))

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def dot(self) -> Path:
        return self.root / ".iphone"

    @property
    def memory_dir(self) -> Path:
        return self.dot / "memory"

    @property
    def config_file(self) -> Path:
        return self.dot / "config.toml"

    @property
    def screenmap_cache(self) -> Path:
        return self.dot / "screenmap.json"

    @property
    def twin_device(self) -> Path:
        """设备层孪生：主屏布局表、系统屏（设计说明）。"""
        return self.dot / "knowledge" / "device"

    @classmethod
    def default(cls) -> Workspace:
        """`IPHONE_WORKSPACE`，否则进程 cwd。

        ⚠ env 在**调用时**读，不在 import 时。import 时读就等于把值锁死在
        「第一个 import 这个模块的人」那一刻 —— 那正是 CONTEXT_MODE 踩过的坑。
        """
        return cls(Path(os.environ.get("IPHONE_WORKSPACE", "").strip() or Path.cwd()))


@dataclass(frozen=True)
class RunConfig:
    """这一次运行怎么跑。进程级常量只当默认值来源，不再当运行时开关。"""

    context_mode: str = "window"        # window | state（设计说明）
    # ⚠ 不能写成 `max_steps: int = config.MAX_STEPS`：那是 import 时求值一次，
    # 绑死在「第一个 import 这个模块的人」看到的值上。field(default_factory=...) 才是
    # 每次构造 RunConfig() 时才读——同一个坑 Workspace.default 的 env 已经踩过。
    max_steps: int = field(default_factory=lambda: config.MAX_STEPS)
    # None 表示「按步数推」，见 effective_timeout_s。显式给了就以它为准。
    timeout_s: float | None = None

    def effective_timeout_s(self) -> float:
        """时限和步数是两个保险丝，得对得上量（设计说明 D8）。"""
        if self.timeout_s is not None:
            return self.timeout_s
        return self.max_steps * config.SECONDS_PER_STEP

    @classmethod
    def from_env(cls, **overrides) -> RunConfig:
        """env / config 常量在**调用时**读，overrides 优先。"""
        base = cls(
            # env 直接给的最优先；没给就退回 config.CONTEXT_MODE ——
            # 那个常量本身就是 import 时读的同一个 env，保留它是为了让
            # 「改 config 模块属性」这条老路（测试、脚本）继续有效。
            context_mode=(os.environ.get("IPHONE_USE_CONTEXT", "").strip()
                          or config.CONTEXT_MODE),
            max_steps=config.MAX_STEPS,
        )
        return replace(base, **overrides) if overrides else base
