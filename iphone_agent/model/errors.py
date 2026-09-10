"""⚠ 这个模块必须保持零 import（连本项目内部的也不要）。

它是整个依赖图的最底层：iphone_agent/config.py 这个共享常量模块 import 它，
而几乎所有模块又 import config。这里一旦 import 任何本项目的东西就成环。
要加东西就加纯粹的异常类。
"""


class ConfigError(RuntimeError):
    """模型/密钥/配置文件层面的错误。在启动时拦下，不让它变成运行中的 model_error。"""

    def __init__(self, message: str, *, code: str = "config.invalid", params: dict | None = None):
        super().__init__(message)
        self.code = code
        self.params = params if params is not None else {"detail": message}

    def as_message(self) -> dict:
        """Language-neutral UI descriptor; keep str(error) compatible with CLI and logs."""
        return {"code": self.code, "params": dict(self.params)}
