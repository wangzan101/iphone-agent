"""桥的统一返回形状。所有 transport 都产出这一个类型；主循环只认它。"""
from __future__ import annotations

from dataclasses import dataclass

from iphone_agent.harness.actions import Action


class ModelError(RuntimeError):
    """模型/协议层的问题。统一成它，不让主循环的兜底 except 误诊成 device_error。"""


@dataclass
class ModelReply:
    actions: list[Action]
    text: str
    model_version: str
    usage: dict
    latency_ms: int
