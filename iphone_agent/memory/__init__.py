"""记忆存储。叶子模块 —— 不 import harness / perceive / driver 中的任何东西。

它只认三样东西：名字、一行描述、正文。将来换 SQLite 或向量检索，
只要这三个接口不变，上层一行都不用改。
"""
from iphone_agent.memory.store import MemoryEntry, MemoryRejected, MemoryStore

__all__ = ["MemoryEntry", "MemoryRejected", "MemoryStore"]
