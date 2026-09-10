"""Anthropic —— 走它官方的 OpenAI 兼容层。声明自文档，未在本项目真机跑通，等 qualify。

⚠ 兼容层不支持 prompt caching、忽略 strict，Anthropic 自述「不适用于生产」。
   子项目 2 换成 anthropic_messages 原生桥后这条改 api_mode。
"""
from iphone_agent.model.profile import ProviderProfile

PROVIDER = ProviderProfile(name="anthropic", aliases=("claude",),
                           base_url="https://api.anthropic.com/v1/", env_vars=("ANTHROPIC_API_KEY",))
