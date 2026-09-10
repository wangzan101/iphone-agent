"""Moonshot（Kimi）。声明自文档，未在本项目真机跑通，等 qualify。"""
from iphone_agent.model.profile import ProviderProfile

PROVIDER = ProviderProfile(name="moonshot", aliases=("kimi",),
                           base_url="https://api.moonshot.cn/v1", env_vars=("MOONSHOT_API_KEY",))
