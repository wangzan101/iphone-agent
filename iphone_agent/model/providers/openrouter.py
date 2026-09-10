"""OpenRouter 聚合。声明自文档，未在本项目真机跑通，等 qualify。"""
from iphone_agent.model.profile import ProviderProfile

PROVIDER = ProviderProfile(name="openrouter", base_url="https://openrouter.ai/api/v1",
                           env_vars=("OPENROUTER_API_KEY",))
