"""OpenAI 官方。声明自文档，未在本项目真机跑通，等 qualify。"""
from iphone_agent.model.profile import ProviderProfile

PROVIDER = ProviderProfile(name="openai", base_url="https://api.openai.com/v1", env_vars=("OPENAI_API_KEY",))
