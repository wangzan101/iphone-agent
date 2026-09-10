"""DeepSeek。声明自文档，未在本项目真机跑通，等 qualify。"""
from iphone_agent.model.profile import ProviderProfile

PROVIDER = ProviderProfile(name="deepseek", base_url="https://api.deepseek.com/v1",
                           env_vars=("DEEPSEEK_API_KEY",))
