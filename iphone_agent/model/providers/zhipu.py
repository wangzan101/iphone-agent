"""智谱（GLM）。声明自文档，未在本项目真机跑通，等 qualify。"""
from iphone_agent.model.profile import ProviderProfile

PROVIDER = ProviderProfile(name="zhipu", aliases=("glm", "bigmodel"),
                           base_url="https://open.bigmodel.cn/api/paas/v4", env_vars=("ZHIPU_API_KEY",))
