"""内置 provider。每家一个文件、纯数据。文件头写明有没有在本项目真机跑通过。"""
from iphone_agent.model.providers import alibaba, anthropic, deepseek, moonshot, openai, openrouter, zhipu

ALL = (alibaba, openai, anthropic, deepseek, moonshot, zhipu, openrouter)
