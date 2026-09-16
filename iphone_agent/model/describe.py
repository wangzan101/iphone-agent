"""模型信息的结构化出口：当前选了谁（`describe`）、有哪些可选（`catalog`）。

为什么单独一个模块：`cmd_model` 原来是直接 print 的，界面消费不了。
而界面比命令行更需要这些字段 —— 尤其是 `calibrated`：

内置表是**不对称的**，只有 `alibaba:qwen3.7-plus` 在真机上标定过坐标约定。
没标定过的模型去指像素，中位数误差 185px（docs/14）。命令行用户会去读 README，
界面用户不会 —— 所以「这个模型标定过吗」必须是一个能渲染成标记的字段，
而不是藏在一行打印里。

⚠ 一个字都不带密钥。`api_key_source` 只说来源（"env:DASHSCOPE_API_KEY"），不说值。
"""
from __future__ import annotations

from iphone_agent.model import registry
from iphone_agent.model.config import ResolvedModel


def describe(r: ResolvedModel) -> dict:
    """当前这一次解析的结果。给设置页和 doctor 用。"""
    m = r.model
    return {
        "spec": r.spec,
        "provider": r.provider.name,
        "model_id": m.id,
        "base_url": r.provider.base_url,
        # 只说来源，不说值 —— 这个项目的密钥泄过一次，见 ResolvedModel 的注释。
        "api_key_source": r.api_key_source,
        "has_key": bool(r.api_key),
        "coord_mode": m.coord_mode,
        "allow_coord_tap": m.allow_coord_tap,
        "calibrated": m.calibrated,
        "supports_tools": m.supports_tools,
        "max_tokens": m.max_tokens,
        "notices": list(r.notices),
    }


def catalog() -> list[dict]:
    """内置表里所有模型。设置页的下拉列表就是它。

    带 `calibrated` 和 `recommended` —— 让界面能把「标定过的那一个」摆在最上面，
    而不是让用户在一串等价的名字里瞎猜。
    """
    out = []
    for spec, m in sorted(registry.MODELS.items()):
        p = registry.PROVIDERS[m.provider]
        out.append({
            "spec": spec,
            "provider": m.provider,
            "model_id": m.id,
            "base_url": p.base_url,
            "coord_mode": m.coord_mode,
            "allow_coord_tap": m.allow_coord_tap,
            "calibrated": m.calibrated,
            "recommended": spec == registry.DEFAULT_MODEL_SPEC,
            # 密钥从哪来 —— 设置页要能告诉用户「去哪拿这个 key」
            "env_vars": list(p.env_vars),
        })
    return out


def providers() -> list[dict]:
    """内置 provider。自定义端点走 config.toml 的 [providers.<name>]，不在这里。"""
    return [{"name": p.name, "aliases": list(p.aliases), "base_url": p.base_url,
             "env_vars": list(p.env_vars)}
            for p in sorted(registry.PROVIDERS.values(), key=lambda x: x.name)]
