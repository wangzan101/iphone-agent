"""验证过的剧本 → 模型看到的 function 定义。"""
from __future__ import annotations

from iphone_agent import config
from iphone_agent.skills.model import Procedure


def tool_name(app: str, name: str) -> str:
    # OpenAI function 名只许 [a-zA-Z0-9_-]，不能有 . 和 /；App id 和剧本名里不会出现双下划线
    return f"{app}__{name}"


def split_tool_name(s: str) -> tuple[str, str] | None:
    if "__" not in s:
        return None
    app, name = s.split("__", 1)
    return (app, name) if app and name else None


def procedure_tool_defs(procs) -> list[dict]:
    defs = []
    for p in procs:
        props = {k: {"type": "string", "description": v.get("description", "")} for k, v in p.params.items()}
        props["reason"] = {"type": "string", "description": "一句话：为什么现在用这条剧本"}
        prov = p.provenance
        desc = (f"{p.description}。验证过的剧本（App：{p.app}），走通 {prov.verified_count} 次"
                + (f"，最后一次 {prov.last_ok}" if prov.last_ok else "")
                + "。返回：" + ("、".join(p.returns) or "无"))
        defs.append({"type": "function", "function": {
            "name": p.tool_name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(p.params) + ["reason"]}}})
    return defs


def select_procedures(catalog, apps: list[str] | None) -> list[Procedure]:
    """进工具列表的范围（spec §4.3）：命中的 App 的全部合格剧本；
    没命中时全库合格剧本 ≤ PROC_TOOLS_MAX 才全放，否则不放。列表任务开始时定死。"""
    if apps:
        return catalog.eligible_procedures(apps)
    allp = catalog.eligible_procedures()
    return allp if len(allp) <= config.PROC_TOOLS_MAX else []
