"""这一帧跑不跑整屏解析、要不要短标注、要不要兜底 —— 只在这里决定（spec 2026-09-14 §3.2，CLAUDE.md §7）。

Perceiver 调 parse_trigger / wants_label，loop 在一处调 fallback_due。executor 和 loop 里不许再写这类判断。
三个函数都只看「程序自己知道的事」（模式、有没有 asker、这帧的 perception），不看屏幕内容（CLAUDE.md §2）。
"""
from __future__ import annotations

MODES = ("always", "on_demand", "off")
DEFAULT_MODE = "always"      # A/B 过闸（spec §10.3）后，由另一次改动只改这一行，并同步 config.py 那段注释


def check_mode(mode: str) -> None:
    """整屏解析模式唯一的合法性检查（controller ruling m7，CLAUDE.md §7 一个规则一个入口）。

    Perceiver.task_scope 和 RunConfig.__post_init__ 都调这里，不各自重写一遍。
    """
    if mode not in MODES:
        raise ValueError(f"整屏解析模式只能是 {MODES}，收到 {mode!r}")


def _perception(obs) -> dict:
    p = getattr(obs, "perception", None)
    return p if isinstance(p, dict) else {}


def parse_trigger(mode: str, requested: bool, fallback: bool, has_asker: bool) -> str | None:
    """返回 "fallback" | "model" | "always" | None（不解析）。
    off 或没有 asker 一律 None；fallback 优先于 requested，requested 优先于 always。"""
    check_mode(mode)
    if mode == "off" or not has_asker:
        return None
    if fallback:
        return "fallback"
    if requested:
        return "model"
    return "always" if mode == "always" else None


def wants_label(mode: str, obs, has_asker: bool) -> bool:
    """只在 on_demand、有 asker、这帧不是 zoom 观察、还没尝试过标注时返回 True。

    「还没尝试过」= perception.label_by 为 None：整屏解析跑过就是 "parse"（失败也算尝试过），
    短标注跑过就是 "label"。zoom 观察的 perception 是空的（spec §5.2），永远不标。
    """
    check_mode(mode)
    p = _perception(obs)
    return mode == "on_demand" and has_asker and bool(p) and p.get("label_by") is None


def fallback_due(mode: str, obs, used: bool, has_asker: bool) -> bool:
    """mode 不是 off、有 asker、本任务还没兜底过，并且这帧没拿到整屏解析结果（vision 不是 ok / empty）。

    ⚠ 2026-09-15（终审发现 1）：zoom 帧自己不解析，按它放大的那一帧算（obs.zoom_base_vision，
      只由 Perceiver.zoom 写）。原来 zoom 帧的 perception 是空的，一律算「没拿到」，
      always 下 zoom 之后 done(failed) 也兜底，白打一次 20–60 秒的整屏解析（spec §3.1 不许）。
    """
    check_mode(mode)
    base = getattr(obs, "zoom_base_vision", None)
    vision = base if base is not None else _perception(obs).get("vision")
    return mode != "off" and has_asker and not used and vision not in ("ok", "empty")
