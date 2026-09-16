"""离线报告（spec 2026-09-12 §9）：全部留档在内存里过一遍，看孪生长成什么样。后续优化的起点，不是闸门。"""
from __future__ import annotations

from collections import Counter

from iphone_agent.twin.record import simulate


def report_text(ws) -> str:
    sim = simulate(ws)
    st = sim.stats
    lines = [f"留档 {st.runs} 个运行。认屏：认出 {st.matched} / 新屏 {st.new} / 没有标注 {st.unlabeled} / "
             f"标注矛盾 {st.label_conflict} / 候选还原失败 {st.candidate_unresolved} / 不认 {st.skipped}",
             f"标注：合格 {st.labeled} / 不合规 {st.label_invalid} / 调用失败 {st.label_failed}",
             f"归属：核身份进入 {st.owner_from_entry} / 标注 {st.owner_from_label} / 候选 {st.owner_from_same_as} / "
             f"动作推断 {st.owner_from_tracker} / 不认识的 App {st.owner_unknown_app}；"
             f"App 名别名 +{st.app_aliases_added}，冲突 {st.app_alias_conflict}",
             f"旧版屏文件 {st.stale_schema} / 不支持的版本 {st.unsupported_schema} / 坏文件 {st.corrupt}"]
    for app in sim.state.all_apps():
        screens = sim.state.screens_of(app)
        if not screens:
            continue
        prov = sum(s.status == "provisional" for s in screens)
        lines.append(f"\n{sim.state.display(app)}（{app}）：{len(screens)} 屏（临时 {prov}）")
        dup = [(n, k) for n, k in Counter(s.name for s in screens).most_common(3) if k > 1]
        if dup:
            lines.append("  同名多屏：" + "，".join(f"「{n}」×{k}" for n, k in dup))
        for s in sorted(screens, key=lambda s: (-s.visits, s.id))[:5]:
            lines.append(f"  · {s.name}  别名={'、'.join(sorted(s.aliases)) or '-'}  来过 {s.visits} 次")
    return "\n".join(lines)
