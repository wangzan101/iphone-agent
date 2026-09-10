"""驱动准入：逐个验证**基本操作**在真机上成立。

这是本项目的主验证工具。目标不是「把某个任务做完」，而是回答一张表：
每一个原语能不能用、可靠不可靠。任务能不能完成是下游的事，原语是地基。

每一步都：执行 → 打印 changed/hamming/text_diff → **停下来问你眼睛看到了什么**。
机器的判定可能骗人（阈值、时序、动画），人眼是这里唯一的基准真值。

跑完输出一张原语清单和一份 JSON。

用法：手机停在**主屏幕**，然后
    python scripts/driver_gate.py
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import _screen
from _screen import CHEVRONS, find, find_back  # noqa: F401  共用，勿在此重写

from iphone_agent.driver.device import Device
from iphone_agent.driver.geometry import image_to_screen
from iphone_agent.harness.actions import Action, validate_action
from iphone_agent.harness.executor import Executor
from iphone_agent.perceive.change import did_change
from iphone_agent.perceive.observe import Perceiver

dev = Device()
per = Perceiver()
ex = Executor(dev, per)
results = []


def look():
    return _screen.look(dev, per)


def frame_of(o):
    return _screen.frame_of(o)


def settle_a_bit(kind="key"):
    _screen.settle_a_bit(dev, kind)


def find_across_pages(text, max_pages=5):
    return _screen.find_across_pages(
        dev, per, text, max_pages,
        on_page=lambda n: print(f"   （往右翻了 {n} 页找到「{text}」）"))


def any_app_label(o):
    """当前页上任意一个像「App 图标标签」的元素，用来在目标 App 不在时仍能测 icon_above。

    判据：文字短、落在图标网格区域（纵向 30%–88%）、不在最左右边缘。
    """
    cands = [e for e in o.elements
             if 1 <= len(e.text.strip()) <= 6
             and o.height_px * 0.30 < e.center[1] < o.height_px * 0.88
             and o.width_px * 0.08 < e.center[0] < o.width_px * 0.92]
    return cands[len(cands) // 2] if cands else None


def step(name, expect, run):
    """执行一个原语并记录结果。expect 是「应该看到什么」，给人核对用。"""
    print(f"\n[{name}]  期望：{expect}")
    before = look()
    note = ""
    try:
        note = run(before) or ""
    except Exception as e:
        print(f"   ✗ 抛异常：{type(e).__name__}: {e}")
        results.append({"primitive": name, "ok": False, "why": f"{type(e).__name__}: {e}"})
        input("   回车继续")
        return None
    time.sleep(1.2)
    after = look()
    r = did_change(before, after)
    print(f"   changed={r.changed} hamming={r.hamming} text_diff={r.text_diff} {note}")
    ans = input("   眼睛看到的是期望的效果吗？(Y/n/s=跳过) ").strip().lower()
    ok = None if ans in ("s", "skip") else ans not in ("n", "no", "否")
    results.append({"primitive": name, "ok": ok, "changed": r.changed,
                    "hamming": r.hamming, "text_diff": r.text_diff, "note": note})
    return after


def tap_el(o, el, target="text"):
    a = validate_action(Action("tap", {"id": el.id, "target": target}, "gate", None, "g"), o)
    sx, sy = image_to_screen(a.args["x"], a.args["y"], frame_of(o))
    dev.tap(sx, sy)
    return f"点 [{el.id}] {el.text!r} target={target} → 图像({a.args['x']},{a.args['y']})"


try:
    print("请把手机停在**主屏幕**，回车开始")
    input()

    # ---- 0. 感知：框和文字对不对得上 ----
    o = look()
    o.marked_image.save("gate_marked.png")
    print(f"\n[感知] 已存 gate_marked.png，{len(o.elements)} 个元素，图像 {o.width_px}x{o.height_px}")
    ans = input("   打开核对：红框和文字对齐吗？(Y/n) ").strip().lower()
    results.append({"primitive": "OCR 框对齐", "ok": ans not in ("n", "no", "否")})

    # ---- 1. 系统键 ----
    step("key home", "回到主屏幕第一页（本来就在则不动）", lambda o: dev.key("home"))
    step("key app_switcher", "打开多任务卡片界面", lambda o: dev.key("app_switcher"))
    step("key home（从多任务退出）", "回到主屏幕", lambda o: dev.key("home"))

    # ---- 2. 横向翻页 ----
    # ⚠ direction 说的是「你想看到的内容在哪边」，不是手指往哪滑。
    #   right = 看右边 = 下一页；left = 看左边 = 上一页（主屏幕第一页往左是 -1 屏小组件页）。
    a = step("scroll right", "翻到**下一页**主屏幕，并且完整停在某一页上", lambda o: dev.scroll("right", "page"))
    if a is not None:
        ans = input("   翻页是否卡在两屏中间？(y=卡住/N=完整停住) ").strip().lower()
        results.append({"primitive": "翻页吸附", "ok": ans not in ("y", "yes", "是")})
    step("scroll left", "翻回**上一页**", lambda o: dev.scroll("left", "page"))

    # ---- 3. tap 的两类目标 ----
    dev.key("home"); settle_a_bit()
    o, label = find_across_pages("设置")
    target_name = "设置"
    if label is None:
        o = look()
        label = any_app_label(o)
        target_name = label.text.strip() if label else None
        if label is not None:
            print(f"\n⚠ 翻遍主屏都没有「设置」，改用当前页的「{target_name}」测 icon_above")

    if label is None:
        results.append({"primitive": "tap target=icon_above", "ok": None, "why": "页面上找不到任何 App 标签"})
        results.append({"primitive": "tap target=text（对照）", "ok": None, "why": "同上"})
    else:
        step(f"tap target=text（对照：点「{target_name}」的标签本身）",
             "**什么都不该发生** —— 标签不是可点区域",
             lambda o: tap_el(o, find(o, target_name, exact=True), "text"))
        step(f"tap target=icon_above（点「{target_name}」的图标）", f"打开 {target_name}",
             lambda o: tap_el(o, find(o, target_name, exact=True), "icon_above"))
        dev.key("home"); settle_a_bit()

    # ---- 4. 竖直滚动 + 文字目标：需要设置 App ----
    o, label = find_across_pages("设置")
    if label is not None:
        tap_el(o, label, "icon_above")
        settle_a_bit("tap")
        # ⚠ iOS 记住 App 上次停留的页面，设置多半不在首页 —— 原来直接查「通用」，
        #   于是明明打开成功却判成「没能进入设置」（2026-09-07 踩到）。
        #   一路点返回退到首页，顺便也把返回控件这个原语用上了。
        for _ in range(8):
            o = look()
            back = find_back(o)
            if back is None:
                break
            print(f"   （设置不在首页，点返回：{back.text!r}）")
            tap_el(o, back, "text")
            settle_a_bit("tap")
        o = look()
    if find(o, "通用") or find(o, "飞行模式"):
        step("scroll down", "设置列表往下滚一屏", lambda o: dev.scroll("down", "page"))
        step("scroll up", "滚回去", lambda o: dev.scroll("up", "page"))
        o = look()
        if find(o, "通用"):
            step("tap target=text（列表行）", "进入「通用」页",
                 lambda o: tap_el(o, find(o, "通用"), "text"))
            o = look()
            if find_back(o):
                step("tap 返回控件", "退回设置首页", lambda o: tap_el(o, find_back(o), "text"))
            else:
                results.append({"primitive": "tap 返回控件", "ok": None, "why": "没找到返回控件"})
        else:
            results.append({"primitive": "tap target=text（列表行）", "ok": None, "why": "没找到「通用」"})
    else:
        print("\n⚠ 没能进入设置 App，跳过竖直滚动与列表点击")
        for n in ("scroll down", "scroll up", "tap target=text（列表行）", "tap 返回控件"):
            results.append({"primitive": n, "ok": None, "why": "没能进入设置 App"})

    # ---- 6. 输入 ----
    dev.key("home"); time.sleep(1.2)
    step("key spotlight", "弹出 Spotlight 搜索", lambda o: dev.key("spotlight"))
    # type 现在优先逐键、非 ASCII 才退回粘贴，两条路要分开测：
    # 逐键实测可用，粘贴实测四种情况全败（见 设计说明）。
    step("type ASCII（走逐键）", "搜索框里出现 safari",
         lambda o: f"路径={dev.type('safari')}")
    dev.key("home"); time.sleep(1.0); dev.key("spotlight"); time.sleep(1.2)
    # ⚠ 分清楚这是**哪一层**打不出中文。
    #   驱动层的 dev.type() 拿到汉字只能退回粘贴，而粘贴实测是坏的 —— 这一条至今成立。
    #   但中文输入**整体是通的**，解法在上一层：harness 把中文转拼音打进 iOS 自己的
    #   输入法，再读候选栏、点选（executor._type_via_ime，2026-09-08 端到端验过）。
    #   所以这一项的「期望」仍然是看不到文字，但它证明的只是「驱动层这条路不通」，
    #   不是「这个项目打不出中文」—— 紧接着下一项就走通的那条路。
    step("type 中文·驱动层（退回粘贴，这一层确实不通）",
         "搜索框里**看不到**「设置」——驱动层的已知缺陷，看不到才算符合预期。"
         "中文输入的真正解法是下一项",
         lambda o: f"路径={dev.type('设置')}")

    # 中文输入的**真实能力**：走 harness 的输入法编排。
    dev.select_all_and_delete(); time.sleep(0.8)

    def do_cn(o):
        res, new = ex.run(Action("type", {"text": "设置"}, "gate", None, "g"), o)
        return (f"→ ok={res.ok} 拼音={res.extra.get('pinyin')} "
                f"选中={res.extra.get('picked')} 候选={res.extra.get('candidates')}")

    step("type 中文·harness（拼音→候选→点选）",
         "搜索框里**真的出现汉字「设置」**（不是拼音字母）", do_cn)

    # ---- 7. 复合动作 ----
    dev.key("home"); time.sleep(1.2)

    def do_open(o):
        res, _ = ex.run(Action("open_app", {"name": "设置"}, "gate", None, "g"), o)
        return f"→ {res.to_json()}"

    step("open_app（复合）", "打开设置 App，且返回里 typed 如实反映输入成没成", do_open)

    # ---- 8. 高阶滚动（复合）----
    # 这两个原语的价值是「把 N 次滚动压成一步」，所以必须连 Executor 一起测：
    # 单独发几次 dev.scroll 验不出内部那个「滚 → 等稳 → 观察 → 判到底」的循环，
    # 而循环里最不确定的恰恰是「滚不动了」判得准不准（用的是 did_change，与单步 scroll 同一套）。
    collected = {}

    def do_collect(o):
        res, _ = ex.run(Action("collect", {"direction": "down"}, "gate", None, "g"), o)
        collected.update(json.loads(res.to_json()))
        return (f"滚了 {collected.get('screens')} 屏 到底={collected.get('reached_end')} "
                f"收到 {collected.get('count')} 行 前几行={collected.get('lines', [])[:5]}")

    step("collect down（复合）",
         "设置列表一路滚到底：屏数 >1、reached_end=true、行数明显多于一屏能显示的",
         do_collect)

    # 目标取 collect 收到的第一行 —— 它来自这台机器这一刻的真实屏幕，
    # 不写死任何 iOS 版本相关的文案。滚到底之后往回找它，正好把另一个方向也验了。
    first = next((t for t in collected.get("lines", []) if len(t.strip()) >= 2), None)
    if first is None:
        results.append({"primitive": "scroll_until up（复合）", "ok": None,
                        "why": "collect 没收到可用的文字"})
    else:
        def do_scroll_until(o):
            res, _ = ex.run(Action("scroll_until", {"direction": "up", "text": first},
                                   "gate", None, "g"), o)
            return f"→ {res.to_json()}"

        step(f"scroll_until up（复合，找「{first}」）",
             f"往回滚到看见「{first}」，返回里 found=true，且当前屏幕上真的有它",
             do_scroll_until)

    dev.key("home")

finally:
    dev.release_all()

# ---- 汇总 ----
print("\n" + "=" * 60)
print("原语清单")
print("=" * 60)
sym = {True: "✅", False: "❌", None: "⏭"}
for r in results:
    line = f"  {sym[r.get('ok')]} {r['primitive']:<32}"
    if r.get("changed") is not None:
        line += f" changed={r['changed']} h={r['hamming']} t={r['text_diff']}"
    if r.get("why"):
        line += f"  ({r['why']})"
    print(line)

bad = [r["primitive"] for r in results if r.get("ok") is False]
skipped = [r["primitive"] for r in results if r.get("ok") is None]
print(f"\n通过 {sum(1 for r in results if r.get('ok'))} / 失败 {len(bad)} / 跳过 {len(skipped)}")
if bad:
    print("失败：" + "、".join(bad))

out = Path("runs") / f"gate-{datetime.now():%Y%m%d-%H%M%S}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n结果已存 {out}")
sys.exit(1 if bad else 0)
