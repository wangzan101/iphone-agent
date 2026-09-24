"""屏幕文字 → 风险档。**全项目只有这一份词表**（CLAUDE.md §7 一个规则一个入口）。

为什么词表在这里而不在 `harness/safety.py`：用它的有三方 ——
执行前的安全闸（`harness/safety.classify`）、自动探索（`eval/explore.py`）、
剧本风险（`skills/model.procedure_risk`）—— 而 `skills/` 不许 import `harness`
（skill 层 spec §6：runner 反向依赖过一次，成环；`test_skill_extract` 有一条测试盯着）。
三方都能 import 的最低一层就是这里。判据、盲点、为什么放在程序侧，仍写在 `harness/safety.py` 的模块注释里。

⚠ 2026-09-16：这段挪家之前，剧本层拿不到词表，于是「这条剧本危不危险」只能跟着 App 的
  风险档走 —— 自动切出来的 App 永远是 draft，草稿就永远是 write，自动验证永远不触发。
  见 `skills/model.procedure_risk`。往词表里加词只改这一个文件。
"""
from __future__ import annotations

import re

# 含这些字的元素**不自动点**：有人就问，没人就拒。自动探索（eval/explore.py）也用它决定不点什么。
DANGER = re.compile(
    r"发送|发布|删除|移除|清除|购买|支付|付款|订阅|提交|确认|拨打|呼叫|退出|注销|登出|"
    r"卸载|重置|抹掉|格式化|转账|充值|提现|下单|结算|开通|升级|续费|"
    r"send|post|delete|remove|buy|pay|purchase|subscribe|submit|confirm|call|"
    r"sign out|log out|logout|uninstall|reset|erase|checkout|order|"
    # 跳出 App 去系统设置的入口。2026-09-09 批跑：Chrome 里点了「前往"设置"」，
    # 系统弹出「设为默认浏览器」的教程视频并推成画中画，此后 20 多个 App 的探索
    # 全在那个浮窗的遮挡下跑 —— 16 个「打不开」，单独跑却都能开。
    r"前往.{0,3}设置|打开.{0,3}设置|去.{0,3}设置|go to settings|open settings"
    , re.I)

# 含这些字的元素**谁说都不点**。钱、账户、整机数据 —— 做错了没有回头路。
NEVER = re.compile(
    r"转账|付款|支付|充值|提现|购买|下单|结算|抹掉|格式化|重置|注销|"
    r"transfer|pay|purchase|buy|checkout|erase all|factory reset|reset|delete account"
    , re.I)

LEVELS = ("read", "write", "never")


def level_for_texts(texts) -> tuple[str, str | None]:
    """一组屏幕文字属于哪一档，取最严的那一条。返回（档位, 命中的那段文字）。
    没有文字 → ("read", None)：判不了不等于安全，调用方各自决定怎么兜（安全闸记 unclassified，
    剧本层按「没导航记录」另行判 write）。"""
    for level, rx in (("never", NEVER), ("write", DANGER)):
        hit = next((t for t in texts if rx.search(t)), None)
        if hit is not None:
            return level, hit
    return "read", None
