"""结构化的自检 —— `iphone doctor` 和网页向导共用的同一份判断。

为什么要有它：原来 `cmd_doctor` 是「边检查边 print」，判断和展示黏在一起。
界面消费不了一串文本 —— 它要知道**哪一项没过、为什么重要、怎么修、能不能一键修**，
才谈得上做一个能让人自己走完的向导（设计说明 P0）。

所以这里只产出事实，一个字都不打印。CLI 和网页各自渲染。

⚠ 每一项都必须自己扛住异常。自检的意义就是在环境坏掉的时候告诉你哪坏了，
它自己先崩了等于没有。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# 直接跳到「系统设置」对应那一页。让用户自己去找「隐私与安全性 → 辅助功能」
# 是产品化里最容易掉人的一步 —— 面板层级深，名字每个大版本还会变。
PREF_ACCESSIBILITY = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
PREF_SCREEN_RECORDING = "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"

PASS, FAIL, INFO = "pass", "fail", "info"


@dataclass(frozen=True)
class Check:
    """一项检查的全部事实。

    `why` / `fix` / `fix_url` 是给界面用的：光说「辅助功能权限：FAIL」，
    不会用电脑的人既不知道那是什么，也不知道下一步该干嘛。
    """
    id: str
    title: str
    status: str                  # pass | fail | info
    detail: str = ""             # 一行现状，比如 "iPhone 16e · 已连接"
    why: str = ""                # 这一项为什么重要
    fix: str = ""                # 人话的修法
    fix_url: str = ""            # 有就给一个「去授权」按钮
    blocking: bool = True        # False = 说明性的，不过也能跑
    messages: dict = field(default_factory=dict)  # Optional UI descriptors; CLI text stays unchanged.

    @property
    def ok(self) -> bool:
        return self.status != FAIL

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "status": self.status,
                "detail": self.detail, "why": self.why, "fix": self.fix,
                "fix_url": self.fix_url, "blocking": self.blocking}


def _messages(cid: str, detail_code: str = "", why_code: str = "", fix_code: str = "", **params) -> dict:
    return {name: {"code": code, "params": params} for name, code in
            (("title", f"check.{cid}.title"), ("detail", detail_code),
             ("why", why_code), ("fix", fix_code)) if code}


def _guard(fn: Callable[[], Check], cid: str, title: str, why: str = "") -> Check:
    """跑一项检查；它自己炸了就记成 fail，而不是把整个自检带走。"""
    try:
        return fn()
    except Exception as e:                           # noqa: BLE001 —— 见模块 docstring
        return Check(id=cid, title=title, status=FAIL,
                     detail=f"{type(e).__name__}: {e}", why=why,
                     fix="这一项检查本身没跑起来。多半是上面某项还没过，先从上往下修。",
                     messages=_messages(cid, "check.failed", fix_code="check.failedFix",
                                        detail=f"{type(e).__name__}: {e}"))


# ── 各项检查 ────────────────────────────────────────────────────────────

def check_accessibility() -> Check:
    from ApplicationServices import AXIsProcessTrusted
    ok = bool(AXIsProcessTrusted())
    return Check(
        id="perm.accessibility", title="辅助功能权限", status=PASS if ok else FAIL,
        detail="已授权" if ok else "未授权",
        why="它要把点击和键盘送进镜像窗口，才动得了手机。",
        fix="" if ok else "系统设置 → 隐私与安全性 → 辅助功能，把当前这个程序打开。",
        fix_url="" if ok else PREF_ACCESSIBILITY,
        messages=_messages("perm.accessibility", "check.granted" if ok else "check.denied",
                           "check.accessibility.why", "" if ok else "check.accessibility.fix"))


def check_screen_recording() -> Check:
    import Quartz
    ok = bool(Quartz.CGPreflightScreenCaptureAccess())
    return Check(
        id="perm.screen_recording", title="屏幕录制权限", status=PASS if ok else FAIL,
        detail="已授权" if ok else "未授权",
        why="它要看得见手机画面，才知道该点哪。",
        fix="" if ok else "系统设置 → 隐私与安全性 → 屏幕录制，把当前这个程序打开。",
        fix_url="" if ok else PREF_SCREEN_RECORDING,
        messages=_messages("perm.screen_recording", "check.granted" if ok else "check.denied",
                           "check.screen.why", "" if ok else "check.screen.fix"))


def check_mirror_window() -> Check:
    from iphone_agent.driver.window import find_mirror_window
    win = find_mirror_window()
    return Check(
        id="mirror.window", title="iPhone 镜像", status=PASS,
        detail=f"窗口 id={win.window_id} pid={win.pid} {win.rect}",
        why="所有操作都发给这一个窗口。",
        messages=_messages("mirror.window", "check.window.detail", "check.window.why",
                           detail=f"id={win.window_id} pid={win.pid} {win.rect}"))


def check_capture(session) -> Check:
    frame = session.dev.capture()
    ok = frame.width_px > 0
    return Check(
        id="mirror.capture", title="抓帧", status=PASS if ok else FAIL,
        detail=f"{frame.width_px}×{frame.height_px} scale={frame.scale:.2f}",
        why="抓不到帧就等于瞎了。",
        fix="" if ok else "确认镜像窗口没有被最小化，且屏幕录制权限已授权。",
        messages=_messages("mirror.capture", why_code="check.capture.why",
                           fix_code="" if ok else "check.capture.fix"))


def check_ocr(session) -> Check:
    obs = session.per.observe(session.dev.capture())
    n = len(obs.elements)
    return Check(
        id="perceive.ocr", title="文字识别", status=PASS if n else FAIL,
        detail=f"读到 {n} 个元素",
        why="它靠读屏上的文字来认路 —— 镜像不给控件树，这是唯一的信息来源。",
        fix="" if n else "当前这一屏可能确实没有文字（比如纯图片页）。换一屏再试。",
        messages=_messages("perceive.ocr", "check.ocr.detail", "check.ocr.why",
                           "" if n else "check.ocr.fix", count=n))


def check_model(session) -> Check:
    from iphone_agent.model.errors import ConfigError
    try:
        r = session.resolved
    except ConfigError as e:
        return Check(
            id="model", title="模型", status=FAIL, detail=str(e),
            why="它的判断全部来自外部大模型，没有模型就只剩一双手。",
            fix="在设置里选一个模型并填上 API 密钥。",
            messages={**_messages("model", why_code="check.model.why", fix_code="check.model.fix"),
                      "detail": e.as_message()})
    m = r.model
    return Check(
        id="model", title="模型", status=PASS,
        detail=(f"{r.spec} · 密钥来自 {r.api_key_source} · "
                f"坐标点击{'开' if m.allow_coord_tap else '关'}"
                f"（{'已标定' if m.calibrated else '未标定'}）"),
        why="模型决定它怎么判断下一步该干什么。",
        messages=_messages("model", "check.model.detail", "check.model.why",
                           spec=r.spec, source=r.api_key_source, coord=str(m.allow_coord_tap)))


def check_injector(session) -> Check:
    """说明性的一条 —— 注入到底生效没有，只读检测不出来，必须人眼确认。"""
    from iphone_agent.driver.skylight import available as sky_available
    try:
        mode = session.dev.injector_mode()
    except Exception:                                # noqa: BLE001 —— 见模块 docstring
        mode = "unknown"
    return Check(
        id="injector", title="注入路径", status=INFO, blocking=False,
        detail=f"{mode} · SkyLight 可用={sky_available()}",
        why="后台注入让点击和键盘都不抢焦点 —— 它操作手机的时候你还能用电脑。",
        fix="注入是否真的生效**只读检测不出来**，需要监督式真机验证。",
        messages=_messages("injector", "check.injector.detail", "check.injector.why",
                           "check.injector.fix", mode=mode, available=str(sky_available())))


# ── 汇总 ────────────────────────────────────────────────────────────────

def run_checks(session) -> list[Check]:
    """按顺序跑完所有检查。顺序即依赖顺序：权限 → 窗口 → 抓帧 → 识别 → 模型。

    前面的没过，后面的多半也过不了 —— 界面应该引导用户**从上往下**修。
    """
    return [
        _guard(check_accessibility, "perm.accessibility", "辅助功能权限",
               "它要把点击和键盘送进镜像窗口，才动得了手机。"),
        _guard(check_screen_recording, "perm.screen_recording", "屏幕录制权限",
               "它要看得见手机画面，才知道该点哪。"),
        _guard(check_mirror_window, "mirror.window", "iPhone 镜像",
               "所有操作都发给这一个窗口。"),
        _guard(lambda: check_capture(session), "mirror.capture", "抓帧",
               "抓不到帧就等于瞎了。"),
        _guard(lambda: check_ocr(session), "perceive.ocr", "文字识别",
               "它靠读屏上的文字来认路。"),
        _guard(lambda: check_model(session), "model", "模型",
               "它的判断全部来自外部大模型。"),
        _guard(lambda: check_injector(session), "injector", "注入路径",
               "后台注入让点击和键盘都不抢焦点。"),
    ]


def all_ok(checks: list[Check]) -> bool:
    """只看拦路的那几项。`injector` 这种说明性的条目不参与判定。"""
    return all(c.ok for c in checks if c.blocking)


def as_dicts(checks: list[Check]) -> list[dict]:
    return [c.as_dict() for c in checks]
