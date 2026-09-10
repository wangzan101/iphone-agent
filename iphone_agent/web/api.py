"""网页的数据面。

每个函数返回 `(status_code, dict)`，不碰 HTTP，也不打印 —— 这样它们能直接测，
server.py 只负责把 dict 变成 JSON 发出去。

⚠ **一个字都不许把密钥发出去。** 这里所有关于密钥的字段都只说「有没有」和
「从哪来」（`has_key` / `api_key_source`），从不说值。这个项目的密钥泄过一次。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from iphone_agent import health
from iphone_agent.model import registry
from iphone_agent.model.config import read_config_file
from iphone_agent.model.configwrite import set_model, set_provider
from iphone_agent.model.describe import catalog, describe, providers
from iphone_agent.model.errors import ConfigError

OK, BAD_REQUEST = 200, 400


def doctor(chat) -> tuple[int, dict]:
    """结构化自检。首启向导和设置页的「连接与权限」都吃这一份。"""
    checks = health.run_checks(chat.session())
    return OK, {"checks": [{**c.as_dict(), "messages": c.messages} for c in checks], "ok": health.all_ok(checks)}


def state(chat) -> tuple[int, dict]:
    """常驻状态条要的那几样。轻量 —— 它会被反复问，不能每次都去抓帧做 OCR。"""
    try:
        r = chat.session().resolved
        model = {"spec": r.spec, "api_key_source": r.api_key_source, "ready": True,
                 "calibrated": r.model.calibrated, "allow_coord_tap": r.model.allow_coord_tap}
    except ConfigError as e:
        # 没配好模型不是异常状态，是**还没配**。界面要据此把人引到设置页去，
        # 而不是弹一个报错。
        model = {"spec": None, "ready": False, "problem": str(e), "problem_message": e.as_message()}
    return OK, {"running": chat.busy.locked(), "stopping": chat._stop.is_set(),
                "paused": chat.paused, "confirm": chat.pending_confirm,
                "confirm_message": chat.pending_confirm_message,
                "confirm_timeout_s": chat.confirm_timeout_remaining,
                "model": model, "step": chat.last_step}


def models(chat) -> tuple[int, dict]:
    """可选的模型。

    ⚠ 内置表是 **7 个 provider、只有 1 个内置模型 profile**。别的 model id 是
    自由输入，resolve 照收但按「未标定、坐标点击关闭」处理。所以设置页不能是一个
    固定的模型下拉：provider 是列表，model id 是输入框，那个唯一标定过的
    spec 作为推荐项摆最前面。
    """
    try:
        current = chat.session().resolved.spec
    except ConfigError:
        current = None

    # 内置的 + **用户自己在 config.toml 里声明过的**。后端一直支持自定义端点
    # （resolve 对不在内置表里的 provider 只要求一条：必须有 base_url），
    # 但界面此前只列内置七个 —— 用户配过的那个自己都看不见。
    provs = [{**p, "custom": False} for p in providers()]
    known = {p["name"] for p in provs}
    for name, table in (read_config_file().get("providers") or {}).items():
        if not isinstance(table, dict) or name in known:
            continue
        provs.append({"name": name, "aliases": [], "base_url": table.get("base_url") or "",
                      # 未知 provider 的环境变量名是 resolve 里现推的：<NAME>_API_KEY
                      "env_vars": [name.upper().replace("-", "_") + "_API_KEY"],
                      "custom": True})
    return OK, {"models": catalog(), "providers": provs,
                "current": current, "default": registry.DEFAULT_MODEL_SPEC}


# 运行目录名的形状：20000101-000000-abcd（合成示例）。
# ⚠ 这是**安全边界**，不是格式洁癖 —— 这个 id 会被拼进文件路径。
#   只认这个形状，再加一道 resolve 后的包含关系检查（见 _run_dir）。
RUN_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-z]{4}$")

# 一次最多列这么多次运行。历史无界，界面不必无界 —— 侧边栏也放不下。
RUNS_LIMIT = 40


def _run_dir(chat, run_id: str) -> Path | None:
    if not RUN_ID_RE.match(run_id or ""):
        return None
    root = Path(chat.workspace.runs).resolve()
    try:
        p = (root / run_id).resolve()
    except OSError:
        return None
    if p.parent != root or not p.is_dir():
        return None
    return p


def _run_brief(d: Path) -> dict | None:
    """一次运行的摘要。读不出来就跳过 —— 半截的运行记录是常态（跑到一半崩了）。"""
    try:
        r = json.loads((d / "run.json").read_text(encoding="utf-8"))
    except Exception:                                # noqa: BLE001 —— 见 docstring
        return None
    end = r.get("end_reason")
    return {
        "id": d.name,
        "task": r.get("task") or "",
        "end_reason": end,
        "ok": end == "done_success",
        # end_reason 为 None = 还在跑或者跑崩了没写完。界面要能区分「失败」和「没结束」。
        "finished": bool(end),
        "steps": r.get("steps") or 0,
        "started_at": r.get("started_at"),
        "ended_at": r.get("ended_at"),
        "model": r.get("model"),
    }


def runs(chat) -> tuple[int, dict]:
    """历史运行列表。侧边栏「最近」和回放的入口都吃这一份。

    只读。目录名带时间戳，倒序排就是最近优先，不用去读每个文件的 mtime。
    """
    root = Path(chat.workspace.runs)
    if not root.is_dir():
        return OK, {"runs": []}
    out = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        if len(out) >= RUNS_LIMIT:
            break
        b = _run_brief(d)
        if b:
            out.append(b)
    return OK, {"runs": out}


def run_detail(chat, run_id: str) -> tuple[int, dict]:
    """一次运行的每一步 —— 回放就是拿这个放。

    帧名在 steps.jsonl 里是**不带目录的**（frame_179.png），而 /api/frame/ 是相对
    runs/ 解析的。这里统一补上 `<run_id>/` 前缀，省得每个调用方各补一次、
    补漏了就是一片取不到的图。
    """
    d = _run_dir(chat, run_id)
    if d is None:
        return 404, {"error": "没有这次运行", "error_message": {"code": "api.runMissing"}}
    brief = _run_brief(d) or {"id": run_id}
    steps = []
    try:
        with (d / "steps.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                         # 坏行跳过，不能因为一行毁掉整次回放
                a = rec.get("action") or {}
                res = rec.get("result") or {}
                m = rec.get("model") or {}
                frame = rec.get("after_frame_file") or (rec.get("observation") or {}).get("frame_file")
                steps.append({
                    "n": rec.get("step"),
                    "name": a.get("name"),
                    "args": a.get("args", a.get("args_raw", {})),
                    "reason": m.get("reason", ""),
                    "changed": res.get("changed"),
                    "error": res.get("error"),
                    "hint": (res.get("hint") or "")[:400],
                    "frame": f"{run_id}/{frame}" if frame else None,
                    # 回放的节奏靠这两个：执行耗时 + 模型思考耗时。
                    "exec_ms": rec.get("exec_ms"),
                    "latency_ms": m.get("latency_ms"),
                })
    except OSError:
        pass                                         # 没有 steps.jsonl 也返回摘要，别 500
    return OK, {**brief, "steps": steps}


def learned(chat) -> tuple[int, dict]:
    """它从跑过的任务里长出来的东西：App、剧本、场景、跨任务记忆。

    这是这个项目最独特的一层，一直只在 `iphone skill list` 里看得见 —— 藏在命令行里
    是浪费（设计说明 P4）。界面要能看见，尤其要能看见**待批准**的那些：
    写类剧本（会改 UI 的操作）必须人点头模型才敢调用，那个「点头」本身就是这一页的功能。
    """
    from iphone_agent.memory.store import MemoryStore
    from iphone_agent.skills.store import SkillStore

    apps: list[dict] = []
    scenarios: list[dict] = []
    pending = 0
    try:
        cat = SkillStore().load()
        for a in cat.visible_apps():
            procs = []
            for p in cat.procedures_of(a.name):
                procs.append({
                    "name": p.name, "description": p.description, "risk": p.risk,
                    "status": p.status, "steps": len(p.steps),
                    "verified": p.provenance.verified_count,
                    # 读类只读屏幕，自动进工具表；写类必须人批准。这条区别要在界面上看得见。
                    "needs_approval": p.risk != "read" and p.status != "approved",
                })
                pending += 1 if procs[-1]["needs_approval"] else 0
            apps.append({"name": a.name, "display": a.display or a.name,
                         "risk": a.risk, "status": a.status, "updated": a.updated,
                         "procedures": procs})
        scenarios = [{"name": s.name, "description": s.description,
                      "apps": list(s.apps), "risk": s.risk, "status": s.status}
                     for s in cat.visible_scenarios()]
    except Exception as e:                           # noqa: BLE001 —— 技能目录是人手可改的，坏了也得能开页面
        apps, scenarios = [], []
        pending = 0
        skills_error = f"{type(e).__name__}: {e}"
    else:
        skills_error = ""

    memories: list[dict] = []
    try:
        entries, _ = MemoryStore(chat.workspace.memory_dir).index()
        memories = [{"name": m.name, "description": m.description, "kind": m.kind,
                     "created": m.created, "source": m.source,
                     "used_success": m.used_success, "used_failed": m.used_failed}
                    for m in entries]
    except Exception as e:                           # noqa: BLE001 —— 同上，记忆是纯文件，人可以手改手删
        memories, mem_error = [], f"{type(e).__name__}: {e}"
    else:
        mem_error = ""

    return OK, {"apps": apps, "scenarios": scenarios, "memories": memories,
                "pending": pending, "skills_error": skills_error, "memory_error": mem_error}


def get_config(chat) -> tuple[int, dict]:
    """当前配置。**只说每个 provider 有没有密钥，不说是什么。**"""
    conf = read_config_file()
    provs = {}
    for name, table in (conf.get("providers") or {}).items():
        if isinstance(table, dict):
            provs[name] = {"has_key": bool(table.get("api_key")),
                           "base_url": table.get("base_url") or ""}
    try:
        current = describe(chat.session().resolved)
    except ConfigError as e:
        current = {"spec": None, "problem": str(e), "problem_message": e.as_message()}
    return OK, {"model": conf.get("model"), "providers": provs, "current": current}


def put_config(chat, body: dict) -> tuple[int, dict]:
    """设置页保存。三样各自独立：换模型、存密钥、清密钥。

    换模型走 Session.select_model —— 它是原子替换，会把 Perceiver 一起换掉。
    模型和 Perceiver 共用一个坐标约定，只换一半就是「一个规则两个入口」那次事故。
    """
    if not isinstance(body, dict):
        return BAD_REQUEST, {"error": "请求体必须是对象", "error_message": {"code": "api.invalidBody"}}
    if chat.busy.locked():
        # 跑到一半换模型会让这一轮的前后半段用两个不同的坐标约定。
        return BAD_REQUEST, {"error": "有任务正在跑，先停下来再改设置。", "error_message": {"code": "api.busy"}}

    changed = []
    try:
        # 密钥和接口地址都写进 [providers.<name>]。接口地址是自定义端点的必要条件 ——
        # 后端本来就支持，之前只是没有写入的口子。
        if "api_key" in body or "base_url" in body:
            prov = (body.get("provider") or "").strip()
            if not prov:
                return BAD_REQUEST, {"error": "没说是哪个服务商", "error_message": {"code": "api.providerMissing"}}
            set_provider(prov, base_url=body.get("base_url"), api_key=body.get("api_key"))
            changed += [k for k in ("api_key", "base_url") if k in body]
        if body.get("model"):
            set_model(str(body["model"]))
            changed.append("model")
    except ConfigError as e:
        # ConfigError 的文案是给人看的（set_api_key 保证里面不含密钥），原样回去。
        return BAD_REQUEST, {"error": str(e), "error_message": e.as_message()}

    # 写完立刻按新配置重解析：让「测试连接」这件事在保存的当下就有答案，
    # 而不是等用户发第一条消息才发现 key 是错的。
    try:
        spec = str(body.get("model") or "") or None
        chat.session().select_model(spec or chat.session().resolved.spec)
    except ConfigError as e:
        return OK, {"ok": True, "changed": changed, "ready": False, "problem": str(e), "problem_message": e.as_message()}
    except Exception as e:                           # noqa: BLE001 —— 保存本身已经成了，别把它报成失败
        return OK, {"ok": True, "changed": changed, "ready": False, "problem": f"{type(e).__name__}: {e}",
                    "problem_message": {"code": "config.invalid", "params": {"detail": f"{type(e).__name__}: {e}"}}}
    return OK, {"ok": True, "changed": changed, "ready": True,
                "current": describe(chat.session().resolved)}


def test_model(chat) -> tuple[int, dict]:
    """「测试连接」。真发一次最小请求 —— 只有真发过才知道 key 对不对。

    ⚠ 不能只检查「key 非空」。填错的 key 和没填的 key，对用户来说是完全不同的两件事，
    而后者才是向导里最常见的。
    """
    try:
        r = chat.session().resolved
    except ConfigError as e:
        return OK, {"ok": False, "problem": str(e), "problem_message": e.as_message()}
    try:
        # tools=[] 而不是 None：默认会把整张工具表带上，探活不需要，白花 token。
        # max_tokens=1 让它一开口就停 —— 我们只想知道这条路通不通。
        reply = chat.session().model.decide(
            [{"role": "user", "content": "hi"}], (1, 1), tools=[], max_tokens=1)
        return OK, {"ok": True, "spec": r.spec, "detail": _brief(reply),
                    "detail_message": {"code": "model.testSuccess"}}
    except Exception as e:                           # noqa: BLE001 —— 任何失败都要变成一句人话
        return OK, {"ok": False, "spec": r.spec, "problem": _explain(e),
                    "problem_message": {"code": _error_kind(e), "params": {"detail": f"{type(e).__name__}: {e}"}}}


def _brief(reply) -> str:
    v = getattr(reply, "model_version", "")
    return f"通了（{v}）" if v else "通了"


def _error_kind(e: Exception) -> str:
    """Stable transport-error code, independent of a browser locale."""
    s = f"{type(e).__name__}: {e}"
    low = s.lower()
    if "401" in s or "invalid" in low and "key" in low or "unauthorized" in low:
        return "model.auth"
    if "403" in s:
        return "model.forbidden"
    if "404" in s:
        return "model.endpoint"
    if "429" in s:
        return "model.rateLimit"
    if "timeout" in low or "timed out" in low:
        return "model.timeout"
    if "ssl" in low or "certificate" in low:
        return "model.tls"
    return "model.error"


_ERROR_TEXT = {
    "model.auth": "密钥不对或者已经失效，检查一下再填一次。",
    "model.forbidden": "这个密钥没有调用这个模型的权限。",
    "model.endpoint": "端点或模型名不对 —— 确认 model id 拼对了，自定义端点确认 base_url。",
    "model.rateLimit": "被限流了，等一会儿再试。",
    "model.timeout": "连不上，超时了。检查网络，或者这个端点在国内是不是需要代理。",
    "model.tls": "TLS 握手失败，多半是代理或证书的问题。"
}


def _explain(e: Exception) -> str:
    """Legacy CLI/diagnostic text; UI uses the corresponding stable code."""
    return _ERROR_TEXT.get(_error_kind(e), f"{type(e).__name__}: {e}")
