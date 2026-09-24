"""三种知识文件的内部表示与校验。校验一律**拒绝**不合规输入，从不规整（同记忆系统）。"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

from iphone_agent import config, fingerprint
from iphone_agent.appids import slugify
from iphone_agent.riskwords import level_for_texts
from iphone_agent.skills import frontmatter

RISKS = ("read", "write", "irreversible")
RISK_RANK = {r: i for i, r in enumerate(RISKS)}
APP_STATUSES = ("draft", "verified")
PROC_STATUSES = ("draft", "verified", "stale")
# 技能的状态：manual = 人写的（可信）；proposed = 模型提议的场景（待审）；
# draft/verified/stale = 自动从运行里提出来的剧本走的三态。一个技能只有一个 status。
SKILL_STATUSES = ("manual", "proposed", "draft", "verified", "stale")
SCENARIO_STATUSES = SKILL_STATUSES        # 旧名，别处还在引用
STEP_DOS = ("open_app", "tap", "scroll_until", "type", "read")
HOWS = ("text", "icon_above", "row_right", "row_left")
DIRECTIONS = ("up", "down", "left", "right")
EXPECT_SOURCES = ("single_run", "intersection")
# 这一步当年在源运行里**做成了什么**。证据，不是执行参数：不进 hash，不改变怎么执行。
# navigated 换了一屏；state_changed 整屏没换、只有点击处变了（开关、勾选）；unknown 判不出来。
# 缺这一栏（老文件）等于「没记录过」，风险照 unknown 算 —— 见 procedure_risk。
EFFECTS = ("navigated", "state_changed", "unknown")
NAVIGATED = "navigated"
# 只挪动位置、不留下痕迹的步骤。tap 另判（要看目标文字和当年的效果）。
NAV_DOS = ("open_app", "scroll_until", "read")
_LEVEL_RISK = {"read": "read", "write": "write", "never": "irreversible"}
_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
_NAME_FORBIDDEN = ("/", "\\", "\0")


class SkillError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_name(name, what: str = "name") -> None:
    if (not isinstance(name, str) or any(c in name for c in _NAME_FORBIDDEN)
            or not config.MEMORY_NAME_RE.match(name)):
        raise SkillError("invalid_name",
                         f"{what} 必须匹配 {config.MEMORY_NAME_RE.pattern}（小写字母数字和连字符），收到 {name!r}")


def _enum(v, allowed, what):
    if v not in allowed:
        raise SkillError("invalid_field", f"{what} 必须是 {allowed} 之一，收到 {v!r}")
    return v


def _str(meta: dict, key: str, what: str, required: bool = True) -> str:
    v = meta.get(key)
    if v is None or v == "":
        if required:
            raise SkillError("missing_field", f"{what} 缺少字段 {key}")
        return ""
    if not isinstance(v, str):
        raise SkillError("invalid_field", f"{what} 的 {key} 必须是字符串，收到 {v!r}")
    return v


def _str_list(meta: dict, key: str, what: str) -> tuple[str, ...]:
    """JSON 数组字段（frontmatter 里写成 `aliases: ["设置", "Settings"]`）。
    缺省是空；写错了**拒绝**，不规整（同文件其余校验的理由）。"""
    v = meta.get(key)
    if v is None or v == "":
        return ()
    if not isinstance(v, list) or any(not isinstance(x, str) or not x.strip() for x in v):
        raise SkillError("invalid_field", f"{what} 的 {key} 必须是非空字符串的数组，收到 {v!r}")
    return tuple(x.strip() for x in v)


# ---- App ----
@dataclass(frozen=True)
class AppProfile:
    """关于一个 App 的**通用知识**：它是什么、打开后在哪、界面结构、全局陷阱、不可逆动作。
    「在这个 App 里怎么做某件事」不在这里 —— 那是技能（Skill）。判断标准：
    没有任何具体任务时这条信息还有用吗？有 → 这里；没有 → 技能。"""
    name: str
    display: str
    open: str
    risk: str
    status: str
    updated: str
    body: str = ""
    description: str = ""       # 一句话，进索引给路由看
    # 这个 App 的其它叫法：中文显示名、英文显示名、2026-09-17 之前用过的旧拼音 id。
    # **别名只有这一处**（CLAUDE.md §7）——`appids.AliasIndex` 就是从这里建索引的。
    # 要支持一种新语言，往这里加它在那种语言下的显示名，不改代码。
    aliases: tuple[str, ...] = ()


def app_visible(app: AppProfile) -> bool:
    """App 对模型可见：写了实际内容——非空正文或非空 description，不问 status（2026-09-17 起）。
    `status` 只管「人是否确认过声明的 risk」，跟能不能进索引是两件事（见 store.Catalog.visible_apps）。
    自动切出的空壳（extract.ensure_app：body="" 且没有 description）永远不可见，防止索引被
    「没人写过一个字」的占位 App 灌满。"""
    return bool(app.body.strip()) or bool(app.description.strip())


def app_from_markdown(text: str, dirname: str) -> AppProfile:
    try:
        meta, body = frontmatter.parse(text)
    except frontmatter.FrontmatterError as e:
        raise SkillError("bad_frontmatter", str(e)) from e
    validate_name(dirname, "App id")
    return AppProfile(
        name=dirname,                       # 以目录名为准；frontmatter 里的 name 只是给人看的冗余
        display=_str(meta, "display", "APP.md"),
        open=_str(meta, "open", "APP.md"),
        risk=_enum(_str(meta, "risk", "APP.md"), RISKS, "risk"),
        status=_enum(_str(meta, "status", "APP.md"), APP_STATUSES, "status"),
        updated=_str(meta, "updated", "APP.md"),
        body=body,
        description=_str(meta, "description", "APP.md", required=False),
        aliases=_str_list(meta, "aliases", "APP.md"),
    )


def app_to_markdown(app: AppProfile) -> str:
    meta = {"name": app.name, "display": app.display, "open": app.open,
            "risk": app.risk, "status": app.status, "updated": app.updated}
    if app.description:
        meta["description"] = app.description
    if app.aliases:
        meta["aliases"] = list(app.aliases)
    return frontmatter.dump(meta, app.body)


# ---- 剧本 ----
@dataclass(frozen=True)
class Step:
    do: str
    target: str | None = None
    how: str = "text"
    find: dict | None = None
    expect: tuple[str, ...] = ()
    direction: str | None = None
    text: str | None = None
    row: str | None = None
    as_: str | None = None
    effect: str | None = None       # 源运行里这一步的效果，见 EFFECTS

    def to_json(self) -> dict:
        d: dict = {"do": self.do}
        if self.target is not None:
            d["target"] = self.target
        if self.do == "tap":
            d["how"] = self.how
        if self.find is not None:
            d["find"] = dict(self.find)
        if self.direction is not None:
            d["direction"] = self.direction
        if self.text is not None:
            d["text"] = self.text
        if self.row is not None:
            d["row"] = self.row
        if self.as_ is not None:
            d["as"] = self.as_
        if self.effect is not None:
            d["effect"] = self.effect
        if self.do != "read":
            d["expect"] = list(self.expect)
        return d

    @staticmethod
    def from_json(d: dict, k: int) -> Step:
        what = f"第 {k} 步"
        if not isinstance(d, dict):
            raise SkillError("invalid_step", f"{what} 必须是对象")
        do = _enum(d.get("do"), STEP_DOS, f"{what} 的 do")
        target = d.get("target")
        how = d.get("how", "text")
        find = d.get("find")
        expect = d.get("expect", [])
        direction = d.get("direction")
        text = d.get("text")
        row = d.get("row")
        as_ = d.get("as")
        effect = d.get("effect")
        if effect is not None:
            _enum(effect, EFFECTS, f"{what} 的 effect")
        if do in ("open_app", "tap", "scroll_until") and not (isinstance(target, str) and target.strip()):
            raise SkillError("invalid_step", f"{what}（{do}）需要非空的 target")
        if do == "tap":
            _enum(how, HOWS, f"{what} 的 how")
            if find is not None:
                if (not isinstance(find, dict) or find.get("direction") not in DIRECTIONS
                        or not isinstance(find.get("max_screens"), int) or isinstance(find.get("max_screens"), bool)
                        or find["max_screens"] < 1):
                    raise SkillError("invalid_step", f"{what} 的 find 必须是 {{direction, max_screens>=1}}")
                find = {"direction": find["direction"], "max_screens": find["max_screens"]}
        elif find is not None:
            raise SkillError("invalid_step", f"{what}：只有 tap 能带 find")
        if do == "scroll_until":
            _enum(direction, DIRECTIONS, f"{what} 的 direction")
        if do == "type" and not (isinstance(text, str) and text):
            raise SkillError("invalid_step", f"{what}（type）需要非空的 text")
        if do == "read":
            if not (isinstance(row, str) and row.strip()) or not (isinstance(as_, str) and as_.strip()):
                raise SkillError("invalid_step", f"{what}（read）需要 row 和 as")
            if "expect" in d:
                raise SkillError("invalid_step", f"{what}（read）不能有 expect")
        else:
            if not isinstance(expect, list) or not all(isinstance(x, str) and x.strip() for x in expect):
                raise SkillError("invalid_step", f"{what} 的 expect 必须是非空字符串数组（可为空数组）")
        return Step(do=do, target=target, how=how if do == "tap" else "text", find=find,
                    expect=tuple(expect) if do != "read" else (), direction=direction,
                    text=text, row=row, as_=as_, effect=effect)


@dataclass
class Provenance:
    runs: list[str] = field(default_factory=list)
    verified_count: int = 0
    last_ok: str | None = None
    last_fail: str | None = None
    fail_streak: int = 0
    macos: str | None = None


@dataclass
class Procedure:
    name: str
    app: str
    description: str
    params: dict
    returns: list[str]
    risk: str
    status: str
    expect_source: str
    provenance: Provenance
    steps: list[Step]
    weak_steps: list[int] = field(default_factory=list)

    @property
    def tool_name(self) -> str:
        return f"{self.app}__{self.name}"

    def hash(self) -> str:
        return step_hash(self.steps)

    def placeholders(self) -> set[str]:
        found: set[str] = set()
        for s in self.steps:
            for v in (s.target, s.text):
                if v:
                    found.update(_PLACEHOLDER.findall(v))
        return found


def weak_of(steps: list[Step]) -> list[int]:
    """指纹弱的步骤号（1 起）：过滤之后锚点不足 WEAK_EXPECT 个，这一步说明不了「到了该到的屏」。
    提取、合并、CLI 共用这一个判据（CLAUDE.md §7）。read 步没有 expect，不参与。"""
    return [k for k, s in enumerate(steps, start=1) if s.do != "read" and fingerprint.is_weak(s.expect)]


def weak_steps(p: Procedure) -> list[int]:
    """给人看的「指纹弱」：算出来的 ∪ 文件里记着的。
    ⚠ 2026-09-16：老文件的 expect 没过过滤（状态栏乱码、视觉图标名都在里面），条数是虚高的 ——
      光按条数算会把一条其实只剩 2 个真锚点的步骤说成不弱。文件里记着的那一份照旧算数，
      重新提取一次之后写回的就是算出来的那一份。"""
    return sorted(set(weak_of(p.steps)) | set(p.weak_steps))


def step_hash(steps: list[Step]) -> str:
    """只看会影响执行的字段；expect / row / as 不参与，改指纹不该变成另一条剧本。"""
    canon = []
    for s in steps:
        d = {"do": s.do}
        for k in ("target", "how", "direction", "text", "find"):
            v = getattr(s, k)
            if v is not None and not (k == "how" and s.do != "tap"):
                d[k] = v
        canon.append(d)
    blob = json.dumps(canon, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _step_risk(s: Step) -> str:
    """一步做了什么。判不出来的一律 write —— 错判成 write 只是多一次人工，错判成 read 是事故。"""
    if s.do == "tap":
        r = _LEVEL_RISK[level_for_texts([s.target or ""])[0]]
        if r != "read":
            return r                            # 危险词命中：至少 write，NEVER 那档仍是 irreversible
        # 文字看着无害，还要问当年它做成了什么：换屏才算「只是走路」，
        # 开关翻转（state_changed）和判不出来的（unknown / 老文件没这一栏）都按写算。
        return "read" if s.effect == NAVIGATED else "write"
    if s.do in NAV_DOS:
        return "read"
    return "write"                              # type，以及将来加进 STEP_DOS 的任何东西


def procedure_risk(steps: list[Step], app: AppProfile | None = None) -> str:
    """**这条剧本自己做了什么**决定它的风险档。全项目只有这一个入口。

    ⚠ 2026-09-16 真机死锁：原来「草稿的 risk 取 App 的 risk，App 是 draft 就取 write」
      （spec §3.2）。自动切出来的 App 永远是 draft，于是 15 条提取出来的剧本全是
      `draft write`；而自动验证要求 `risk == "read"`（extract.ingest_draft），
      `approve` 又只改 status 不改 risk（store.approve_procedure）—— 「学到的剧本变成工具」
      这条路一次都没通过。风险描述的是动作的副作用，不是 App 的身份，所以改成按步骤算。

    App 的档只剩一个作用：**人批准过**（status=verified）的 App 声明了更严的档就按更严的算。
    draft 的 App 那个 risk 是自动填的占位值，不是任何人的声明，不参与。
    """
    rank = RISK_RANK["read"]
    for s in steps:
        rank = max(rank, RISK_RANK[_step_risk(s)])
    if app is not None and app.status == "verified":
        rank = max(rank, RISK_RANK[app.risk])    # 只会更严，永不放松
    return RISKS[rank]


def is_callable(p: Procedure) -> bool:
    """能不能进模型的工具列表：verified **且** read（spec §4.3，写类本期不自动执行）。
    索引行、eligible_procedures、CLI 的「可批准」标记都问这一个函数。"""
    return p.status == "verified" and p.risk == "read"


def manual_only_note(p: Procedure) -> str | None:
    """批准之后仍然不能自动调用时，这句话必须说出来 —— 否则人以为批准完就能用了。"""
    if p.risk == "read":
        return None
    return (f"但它是写类剧本（risk={p.risk}），本期不进模型的工具列表："
            f"只能在人在场时 `iphone skill run {p.app}/{p.name}` 手动跑。")


def procedure_from_json(text: str, app: str) -> Procedure:
    validate_name(app, "App id")
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise SkillError("bad_json", f"剧本不是合法 JSON：{e}") from e
    if not isinstance(d, dict) or d.get("schema_version") != 1:
        raise SkillError("bad_schema", "剧本顶层必须是对象且 schema_version 为 1")
    name = _str(d, "name", "剧本")
    validate_name(name, "剧本名")
    params = d.get("params", {})
    if not isinstance(params, dict) or not all(
            isinstance(k, str) and isinstance(v, dict) and v.get("type") == "string" for k, v in params.items()):
        raise SkillError("invalid_field", "params 必须是 {名字: {type: 'string', description: ...}}")
    returns = d.get("returns", [])
    if not isinstance(returns, list) or not all(isinstance(r, str) for r in returns):
        raise SkillError("invalid_field", "returns 必须是字符串数组")
    steps_raw = d.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise SkillError("invalid_field", "steps 必须是非空数组")
    steps = [Step.from_json(s, k) for k, s in enumerate(steps_raw, start=1)]
    read_names = [s.as_ for s in steps if s.do == "read"]
    if sorted(read_names) != sorted(returns) or len(set(read_names)) != len(read_names):
        raise SkillError("returns_mismatch", f"returns {returns} 必须与 read 步的 as {read_names} 一一对应")
    prov_raw = d.get("provenance", {})
    if not isinstance(prov_raw, dict):
        raise SkillError("invalid_field", "provenance 必须是对象")
    prov = Provenance(
        runs=list(dict.fromkeys(prov_raw.get("runs", []))),
        verified_count=int(prov_raw.get("verified_count", 0)),
        last_ok=prov_raw.get("last_ok"), last_fail=prov_raw.get("last_fail"),
        fail_streak=int(prov_raw.get("fail_streak", 0)), macos=prov_raw.get("macos"))
    weak = d.get("weak_steps", [])
    if not isinstance(weak, list) or not all(isinstance(w, int) for w in weak):
        raise SkillError("invalid_field", "weak_steps 必须是整数数组")
    p = Procedure(
        name=name, app=app, description=_str(d, "description", "剧本"),
        params=params, returns=list(returns),
        risk=_enum(d.get("risk"), RISKS, "risk"),
        status=_enum(d.get("status"), PROC_STATUSES, "status"),
        expect_source=_enum(d.get("expect_source", "single_run"), EXPECT_SOURCES, "expect_source"),
        provenance=prov, steps=steps, weak_steps=list(weak))
    undeclared = p.placeholders() - set(params)
    if undeclared:
        raise SkillError("undeclared_param", f"步骤里用了未声明的参数 {sorted(undeclared)}")
    return p


def procedure_to_json(p: Procedure) -> str:
    d = {
        "schema_version": 1, "name": p.name, "description": p.description,
        "params": p.params, "returns": p.returns, "risk": p.risk, "status": p.status,
        "expect_source": p.expect_source, "provenance": asdict(p.provenance),
        "weak_steps": p.weak_steps, "steps": [s.to_json() for s in p.steps],
    }
    return json.dumps(d, ensure_ascii=False, indent=2) + "\n"


def auto_name(text: str, h: str, n: int = 4) -> str:
    """自动生成的剧本名：前 20 字的拼音 slug + 步骤 hash 前 n 位（默认 4）。永不用来覆盖同名文件。
    n 可以调大——仅供 pending 目录里化解撞名用（extract._queue_pending）；总长要留在
    MEMORY_NAME_RE 的 48 字符上限内，所以 slug 长度跟着 n 一起缩。"""
    return f"{slugify(text[:20], max_len=max(1, 47 - n))}-{h[:n]}"


# ---- 技能 ----
@dataclass(frozen=True)
class Skill:
    """**怎么做一件事**。单 App 和跨 App 同一形态，靠 apps 声明用到谁。

    apps 是**出处声明**不是使用限制：这份步骤是在这些 App 上写的；命中时把它们的 APP.md
    一起带出来。模型在别的 App 上照样可以参考它的「需要的信息」和「做成的标志」。
    procedure 可选：有就是 harness 可逐步核对指纹的可执行版本，没有就照 body 的 prose 走。"""
    name: str
    description: str
    apps: tuple[str, ...]
    risk: str
    status: str
    body: str = ""
    procedure: Procedure | None = None


Scenario = Skill        # 旧名


def skill_from_markdown(text: str, stem: str, procedure: Procedure | None = None) -> Skill:
    try:
        meta, body = frontmatter.parse(text)
    except frontmatter.FrontmatterError as e:
        raise SkillError("bad_frontmatter", str(e)) from e
    validate_name(stem, "技能名")
    apps = meta.get("apps")
    if not isinstance(apps, list) or not apps or not all(isinstance(a, str) for a in apps):
        raise SkillError("invalid_field", 'apps 必须是非空 JSON 数组，如 apps: ["alipay", "jizhangben"]')
    for a in apps:
        validate_name(a, "apps 里的 App id")
    return Skill(
        name=stem, description=_str(meta, "description", "SKILL.md"),
        apps=tuple(apps),
        risk=_enum(_str(meta, "risk", "SKILL.md"), RISKS, "risk"),
        status=_enum(_str(meta, "status", "SKILL.md"), SKILL_STATUSES, "status"),
        body=body, procedure=procedure)


scenario_from_markdown = skill_from_markdown        # 旧名


def skill_to_markdown(s: Skill) -> str:
    meta = {"name": s.name, "description": s.description, "apps": list(s.apps),
            "risk": s.risk, "status": s.status}
    return frontmatter.dump(meta, s.body)


scenario_to_markdown = skill_to_markdown            # 旧名


def scenario_risk(declared: str, referenced: list[str], body: str) -> str:
    """声明值、引用剧本的风险、正文关键词，三者取最高。人只能往高改不能往低改。"""
    rank = RISK_RANK[declared]
    for r in referenced:
        rank = max(rank, RISK_RANK[r])
    for level, words in config.SCENARIO_RISK_WORDS.items():
        # Latin keywords use word boundaries: "post" must not match "posture".
        if any((re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", body, re.I)
                if w.isascii() else w in body) for w in words):
            rank = max(rank, RISK_RANK[level])
    return RISKS[rank]
