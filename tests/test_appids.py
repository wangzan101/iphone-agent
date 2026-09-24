"""App id 的唯一入口（appids.resolve_app_id）。

这一组测试盯的是 2026-09-17 之前的一个事实错误：App id 从屏幕上的**显示名**拼音化而来，
所以中文机上 设置 → `she-zhi`、英文机上同一个 App → `settings`。仓库里那份共享知识
于是只对中文机有效 —— 对一个开源项目来说这是错的。
"""
from pathlib import Path

import pytest

from iphone_agent import appids
from iphone_agent.appids import AppIdConflict, resolve_app_id, slugify

REPO = Path(__file__).resolve().parent.parent


# ---- 解析器本身 ----

def test_canonical_id_resolves_to_itself():
    for aid in appids.CANONICAL_APP_IDS:
        assert resolve_app_id(aid) == aid


def test_chinese_english_and_legacy_pinyin_all_reach_the_same_id():
    for name in ("设置", "Settings", "settings", "she-zhi"):
        assert resolve_app_id(name) == "settings"
    for name in ("备忘录", "Notes", "bei-wang-lu"):
        assert resolve_app_id(name) == "notes"
    for name in ("提醒事项", "Reminders", "ti-xing-shi-xiang"):
        assert resolve_app_id(name) == "reminders"


def test_case_and_whitespace_differences_still_resolve():
    assert resolve_app_id("  settings  ") == "settings"
    assert resolve_app_id("SETTINGS") == "settings"
    assert resolve_app_id(" 设置 ") == "settings"
    assert resolve_app_id("She-Zhi") == "settings"


def test_unknown_third_party_display_name_falls_back_to_slugify():
    assert resolve_app_id("记账本") == slugify("记账本") == "ji-zhang-ben"
    assert resolve_app_id("购物商城") == slugify("购物商城")
    assert resolve_app_id("App Store") == slugify("App Store") == "app-store"


def test_empty_name_does_not_blow_up():
    assert resolve_app_id("") == slugify("")


# ---- 别名索引来自知识目录 ----

def test_index_is_built_from_the_catalog_not_hardcoded():
    idx = appids.AliasIndex.build({"chat-app": ["聊天", "Chat"]})
    assert idx.resolve("聊天") == "chat-app" and idx.resolve("Chat") == "chat-app"
    # 索引里没有的仍然退回 slugify
    assert idx.resolve("天气") == slugify("天气")


def test_duplicate_alias_pointing_at_two_apps_is_a_visible_error():
    """CLAUDE.md §3：判不出来不能随便挑一个 —— 挑错了没人看得见。"""
    idx = appids.AliasIndex.build({"a-app": ["共用别名"], "b-app": ["共用别名"]})
    assert idx.conflicts == {"共用别名": ("a-app", "b-app")}
    with pytest.raises(AppIdConflict) as e:
        idx.resolve("共用别名")
    assert "a-app" in str(e.value) and "b-app" in str(e.value)


def test_an_app_id_can_never_be_hijacked_by_another_apps_alias():
    """别名指向一个**真 App 自己的 id**：id 一律优先（twin.record 那条 probe B 的同一条规矩）。"""
    idx = appids.AliasIndex.build({"settings": ["设置"], "notes": ["settings"]})
    assert idx.resolve("settings") == "settings"


# ---- 仓库里的每一份知识 ----

def test_every_knowledge_app_loads_and_every_alias_resolves_back(tmp_path, monkeypatch):
    from iphone_agent.skills import model as M
    monkeypatch.chdir(REPO)
    dirs = sorted(p for p in (REPO / "knowledge" / "apps").iterdir() if p.is_dir())
    assert dirs, "knowledge/apps 下一份知识都没有"   # 不钉数量：知识会一直加
    for d in dirs:
        app = M.app_from_markdown((d / "APP.md").read_text(encoding="utf-8"), d.name)
        assert app.name == d.name, f"{d.name}：frontmatter 的 name 与目录名分叉"
        assert app.aliases, f"{d.name}：没有 aliases，换台英文机就认不出来了"
        for alias in app.aliases:
            assert resolve_app_id(alias) == d.name, f"{d.name}：别名 {alias} 回不来"


def test_apple_system_apps_are_english_slugs_on_disk():
    names = {p.name for p in (REPO / "knowledge" / "apps").iterdir() if p.is_dir()}
    assert names >= appids.CANONICAL_APP_IDS
    assert not (names & {"she-zhi", "bei-wang-lu", "ti-xing-shi-xiang", "tian-qi", "shi-zhong"})


# ---- 没人绕过解析器 ----

def test_nothing_derives_an_app_id_behind_the_resolvers_back():
    """CLAUDE.md §7 一个规则一个入口：App id 的定义只许在 appids.py 里有一份，
    别处要 App id 只能 import 这里的 app_id_for / resolve_app_id。

    （executor 里还有 lazy_pinyin，那是 Spotlight 打字和候选栏匹配，与 App id 无关。）"""
    offenders_def, offenders_import = [], []
    for p in sorted((REPO / "iphone_agent").rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        rel = p.relative_to(REPO).as_posix()
        if rel == "iphone_agent/appids.py":
            continue
        if "def app_id_for" in text or "def slugify" in text or "def resolve_app_id" in text:
            offenders_def.append(rel)
        if "app_id_for" not in text:
            continue
        if "from iphone_agent.appids import" not in text and "from iphone_agent import appids" not in text:
            offenders_import.append(rel)
    assert not offenders_def, f"App id 的定义出现了第二份：{offenders_def}"
    assert not offenders_import, f"这些文件的 app_id_for 不是从 appids 来的：{offenders_import}"


# ---- 所有入口都落到规范 id ----

def test_screenmap_ownership_and_apps_seen_use_canonical_ids():
    from iphone_agent.memory.screenmap import apps_seen, ownership
    steps = [{"step": 1, "action": {"name": "open_app", "args": {"name": "设置"}},
              "result": {"ok": True, "changed": True}, "observation": {"elements": []}},
             {"step": 2, "action": {"name": "open_app", "args": {"name": "Notes"}},
              "result": {"ok": True, "changed": True}, "observation": {"elements": []}}]
    assert ownership(steps) == ["system", "settings"]
    assert apps_seen(steps) == {"settings": "设置", "notes": "Notes"}


def test_twin_layout_and_identify_use_canonical_ids():
    from iphone_agent.twin.identify import plain_app_id
    assert plain_app_id("设置") == "settings"
    assert plain_app_id("Settings") == "settings"


def test_executor_open_path_asks_the_resolver_for_the_id():
    """executor._identity 拿 app_id_for(name) 和标注换算出来的 id 比：两边都得是规范 id，
    否则中文机上开 Settings、英文机上开 设置 永远核对不上。"""
    from iphone_agent.harness import executor
    assert executor.app_id_for("设置") == executor.app_id_for("Settings") == "settings"


def test_ensure_app_lands_on_the_canonical_id_and_remembers_the_display_name(tmp_path):
    from iphone_agent.skills import extract as X
    from iphone_agent.skills.store import SkillStore
    store = SkillStore()                 # conftest 已把 cwd 换到 tmp
    cat = store.load()
    app = X.ensure_app(store, cat, "she-zhi", "设置")
    assert app.name == "settings"
    assert (store.app_dir("settings") / "APP.md").exists()
    assert "设置" in app.aliases
    # 同一个 App 换个叫法再来一次：不该切出第二个
    again = X.ensure_app(store, cat, "settings", "Settings")
    assert again.name == "settings" and len(cat.apps) == 1


def test_a_personal_knowledge_dir_still_named_with_a_legacy_id_is_reported(tmp_path):
    """用户不迁移也不能坏（旧 id 照样解析得回来），但「盘上还是旧名」这件事必须看得见。"""
    from iphone_agent.skills import model as M
    from iphone_agent.skills.store import SkillStore
    store = SkillStore()
    d = store.knowledge_personal / "apps" / "she-zhi"
    d.mkdir(parents=True)
    (d / "APP.md").write_text(M.app_to_markdown(
        M.AppProfile("she-zhi", "设置", "设置", "read", "draft", "2026-09-17", "老目录")), encoding="utf-8")
    cat = store.load()
    assert cat.legacy_dirs == {"she-zhi": "settings"}
    assert resolve_app_id("she-zhi") == "settings"


def test_a_twin_only_legacy_dir_without_an_app_md_is_still_reported(tmp_path):
    """真机上的旧目录大多**只有孪生的 screens/**、没有 APP.md（`iphone skill list` 那条
    提醒要是躲在 APP.md 那道门后面，最该提醒的那一种就一个字都不说）。"""
    from iphone_agent.skills.store import SkillStore
    store = SkillStore()
    (store.knowledge_personal / "apps" / "tian-qi" / "screens").mkdir(parents=True)
    assert store.load().legacy_dirs == {"tian-qi": "weather"}
