"""记忆存储：校验是安全边界，排在最前面。

⚠ name 不合规必须**拒绝**而不是规整。规整会让两个不同的名字撞成同一个文件、
静默覆盖别人的记忆（spec §3.2）。拒绝了模型自己会重起一个。
"""
from pathlib import Path

import pytest

from iphone_agent import config
from iphone_agent.memory import MemoryRejected, MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory")


OK_NAME = "settings-entry"
OK_DESC = "设置 App 在主屏幕第 2 页"
OK_BODY = "往右滑一页能看到设置。"


@pytest.mark.parametrize("bad", [
    "../../.ssh/id_rsa",      # 路径穿越
    "/etc/passwd",
    "a/b",
    "MEMORY",                 # 保留名（且含大写）
    "memory", "trash", "index",
    "Settings-Entry",         # 大写：macOS 文件系统默认不区分大小写，会撞
    "设置入口",                # 非 ASCII：Unicode 规范化会造成撞名
    "-leading-dash",          # 必须以字母数字开头
    "",                       # 空
    "a" * 49,                 # 超 48 字符
    "memory\n",               # ⚠ $ 会匹配末尾换行之前，这条能绕过正则+保留名两道检查
    "settings-entry\n",
    "ok-name\r",
])
def test_bad_name_is_rejected(store, bad):
    with pytest.raises(MemoryRejected) as e:
        store.validate(bad, OK_DESC, OK_BODY)
    assert e.value.code == "invalid_name"


def test_good_name_passes(store):
    store.validate(OK_NAME, OK_DESC, OK_BODY)      # 不抛即通过
    store.validate("a", OK_DESC, OK_BODY)
    store.validate("a" * 48, OK_DESC, OK_BODY)


@pytest.mark.parametrize("bad", [
    "含换行\n伪造: 字段",        # 换行能伪造 frontmatter 里的 source/created
    "多行\r也不行",
    "---",                      # 能提前闭合 frontmatter
    "前面有---也不行",
    ": 冒号开头",
    "- 减号开头",                # 能伪造出一条不存在的索引项
    "x" * 101,                  # 超 100 字符
    "",                         # 空
])
def test_bad_description_is_rejected(store, bad):
    with pytest.raises(MemoryRejected) as e:
        store.validate(OK_NAME, bad, OK_BODY)
    assert e.value.code == "invalid_description"


def test_body_over_the_limit_is_rejected(store):
    with pytest.raises(MemoryRejected) as e:
        store.validate(OK_NAME, OK_DESC, "字" * (config.MEMORY_BODY_MAX + 1))
    assert e.value.code == "content_too_long"
    # 拒绝的理由要说清楚上限是多少，否则改代码的人会随手调大
    assert str(config.MEMORY_BODY_MAX) in e.value.message


def test_body_of_exactly_the_limit_passes(store):
    store.validate(OK_NAME, OK_DESC, "字" * config.MEMORY_BODY_MAX)


def test_write_then_read_roundtrip(store):
    store.write(OK_NAME, OK_DESC, OK_BODY, "runs/x", "done_success")
    assert store.read(OK_NAME) == OK_BODY


def test_read_missing_returns_none(store):
    assert store.read("nothing-here") is None


def test_write_fills_frontmatter(store):
    p = store.write(OK_NAME, OK_DESC, OK_BODY, "runs/x", "done_failed")
    text = p.read_text(encoding="utf-8")
    assert "name: settings-entry" in text
    assert f"description: {OK_DESC}" in text
    assert "source: runs/x" in text
    assert "source_outcome: done_failed" in text
    assert "created: " in text


def test_write_validates_before_touching_disk(store):
    """校验必须在写盘之前 —— 否则一个非法名字已经建过目录/文件了。"""
    with pytest.raises(MemoryRejected):
        store.write("../evil", OK_DESC, OK_BODY, "runs/x", "done_success")
    assert not store.root.exists() or list(store.root.glob("*.md")) == []


def test_overwrite_moves_the_old_file_to_trash(store):
    """覆盖也要保历史。v1 一边给删除做了垃圾箱、一边让覆盖直接毁内容，
    两套理由冲突（spec §6.1）。"""
    store.write(OK_NAME, OK_DESC, "旧内容", "runs/a", "done_success")
    store.write(OK_NAME, OK_DESC, "新内容", "runs/b", "done_success")
    assert store.read(OK_NAME) == "新内容"
    trashed = list(store.trash.glob(f"{OK_NAME}-*.md"))
    assert len(trashed) == 1
    assert "旧内容" in trashed[0].read_text(encoding="utf-8")


def test_repeated_create_delete_does_not_clobber_trash(store):
    """反复「建了删、删了建」时，trash 里的旧版本不能互相覆盖，所以带时间戳。"""
    for i in range(3):
        store.write(OK_NAME, OK_DESC, f"第{i}版", "runs/x", "done_success")
        store.move_to_trash(OK_NAME, "测试")
    assert len(list(store.trash.glob(f"{OK_NAME}-*.md"))) == 3


def test_move_to_trash_records_why_and_removes_from_store(store):
    store.write(OK_NAME, OK_DESC, OK_BODY, "runs/x", "done_success")
    assert store.move_to_trash(OK_NAME, "记错了") is True
    assert store.read(OK_NAME) is None
    trashed = list(store.trash.glob(f"{OK_NAME}-*.md"))[0].read_text(encoding="utf-8")
    assert OK_BODY in trashed and "记错了" in trashed


def test_move_to_trash_missing_returns_false(store):
    assert store.move_to_trash("nothing-here", "x") is False


def test_write_leaves_no_partial_file_when_archiving_fails(store, monkeypatch):
    """os.replace 无论调用几次都失败：覆盖写入时第一次撞见的 os.replace 调用是
    「把旧内容归档进 trash」这一步（_archive_to_trash 内部），这一步失败时新内容
    还只待在临时文件里、旧文件完全没被动过，所以正式文件应该还是旧内容，
    且没有 .tmp 残留。（注：这条测试原名叫 xxx_serialization_fails，
    review 指出它当时拦到的调用其实不是「写新内容」那一步，改名 + 补充这段注释
    让它名副其实。）"""
    import iphone_agent.memory.store as mod

    def boom(*a, **k):
        raise OSError("disk full")

    store.write(OK_NAME, OK_DESC, "好内容", "runs/a", "done_success")
    monkeypatch.setattr(mod.os, "replace", boom)
    with pytest.raises(OSError):
        store.write(OK_NAME, OK_DESC, "新内容", "runs/b", "done_success")
    # 正式文件仍是旧内容，没有被写坏
    assert store.read(OK_NAME) == "好内容"
    assert list(store.root.glob("*.tmp*")) == []


def test_write_keeps_old_content_when_final_replace_fails(store, monkeypatch):
    """覆盖写入真正危险的那一步：把新内容 rename 到正式路径的最后一次 os.replace。

    旧实现的顺序是 move_to_trash（归档旧内容 + unlink 正式路径）在前、写新内容在后：
    一旦写新内容这一步失败，正式路径已经被 unlink 掉了，记忆就从存储里凭空消失
    （read() 返回 None），trash 里的旧版本也不会被当成「当前」记忆用。

    这里只让「目标是正式路径」的那次 os.replace 失败（前面「归档旧内容进 trash」
    的那次 os.replace 放行），复现的正是 review 描述的场景：新内容已经稳妥地在
    临时文件里、旧内容也已经归档进了 trash，但最后一步重命名失败。
    这种情况下正式路径必须还是旧内容 —— 不能因为「归档」先做了就把旧文件也丢了。
    """
    import iphone_agent.memory.store as mod

    store.write(OK_NAME, OK_DESC, "旧内容", "runs/a", "done_success")
    official_path = store._path(OK_NAME)
    real_replace = mod.os.replace

    def flaky_replace(src, dst, *a, **k):
        if Path(dst) == official_path:
            raise OSError("disk full")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(mod.os, "replace", flaky_replace)
    with pytest.raises(OSError):
        store.write(OK_NAME, OK_DESC, "新内容", "runs/b", "done_success")

    assert store.read(OK_NAME) == "旧内容"
    assert official_path.exists()
    assert list(store.root.glob("*.tmp*")) == []


def test_index_is_computed_from_files_not_stored(store):
    """索引不落盘、每次现算。v1 是「写文件 + 重建索引」两次写入，
    中间崩溃就漂移，却声称永远不会漂移（spec §3.3）。"""
    store.write("aaa", "第一条", "x", "runs/1", "done_success")
    store.write("bbb", "第二条", "y", "runs/2", "done_failed")
    entries, broken = store.index()
    assert [e.name for e in entries] == ["aaa", "bbb"]
    assert [e.source_outcome for e in entries] == ["done_success", "done_failed"]
    assert broken == 0

    # 手动删掉一个文件，索引立刻反映 —— 因为没有第二份数据
    (store.root / "aaa.md").unlink()
    entries, broken = store.index()
    assert [e.name for e in entries] == ["bbb"]


def test_index_skips_broken_files_and_counts_them(store):
    """一个坏文件不能毁掉整次召回。"""
    store.write("good", "好的", "x", "runs/1", "done_success")
    (store.root / "bad.md").write_text("这不是 frontmatter", encoding="utf-8")
    (store.root / "half.md").write_text("---\nname: half\n", encoding="utf-8")
    entries, broken = store.index()
    assert [e.name for e in entries] == ["good"]
    assert broken == 2


def test_index_ignores_trash(store):
    store.write("gone", "要删的", "x", "runs/1", "done_success")
    store.move_to_trash("gone", "测试")
    entries, broken = store.index()
    assert entries == [] and broken == 0


def test_index_on_missing_dir_is_empty(store):
    assert store.index() == ([], 0)


def test_count_is_used_for_the_50_cap(store):
    for i in range(3):
        store.write(f"m{i}", "d", "x", "runs/1", "done_success")
    assert store.count() == 3


def test_index_name_follows_the_filename_not_the_frontmatter(store):
    """手工把 frontmatter 里的 name 改掉、文件名不动：index() 必须报告文件名，
    否则会展示一个 read() 解析不了的名字（真正能读到的是文件名对应的那条）。"""
    store.write("aaa", "第一条", "x", "runs/1", "done_success")
    path = store.root / "aaa.md"
    path.write_text(path.read_text(encoding="utf-8").replace("name: aaa", "name: bbb"),
                     encoding="utf-8")
    entries, broken = store.index()
    assert [e.name for e in entries] == ["aaa"]
    assert broken == 0
    assert store.read("aaa") is not None


def test_renaming_the_file_keeps_the_memory_usable(store):
    """重命名文件（frontmatter 不动）是这个项目刻意支持人手改的操作之一：
    重命名后记忆必须继续可用，而不是因为和 frontmatter 里的旧 name 不符就消失。"""
    store.write("aaa", "第一条", "x", "runs/1", "done_success")
    (store.root / "aaa.md").rename(store.root / "ccc.md")
    entries, broken = store.index()
    assert [e.name for e in entries] == ["ccc"]
    assert broken == 0
    assert store.read("ccc") is not None


def test_index_skips_files_that_are_not_utf8(store):
    """读不出来和 frontmatter 格式错是同一类「这个文件坏了」，必须走同一条路径。

    这个项目**刻意**让人手改记忆文件，而内容是中文 —— 中文环境下编辑器存成 GBK
    是很常见的一次手滑。index() 已经防了格式错却没防读文件本身失败，
    同一句「一个坏文件不该拖累其他文件」的承诺只兑现了一半。
    """
    store.write("good", "好的", "x", "runs/1", "done_success")
    (store.root / "broken.md").write_bytes("---\nname: x\n---\n中文正文".encode("gbk"))
    entries, broken = store.index()
    assert [e.name for e in entries] == ["good"]
    assert broken == 1
    assert store.count() == 1


GBK_BYTES = "---\nname: settings-entry\ndescription: 手改过的\n---\n中文正文".encode("gbk")


def test_read_returns_none_for_a_non_utf8_file(store):
    """读不出来就是「这个文件用不了」，返回 None 而不是抛。

    抛的话 CLI 的 memory show 直接崩（Task 10 会直接调 read()），
    commit_memories 也会在「要写的名字恰好是那个坏文件」时整批失败。
    """
    (store.root).mkdir(parents=True, exist_ok=True)
    (store.root / "settings-entry.md").write_bytes(GBK_BYTES)
    assert store.read("settings-entry") is None


def test_writing_over_a_non_utf8_file_preserves_it_in_trash(store):
    """不静默销毁，也不崩溃：坏文件原样进 trash，新内容正常落盘。

    人手工编辑过的东西一个字节都不丢，只是挪了位置 —— 与「覆盖也进 trash、
    永不硬删」的既有立场一致。
    """
    (store.root).mkdir(parents=True, exist_ok=True)
    (store.root / "settings-entry.md").write_bytes(GBK_BYTES)

    store.write("settings-entry", "新的一条", "新正文", "runs/1", "done_success")
    assert store.read("settings-entry") == "新正文"

    trashed = list(store.trash.glob("*.md"))
    assert len(trashed) == 1
    raw = trashed[0].read_bytes()
    # ⚠ 用 read_bytes 比对：原文件不是 UTF-8，read_text 会在这里重演被修掉的那个 bug。
    #   原字节必须原封不动地在最前面 —— 归档只在末尾追加一句理由，不重编码、不丢字节。
    assert raw.startswith(GBK_BYTES)
    assert "移入 trash 的理由".encode() in raw


# ---- 读取路径的越权边界（spec §2.2 第 1 条：模型给不了路径） ----

@pytest.mark.parametrize("bad", [
    "../private",
    "../../.ssh/id_rsa",
    "/etc/passwd",
    "..\\private",            # Windows 风格：POSIX 上是合法文件名，但仍然拒绝
    "a/b",
    "..",
])
def test_read_of_a_traversing_name_returns_none(store, bad):
    """recall 的名字是模型给的，不能当路径用。

    写入路径靠 validate() 挡住了非法名字，读取路径以前直接拼 —— 于是
    read("../private") 能读到记忆目录之外的 .md，内容随即进模型上下文，
    而模型有 type 工具能把它打进任意 App。
    """
    store.root.mkdir(parents=True, exist_ok=True)
    secret = "私密笔记：银行卡尾号 1234"
    # 越权目标就放在记忆目录的**外面**（父目录），确保这条测试在修复前是红的。
    (store.root.parent / "private.md").write_text(secret, encoding="utf-8")
    # POSIX 上 `..\private.md` 只是一个奇怪的文件名、并不越权，
    # 所以按它的字面名字在目录里造一个，否则这条参数在修复前不会红。
    (store.root / "..\\private.md").write_text(secret, encoding="utf-8")

    assert store.read(bad) is None


def test_move_to_trash_of_a_traversing_name_does_not_touch_the_outside_file(store):
    """删除走的是同一个 _path()：越权名字既不能读，也不能顺手把外面的文件挪走。"""
    store.root.mkdir(parents=True, exist_ok=True)
    outside = store.root.parent / "private.md"
    outside.write_text("私密笔记", encoding="utf-8")

    assert store.move_to_trash("../private", "手动删除") is False
    assert outside.exists()


def test_write_of_a_traversing_name_still_raises(store):
    """写入的契约不变：非法名字抛 MemoryRejected（validate() 本来就先跑）。"""
    with pytest.raises(MemoryRejected) as e:
        store.write("../private", OK_DESC, OK_BODY, "runs/1", "done_success")
    assert e.value.code == "invalid_name"
    assert not (store.root.parent / "private.md").exists()


@pytest.mark.parametrize("hand_name", ["Foo", "设置入口", "a.b"])
def test_renaming_the_file_to_an_unusual_name_keeps_it_readable(store, hand_name):
    """⚠ 回归护栏：修复用的判据必须是 containment，**不能**套 MEMORY_NAME_RE。

    index() 刻意支持人手改文件名（见 test_renaming_the_file_keeps_the_memory_usable），
    而 MEMORY_NAME_RE 只收小写 ASCII —— 手工改成 Foo.md 或中文名之后，
    索引会列出它、正则却会拒绝读取，那就是又造一次「索引和读取判据分裂」。
    只挡越权，不挡手改。
    """
    store.write("aaa", "第一条", "x", "runs/1", "done_success")
    (store.root / "aaa.md").rename(store.root / f"{hand_name}.md")

    entries, broken = store.index()
    assert [e.name for e in entries] == [hand_name]
    assert broken == 0
    assert store.read(hand_name) == "x"


# ---- kind / 使用计数 / 相似度搜索 / 敏感内容拦截 ----

def test_write_with_kind_and_usage_round_trip(store):
    store.write("pay-code", "付款码位置", "首页右上", "runs/a", "done_success", kind="playbook")
    e = store.index()[0][0]
    assert e.kind == "playbook" and e.used_success == 0 and e.last_used == ""
    assert store.update_usage("pay-code", "success", "runs/b") is True
    assert store.update_usage("pay-code", "failed", "runs/c") is True
    e = store.index()[0][0]
    assert (e.used_success, e.used_failed, e.last_used) == (1, 1, "runs/c")
    assert store.read("pay-code") == "首页右上"  # 正文没动
    assert store.update_usage("nope", "success", "runs/d") is False


def test_update_usage_on_a_file_index_treats_as_broken_refuses_to_heal_it(store):
    """update_usage 和 index() 必须用同一条"损坏"判据：used_success/used_failed
    转不成 int 时，两边都要判定为损坏——不能一边拒绝、一边悄悄改成 0 再写回去。
    """
    store.write("pay-code", "付款码位置", "首页右上", "runs/a", "done_success")
    path = store.root / "pay-code.md"
    original = path.read_text(encoding="utf-8")
    # 手改进一条转不成 int 的 used_success：original 里本没有这一行（write()
    # 不落缺省值），插进 frontmatter 收尾的 "---\n" 之前。
    text = original.replace("kind: knowledge\n---\n",
                             "kind: knowledge\nused_success: abc\n---\n")
    assert text != original  # 确认真的插进去了，不是空操作
    path.write_text(text, encoding="utf-8")
    corrupted = path.read_bytes()

    assert store.update_usage("pay-code", "success", "runs/b") is False
    assert path.read_bytes() == corrupted  # 一个字节都没被"自愈"改动

    entries, broken = store.index()
    assert broken == 1
    assert [e.name for e in entries] == []


def test_old_files_without_new_fields_still_parse(store):
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "old.md").write_text(
        "---\nname: old\ndescription: d\nsource: runs/x\n"
        "source_outcome: done_success\ncreated: 2026-09-01\n---\n\nbody\n",
        encoding="utf-8")
    e = store.index()[0][0]
    assert e.kind == "knowledge" and e.used_success == 0


def test_search_ranks_by_bigram_overlap(store):
    store.write("wechat-groups", "微信置顶群在首页顶部", "…", "runs/a", "done_success")
    store.write("settings-entry", "设置入口在主屏第一页", "…", "runs/a", "done_success")
    hits = store.search("微信群消息", k=5)
    assert [e.name for e, _ in hits][0] == "wechat-groups"
    assert all(sc > 0 for _, sc in hits)
    assert store.search("zzz", k=5) == []


@pytest.mark.parametrize("bad", [
    "验证码是 483920",
    "收到 6 位验证码 123456 请填写",
    "联系电话 13812345678",
    "密码: hunter2",
    "password = abc",
])
def test_sensitive_content_is_rejected_without_rephrase_hint(store, bad):
    with pytest.raises(MemoryRejected) as ei:
        store.write("x", "d", bad, "runs/a", "done_success")
    assert ei.value.code == "sensitive_content" and "换" not in ei.value.message


@pytest.mark.parametrize("ok_body", [
    "2026 年验证码策略",
    "2026年验证码策略讨论",
])
def test_sensitive_regex_does_not_false_positive_on_years(store, ok_body):
    """"2026 年验证码策略"里的 2026 是年份不是验证码，四位数字后紧跟
    （可选空白 +）"年"就不该算命中——但别的 4-8 位数字仍然要拦。
    """
    store.write("x", "d", ok_body, "runs/a", "done_success")  # 不抛即通过


@pytest.mark.parametrize("bad", [
    "验证码 483920",
    "483920 是验证码",
])
def test_sensitive_regex_still_rejects_real_verification_codes(store, bad):
    with pytest.raises(MemoryRejected) as ei:
        store.write("x", "d", bad, "runs/a", "done_success")
    assert ei.value.code == "sensitive_content"


def test_body_limit_is_2000(store):
    store.write("long", "d", "x" * 2000, "runs/a", "done_success")
    with pytest.raises(MemoryRejected):
        store.write("long2", "d", "x" * 2001, "runs/a", "done_success")
