"""孪生的读写底座：随机 id、原子替换、revision 检查。spec docs/32 §2、§7 不变式 3。"""
import json
import re

import pytest

from iphone_agent.twin import schema


def test_new_id_has_prefix_and_six_random_chars():
    a, b = schema.new_id("p"), schema.new_id("p")
    assert re.fullmatch(r"p_[a-z0-9]{6}", a)
    assert a != b


def test_read_missing_or_broken_returns_none_not_raises(tmp_path):
    assert schema.read_json(tmp_path / "nope.json") is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert schema.read_json(tmp_path / "bad.json") is None


def test_write_is_atomic_and_bumps_revision(tmp_path):
    p = tmp_path / "layout.json"
    r1 = schema.write_json(p, {"a": 1}, expect_revision=None)
    assert r1 == 1
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1, "revision": 1}
    r2 = schema.write_json(p, {"a": 2}, expect_revision=1)
    assert r2 == 2 and schema.read_json(p)["a"] == 2
    assert not list(tmp_path.glob("*.tmp")), "临时文件必须在 replace 之后消失"


def test_write_with_stale_revision_is_refused(tmp_path):
    p = tmp_path / "layout.json"
    schema.write_json(p, {"a": 1}, expect_revision=None)
    schema.write_json(p, {"a": 2}, expect_revision=1)
    with pytest.raises(schema.RevisionConflict):
        schema.write_json(p, {"a": 3}, expect_revision=1)      # 拿着旧快照写回
    assert schema.read_json(p)["a"] == 2, "被拒的写入不能动磁盘"


def test_write_creates_parent_dirs(tmp_path):
    p = tmp_path / "knowledge" / "device" / "layout.json"
    schema.write_json(p, {"x": 1}, expect_revision=None)
    assert p.exists()


def test_workspace_has_a_device_dir(tmp_path):
    from iphone_agent.workspace import Workspace
    assert Workspace(tmp_path).twin_device == tmp_path / ".iphone" / "knowledge" / "device"


@pytest.mark.parametrize("bad_revision", ["weird", True, [1], {"x": 1}, 1.5])
def test_write_json_sanitizes_corrupted_on_disk_revision(tmp_path, bad_revision):
    """磁盘上的 revision 是不可信输入：write_json 自己重读磁盘拿 on_disk，跟调用方传入的
    expect_revision 无关。类型不对（含 bool，它是 int 子类要单独排掉）就当「没有 revision」，
    不能让 (on_disk or 0) + 1 直接对它做算术抛 TypeError —— Layout.save 只接
    RevisionConflict/OSError，接不住 TypeError，会把不变式 5（写坏 = False，绝不抛）击穿。"""
    p = tmp_path / "layout.json"
    p.write_text(json.dumps({"a": 1, "revision": bad_revision}), encoding="utf-8")
    r = schema.write_json(p, {"a": 2}, expect_revision=None)
    assert r == 1, "坏 revision 当作没有 revision：新 revision 从 1 开始"
    assert schema.read_json(p) == {"a": 2, "revision": 1}


def test_unserializable_data_never_creates_a_temp_file(tmp_path, monkeypatch):
    """dumps 挪到 mkstemp 之前（学 model/configwrite.py）：数据序列化不了，就连临时文件都不该建。
    原先 dumps 在 fdopen 里面，靠 except 分支 unlink 兜底 —— 兜底本身再失败就留下一个孤儿 .tmp。"""
    made = []
    real = schema.tempfile.mkstemp
    monkeypatch.setattr(schema.tempfile, "mkstemp", lambda *a, **k: made.append(1) or real(*a, **k))
    p = tmp_path / "layout.json"
    with pytest.raises(TypeError):
        schema.write_json(p, {"a": {1, 2}}, expect_revision=None)      # set 不能 JSON 序列化
    assert made == [], "dumps 抛错之前不该建临时文件"
    assert not list(tmp_path.glob("*.tmp")) and not p.exists()
