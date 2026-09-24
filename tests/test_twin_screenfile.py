"""屏文件 schema 2（spec 2026-09-12 §4.2、§7）。"""
import pytest

from iphone_agent.twin.screenfile import (
    AppMeta,
    Screen,
    StaleSchema,
    Transition,
    UnsupportedSchema,
    screen_id,
    utc,
)


def mk(**kw):
    base = {"id": "s_1", "app": "settings", "first_event": "r1:f2.png", "created": "t", "updated": "t", "name": "通用"}
    return Screen(**{**base, **kw})


def test_screen_id_is_deterministic_64bit_and_salt_changes_it():
    a = screen_id("settings", "r1:f2.png")
    assert a == screen_id("settings", "r1:f2.png") and a.startswith("s_") and len(a) == 18
    assert screen_id("settings", "r1:f2.png", 1) != a


def test_utc_format():
    assert utc(0) == "1970-01-01T00:00:00Z"


def test_round_trip_and_stable_order():
    s = mk(aliases={"通用设置", "Ab"}, anchors={"通用": 3, "关于本机": 2}, visits=3, status="confirmed",
           applied_runs={"r2", "r1"})
    t = Transition("tap", "关于本机", True, "navigated", "s_2", 2, 0, "t", ["r1:3"])
    s.transitions[t.key()] = t
    d = s.to_json()
    assert d["schema"] == 2 and d["identity"] == {"name": "通用", "aliases": ["Ab", "通用设置"],
                                                  "anchors": {"关于本机": 2, "通用": 3}}
    assert d["applied_runs"] == ["r1", "r2"]
    back = Screen.from_json(d)
    assert back.to_json() == d and back.names() == {"通用", "通用设置", "Ab"}


def test_schema_classification():
    d = mk().to_json()
    with pytest.raises(StaleSchema):
        Screen.from_json({**d, "schema": 1})
    for bad in ({k: v for k, v in d.items() if k != "schema"}, {**d, "schema": 3}, {**d, "schema": True}):
        with pytest.raises(UnsupportedSchema):
            Screen.from_json(bad)
    broken = {**d, "identity": {"aliases": []}}
    with pytest.raises(ValueError) as e:
        Screen.from_json(broken)
    assert not isinstance(e.value, (StaleSchema, UnsupportedSchema))


def test_unstable_when_off_track_is_more_than_half():
    assert Transition("tap", "x", True, "navigated", "s", 3, 2).unstable
    assert not Transition("tap", "x", True, "navigated", "s", 4, 2).unstable


def test_app_meta_round_trip_and_schema():
    m = AppMeta("app-store", "App Store", {"应用商店"})
    d = m.to_json()
    assert d == {"schema": 2, "app": "app-store", "display": "App Store", "aliases": ["应用商店"]}
    assert AppMeta.from_json({**d, "revision": 4}).revision == 4
    with pytest.raises(StaleSchema):
        AppMeta.from_json({**d, "schema": 1})
    with pytest.raises(UnsupportedSchema):
        AppMeta.from_json({**d, "schema": 9})
    with pytest.raises(ValueError):
        AppMeta.from_json({"schema": 2})


def _s(**kw):
    return Screen(id="s_1", app="settings", first_event="r:f.png", created="t", updated="t", name="通用", **kw)


def test_level_is_the_majority_vote_and_none_on_a_tie():
    """spec 打开App落点 §3.3：票数最多的层级；平票或没票 → None（宁可不说，不说错）。"""
    assert _s().level is None
    assert _s(levels={"inner": 2, "top": 1}).level == "inner"
    assert _s(levels={"inner": 1, "top": 1}).level is None


def test_levels_round_trip_and_are_omitted_when_empty():
    """老屏文件没有 levels；空的时候也不写出，老文件与重建结果逐字节不变。"""
    d = _s().to_json()
    assert "levels" not in d["identity"]
    assert Screen.from_json(d).levels == {}
    d2 = _s(levels={"home": 3}).to_json()
    assert d2["identity"]["levels"] == {"home": 3}
    assert Screen.from_json(d2).levels == {"home": 3}
