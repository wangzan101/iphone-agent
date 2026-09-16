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
    base = {"id": "s_1", "app": "she-zhi", "first_event": "r1:f2.png", "created": "t", "updated": "t", "name": "通用"}
    return Screen(**{**base, **kw})


def test_screen_id_is_deterministic_64bit_and_salt_changes_it():
    a = screen_id("she-zhi", "r1:f2.png")
    assert a == screen_id("she-zhi", "r1:f2.png") and a.startswith("s_") and len(a) == 18
    assert screen_id("she-zhi", "r1:f2.png", 1) != a


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
