"""点击结果统计：口径与 docs/superpowers/specs/2026-09-11-点击预期核对-design.md §0 一致。"""
import json

from iphone_agent.eval import taps as T


def _tap(step, eid, texts, changed, expect=None, **result):
    r = {"ok": True, "changed": changed}
    r.update(result)
    return {"step": step, "action": {"name": "tap", "args": {"id": eid}},
            "model": {"reason": "r", "expect": expect},
            "observation": {"elements": [{"id": i + 1, "text": t} for i, t in enumerate(texts)]},
            "result": r}


def _run(tmp_path, name, steps):
    d = tmp_path / name
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"task": "t"}), encoding="utf-8")
    (d / "steps.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in steps), encoding="utf-8")
    return d


def test_counts_unchanged_repeat_back_and_expect(tmp_path):
    _run(tmp_path, "20260911-100000-aaaa", [
        _tap(1, 1, ["通用", "通知"], False),                       # 没变
        _tap(2, 1, ["通用", "通知"], True, expect="进入通用页，出现「关于本机」",
             expect_check={"met": True, "by": "text", "matched": "关于本机"}),   # 原样重点，变了
        _tap(3, 2, ["关于本机", "返回"], True),                    # 点「返回」（id 2 就是「返回」）
    ])
    s = T.tap_outcomes(tmp_path)
    c = s["counts"]
    assert c["runs"] == 1 and c["taps"] == 3
    assert c["unchanged"] == 1 and c["repeat_after_unchanged_2"] == 1
    # 只有第 2 步「变了之后 2 步内按了返回」；第 3 步自己就是返回，后面没有步了
    assert c["changed"] == 2 and c["back_after_change_2"] == 1
    assert c["with_expect"] == 1
    assert s["expect_check"] == {"text:True": 1}


def test_rejected_taps_and_non_runs_are_not_counted(tmp_path):
    rej = _tap(1, 9, ["x"], None)
    rej["validation"] = "repeated_action"
    _run(tmp_path, "20260911-100000-bbbb", [rej])
    (tmp_path / "not-a-run").mkdir()
    assert T.tap_outcomes(tmp_path)["counts"].get("taps", 0) == 0


def test_since_filters_by_run_name_prefix(tmp_path):
    _run(tmp_path, "20260910-100000-old", [_tap(1, 1, ["a"], True)])
    _run(tmp_path, "20260911-100000-new", [_tap(1, 1, ["a"], True)])
    assert T.tap_outcomes(tmp_path, since="20260911")["counts"]["taps"] == 1


def test_render_reports_rates_with_denominators(tmp_path):
    _run(tmp_path, "20260911-100000-cccc", [_tap(1, 1, ["a"], False), _tap(2, 2, ["b"], True)])
    text = "\n".join(T.render(T.tap_outcomes(tmp_path)))
    assert "1/2" in text and "点完没变" in text


def test_match_in_second_step_only(tmp_path):
    """钉住点击统计的「2 步内」窗口：§0 基线指标的规则。

    见 docs/superpowers/specs/2026-09-11-点击预期核对-design.md §0：
    - repeat_after_unchanged_2：点完没变后，2 步内（i+1 到 i+2）原样重点
    - back_after_change_2：点完变了后，2 步内（i+1 到 i+2）按返回

    这条测试覆盖「第二步（i+2）才有匹配」的情况，确保窗口长度 steps[i+1:i+3] 真的看 2 步，
    不会退化成 steps[i+1:i+2]（只看 1 步）。设计稿定义的两个基线指标（repeat_after_unchanged_2、
    back_after_change_2）都以这个 2 步窗口为前提：如果窗口缩到 1 步，这两个指标会错低。
    """
    # 场景 1：没变 → 接下来第一步是别的点击 → 第二步原样重点
    _run(tmp_path, "20260911-200001-r2nd", [
        _tap(1, 1, ["a"], False),      # 点 id=1，没变 → unchanged += 1
        _tap(2, 2, ["b"], True),       # 接下来第一步点 id=2（不同目标）
        _tap(3, 1, ["a"], True),       # 接下来第二步点 id=1（同一个目标）→ repeat_after_unchanged_2 += 1
    ])

    # 场景 2：变了 → 紧邻的下一步（i+1）就按返回。
    # ⚠ 名字叫 b2nd，但返回在 i+1，不在 i+2 —— back_after_change_2 的「i+2 才命中」这里没有覆盖。
    _run(tmp_path, "20260911-200002-b2nd", [
        _tap(1, 1, ["a"], True),       # 点 id=1，变了 → changed += 1
        _tap(2, 2, ["别的", "返回"], False),  # 紧邻的下一步（i+1）点返回（id=2）；这步自己没变，后面没有步可重点
    ])

    # 反例：第三步（i+3）才有匹配（不计入，超出 2 步窗口）
    _run(tmp_path, "20260911-200003-nope", [
        _tap(1, 1, ["a"], False),      # 点 id=1，没变 → unchanged += 1
        _tap(2, 2, ["b"], True),       # 接下来第一步点别的
        _tap(3, 3, ["c"], True),       # 接下来第二步点别的
        _tap(4, 1, ["a"], True),       # 第三步（i+3）才重点，窗口外，不计入 repeat_after_unchanged_2
    ])

    s = T.tap_outcomes(tmp_path)
    c = s["counts"]
    # 三个 run，共 9 个点击（3 + 2 + 4）
    assert c["runs"] == 3 and c["taps"] == 9
    # unchanged：3 个（场景 1 第 1 步、场景 2 第 2 步、反例第 1 步），repeat_after_unchanged_2：1 个（只有场景 1 在窗口内）
    assert c["unchanged"] == 3 and c["repeat_after_unchanged_2"] == 1
    # changed：6 个（所有其他点击），back_after_change_2：1 个（只有场景 2 的第 1 步在窗口内有返回）
    assert c["changed"] == 6 and c["back_after_change_2"] == 1


def test_procedure_taps_are_not_model_taps(tmp_path):
    """剧本子步骤（kind=procedure_step）没有 model 键、expect 为 None，执行层给它们写 by=missing。
    算进来会拉低「带预期」、抬高 missing —— 剧本的点击不是模型的点击，单独计数（终审 2026-09-11）。"""
    proc = _tap(2, 1, ["通用"], True, expect_check={"met": None, "by": "missing"})
    proc["kind"] = "procedure_step"
    del proc["model"]
    _run(tmp_path, "20260911-300000-proc", [
        _tap(1, 1, ["设置"], True, expect="进入设置",
             judged={"worked": True, "confidence": 0.9, "why": "到了", "on_change": True},
             expect_check={"met": True, "by": "vision"}),
        proc,
        _tap(3, 2, ["通用", "关于本机"], True, expect="出现「关于本机」",
             expect_check={"met": True, "by": "text", "matched": "关于本机"}),
        _tap(4, 1, ["a"], False, expect="出现「b」",
             judged={"worked": False, "confidence": 0.9, "why": "两张图一样"},   # 没变那一侧的看图，不算
             expect_check={"met": False, "by": "vision"}),
    ])
    s = T.tap_outcomes(tmp_path)
    c = s["counts"]
    assert c["taps"] == 3 and c["procedure_taps"] == 1
    assert c["with_expect"] == 3, "分母是模型的 3 次点击，剧本那次不拉低带预期比例"
    assert s["expect_check"] == {"vision:True": 1, "text:True": 1, "vision:False": 1}
    assert c["changed"] == 2 and c["vision_on_change"] == 1
    lines = T.render(s)
    assert any(line.strip() == "剧本内点击（未计入上面） 1" for line in lines), lines
    vis = next(line for line in lines if "变了之后看图" in line)
    assert vis.endswith("1/2（50.0%）"), vis
    assert any(line.strip().startswith("带预期") and "3/3" in line for line in lines), lines


def test_render_leaves_out_the_procedure_line_when_there_were_none(tmp_path):
    _run(tmp_path, "20260911-300001-none", [_tap(1, 1, ["a"], True)])
    lines = T.render(T.tap_outcomes(tmp_path))
    assert not any("剧本内点击" in line for line in lines)
    assert any("变了之后看图" in line and "0/1" in line for line in lines), lines
