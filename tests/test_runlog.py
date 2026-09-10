import json

from PIL import Image

from iphone_agent.driver.geometry import Frame, Rect
from iphone_agent.harness.runlog import RunLog


def test_runlog_writes_run_json_steps_and_frames(tmp_path):
    log = RunLog(tmp_path, task="t", model="m", config_snapshot={"MAX_STEPS": 30}, prompt_hash="abc")
    assert (log.dir / "run.json").exists()
    f = Frame(Image.new("RGB", (10, 10)), 10, 10, Rect(0, 0, 5, 5), 0.0, 3)
    name = log.save_frame(f)
    assert name == "frame_003.png" and (log.dir / name).exists()
    assert log.save_frame(f) == name  # 同一 frame 不重写
    log.step({"step": 1, "action": {"name": "tap"}})
    log.step({"step": 2, "action": {"name": "done"}})
    log.finish("done_success", steps=2, done_status="success", done_result="18.6",
               model_version="m-1", usage_total={"prompt_tokens": 1})
    run = json.loads((log.dir / "run.json").read_text())
    assert run["end_reason"] == "done_success" and run["steps"] == 2 and run["done"]["result"] == "18.6"
    assert run["config"]["MAX_STEPS"] == 30 and run["prompt_hash"] == "abc"
    steps = RunLog.read_steps(log.dir)
    assert [s["step"] for s in steps] == [1, 2]


def test_finish_can_be_called_on_interrupt(tmp_path):
    log = RunLog(tmp_path, task="t", model="m", config_snapshot={}, prompt_hash="h")
    log.finish("interrupted", steps=0, done_status=None, done_result=None, model_version="", usage_total={})
    assert json.loads((log.dir / "run.json").read_text())["end_reason"] == "interrupted"


def test_procedure_step_records_carry_kind_and_knowledge_lands_in_run_json(tmp_path):
    import json

    from iphone_agent.harness.runlog import RunLog
    log = RunLog(tmp_path, task="t", model="m", config_snapshot={}, prompt_hash="h")
    log.step({"step": 1, "observation": {"elements": []}})
    log.procedure_step({"parent_call_id": "c1", "procedure": "settings__x", "step": 1,
                        "observation": {"elements": []}, "action": {"name": "tap", "args": {}}, "result": {"ok": True}})
    steps = RunLog.read_steps(log.dir)
    assert "kind" not in steps[0] and steps[1]["kind"] == "procedure_step" and steps[1]["parent_call_id"] == "c1"
    log.set_knowledge({"routing": {"ran": False}})
    assert json.loads((log.dir / "run.json").read_text())["knowledge"] == {"routing": {"ran": False}}


def test_run_json_starts_with_knowledge_none(tmp_path):
    import json

    from iphone_agent.harness.runlog import RunLog
    log = RunLog(tmp_path, task="t", model="m", config_snapshot={}, prompt_hash="h")
    assert json.loads((log.dir / "run.json").read_text())["knowledge"] is None
