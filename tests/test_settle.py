from iphone_agent.driver.timing import Timing
from iphone_agent.harness.settle import settle
from iphone_agent.perceive.hashing import ahash


def test_action_env_advances_only_on_configured_actions(action_env):
    dev, _, frames = action_env([["source"], ["target"]], {(0, "tap"): 1})
    for _ in range(20):
        assert dev.capture() is frames[0][0]
    dev.key("home")
    assert dev.capture() is frames[0][0]
    dev.tap(10, 20)
    for _ in range(20):
        assert dev.capture() is frames[1][0]
    assert dev.actions == [("tap", 10, 20)]


def test_settle_returns_stable_frame(fake_env):
    dev, per, frames = fake_env([["a"], ["b"], ["b"], ["b"]])
    before = dev.capture()
    f, settled = settle(dev, before, Timing(0, 1, 2, 500), ahash)
    assert settled and f.frame_id >= 3


def test_settle_times_out_when_never_stable(fake_env):
    specs = [[f"t{i}"] for i in range(40)]
    dev, per, frames = fake_env(specs)
    before = dev.capture()
    f, settled = settle(dev, before, Timing(0, 1, 10, 30), ahash)
    assert settled is False and f is not None
