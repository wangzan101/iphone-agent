from iphone_agent.harness.usage import accumulate


def test_top_level_ints_add_up():
    t = {}
    accumulate(t, {"prompt_tokens": 10, "completion_tokens": 3})
    accumulate(t, {"prompt_tokens": 5, "completion_tokens": 1})
    assert t == {"prompt_tokens": 15, "completion_tokens": 4}


def test_nested_dict_is_flattened_one_level():
    """cached_tokens 藏在 prompt_tokens_details 里；原来的累加器只收顶层 int，把它整个丢了。
    这就是我们一直看不见缓存命中率的原因(设计说明)。"""
    t = {}
    accumulate(t, {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 60}})
    accumulate(t, {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}})
    assert t["prompt_tokens"] == 200
    assert t["prompt_tokens_details.cached_tokens"] == 60


def test_non_int_and_none_are_ignored():
    t = {}
    accumulate(t, None)
    accumulate(t, {"model": "qwen", "ratio": 0.5, "flag": True,
                   "prompt_tokens_details": {"note": "x", "deep": {"a": 1}}})
    assert t == {}
