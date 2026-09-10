"""模型返回的 usage 累加。

⚠ 原来的累加器只收顶层 int（loop.py:278 的 isinstance(v, int)），
把 prompt_tokens_details 这个嵌套 dict 整个丢了 —— 而 cached_tokens 就在里面。
结果是 run.json 里从来没有过缓存命中数（设计说明）。

只展一层，不递归：cached_tokens 是 prompt_tokens 的子集，键名必须保留
"prompt_tokens_details.cached_tokens" 这个语义，不能和 prompt_tokens 混在一起加。
bool 是 int 的子类，要显式排除。
"""
from __future__ import annotations


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def accumulate(total: dict[str, int], usage: dict | None) -> None:
    for k, v in (usage or {}).items():
        if _is_int(v):
            total[k] = total.get(k, 0) + v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if _is_int(v2):
                    key = f"{k}.{k2}"
                    total[key] = total.get(key, 0) + v2
