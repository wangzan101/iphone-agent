"""字符二元组（bigram）相似度。recap 排最近记忆、search 排候选记忆，共用同一套打分——
两处若各写一套，标准就会悄悄分裂（当年 index/read 的越权判据分裂过一次，教训见 store.py）。

⚠ 不用编辑距离 / TF-IDF 之类更"准"的算法：这里比的是几十字的 name+description，
字符集是中英混排，bigram 交集比例已经够分出「相关」和「不相关」，
换更重的算法只是多一个依赖、多一次性能开销，换不来可感知的排序质量提升。
"""
from __future__ import annotations


def bigrams(s: str) -> set[str]:
    """把 s 变成相邻字符对的集合。

    先去空白、转小写——空白只是排版噪音，不该参与匹配；大小写在中文场景下
    基本不影响语义，统一小写让 "WiFi" 和 "wifi" 能对上。
    长度 < 2 的（去空白后可能是空串，也可能是单字）没法切出二元组，
    直接把整个（可能是空的）字符串当唯一元素返回——这样它至少能和自己完全相同的
    查询匹配上，而不是永远得零分。
    """
    s = "".join(s.split()).lower()
    if len(s) < 2:
        return {s}
    return {s[i:i + 2] for i in range(len(s) - 1)}


def score(query: str, text: str) -> float:
    """query 命中 text 的比例：|B(query) ∩ B(text)| / |B(query)|。

    分母用 query 的 bigram 数、不用 text 的——这样长 text（比如正文很长的记忆）
    不会因为分母被拉大而稀释掉短查询的命中率；用户关心的是「我这句话有多少
    落在了这条记忆里」，不是反过来。

    query 为空是特例：空字符串的 bigrams() 会返回 {""}，如果不特判，
    任何 text 只要也去空白后为空就能凑出 1.0 的满分——那是误导，
    空查询本来就不该匹配任何东西，直接判 0.0。
    """
    if not query:
        return 0.0
    q_bigrams = bigrams(query)
    if not q_bigrams:
        return 0.0
    t_bigrams = bigrams(text)
    return len(q_bigrams & t_bigrams) / len(q_bigrams)
