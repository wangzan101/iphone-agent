from iphone_agent.memory.similarity import bigrams, score


def test_bigrams_chinese_and_ascii():
    assert bigrams("微信群") == {"微信", "信群"}
    assert bigrams("a b") == {"ab"}
    assert bigrams("x") == {"x"}


def test_score_is_fraction_of_query_bigrams_found():
    # ⚠ 原简报例句是「打开微信，看群消息」——"看" 隔在 "信" 和 "群" 之间，
    #   "信群" 这个二元组在该句里根本不存在，2/4 的期望值和输入对不上（已用脚本核实：
    #   逐字符滑窗算出来只有 "微信" 命中，是 1/4）。换成语义更自然的
    #   「打开微信群，看消息」（"微信群"连写），"信群" 才真的是相邻二元组，2/4 才成立。
    assert score("微信群摘要", "打开微信群，看消息") == 2 / 4  # 微信、信群 命中；群摘、摘要 未命中
    assert score("", "anything") == 0.0
    assert score("设置", "") == 0.0
