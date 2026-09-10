from PIL import Image


def ahash(img: Image.Image, crop_top_ratio: float = 0.0) -> int:
    """8x8 平均哈希。crop_top_ratio 裁掉顶部（状态栏时钟等自变内容）。"""
    if crop_top_ratio > 0:
        top = int(img.height * crop_top_ratio)
        img = img.crop((0, top, img.width, img.height))
    small = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    px = list(small.getdata())
    avg = sum(px) / 64
    bits = 0
    for i, p in enumerate(px):
        if p >= avg:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def state_key(ahash_value: int, texts) -> int:
    """画面**状态**的身份：aHash + OCR 文字集合。整个项目里「这是不是同一个画面」只认这一个键。

    ⚠ 为什么不能只用 aHash：aHash 是 8×8 灰度，分辨率是整屏 64 格。金额 0.00→1→13、
      tab 选中态、输入框里多一个字，全在它的分辨率之下 —— 它会把状态不同的画面判成同一个。
      2026-09-09 真机（记一笔账）：点 App 自带键盘的每一位数字都被熔断记成「回到见过的画面」，
      NO_PROGRESS_STOP=6 —— 金额六位数，第六位按下去任务就会被终止，而每一步都是对的。
      同一个键还用在同屏同动作去重和探索的「点过就不再点」上，三处一起换。
    OCR 噪声会让同一画面偶尔出两个键（少抓一次「原地打转」，安全方向）；
    变化在两者分辨率之下的（开关翻转）见 guard.record_result 的处理。
    """
    return hash((ahash_value, frozenset(texts)))
