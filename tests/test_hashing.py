from PIL import Image, ImageDraw

from iphone_agent.perceive.hashing import ahash, hamming


def solid(color, size=(64, 128)):
    return Image.new("RGB", size, color)


def test_identical_images_zero_distance():
    assert hamming(ahash(solid("white")), ahash(solid("white"))) == 0


def test_different_images_large_distance():
    a = solid("white")
    b = solid("white")
    ImageDraw.Draw(b).rectangle([0, 64, 64, 128], fill="black")
    assert hamming(ahash(a), ahash(b)) >= 24


def test_crop_top_ignores_status_bar_change():
    a = solid("white")
    b = solid("white")
    ImageDraw.Draw(b).rectangle([0, 0, 64, 4], fill="black")  # 顶部 3%
    assert hamming(ahash(a, crop_top_ratio=0.05), ahash(b, crop_top_ratio=0.05)) == 0
