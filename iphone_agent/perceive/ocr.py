# 本任务只写纯函数；Vision 调用在 Task 5 补
from __future__ import annotations

import io
from dataclasses import dataclass

import Quartz
import Vision
from PIL import Image


@dataclass(frozen=True)
class RawBox:
    """Vision 原始结果：归一化坐标，左下原点。"""
    text: str
    confidence: float
    x: float
    y: float
    w: float
    h: float


def vision_box_to_pixels(box: RawBox, W: int, H: int) -> tuple[int, int, int, int]:
    """spec §5.1 公式：x1=x·W，y1=(1−y−h)·H，x2=(x+w)·W，y2=(1−y)·H。左上原点。"""
    x1 = round(box.x * W)
    x2 = round((box.x + box.w) * W)
    y1 = round((1 - box.y - box.h) * H)
    y2 = round((1 - box.y) * H)
    return x1, y1, x2, y2



def _pil_to_cgimage(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = Quartz.CFDataCreate(None, buf.getvalue(), len(buf.getvalue()))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def run_vision_ocr(img: Image.Image) -> list[RawBox]:
    """Vision 文字识别：zh-Hans + en，accurate，方向 up，不裁剪（spec §5.1）。"""
    cg = _pil_to_cgimage(img)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setRecognitionLanguages_(["zh-Hans", "en-US"])
    req.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(f"Vision OCR 失败：{err}")
    out: list[RawBox] = []
    for obs in req.results() or []:
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        bb = obs.boundingBox()
        out.append(RawBox(text=str(cands[0].string()), confidence=float(cands[0].confidence()),
                          x=float(bb.origin.x), y=float(bb.origin.y),
                          w=float(bb.size.width), h=float(bb.size.height)))
    return out
