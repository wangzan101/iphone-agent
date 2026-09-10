from __future__ import annotations

from PIL import Image

from iphone_agent import config
from iphone_agent.driver.geometry import Frame
from iphone_agent.perceive.elements import (
    Observation,
    build_observation,
    build_zoom_observation,
    zoom_size,
)
from iphone_agent.perceive.ocr import run_vision_ocr
from iphone_agent.perceive.screen import parse_screen


class Perceiver:
    """observe(frame)：OCR + 屏幕解析 → 融合 → 元素 → 标记图 → 哈希。observation_id 递增。

    coord_mode 是当前模型的坐标约定，由 Session 注入。perceive 层收到的只是一个格式化参数，
    不认识模型。

    asker（VisionAsker）是**可选**的第二路感知：让视觉模型把画面读成结构化元素，
    补上 OCR 给不出的图标、无字按钮、开关状态。给 None 就退回纯 OCR，行为和以前
    逐字节一致 —— 这条降级路必须一直留着：视觉那一路会超时、会抽风、会被用户关掉。
    """

    def __init__(self, ocr=run_vision_ocr, coord_mode: str = "norm1000", asker=None):
        self._ocr = ocr
        self._next_id = 1
        self.coord_mode = coord_mode
        self.asker = asker

    def observe(self, frame: Frame) -> Observation:
        boxes = self._ocr(frame.image)
        # ⚠ 顺序无所谓，但**两路都失败不能等于观察失败**：parse_screen 自己吞掉所有异常
        #   返回空列表，OCR 才是命脉。
        # ⚠ config.SCREEN_PARSE 只管**这里**（每次观察都整屏解析），不管 zoom。
        #   这两件事的成本和价值完全不同：整屏解析 34 秒、大部分元素用不上；
        #   zoom 是模型自己开口要看清一小块，5 秒、每个元素都是它要的。
        #   一度把开关做在 VisionAsker.enabled 上，结果关掉整屏解析连 zoom 一起废了。
        items = ([] if not config.SCREEN_PARSE or self.asker is None
                 else parse_screen(frame.image, self.asker))
        obs = build_observation(frame, boxes, observation_id=self._next_id,
                                coord_mode=self.coord_mode, screen_items=items)
        self._next_id += 1
        return obs

    def zoom(self, base: Observation, box: tuple[int, int, int, int]) -> Observation:
        """把 base 的某一块放大重看。不碰设备 —— 用的是它当时那张图。

        为什么重新跑一遍 OCR：**放大之后它明显更准**。小字、图标旁边的标签、
        候选栏里挤在一起的词，在 624x1388 的整图上常常读错或读不到，
        放大到长边 1400 再读就出来了。这正是 zoom 的价值所在，不是顺手做的。

        局部屏幕解析也在这儿：区域小、元素少，输出天然短 ——
        不会像整屏解析那样撞上 max_tokens 然后整屏归零（那是 SCREEN_PARSE
        默认关掉的原因，见 config）。所以这里**不看 SCREEN_PARSE 开关**：
        它管的是「每次观察都全屏解析」那件事，而这里是模型自己开口要看清楚。
        """
        crop = base.image.crop(box)
        zoomed = crop.resize(zoom_size(crop.width, crop.height), Image.LANCZOS)
        boxes = self._ocr(zoomed)
        items = parse_screen(zoomed, self.asker) if self.asker is not None else []
        obs = build_zoom_observation(base, box, zoomed, boxes,
                                     observation_id=self._next_id, screen_items=items)
        self._next_id += 1
        return obs
