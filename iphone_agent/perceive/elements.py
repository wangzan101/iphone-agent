from __future__ import annotations

from dataclasses import dataclass, field, replace

from PIL import Image

from iphone_agent import config
from iphone_agent.driver.geometry import Frame, Rect
from iphone_agent.perceive.hashing import ahash, state_key
from iphone_agent.perceive.marking import draw_marks
from iphone_agent.perceive.ocr import RawBox, vision_box_to_pixels


@dataclass(frozen=True)
class Element:
    id: int
    text: str
    confidence: float
    box: tuple[int, int, int, int]
    center: tuple[int, int]
    # ---- 下面三个来自屏幕解析层（perceive/screen.py），OCR 那一路给不出 ----
    # kind: button / icon / input / switch / cell / tab / image / text / other，不知道就是 None
    kind: str | None = None
    # switch 的 on/off、控件的 disabled 之类
    state: str | None = None
    # 这个元素是谁看出来的：ocr / vision / both。回放评测要靠它分开算两路的贡献。
    source: str = "ocr"


@dataclass
class Observation:
    observation_id: int
    frame_id: int
    image: Image.Image
    width_px: int
    height_px: int
    elements: list[Element]
    marked_image: Image.Image
    elements_text: str
    ahash: int
    window_rect: Rect
    text_set: set[str] = field(default_factory=set)
    # 这次观察是用哪种坐标约定告诉模型的。头部那行按它写，actions.py 拒绝时的提示语也读它 ——
    # 一个规则一个入口。2026-09-08 真机踩过两个入口只改一个的坑（见下面 build_observation）。
    coord_mode: str = "norm1000"
    # 各路感知这一帧怎么样：{"ocr": 元素数, "vision": off/failed/empty/ok}。zoom 观察不填。
    perception: dict = field(default_factory=dict)
    # 这一屏是什么（perceive.screen.ScreenLabel，spec 2026-09-12 §3）。整屏解析没给 / 没做就是 None。
    # 类型写 object：elements 不 import screen，避免两个模块互相依赖。
    screen: object | None = None
    # 这一帧提示词里给过的候选（perceive.screen.Candidate）。重放靠它把 same_as 编号还原成屏 id。
    screen_candidates: list = field(default_factory=list)
    # 只有 zoom 观察填：它放大的那一帧的整屏解析结果（那一帧的 perception.vision）。非 zoom 观察是 None。
    # ⚠ 2026-09-15（终审发现 1）：zoom 自己不解析、perception 是空的，policy.fallback_due 原来把它当成
    #   「没拿到整屏解析」，always 下 zoom 之后 done(failed) 也兜底，白打一次整屏解析（spec §3.1 不许）。
    #   兜底要按被放大的那一帧判。单独一个字段而不是塞进 perception：wants_label 靠 perception 为空认 zoom 帧，
    #   zoom 帧永远不标。只由 Perceiver.zoom 写（grep 测试守着）。
    zoom_base_vision: str | None = None
    # 推给模型之后置 True（Perceiver.finalize，spec 2026-09-14 §6.3）。之后再补写感知字段 = 程序 bug。
    # 这是约定不是语言层面的不可变：Perceiver 之外不许写 screen / screen_candidates / perception（grep 测试守着）。
    finalized: bool = False

    @property
    def state_key(self) -> int:
        """「这是不是同一个画面」的键。guard / explore 都用它，不要直接拿 ahash 比（hashing.state_key 的注释）。"""
        return state_key(self.ahash, self.text_set)

    def element(self, eid: int) -> Element | None:
        for e in self.elements:
            if e.id == eid:
                return e
        return None


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


FUSE_IOU = 0.3      # 两个框互相盖住这么多就算同一个东西（用的是「小框被盖住的比例」）

# 元素表头部那一句与视觉项分隔行（spec 2026-09-14 §4.2）。只由 build_observation 生成 —— 一个入口。
HEAD_FULL = "（整屏看过：含图标、按钮、开关）"
HEAD_OCR_ONLY = "（只有 OCR 文字；图标、无字按钮、开关状态可能不在列表里）"
VISION_SEPARATOR = "（下面是看全屏补上的：图标、无字按钮、开关）"


def _covered(small: tuple[int, int, int, int], big: tuple[int, int, int, int]) -> float:
    """small 有多大比例落在 big 里面（交集 ÷ small 的面积）。

    ⚠ 为什么不用 IoU：OCR 的文字框套在视觉的按钮框**里面**，是这里最常见的形状 ——
      「完成」两个字的框可能只有按钮的 8%，IoU 永远够不到 0.3，于是同一个按钮
      被拆成两个元素、两个编号。2026-09-09 真机实测：'+'(OCR) 和「加号图标按钮」
      (视觉) 中心只差 13px，却没能合并。
      这不是个例 —— **每一个带文字的按钮都会中招**。IoU 度量的是「两个框有多像」，
      而我们要问的是「这俩说的是不是同一个东西」，后者该看包含关系。
    """
    ix1, iy1 = max(small[0], big[0]), max(small[1], big[1])
    ix2, iy2 = min(small[2], big[2]), min(small[3], big[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    area = (small[2] - small[0]) * (small[3] - small[1])
    return ((ix2 - ix1) * (iy2 - iy1)) / area if area > 0 else 0.0


def fuse(ocr_els: list[Element], items) -> list[Element]:
    """OCR 元素 + 屏幕解析元素，合成一份列表（还没编号，编号由调用方统一给）。

    分工是清楚的，别搞反：
    · **文字以 OCR 为准** —— 它的转录准确度高于视觉模型，'1 你好' 这种一个字都不能错。
    · **kind / state 只能由视觉给** —— OCR 根本不知道什么是按钮。
    · 视觉看到、OCR 没读到的（图标、无字按钮、开关），作为新元素补进来。

    重叠判定用 IoU：一个 OCR 元素最多被一个视觉元素认领，认领了就标 source='both'。
    """
    out = list(ocr_els)
    taken: set[int] = set()
    for item in items:
        best, best_iou = -1, FUSE_IOU
        for i, e in enumerate(out):
            if i in taken:
                continue
            # 谁小就问谁被盖住了多少 —— OCR 文字框通常小，视觉按钮框通常大。
            a_area = (e.box[2] - e.box[0]) * (e.box[3] - e.box[1])
            b_area = (item.box[2] - item.box[0]) * (item.box[3] - item.box[1])
            v = (_covered(e.box, item.box) if a_area <= b_area
                 else _covered(item.box, e.box))
            if v > best_iou:
                best, best_iou = i, v
        if best >= 0:
            taken.add(best)
            e = out[best]
            # ⚠ 只贴 kind/state，**不碰 text** —— 视觉模型的转录没有 OCR 准。
            out[best] = replace(e, kind=item.kind, state=item.state, source="both")
        else:
            out.append(Element(id=0, text=item.label, confidence=0.0, box=item.box,
                               center=item.center, kind=item.kind, state=item.state,
                               source="vision"))
    return out


def number_elements(elements: list[Element], height: int, start: int = 1) -> list[Element]:
    """编号的唯一入口（spec 2026-09-14 §6.1）：build_observation 与 build_zoom_observation 共用。

    OCR 来源（ocr / both）先编，只按它们自己的框排序（上到下按 y1 分带，带高 = 图高 2%，带内按 x1）；
    纯视觉项（vision）按同样的排序接在后面。视觉项只给认领到的元素贴 kind/state，从不改它们的编号。
    ⚠ 2026-09-14：原来 OCR 与视觉项混排后统一编号 —— 同一帧解析跑没跑，「通用」的编号就不同。
      on_demand 下同一屏看全屏前后编号会变，模型拿旧编号去点就点到别处；OCR 优先后，
      同一屏再看一次，OCR 读到的框一样，编号就一样（§6.3 旧编号的第一道防线）。
    """
    band = max(1, int(height * 0.02))

    def key(e: Element) -> tuple[int, int]:
        return e.box[1] // band, e.box[0]

    ocr = sorted((e for e in elements if e.source != "vision"), key=key)
    vis = sorted((e for e in elements if e.source == "vision"), key=key)
    return [replace(e, id=i) for i, e in enumerate([*ocr, *vis], start=start)]


def display_xy(x: int, y: int, width: int, height: int, coord_mode: str) -> tuple[int, int]:
    """图像像素 → 告诉模型的那套坐标。唯一入口：元素表、zoom 视图、tapped 回显共用（CLAUDE.md §7）。"""
    if coord_mode == "norm1000":
        return round(x / width * 1000), round(y / height * 1000)
    return x, y


def elements_at(obs, x: int, y: int) -> list[Element]:
    """框包含 (x, y) 的所有元素（边界算在内），不论来源。坐标点击的安全分类只经这一个几何入口（spec §6.2）。"""
    return [e for e in obs.elements if e.box[0] <= x <= e.box[2] and e.box[1] <= y <= e.box[3]]


def build_observation(frame: Frame, boxes: list[RawBox], observation_id: int,
                       coord_mode: str = "norm1000", screen_items=None,
                       full_screen: bool = False) -> Observation:
    W, H = frame.width_px, frame.height_px
    pix = [(b, vision_box_to_pixels(b, W, H)) for b in boxes]
    ocr_els = [Element(id=0, text=b.text, confidence=round(b.confidence, 2),
                       box=(x1, y1, x2, y2), center=((x1 + x2) // 2, (y1 + y2) // 2))
               for b, (x1, y1, x2, y2) in pix]
    # ⚠ text_set 在下面要用，而它**只能装 OCR 的文字**：视觉模型每次措辞都可能不一样
    #   （'返回箭头' / '返回按钮' / '＜ 图标'），混进变化判定就是满屏假变化，
    #   而假变化会直接喂给熔断器（NO_PROGRESS_STOP）。这条线不能越。
    ocr_texts = {e.text for e in ocr_els if e.center[1] > H * config.STATUS_BAR_CROP_RATIO}
    merged = fuse(ocr_els, screen_items or [])
    elements: list[Element] = number_elements(merged, H)
    # 头部必须带图像尺寸：模型要用坐标点「OCR 看不见的图标」时，得知道坐标空间多大。
    # 2026-09-07 实测：不给尺寸，模型按 850 宽估坐标（实际 644），越界被拒。
    #
    # ⚠ 坐标约定必须**跟着 coord_mode 走**，不能写死。2026-09-08 真机踩过：
    #   当时还是全局配置项，标定翻成了 norm1000，这行却还写着「坐标就用这个空间（像素）」。
    #   模型照头部给了像素 y=1280，适配器按 norm1000 换算成 1280/1000*1388=1777，
    #   越界被拒；模型再试再拒，连续五次被拒，整个任务终止。
    #   —— 一个规则两个入口、只改了一个。这行现在从 coord_mode 参数推，
    #   同一个值也记在 Observation.coord_mode 上。
    if coord_mode == "norm1000":
        head = (f"observation #{observation_id}  图像 {W}x{H} 像素；"
                f"**下面每个元素的坐标、以及你给 tap 的 x,y，统一用 0-1000 的归一化值**"
                f"（x 按宽、y 按高各自折算）。")
    else:
        head = f"observation #{observation_id}  图像 {W}x{H} 像素（坐标就用这个空间）"
    # ⚠ 2026-09-14（spec 按需看图 §4.2）：on_demand 下元素表默认只有 OCR，头部必须如实说，模型才知道
    #   截图上看得见、列表里没有的东西要去 zoom / observe。解析跑了但失败也写「只有 OCR」（计划 R12）。
    lines = [head + (HEAD_FULL if full_screen else HEAD_OCR_ONLY)]
    # ⚠ 元素坐标必须和 tap 参数用**同一套约定**。
    #   2026-09-08 真机踩到：头部写着「给坐标时用 0-1000 的归一化值」，
    #   而元素行印的是**像素**中心 `(187,673)` —— 同一段文字里两套约定。
    #   模型从列表读到那种数、在那个尺度上做空间推理，然后自然按同一尺度给坐标，
    #   连续 5 次 out_of_image 被熔断。它不是没看提示，是**列表本身在教它用像素**。
    # 元素表先列 OCR 项，再加一行分隔，然后列看全屏补上的视觉项（spec §6.1 第 2 条）
    first_vision = next((e.id for e in elements if e.source == "vision"), None)
    for e in elements:
        if e.id == first_vision:
            lines.append(VISION_SEPARATOR)
        cx, cy = display_xy(*e.center, W, H, coord_mode)
        # kind 跟在文字后面，用 · 引出；开关这类再带上状态。
        # 视觉那一路没有 OCR 置信度，印 `-` 而不是 0.00 —— 后者会被读成「很不确定」，
        # 而它其实只是「这个数不适用」。
        tag = f" ·{e.kind}{':' + e.state if e.state else ''}" if e.kind else ""
        conf = f"{e.confidence:.2f}" if e.source != "vision" else "-"
        lines.append(f"[{e.id}] {e.text}{tag} ({cx},{cy}) {conf}")
    return Observation(
        observation_id=observation_id,
        frame_id=frame.frame_id,
        image=frame.image,
        width_px=W,
        height_px=H,
        elements=elements,
        marked_image=draw_marks(frame.image, elements),
        elements_text="\n".join(lines),
        ahash=ahash(frame.image, crop_top_ratio=config.STATUS_BAR_CROP_RATIO),
        window_rect=frame.window_rect,
        # ⚠ 变化判定用的文字集合必须和 aHash 一样裁掉状态栏。
        #   原来 aHash 裁了、这里没裁 —— 同一条规矩两个入口只在一个入口执行，
        #   这个项目今晚已经栽了五次。时钟每分钟变一次就贡献 2 的文本差。
        #   注意只影响**判定**：模型看到的 elements_text 仍然是全部元素。
        text_set=ocr_texts,
        coord_mode=coord_mode,
    )


# ---- 放大观察 ----
# 为什么需要它：在这之前，「看」是一次性的 —— 一张 624x1388 的缩略图 + 一个上百项的
# 列表，模型必须一次猜对。2026-09-09 真机上它把计算器的 '−' 当成了「添加账户 +」，
# 因为 OCR 只能给它两个孤零零的符号；而 OS-Atlas 的对比里，这种「全屏枚举 + 选编号」
# 的做法（Set-of-Mark）成功率只有 4.59%。
#
# 人不是这么看屏幕的：扫一眼 → 那块可能是 → **凑近看** → 确认 → 点。
# 缺的就是「凑近看」。

ZOOM_MIN_PX = 24        # 比这更小的框放大了也没意义，多半是模型给错了坐标
ZOOM_TARGET_LONG = 1400  # 放大后长边的目标像素；再大只是白烧视觉 token


def zoom_size(w: int, h: int, target_long: int = ZOOM_TARGET_LONG) -> tuple[int, int]:
    """放大到长边 target_long，但**绝不缩小** —— 裁出来已经够大就原样用。"""
    long_side = max(w, h)
    if long_side >= target_long:
        return w, h
    k = target_long / long_side
    return max(1, round(w * k)), max(1, round(h * k))


def build_zoom_observation(base: Observation, box: tuple[int, int, int, int],
                           zoomed: Image.Image, boxes: list[RawBox],
                           observation_id: int, screen_items=None) -> Observation:
    """放大某一块之后的观察。

    ## 三条不能搞错的规矩

    1. **坐标空间仍然是原图。** width_px/height_px、元素坐标、模型给 tap 的 x,y ——
       全都在原图那套里。模型看清楚了要能**直接点**，不必先换算回去。
       （`image_to_screen` 只认 width_px/height_px 和 window_rect，不看 image。）

    2. **image 保持原图，只有 marked_image 换成放大图。** loop 发给模型的是
       marked_image，而 image 是 did_change / local_mad 拿去做像素比对的。
       放大图混进变化判定，下一步就会报出一个巨大的假变化 —— 那会直接喂给熔断器。

    3. **区域内的元素用放大后的重读结果替换，区域外的原样保留。**
       放大后 OCR 明显更准（小字、图标标签），这正是 zoom 的价值；
       但区域外的东西模型可能还要点，不能因为放大了一块就把别处弄丢。
    """
    x1, y1, x2, y2 = box
    W, H = base.width_px, base.height_px
    zw, zh = zoomed.size
    sx = (x2 - x1) / zw if zw else 1.0
    sy = (y2 - y1) / zh if zh else 1.0

    inside: list[Element] = []
    for b in boxes:
        bx1, by1, bx2, by2 = vision_box_to_pixels(b, zw, zh)
        # 放大图坐标 → 原图坐标
        ox1, oy1 = round(x1 + bx1 * sx), round(y1 + by1 * sy)
        ox2, oy2 = round(x1 + bx2 * sx), round(y1 + by2 * sy)
        inside.append(Element(id=0, text=b.text, confidence=round(b.confidence, 2),
                              box=(ox1, oy1, ox2, oy2),
                              center=((ox1 + ox2) // 2, (oy1 + oy2) // 2),
                              source="ocr"))
    if screen_items:
        # 局部解析：区域小、元素少，输出天然短 —— 不会像整屏解析那样撞 max_tokens。
        mapped = [replace(it, box=(round(x1 + it.box[0] * sx), round(y1 + it.box[1] * sy),
                                   round(x1 + it.box[2] * sx), round(y1 + it.box[3] * sy)))
                  for it in screen_items]
        inside = fuse(inside, mapped)

    def _outside(e: Element) -> bool:
        cx, cy = e.center
        return not (x1 <= cx <= x2 and y1 <= cy <= y2)

    # ⚠ 2026-09-14（spec §6.1 第 3 条）：原来这里把区域外 + 区域内整体重新编号，区域外的编号也跟着变。
    #   现在区域外原样保留，区域内从 base 最大编号之后接着编；被替换的旧编号不再使用，拿它点得到 stale_element_id。
    start = max((e.id for e in base.elements), default=0) + 1
    elements = [e for e in base.elements if _outside(e)] + number_elements(inside, H, start=start)

    head = (f"observation #{observation_id}  **这是放大视图**：把原图 {W}x{H} 的 "
            f"({x1},{y1})-({x2},{y2}) 这一块放大了看。图上是放大后的样子，"
            f"下面的坐标**仍然是原图那一套**，看准了直接 tap 就行。")
    if base.coord_mode == "norm1000":
        head += "坐标统一用 0-1000 的归一化值（x 按宽、y 按高各自折算）。"
    lines = [head]
    for e in elements:
        cx, cy = display_xy(*e.center, W, H, base.coord_mode)
        tag = f" ·{e.kind}{':' + e.state if e.state else ''}" if e.kind else ""
        conf = f"{e.confidence:.2f}" if e.source != "vision" else "-"
        mark = "" if _outside(e) else " ←放大区内"
        lines.append(f"[{e.id}] {e.text}{tag} ({cx},{cy}) {conf}{mark}")
    return Observation(
        observation_id=observation_id,
        frame_id=base.frame_id,
        image=base.image,                      # ⚠ 原图：变化判定要拿它比对
        width_px=W, height_px=H,               # ⚠ 原图尺寸：坐标空间不变
        elements=elements,
        marked_image=draw_marks(zoomed, [      # ⚠ 放大图：这张才是给模型看的
            replace(e, box=(round((e.box[0] - x1) / sx), round((e.box[1] - y1) / sy),
                            round((e.box[2] - x1) / sx), round((e.box[3] - y1) / sy)))
            for e in elements if not _outside(e)]),
        elements_text="\n".join(lines),
        ahash=base.ahash,                      # 画面没动过，沿用
        window_rect=base.window_rect,
        text_set=base.text_set,                # 同上：zoom 不改变屏幕
        coord_mode=base.coord_mode,
    )
