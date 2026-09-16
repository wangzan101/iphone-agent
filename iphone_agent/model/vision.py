"""看图问答：把截图和一个问题交给视觉模型，要回一段 JSON。

屏幕解析（perceive/screen.py）和各种判定（harness/judge.py）都走这里。

设计上只有两条硬规矩：

1. **绝不把主循环带走。** 视觉调用是增强，不是命脉 —— 端点抽风、超时、模型胡说八道，
   一律记 `last_error` 然后返回 None，让调用方退回原来那条路（OCR / 汉明距离）。
   这条比什么都重要：新链路刚上线时一定会有没见过的失败方式，不能让它们变成任务终止。

2. **模型的回答是数据，不是指令。** 它读的是手机屏幕上的任意文字（网页、聊天记录、
   通知），那些内容随时可能写着「忽略之前的指令」。所以这里只从回复里**取值**，
   不把它的话当成要执行的东西。
"""
from __future__ import annotations

import base64
import io
import json
import re

from iphone_agent.model.reply import ModelError

# 模型很爱把 JSON 裹进 ```json ... ``` 里，即使你让它别这么干。
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

# 一次调用失败的种类（spec 2026-09-14 §8.1）：和 ask_json 的几个分支一一对应。
OUTCOME_KINDS = ("transport", "exception", "truncated", "unparsable", "partial")


def png_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_json(text: str):
    """从模型回复里挖出 JSON。挖不出来返回 None —— 不抛，调用方要的是降级不是崩。

    三层尝试：整段直接解析 → 剥掉代码围栏 → 截取第一个 { 或 [ 到最后一个 } 或 ]。
    最后那层是为了应付「好的，这是结果：{...}」这种前言。
    """
    if not text:
        return None
    for cand in (text, *(m.group(1) for m in _FENCE.finditer(text))):
        try:
            return json.loads(cand)
        except (ValueError, TypeError):
            pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    ends = [i for i in (text.rfind("}"), text.rfind("]")) if i >= 0]
    if starts and ends:
        try:
            return json.loads(text[min(starts):max(ends) + 1])
        except (ValueError, TypeError):
            pass
    # 被截断的数组：'[{..},{..},{"kind": "button", "label": "标' —— 截到最后一个完整的
    # 对象再补上 ]，把已经说完的那些抢救出来。半个列表远好过一个空列表：
    # 2026-09-09 真机上这种截断让整屏视觉贡献归零，而前 40 个元素其实是好的。
    lb = text.find("[")
    if lb >= 0:
        last = text.rfind("}")
        if last > lb:
            try:
                return json.loads(text[lb:last + 1] + "]")
            except (ValueError, TypeError):
                pass
    return None


class VisionAsker:
    """包一个 transport，提供「给张图问个问题，要一段 JSON」。

    calls / failures 是给回放评测和 doctor 看的 —— 上线一个会自己失败的东西，
    总得能查它到底失败了多少次。
    """

    def __init__(self, transport, enabled: bool = True, cache_dir=None):
        self._t = transport
        self.enabled = enabled
        self.calls = 0
        self.failures = 0
        self.truncations = 0
        self.cache_hits = 0
        self.last_error: str | None = None
        # ⚠ 2026-09-14：last_error 是共享的，会被下一次调用覆盖、成功时也不清空 —— 拿它当「这一帧为什么失败」
        #   就会把上一帧的原因安到这一帧头上。last_outcome 在每次调用开始时清空，Perceiver 调用一返回就抄进本帧。
        self.last_outcome: dict | None = None
        # ⚠ 缓存放在**这一层**，缓存的是模型的原始回复，不是解析/融合之后的结果。
        #   评测集靠它做到「改融合、改坐标映射、改 OCR 后处理都不花钱」：
        #   那些都在这层之上，重跑是本地几百毫秒；只有改 prompt 或换模型才真调。
        #   一度把缓存做在融合之后，结果改一行 fuse() 就得把几百帧全部重调一遍。
        self.cache_dir = cache_dir

    def _cache_key(self, prompt: str, images: list, max_tokens) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(str(getattr(getattr(self._t, "resolved", None), "model", None) and
                     self._t.resolved.model.id).encode())
        h.update(str(max_tokens).encode())
        h.update(prompt.encode("utf-8"))
        for img in images:
            h.update(img.tobytes())
            h.update(str(img.size).encode())
        return h.hexdigest()[:24]

    def ask_json(self, prompt: str, images: list, max_tokens: int | None = None):
        """images 是 PIL 图，按顺序附在问题后面。返回解析好的 JSON，失败返回 None。"""
        self.last_outcome = None
        if not self.enabled or self._t is None:
            self.last_outcome = {"kind": "transport", "detail": "视觉未启用或没有 transport"}
            return None
        cache_path = None
        if self.cache_dir is not None:
            from pathlib import Path
            cache_path = Path(self.cache_dir) / f"{self._cache_key(prompt, images, max_tokens)}.json"
            if cache_path.exists():
                self.cache_hits += 1
                data = extract_json(cache_path.read_text(encoding="utf-8"))
                if data is None:
                    self.last_outcome = {"kind": "unparsable", "detail": f"缓存 {cache_path.name} 解析不出 JSON"}
                return data
        parts: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/png;base64,{png_b64(img)}"}})
        self.calls += 1
        try:
            text = self._t.ask([{"role": "user", "content": parts}], max_tokens=max_tokens)
        except ModelError as e:
            self.failures += 1
            self.last_error = str(e)[:400]
            self.last_outcome = {"kind": "transport", "detail": self.last_error}
            return None
        except Exception as e:      # 连接层的意外：一样只记不抛
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"[:400]
            self.last_outcome = {"kind": "exception", "detail": self.last_error}
            return None
        truncated = getattr(self._t, "last_finish_reason", None) == "length"
        if truncated:
            self.truncations += 1
        data = extract_json(text)
        # 只缓存能解析出来的回复。截断的、答非所问的不缓存 —— 那种要重试，不该被钉死。
        if cache_path is not None and data is not None and not truncated:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
        if data is None:
            self.failures += 1
            # 截断和「模型答非所问」是两种病，说法必须分开：前者加额度，后者改 prompt。
            self.last_error = (
                f"回复被 max_tokens={getattr(self._t, 'last_max_tokens', '?')} 截断，"
                f"连一条完整记录都没抢救出来（已生成 {len(text)} 字符）"
                if truncated else f"回复里没有可解析的 JSON：{text[:200]!r}")
            self.last_outcome = {"kind": "truncated" if truncated else "unparsable", "detail": self.last_error}
        elif truncated:
            # 抢救成功，但这次结果是**不完整**的 —— 该说一声，别让人以为这屏就这么点东西。
            self.last_error = (f"回复被 max_tokens={getattr(self._t, 'last_max_tokens', '?')} "
                               f"截断，只用上了能解析出来的那部分")
            self.last_outcome = {"kind": "partial", "detail": self.last_error}
        return data

    def stats(self) -> dict:
        d = {"calls": self.calls, "failures": self.failures}
        if self.cache_hits:
            d["cache_hits"] = self.cache_hits
        if self.truncations:
            d["truncations"] = self.truncations
        if self.last_error:
            d["last_error"] = self.last_error
        return d
