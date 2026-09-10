"""append-only 消息流；滑窗只作用于发送副本（spec §6.3，学 next_main B047）。"""
from __future__ import annotations

import copy
import hashlib
import json

from iphone_agent import config

IMAGE_PLACEHOLDER = "[图已滑窗]"
ELEMENTS_PLACEHOLDER = "[元素列表已省略]"

# 进程内标记，永远不出网、不落盘。新增内部键必须加进这个集合，否则
# transport 的防御性断言（任务 2）会在它漏出去的那一刻炸。
INTERNAL_KEYS = frozenset({"_kind", "_seg", "_obs_parts", "_px"})


class MessageLog:
    def __init__(self):
        self._msgs: list[dict] = []
        self._frozen = False
        # freeze() 时拍下的前缀 hash；只在 state 模式下由 loop.py 用来核对
        # 「前缀真的没变」（见 prefix_hash 的说明）。None 表示还没 freeze 过。
        self.frozen_prefix_hash: str | None = None

    def system(self, text: str):
        self._assert_open()
        self._msgs.append({"role": "system", "content": text, "_seg": "system"})

    def user_text(self, text: str, *, seg: str):
        self._assert_open()
        # `_kind` 决定进不进 state 视图，`_seg` 决定记到哪个段。两个都在
        # to_wire() 里被剥掉，发出去的字节不变（黄金快照钉着）。
        self._msgs.append({"role": "user", "content": [
            {"type": "text", "text": text, "_kind": "prefix", "_seg": seg}]})

    def user_note(self, text: str):
        """任务中途用户插的一句话（网页「暂停 → 补一句话 → 继续」）。

        ⚠ 不是 prefix：前缀任务开始就冻结了（freeze），中途加进去会砸掉缓存前缀，
          prefix_hash 也会报 drift。它是一条普通的用户消息，只在 window 视图里发出去；
          state 视图每轮重建状态报告，插话由 build_state_parts 的 user_notes 段承载。
        """
        self._msgs.append({"role": "user", "content": [
            {"type": "text", "text": "【用户补充】" + text, "_kind": "note", "_seg": "user_note"}]})

    def prefix_hash(self) -> str:
        """前缀（system + 所有 prefix user 消息）的规范化 hash。

        ⚠ freeze() 不是真不可变：raw() 暴露内部 list，user_observation /
          assistant_tool_call / tool_result 冻结后照常能写。所以「前缀不变」
          是**当前 loop 调用路径的性质**，不是 MessageLog 的不变式。这个
          hash 把那条性质变成可验证的（spec §2.2）：freeze() 时拍一次，
          之后每次请求前再拍一次，两次不一致就是 prefix_drift。

        ⚠ 观测设施绝不能抛出去砸任务：消息里混进不可 json 序列化的东西
          （理论上不该发生，但别赌）也只把这次 hash 降级成一个稳定的
          错误占位符，不让异常往上冒。
        """
        try:
            prefix = [m for m in self._msgs
                      if m["role"] == "system"
                      or (m["role"] == "user" and isinstance(m["content"], list)
                          and all(p.get("_kind") == "prefix" for p in m["content"]))]
            blob = json.dumps(prefix, sort_keys=True, ensure_ascii=False, default=str)
            return hashlib.sha256(blob.encode("utf-8")).hexdigest()
        except Exception as e:      # noqa: BLE001 —— 观测设施不得抛出去
            return f"hash_error:{type(e).__name__}"

    def freeze(self):
        """设计说明：前缀在一次任务里冻结，绝不回头改 —— 改了就砸掉缓存前缀。"""
        self._frozen = True
        self.frozen_prefix_hash = self.prefix_hash()

    def _assert_open(self):
        if self._frozen:
            raise RuntimeError("前缀已冻结：任务开始后不能再往 system/前缀里加东西（冻结前缀约束）")

    def user_observation(self, obs_parts: list[tuple[str, str]], image_b64: str,
                        *, px: tuple[int, int] = (0, 0)):
        """obs_parts 是 [(seg, text), …]，按顺序用 "\\n\\n" 拼成模型看到的那一段。

        ⚠ 拼接结果必须与拆分前逐字符相同（loop.push_obs 原来就是这么拼的）。
        ⚠ px 是图片分辨率，estimate_image 要它。它是内部键 `_px`，to_wire 会剥掉。
        """
        content = [{"type": "text", "text": "\n\n".join(t for _, t in obs_parts),
                    "_kind": "elements", "_seg": obs_parts[0][0],
                    "_obs_parts": obs_parts}]
        content.append({"type": "image_url", "_seg": "image", "_px": list(px),
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
        self._msgs.append({"role": "user", "content": content})

    def assistant_tool_call(self, call_id: str, name: str, args_json: str, content_text: str = ""):
        self._msgs.append({"role": "assistant", "content": content_text or None,
                           "_seg": "assistant_text",
                           "tool_calls": [{"id": call_id, "type": "function",
                                           "function": {"name": name, "arguments": args_json}}]})

    def tool_result(self, call_id: str, text: str):
        self._msgs.append({"role": "tool", "tool_call_id": call_id,
                           "_seg": "tool_result", "content": text})

    def raw(self) -> list[dict]:
        return self._msgs

    def state_view(self, state_text: str, image_b64: str,
                   *, parts: list[tuple[str, str]] | None = None,
                   px: tuple[int, int] = (0, 0)) -> list[dict]:
        """冻结前缀 + 一条状态报告（设计说明）。

        raw 里的 observation / assistant / tool 消息一条都不进 —— 它们仍然照常记着
        （日志、window 模式都靠它），只是不发给模型：状态报告已经把该说的重建了一遍。

        parts 不传时整段记 "state_report"（兼容旧调用）。传了就逐段标记，但
        **返回前合并回一个 text part** —— 分段是记账的需要，不是协议的需要；
        协议结构（一个 text part + 一个 image part）一字不动，分段信息挂在
        合并后那个 part 的 `_obs_parts` 上（和 user_observation 同一个手法）。
        """
        out: list[dict] = []
        for m in self._msgs:
            if m["role"] == "system":
                out.append(dict(m))
            elif (m["role"] == "user" and isinstance(m["content"], list)
                    and all(p.get("_kind") == "prefix" for p in m["content"])):
                out.append({"role": "user", "content": [dict(p) for p in m["content"]]})
        pairs = parts if parts is not None else [("state_report", state_text)]
        joined = "\n\n".join(t for _, t in pairs)
        # parts 和 state_text 都给了时，两者必须是同一段字节的两种形态
        # （build_state_parts 的 join 就是 build_state_text）。不加这条断言的话
        # state_text 会被静默丢弃：将来两个 builder 一旦分叉，谁都发现不了。
        if parts is not None and state_text != joined:
            raise ValueError(
                "state_view: state_text 与 parts 拼接结果不一致 —— "
                "build_state_text 和 build_state_parts 已经分叉。\n"
                f"state_text（{len(state_text)} 字）: {state_text[:200]!r}\n"
                f"join(parts)（{len(joined)} 字）: {joined[:200]!r}")
        content = [{"type": "text", "text": joined, "_seg": pairs[0][0], "_obs_parts": pairs}]
        content.append({"type": "image_url", "_seg": "image", "_px": list(px),
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
        out.append({"role": "user", "content": content})
        return out

    def windowed(self, keep_images: int = 1, keep_elements: int = 1,
                 max_tool_chars: int = config.TOOL_RESULT_MAX_CHARS,
                 max_latest_tool_chars: int = config.LATEST_TOOL_RESULT_MAX_CHARS) -> list[dict]:
        """工具结果分两档截断：最新一条给足，更老的截短。

        原来一档 500 字符**含最新那条** —— 模型正要据以行动的结果被砍掉一半。
        collect 这类一次返回一整屏汇总的工具在这一档下根本没法用。
        更老的仍截到 500：那些模型已经用过了，留着只是历史。
        """
        msgs = copy.deepcopy(self._msgs)
        # 只认最后一条 tool 消息。多工具调用被拒时会连着推好几条，但那些都是几十字的
        # 拒绝理由，谁拿到大额度都无所谓。
        latest_tool = max((i for i, m in enumerate(msgs) if m["role"] == "tool"), default=None)
        img_idx = [i for i, m in enumerate(msgs) if m["role"] == "user" and isinstance(m["content"], list)
                   and any(p.get("type") == "image_url" for p in m["content"])]
        el_idx = [i for i, m in enumerate(msgs) if m["role"] == "user" and isinstance(m["content"], list)
                  and any(p.get("_kind") == "elements" for p in m["content"])]
        strip_img = set(img_idx[:-keep_images]) if keep_images > 0 else set(img_idx)
        strip_el = set(el_idx[:-keep_elements]) if keep_elements > 0 else set(el_idx)
        for i, m in enumerate(msgs):
            if m["role"] == "tool":
                cap = max_latest_tool_chars if i == latest_tool else max_tool_chars
                if len(m["content"]) > cap:
                    m["content"] = m["content"][:cap] + "…"
            if not isinstance(m.get("content"), list):
                continue
            parts = []
            for p in m["content"]:
                if p.get("type") == "image_url":
                    if i in strip_img:
                        # 已经是文本了，不能再按图片统计（Codex #6）
                        parts.append({"type": "text", "text": IMAGE_PLACEHOLDER,
                                      "_seg": "image_placeholder_text"})
                    else:
                        parts.append(dict(p))
                elif p.get("_kind") == "elements":
                    text = p["text"]
                    new = {"type": "text", "text": text}
                    # 用 `in` 判断而不是 .get()：漏标时要缺键，不能静默写进一个
                    # `_seg: None` —— 那会让「每个 part 都有 _seg」的防线出个洞。
                    if "_seg" in p:
                        new["_seg"] = p["_seg"]
                    if i in strip_el:
                        head = text.splitlines()[0] if text else ""
                        new["text"] = f"{head}\n{ELEMENTS_PLACEHOLDER}"
                        # 截断后文本已经不是原来那几段了，分段信息不再对应实际
                        # 字节，带过去只会把 token 记到对不上的段上 —— 丢掉。
                    elif "_obs_parts" in p:
                        # 未截断：分段必须带过去，否则 window 模式下多段会被压回
                        # 顶层单一 _seg，几段的 token 全记到第一段头上（静默错账）。
                        new["_obs_parts"] = p["_obs_parts"]
                    parts.append(new)
                else:
                    parts.append(dict(p))
            m["content"] = parts
        return msgs


def middle_truncate(s: str, cap: int) -> str:
    """头尾各留一半，中间挖掉。

    失败的工具结果关键行常在尾部（报错 dump），一律尾截会正好把它砍掉（OpenHands 的做法）。
    """
    if len(s) <= cap:
        return s
    half = cap // 2
    return s[:half] + f"\n[... 已省略 {len(s) - cap} 字 ...]\n" + s[-(cap - half):]


def build_state_parts(*, history: str, memory: str | None, last_result: str | None,
                      transition: str | None, elements_text: str, step: int,
                      max_steps: int, user_notes: list[str] | None = None) -> list[tuple[str, str]]:
    """每步重建的状态报告，按段拆开（设计说明）。

    段落顺序固定：历史 → 备忘 → 用户补充 → 上一步结果 → 上一步之后 → 当前屏幕 → 步数。
    history 自带【到目前为止】标题，transition 自带【上一步之后】标题。
    user_notes 是任务中途用户插的话（网页暂停后补的），每轮都带上 —— 它和任务一样是指令，
    不是历史里那种「数据」。

    ⚠ 段名要和 budget.py 的段表对上。拼接产生的 "\n\n" 分隔符不属于任何段，
      由 budget 记进 protocol_residual。
    """
    parts: list[tuple[str, str]] = [("state_history", history)]
    if memory:
        parts.append(("state_memo", "【备忘】\n" + memory))
    if user_notes:
        parts.append(("state_user_note", "【用户补充】\n" + "\n".join(f"- {n}" for n in user_notes)))
    if last_result:
        parts.append(("state_last_result", "【上一步结果】\n" + last_result))
    if transition:
        parts.append(("state_transition", transition))
    parts.append(("state_elements", "【当前屏幕】\n" + elements_text))
    parts.append(("state_step", f"第 {step}/{max_steps} 步"))
    return parts


def build_state_text(*, history: str, memory: str | None, last_result: str | None,
                     transition: str | None, elements_text: str, step: int, max_steps: int,
                     user_notes: list[str] | None = None) -> str:
    """build_state_parts 的 join。签名和返回值与拆分前逐字符相同。"""
    return "\n\n".join(text for _, text in build_state_parts(
        history=history, memory=memory, last_result=last_result, transition=transition,
        elements_text=elements_text, step=step, max_steps=max_steps, user_notes=user_notes))


def to_wire(view: list[dict]) -> list[dict]:
    """发送前的唯一净化点：剥掉内部标记，其余逐字节不动。

    ⚠ 只删 INTERNAL_KEYS 里的键，不笼统删任意 `_` 开头的键 —— provider 将来
      真需要一个下划线开头的字段时，笼统删会静默把它吃掉。

    ⚠ 必须覆盖四条 v1 漏掉的路径（Codex 阻断 #1）：system 的顶层键、
      windowed 里被保留而原样 append 的 image_url part、tool/assistant 那种
      content 不是 list 的消息、state_view 新建的状态消息。所以这里对
      **每条消息的顶层**和**每个 list part** 都过一遍，不按消息类型分支。
    """
    out = copy.deepcopy(view)
    for m in out:
        for k in INTERNAL_KEYS & m.keys():
            del m[k]
        content = m.get("content")
        if isinstance(content, list):
            for p in content:
                if isinstance(p, dict):
                    for k in INTERNAL_KEYS & p.keys():
                        del p[k]
    return out
