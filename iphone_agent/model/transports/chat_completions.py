"""OpenAI Chat Completions 形状的桥。DashScope、DeepSeek、Kimi、GLM、vLLM、OpenRouter……都走它。

请求参数**全部**从 ResolvedModel 的两张表拼；这里不写任何一家的名字。
"""
from __future__ import annotations

import json
import time

from iphone_agent import config
from iphone_agent.harness.actions import Action
from iphone_agent.harness.tools import tool_defs
from iphone_agent.model.config import ResolvedModel
from iphone_agent.model.coords import to_pixels
from iphone_agent.model.reply import ModelError, ModelReply


def assert_no_internal_keys(o, _path: str = "messages") -> None:
    """出站 payload 里不许有任何 `_` 开头的键。

    to_wire() 是约定，这里是保证：将来新增视图函数或调用点忘了过 to_wire，
    在这里立刻炸，而不是把标记发给 provider、让分段账悄悄测的是另一份输入。
    这里查的是**所有** `_` 开头的键，比 INTERNAL_KEYS 更严 —— 出站 payload
    本来就不该有任何下划线私有键。
    """
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(k, str) and k.startswith("_"):
                raise ModelError(f"出站 payload 含内部标记 {_path}.{k}：忘了过 messages.to_wire()")
            assert_no_internal_keys(v, f"{_path}.{k}")
    elif isinstance(o, list):
        for i, x in enumerate(o):
            assert_no_internal_keys(x, f"{_path}[{i}]")


class ChatCompletionsTransport:
    def __init__(self, resolved: ResolvedModel):
        self.resolved = resolved
        self._client = None
        # 最近一次 ask() 的收尾原因。'length' = 撞上 max_tokens 被截断。
        self.last_finish_reason: str | None = None
        self.last_max_tokens: int | None = None

    def _client_or_create(self):
        if self._client is None:
            import openai
            p = self.resolved.provider
            self._client = openai.OpenAI(api_key=self.resolved.api_key, base_url=p.base_url,
                                         timeout=config.MODEL_TIMEOUT_S,
                                         default_headers=dict(p.default_headers) or None)
        return self._client

    def request_kwargs(self, messages: list[dict], tools: list[dict] | None = None,
                       max_tokens: int | None = None) -> dict:
        """tools / max_tokens 传了就压过 profile：主循环要按任务给不同的工具表（剧本工具），
        路由预调用还要一个很小的 token 预算。传 None 时保持 profile 的默认，行为不变。"""
        p, m = self.resolved.provider, self.resolved.model
        kw = {"model": m.id, "messages": messages,
              "tools": tool_defs(m.allow_coord_tap) if tools is None else tools,
              "tool_choice": "auto", "parallel_tool_calls": False,
              "max_tokens": m.max_tokens if max_tokens is None else max_tokens}
        for k in p.omit_params:                       # 有的端点收到不认识的顶层字段直接 400
            kw.pop(k, None)
        extra = dict(p.extra_body) | dict(m.extra_body)   # 模型级覆盖端点级
        if extra:
            kw["extra_body"] = extra
        assert_no_internal_keys(kw["messages"])
        return kw

    def decide(self, messages: list[dict], image_size: tuple[int, int],
               tools: list[dict] | None = None, max_tokens: int | None = None) -> ModelReply:
        t0 = time.time()
        try:
            resp = self._client_or_create().chat.completions.create(
                **self.request_kwargs(messages, tools=tools, max_tokens=max_tokens))
            reply = self.parse_response(resp.model_dump(), image_size)
        except ModelError:
            raise
        except Exception as e:  # openai 的各类异常、以及响应格式异常，统一为 ModelError：
            # 这是模型/协议层的问题，不该被主循环的兜底 except 误诊成 device_error。
            raise ModelError(self._describe(e)) from e
        reply.latency_ms = int((time.time() - t0) * 1000)
        return reply

    def ask(self, messages: list[dict], max_tokens: int | None = None) -> str:
        """不带工具的一次问答，返回纯文本。屏幕解析和各种判定都走它。

        ⚠ 为什么不复用 decide()：decide 永远带着 tool_defs，模型会倾向于**调工具**
          而不是回答问题 —— 我们这里要的是一段 JSON，不是一个动作。
          `tools` 字段整个不出现，比传空列表安全：有的端点收到 `tools: []` 直接 400。

        provider 的 omit_params / extra_body 照样生效 —— 那些是端点的脾气，
        跟这次调用问什么无关。
        """
        p, m = self.resolved.provider, self.resolved.model
        kw = {"model": m.id, "messages": messages}
        # ⚠ max_tokens 传 None = **整个字段不出现** = 不限制，让端点用模型自己的上限。
        #   这里一度默认拿 profile 的 m.max_tokens 来顶，那个数（1024）是给
        #   decide()「回一个工具调用」定的，屏幕解析要列几十个元素，一列就超。
        #   2026-09-09 真机：记账页 51 个元素输出 1000+ token 被硬截断，JSON 断在
        #   半句话，那一屏视觉贡献归零 —— 而**「工资账户」这个元素只有视觉读得到**，
        #   OCR 根本没有它。三次任务失败就卡在这儿。
        #   把 1024 改成 4000 是打补丁：51 个元素要 1000 token，80 个元素的界面呢？
        #   只是把断点往后挪一格。要限制的时候调用方自己传。
        if max_tokens is not None:
            kw["max_tokens"] = max_tokens
        for k in p.omit_params:
            kw.pop(k, None)
        extra = dict(p.extra_body) | dict(m.extra_body)
        if extra:
            kw["extra_body"] = extra
        try:
            resp = self._client_or_create().chat.completions.create(**kw)
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(self._describe(e)) from e
        d = resp.model_dump()
        choice = d["choices"][0]
        # ⚠ 撞上 max_tokens 必须**能被看见**，但**不能因此丢掉已经说出来的部分**。
        #   2026-09-09 真机踩到：屏幕解析在元素多的界面上输出 1000+ token，超过
        #   profile 的 max_tokens=1024，JSON 被硬截断，调用方只看到「解析不出 JSON」——
        #   而真因是「话没说完」，两者的修法完全不同（改 prompt vs 加额度）。
        #   更毒的是它**反向选择**：界面越复杂越需要抓手，输出越长越会被截断。
        #
        #   这里记在实例上而不是抛异常：截断的 JSON 数组仍然能抢救出前面完整的那些
        #   （见 model/vision.py 的 extract_json），半个列表远好过一个空列表。
        #   抛了就等于把能用的部分一起扔掉。
        self.last_finish_reason = choice.get("finish_reason")
        self.last_max_tokens = kw.get("max_tokens")
        return (choice["message"].get("content") or "").strip()

    @staticmethod
    def _describe(e: Exception) -> str:
        """带上 HTTP 响应体原文：用户要靠它判断是该填 omit_params 还是模型不吃图片。"""
        msg = f"{type(e).__name__}: {e}"
        body = getattr(e, "body", None)
        if body:
            msg += f"\n响应体：{json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else body}"
        if getattr(e, "status_code", None) == 400:
            msg += ("\n（端点拒绝了请求。如果它不认识某个顶层参数，在 config.toml 的 [providers.<name>] "
                    "里加 omit_params；如果这个模型不支持图片输入，本期不支持，等子项目 4）")
        return msg

    def parse_response(self, resp: dict, image_size: tuple[int, int]) -> ModelReply:
        msg = resp["choices"][0]["message"]
        actions: list[Action] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args = {"_parse_error": True}
            reason = str(args.pop("reason", "")) if "_parse_error" not in args else ""
            expect = args.pop("expect", None) if "_parse_error" not in args else None
            # eval / memory 与 reason、expect 同属「模型的话」而不是动作参数，必须一起弹出，
            # 否则会混进 args 被 validate_action 当成非法参数拒掉。
            ev = args.pop("eval", None) if "_parse_error" not in args else None
            mem = args.pop("memory", None) if "_parse_error" not in args else None
            # ⚠ 这里**不按动作名分支**。2026-09-16 之前写的是 `if fn["name"] == "tap"`，
            #   于是 zoom 的框从来没被换算过（事故与证据见 model/coords.py 的注释）。
            #   哪些键是坐标由 coords.py 一处说了算，桥只管把 args 交给它。
            args = to_pixels(args, self.resolved.model.coord_mode, image_size)
            actions.append(Action(name=fn["name"], args=args, reason=reason,
                                  expect=str(expect) if expect else None, call_id=tc["id"],
                                  eval=str(ev) if ev else None, memory=str(mem) if mem else None))
        return ModelReply(actions=actions, text=msg.get("content") or "",
                          model_version=resp.get("model", ""), usage=resp.get("usage") or {}, latency_ms=0)
