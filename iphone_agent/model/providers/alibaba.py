"""阿里云 DashScope（OpenAI 兼容端点）。真机验证：**是**（2026-09-08 起，qwen3.7-plus）。"""
from iphone_agent.model.profile import ModelProfile, ProviderProfile

PROVIDER = ProviderProfile(
    name="alibaba", aliases=("dashscope",),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    env_vars=("DASHSCOPE_API_KEY",),
)

MODELS = (
    # 2026-09-08 标定（scripts/calibrate_coords.py，iPhone 16e，图像 624x1388，逐条数据
    # 在 runs/calibrate-coords.json）：让模型直接报坐标，两种约定的误差差了 20 倍 ——
    #
    #   目标   约定        中位数    p90
    #   文字   norm1000    8.4px    18.3px
    #   文字   pixel     185.3px   328.6px
    #   图标   norm1000   35.0px   116.6px
    #   图标   pixel     177.0px   314.2px
    #
    # 像素约定基本等于乱指（图像才 624 宽，误差 180px 起步）。默认必须是 norm1000。
    # enable_thinking 是 Qwen3 混合思考的开关 —— 模型的属性不是端点的：
    #   同一个端点上的 qwen-vl-max 没有这个参数。
    ModelProfile(id="qwen3.7-plus", provider="alibaba", coord_mode="norm1000",
                 allow_coord_tap=True, calibrated=True,
                 extra_body={"enable_thinking": False}),
)
