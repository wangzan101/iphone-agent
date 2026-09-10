"""环境变量与常量。待标定常量集中在此，每个附来源与日期。"""
from __future__ import annotations

import os
import re as _re
from pathlib import Path as _Path

from iphone_agent.model.errors import ConfigError  # noqa: F401  旧路径别名，外部 import 不断

# 熔断与时限（spec M11）
# ⚠ 这两个是**保险丝**，不是「任务该多长」的设计假设。上下文这套结构（设计说明）按步数无界设计，
#   每步发给模型的东西大小固定；这里只防失控烧钱。原来 30 / 600 是按早期的简单任务定的
#   （历史里成功 run 中位数 6 步），拿它反过来限制复杂任务是本末倒置。按任务传 max_steps 覆盖。
MAX_STEPS = 100
# ⚠ 实际时限不读这个常量 —— RunConfig.effective_timeout_s 按 max_steps ×
#   SECONDS_PER_STEP 推（见下）。TASK_TIMEOUT_S 现在只是 _config_snapshot 里
#   留档用的历史字段，改它不会改变任何一次运行真正跑多久。
TASK_TIMEOUT_S = 1800
# 每步的时间预算，用来从 max_steps 推默认时限（RunConfig.effective_timeout_s）。
# ⚠ 时限和步数是两个保险丝，必须对得上量：原来 30 步/600s 和 100 步/1800s 都是
#   各自拍脑袋定的，100 步 ×（模型 3.5s + settle）会先撞时限 —— 于是步数上限
#   形同虚设，任务总以 timeout 收场，看不出是「走不完」还是「走太慢」。
SECONDS_PER_STEP = 30
MODEL_TIMEOUT_S = 120
OP_TIMEOUT_S = 10
NO_PROGRESS_WARN = 3
NO_PROGRESS_STOP = 6
# 连续这么多次点击完全没反应，就走恢复阶梯（harness/recovery.py），而不是告诉模型「可能没点中」。
# 取 3：一次可能是点空了，两次可能是运气，三次连画面都不动就不是模型的事了。
DEAD_TAPS_ALERT = 3
# 一次任务里最多重启镜像几次（harness/recovery.py 阶梯第 ③ 级）。重启一次约半分钟到一分钟；
# 两次都救不回来，多半不是重启能解决的事，结束任务让人看。
RECOVERY_MAX_RESTARTS = 2

# open_app 打不出字时的退路：回主屏、翻页找图标。只用**点击和滚动**，不用打字。
# 2026-09-08 批跑实测：打字整条通道死掉时（39/40 失败），点击和滚动是好的
# （日历 6/6、音乐 9/10）—— 所以退路必须走**另一条通道**才有意义。
HOME_PAGES_MAX = 6
# ⚠ 横向翻页之后必须多等一下再点，否则点击会落到错的地方。
#   2026-09-07 实测：翻到有「设置」的那一页，标签在 (104,894)，**2 秒后再观察
#   它仍在 (104,894)**（页面确实已经静止），可紧接着点却打开了另一个 App，
#   等 2 秒再点才对。连续三次复现 —— settle 检测不到这段，只能硬等。
AFTER_PAGE_FLIP_S = 2.0
MAX_CONSECUTIVE_REJECTIONS = 5  # 校验失败/同屏重复/多工具调用连续拒绝这么多次即终止（不占用总步数预算）
WEB_CONFIRM_TIMEOUT_S = 120     # 网页上等人确认写操作的时限；没人应答按拒绝 —— 「无人只读」在网页这头的形态

# schema 校验
CLAMP_TOLERANCE = 0.02  # 越界 2% 以内夹回，超出拒绝（spec §6.1）
MAX_TYPE_TEXT = 500
# erase 一次最多退几个字符。设上限不是怕慢，是**怕删过头** —— 退格删掉的东西没有
# 回头路，而模型对「输入框里现在有几个字」的估计经常是错的。宁可让它退两次。
MAX_ERASE = 100
WAIT_MIN_S, WAIT_MAX_S = 0.5, 10.0

# ---- 消息滑窗里工具结果的两档预算 ----
# 分两档的理由：**最新那条结果是模型正要据以行动的东西**，砍掉它等于让它盲着走下一步；
# 更老的结果模型已经用过了，留在上下文里只是历史，截短纯赚。
# 一档的时候（全都 500）踩过：recall 的正文、collect 的汇总都在第一次返回时就被砍一半。
TOOL_RESULT_MAX_CHARS = 500
LATEST_TOOL_RESULT_MAX_CHARS = 20000

# ---- 高阶滚动（scroll_until / collect）----
# 存在的理由：单步 scroll 滚一屏 = 一步 + 一次模型调用（约 3.5 秒），
# 20 屏的长列表能把 MAX_STEPS 烧光，长列表任务因此根本做不了。
#
# ⚠ 屏数上限只是**防死循环的兜底**，不是正常的停止条件。正常停在「滚不动了」——
#   判据是 did_change，与单步 scroll 判 changed 的那一套完全相同（见 perceive/change.py）。
SCROLL_MAX_SCREENS = 25
# 连续这么多屏没变化才算「滚到底」。不能只看一次：「画面没变」既可能是真到底，
# 也可能是这一次滚动没生效，一次就收工会把半个列表当成全部交出去。
# 和熔断的 NO_PROGRESS_WARN 是同一个形状 —— 几次都没变化才下结论。
SCROLL_END_CONFIRMATIONS = 2
SCROLL_UNTIL_TEXT_MAX = 100
# collect 的汇总要一次性喂给模型，必须留在 LATEST_TOOL_RESULT_MAX_CHARS 之内 ——
# 否则滑窗会从中间把 JSON 切断，模型拿到半截结果比拿到「收了一部分」更坏。
# 差出来的这 4000 给 JSON 转义和其余字段。
COLLECT_MAX_CHARS = LATEST_TOOL_RESULT_MAX_CHARS - 4000

# 待标定（spec §11）。初值仅供起步，标定脚本跑完后覆盖并写明来源。
# 阈值标定（2026-09-07，macOS 15.6.1，窗口 322x718，
#   原始数据见 内部标定记录（未公开））
#
#   判据            噪声上限(35样本)   真实变化(9样本)
#   文本集合差异          2              27 - 51
#   aHash 汉明            0               0 - 34
#
# 文本判据非常干净：13 倍分离度。它是主判据。
# aHash 是「有信号即可信，无信号不可信」—— 文字密集页滚动前后缩成 8x8 灰度几乎一样
#   （关于本机滚半屏：aHash=0 而文本=36），但图片页一滚就是 34。
#   两个判据互补而非冗余：将来遇到「只换图片不换文字」的页面（相册、视频列表），
#   文本会失明而 aHash 救场。所以 did_change 用「或」是对的。
#
# ⚠ 标定时有一个无效样本：备忘录空白新建页「滚半屏」得到 aHash=0/文本=2 ——
#   空页面本来就没有内容可滚，屏幕不该变，那是噪声不是变化。取最小真实变化时已剔除。
#
# aHash 阈值取 2 而非 1：八帧采样窗口很短，可能低估真实噪声（电量、信号、通知角标）。
#   文本判据够强，aHash 宁可保守 —— 假阳性会让模型以为动作成功了继续往下走，
#   比假阴性坏得多；假阴性只让它换个方法再试，无进展熔断兜得住。
AHASH_CHANGED_THRESHOLD = 2
STABLE_THRESHOLD = 1          # 静止时相邻帧汉明恒为 0，收紧让 settle 判得更准
TEXT_DIFF_THRESHOLD = 8       # 噪声 2、最小真实变化 27，取中间偏保守

# 状态栏裁剪比例（aHash 前裁掉顶部这一段）
# ⚠ 这个值原来是 0.05，**盖不住状态栏**。2026-09-08 实测（图像 624x1388）：
#   顶部有约 104px 是机身黑边，状态栏文字在 y=104~133，折算比例 0.075~0.096。
#   裁 5% 只裁掉了黑边，时钟和电量原样留在里面 —— 而 aHash 一直用的就是这个比例。
#   时钟每分钟变一次，OCR 还读不稳（'12:221' / '12:22⑦'、'17:054' / '17:06'），
#   于是「什么都没发生」也会有 2 的文本差和非零的汉明。
#   取 0.11：盖住 133px 还留一点余量。
STATUS_BAR_CROP_RATIO = 0.11

# 注入路径：background 走 SkyLight（点击不抢焦点），foreground 走公开 CGEvent。
# 真机实测 background 的点击有反应、前台不变、用户可继续打字。
INJECT_MODE = os.environ.get("IPHONE_USE_INJECT", "background")

# ---- 屏幕解析（perceive/screen.py）----
# 每次观察多花一次视觉模型调用，换来 OCR 给不出的图标/无字按钮/开关状态。
# 关掉就是 2026-09-09 之前的纯 OCR 行为，一行不差 —— 这条降级路要一直留着：
# 端点会抽风、会超时，而它只是增强，不是命脉。
# 每次观察都让视觉模型把整屏读成结构化元素。**默认开着**，因为它是成功的直接条件：
#
#   回归场景（名称为合成示例）：OCR 漏掉「示例银行」列表项，视觉解析能补全。
#   若解析输出被 max_tokens 截断，结构化元素可能全部丢失（见 transports 注释）。
#
#   我一度把它默认关掉、改用 zoom 按需放大，理由是它慢（每步 20~34s）。那是错的：
#   zoom 解决「看不清」，不解决「不知道有」—— 模型得先知道列表里有「示例银行」，
#   才谈得上去放大哪一块。功能优先于速度，慢可以再优化，丢了元素就是任务失败。
#
#   关掉它（IPHONE_SCREEN_PARSE=off）会退回纯 OCR：快很多，但 OCR 读不到的图标、
#   无字按钮、以及像上面那样被 OCR 漏掉的列表项，模型就只剩 zoom 和估坐标两条路。
SCREEN_PARSE = os.environ.get("IPHONE_SCREEN_PARSE", "on").lower() not in ("0", "off", "false")

# ---- 从 OCR 结构派生无文字目标（2026-09-08 真机实测）----
# 列表行的版式很规整：文字在左、chevron/开关/数值在右、图标在最左，都在同一个 y 上。
# 设置首页（图像宽 624）实测：文字 x≈150-250，chevron x≈553-555，图标 x≈82；
# 键盘页的 '4>' 在 x=538、'关〉' 在 x=535。折算成比例就是右端 0.88、左端 0.13。
#
# ⚠ 为什么必须有 row_right：**开关点行文字没用，只有点右端才切换**（实测，
#   点文字开关区颜色不变，点右端从灰 (245,245,246) 变绿 (211,236,212)）。
#   而开关在 OCR 里根本没有元素 —— 这就是「无文字目标」的典型。
ROW_RIGHT_RATIO = 0.88
ROW_LEFT_RATIO = 0.13

# ---- 局部变化：整屏判据看不见小控件 ----
# 2026-09-08 标定：点空白处（噪声）局部 MAD 三次都是 0.0；开关翻转（信号）两次是
# 17.0 / 16.98。而**同样这两次开关翻转，整屏 aHash 汉明 = 0、文本差 = 0** ——
# 两个现有判据全瞎了，只有局部看得见。
#
# 后果不修就是：模型点了开关，工具说「没变化」，它以为没点中，再点一次 —— 又翻回去了，
# 而且熔断还把这算成无进展。
#
# 噪声 0 / 信号 17，取 4：离噪声足够远，离信号足够远。
# ⚠ 必须是比例，不能是绝对像素。半径要「够罩住一个开关、又不把整行算进去」，
#   而开关是按点算的固定尺寸（约 51pt 宽），窗口按点算也基本固定 —— 所以
#   「开关占窗口宽的比例」才是稳定量，绝对像素不是。
#   标定这台：窗口 312pt / 图像宽 624px，40px 罩住直径 12.8%，开关约占 16%，正好。
#   写死 40 的话，**换台屏更大的手机、甚至只是把镜像窗口拉大**，图像宽变 900，
#   40px 就只罩住开关一角，MAD 被周围没变的像素稀释 → 漏判。
LOCAL_PATCH_RATIO = 0.064    # 40 / 624，见上
LOCAL_CHANGED_MAD = 4.0

# ---- 输入法候选栏的几何特征（2026-09-08 探针实测，runs/ime-q*.png）----
# 候选栏是**一条水平带**：所有候选的 y 几乎相同。实测四组样本跨度都只有 2px
# （'1设置'@1218 '2 摄制'@1217 '3 摄制组'@1218，图像高 1388），
# 而它上面最近的一行在 1185，差 32px。取图像高的 1% 当聚带容差，两边都很宽裕。
CAND_BAND_TOL_RATIO = 0.01
# ⚠ 这里原来还有一个 CAND_MIN_Y_RATIO = 0.70（「候选栏贴着键盘」），2026-09-09 删掉了：
#   候选栏贴的是**光标**，不是键盘。备忘录里它出现在 y/H=0.229，被那个下限整条筛没，
#   于是中文输入在 App 内容框里 8 次尝试 8 次全挂。详见 executor._candidates 的注释。


# ---- 记忆系统（spec 内部标定记录（未公开））----
# 记忆目录不再是常量：位置由 Workspace 决定（MemoryStore(root=None) → 默认工作区）。

# ⚠ 只收小写 ASCII。大写在 macOS 默认不区分大小写的文件系统上会和小写撞；
#    非 ASCII 有 NFC/NFD 两种写法，同一个词可能对应两个文件名。
#    校验只拒绝、不规整 —— 规整会把两个不同的名字并成一个，静默覆盖别人的记忆。
# ⚠ 用 \Z 不是 $：Python 的 $ 会匹配「末尾换行符之前」，
#    于是 'memory\n' 能过正则，而保留名检查用的是精确比较、也认不出它 ——
#    两道检查被同一个换行符从中间穿过去（2026-09-07 review 实测复现）。
MEMORY_NAME_RE = _re.compile(r"^[a-z0-9][a-z0-9-]{0,47}\Z")
MEMORY_RESERVED_NAMES = frozenset({"memory", "trash", "index"})

MEMORY_DESC_MAX = 100     # 描述直接进 frontmatter 和索引，必须单行且短
# ⚠ 400 是按 window 模式（旧的滑窗视图）标定的：一条记忆要跨好几步用，所以按
#   TOOL_RESULT_MAX_CHARS（更老结果那一档，500）留余量定的。state 模式下
#   （设计说明，CONTEXT_MODE=state）recall 的正文**只在返回的那一步出现一次**——
#   它不进历史消息、不会被滑窗反复截断，模型要用就得当场把要点抄进 memory 字段。
#   400 字符装不下一条像样的操作步骤说明，改成 2000（设计说明 B2）。
MEMORY_BODY_MAX = 2000
# ⚠ 这是保险丝，不是「记忆库该有多大」的设计假设（设计说明 B1）：索引现算、不落盘，
#   entries 数量只影响 index() 单次扫描目录的开销，不存在「装不下」的硬上限。
#   50 是早期按「索引全部注入 prompt、不截断」的假设定的；随着规模扩大，注入方式
#   本身会先变（分页/相关性截断），届时这个数字该配合注入策略一起再调。
#   5000 只防真正失控（比如某次写入死循环）：正常使用不会撞到它。
MEMORY_MAX_ITEMS = 5000
MEMORY_WRITE_PER_RUN = 3
# 一次运行最多自报用了几条记忆。它是反馈信号不是清单：报一屏名字没有信息量，
# 还会把 run.json 撑大。
MEMORY_USED_PER_RUN = 10
RECALL_PER_RUN = 3
# 记忆「转正 / 淘汰」的阈值。⚠ shadow 期间**只用于记账**：够阈值只往 run.json 的
# memory.shadow 里记一笔 would_verify / would_trash，代码一条记忆都不动 ——
# 先攒够真实数据看看这两个数是不是对的，再谈自动化。记忆永远不自动删、不自动移 trash。
MEMORY_VERIFY_AFTER = 3
MEMORY_TRASH_AFTER = 3
# 索引按需检索的阈值（计划 B Task 3）。⚠ 这也是保险丝不是设计假设：条目数 ≤ 这个
# 数就全量注入（现在的行为，跨任务稳定，吃得到前缀缓存）；超过才按任务相关度截断
# ——截断本身有代价（下面 recap.build_memory_message 的注释细说），所以阈值要
# 留得住大多数正常规模的记忆库，只在真正长出规模时才切换策略。
MEMORY_INJECT_TOPK = 20
# search_memory 工具一次最多返回几条命中。比 MEMORY_INJECT_TOPK 小：它是模型
# 主动发起的一次查询，不是被动注入，回一屏都读不完的结果没有意义。
SEARCH_MEMORY_TOPK = 10
# 网页对话里往回带几轮。带太多会稀释当前任务，带太少就没有「对话」的感觉。
CHAT_HISTORY_TURNS = 6
# ---- 发给模型的上下文视图（设计说明）----
# window：现在的滑窗，每步改历史消息。state：冻结前缀 + 每步重建一份有界状态报告。
# 默认 window：对照实验（设计说明 第 6 步）跑完之前不换默认。
# ⚠ 这是 RunConfig.from_env() 的**默认值来源**，不是运行时开关：loop 读的是
#   RunConfig.context_mode，同一进程里两种视图可以并存（对照实验要的就是这个）。
CONTEXT_MODE = os.environ.get("IPHONE_USE_CONTEXT", "window")
HISTORY_KEEP = 12          # 状态报告里保留的历史行数；超出：第 1 行 + 【更早】段 + 最近 11 行
EARLIER_LIST_MAX = 8       # 【更早】段每个列表最多列几项；超出报"共 N"
MEMORY_FIELD_MAX = 600     # 模型自写备忘的字符上限，整段替换
TRANSITION_LIST_MAX = 15   # 【上一步之后】新出现/消失的文字各列几项
WEB_BACKLOG = 400        # 新连上来补看最近这么多条事件，刷新页面不至于一片空白
WEB_PORT = 8765          # 只监听 127.0.0.1：这个界面能操作你的真手机
RECAP_RUNS = 5            # 注入时附最近几次运行的摘要
RECALL_RUNS_MAX = 20

# ---- skill 层（spec 内部标定记录（未公开））----
# ---- 知识与技能的两层目录（2026-09-09 重排）----
# 知识 = 关于 App 的通用描述（APP.md）；技能 = 怎么做一件事（SKILL.md + 可选 procedure.json）。
# 两者各有结构层（进 git，人写）和个人层（不进 git，自动沉淀落这里），加载时合并、同名个人层优先。
KNOWLEDGE_DIR_SHARED = _Path("knowledge")    # knowledge/apps/<app>/APP.md
KNOWLEDGE_DIR = _Path(".iphone/knowledge")
SKILLS_DIR_SHARED = _Path("skills")          # skills/<skill>/SKILL.md [+ procedure.json]
SKILLS_DIR = _Path(".iphone/skills")
SKILL_INDEX_MAX_LINES = 60                   # 注入索引的行数上限，同 MEMORY.md「只加载前 N 行」的形状
PROC_TOOLS_MAX = 15                          # 路由未命中时，全库合格剧本不超过这么多才全放进工具列表
PROC_MAX_ACTIONS = 12                        # 一条剧本内部动作数上限（含 find 的滚动）：一次调用不能做太多事
PROC_STALE_STREAK = 2                        # 连续这么多次 expect 对不上就转 stale；与 SCROLL_END_CONFIRMATIONS 同形
SKILL_USE_PER_RUN = 3                        # use_skill 每次运行的预算，同 RECALL_PER_RUN
ROUTE_MAX_TOKENS = 200                       # 路由调用只要一个小 JSON
# read 步取「同一行」的 y 容差（占图高比例）。
# ⚠ 初值没标定：候选栏那个 CAND_BAND_TOL_RATIO 是按候选栏几何量的，不能借用。
#   按设置页 / 关于本机页实测后改，标定数据进 runs/calibrate-read-*.json。
READ_ROW_TOL_RATIO = 0.015
# 场景正文里出现这些词，风险档至少提到对应档；人只能往高改不能往低改。
SCENARIO_RISK_WORDS = {
    "write": ("发送", "回复", "记一笔", "修改", "保存", "send", "reply", "post", "edit", "save", "submit"),
    "irreversible": ("支付", "付款", "转账", "删除", "下单", "购买",
                     "pay", "payment", "transfer", "delete", "purchase", "buy", "place an order"),
}
# run_summaries 扫描 runs/ 时，目录名（时间戳）倒序只看前这么多个就去读 run.json，
# 不管全目录下总共有多少个（计划 B Task 4，设计说明 C5）。⚠ 这也是保险丝不是设计
# 假设：目录数量本身无界，扫全量、读每个 run.json、再排序丢掉大半，是纯浪费的
# O(全部 run 数) 开销，而调用方最终只要最近/最相关的 limit 条。200 留了足够宽的
# 余量——正常一天几十次运行也要连续跑好几天才会把真正想找的那条推出扫描窗口，
# 真撞到这个上限时该做的是先把旧 run 归档，而不是不断调大这个数。
RECAP_SCAN_MAX = 200
