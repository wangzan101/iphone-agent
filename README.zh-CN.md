<p align="center">
  <img src="assets/logo/mark.svg" width="88" alt="iPhone Agent">
</p>

<h1 align="center">iPhone Agent</h1>
<p align="center"><strong>给你的 iPhone，一个自己的 Agent。</strong></p>
<p align="center">让它操作真实的手机，也一起探索：它能不能逐渐学会使用这台手机？</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/platform-macOS-black" alt="macOS">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/status-Alpha-orange" alt="Alpha">
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="https://iphone-agent.com">官网</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#手机的数字孪生">数字孪生</a> ·
  <a href="#一起做边做边学">一起做</a>
</p>

会写代码的 Agent 已经很多了。**我们想让 Agent 也能用你的手机。**

iPhone Agent 通过 macOS「iPhone 镜像」观察真实屏幕，理解任务，点击、输入、检查结果，并留下可以回看的记录。你可以在本机网页或终端里给它一个任务，模型由你自己选择。

**不越狱，手机上不装 Agent，通过 Mac 后台输入操作。**

我们想继续往前走：让它认识这台手机，记住走得通的路径，在界面变化时修正自己的知识，把一次次使用沉淀成可以复用的技能。这是我们正在公开建设的方向，第一步从主屏布局开始。

> **Alpha，持续迭代中。** 已有任务循环、本机界面、记忆、技能和第一期设备布局层。不同机型、App 和模型的兼容性还在扩展，欢迎带着真实问题加入。[当前范围](docs/digital-twin.md) · [已知限制](docs/limitations.md)

## 先看看它怎么工作

[官网的动态演示](https://iphone-agent.com)展示了产品交互思路。其中手机画面是概念动画，不是真机执行录像。

你可以从一个简单任务开始：

```sh
iphone run "打开设置，进通用，进关于本机，告诉我 iOS 版本号"
```

下面是工作过程示意，不是一次实际运行的日志：

```text
你：       告诉我这台手机的 iOS 版本。
Agent：    观察 → 打开设置 → 导航 → 读取 → 核对 → 回答
Mac：      运行 Agent 和本机界面。
iPhone：   在真实的「设置」App 中完成操作。
任务之后：  回看它观察了什么、做了什么、哪里没有达到预期。
```

如果你跑通了一个有趣的任务，欢迎[分享脱敏后的演示或技能](docs/sharing.md)。我们希望这里逐渐长出来自不同设备、不同 App 的真实案例。

## 为什么值得做

手机里有很多已经存在、每天都在使用的工作流，却不一定有方便调用的 API。通过屏幕交互，Agent 有机会进入这些场景。

真正难的地方，是接下来的问题：

- 现在究竟在哪个页面？
- 刚才打开的是不是目标 App？
- 这条路径以前有没有走通过？
- 页面变了，旧经验还能不能用？

我们希望成功的任务留下知识，失败的任务留下证据，让下一次尝试有更好的起点。

| 能力 | 对你意味着什么 |
|---|---|
| **真实 iPhone** | 通过镜像操作手机上已有的 App，不需要给每个 App 单独对接。 |
| **后台输入** | 常规输入路径尽量保留 Mac 的光标和焦点；连接、恢复时仍可能需要你处理。 |
| **看得见的执行过程** | 能检查它看到了什么、尝试了什么、怎么判断结果。 |
| **记忆和技能** | 跨任务保留有用信息，使用可阅读、可整理的文件积累经验。 |
| **自选模型** | 通过配置切换服务商或兼容端点，复用相同的任务入口。 |
| **逐步认识设备** | 从主屏布局开始，为后续的屏幕、路径和状态建模积累基础。 |

## 可以先试什么

先选一个小任务，在旁边观察一次。下面是可尝试的提示词，并不代表所有设备和 App 版本都已通过验收。

| 对它说 | 观察什么 |
|---|---|
| “打开设置，告诉我 iOS 版本号。” | 打开 App、导航、读取和回答。 |
| “当前主屏上能看到哪些 App？” | 对当前画面的理解。 |
| “读一下这个页面的文字，帮我总结。” | OCR 与模型理解。 |

示例以中文系统为主要语境，其他语言请调整任务描述。第一次成功后，可以回看记录，再尝试相关任务。涉及写入时，先使用测试内容，并在旁边观察。

## 快速开始

### 准备条件

- 一台已经能正常使用 **iPhone 镜像** 的 Mac 和 iPhone。系统、硬件、账号及地区条件见[苹果官方说明](https://support.apple.com/zh-cn/120421)。
- **Python 3.12 或更新版本**。
- 你自己的模型 API 和密钥。模型调用可能收费，聊天订阅不一定包含 API 额度。

驱动运行在 macOS 上；这个项目不需要在 iPhone 上安装软件。

### 方式一：让 Coding Agent 帮你安装

可以把这段话交给 Codex、Claude Code 等能操作本地文件和终端的助手：

```text
帮我在这台 Mac 上安装 https://github.com/wangzan101/iphone-agent。
先阅读 README.zh-CN.md 和 docs/getting-started.md，检查已有目录，
不要覆盖已有项目。检查 Python 环境，创建项目虚拟环境，安装依赖，
运行 iphone doctor。需要 iPhone 镜像配对或 macOS 权限时引导我完成。
让我在本机设置页填写模型密钥，然后帮我启动 iphone serve。
等我选择第一次要执行的手机任务。
```

这是辅助安装提示词，不代表项目已经提供 MCP 插件。Coding Agent 帮你搭好环境，iPhone Agent 负责执行手机任务。

### 方式二：自己安装

```sh
git clone https://github.com/wangzan101/iphone-agent.git
cd iphone-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[repl]'
iphone doctor
```

根据诊断结果完成镜像连接和权限检查，然后启动：

```sh
iphone serve
```

Web 界面支持 **English / 简体中文**，可以在侧栏或首次配置弹窗切换。切换不改变手机语言，也不翻译任务原文和历史记录。详见[语言支持](docs/languages.md)。

打开 **http://127.0.0.1:8765**，进入设置，配置服务商、模型与 API Key。再回到任务界面，输入一个简单任务。

如果喜欢文件配置，可参考[配置示例](examples/config.example.toml)。完整步骤、模型配置及排障见[启动指南](docs/getting-started.md)。

### 选择顺手的入口

| 入口 | 命令 | 适合什么 |
|---|---|---|
| 本机网页 | `iphone serve` | 看手机画面、输入任务、管理设置。 |
| 交互终端 | `iphone` | 在终端持续对话。 |
| 单次任务 | `iphone run "任务"` | 从命令行发起一个明确请求。 |

终端里按 Ctrl-C 请求停止，正在执行的动作可能会先完成。想检查已经结束的任务：

```sh
iphone replay runs/<run-id>
```

把 `<run-id>` 换成实际生成的运行目录名。这个命令读取并展示记录，不会重新操作手机。

## 手机的数字孪生

**如果 Agent 不用每次都重新认识你的手机，会怎么样？**

我们想建立一份可更新的设备知识：App 在哪里、屏幕表达什么、哪些路径有效，以及什么时候需要重新确认。

它不应该只有记忆力，还需要知道自己的记忆可能已经过时。

| 层次 | 当前进度 |
|---|---|
| 主屏布局 | 已实现 schema、网格推断、持久化和扫描。 |
| 利用布局打开 App | 已实现：先试主屏，核对身份，布局不对时降级。 |
| 屏幕知识 | 已实现：屏幕由视觉标注命名，可从留档重建，并有错合 / 错归两道闸门。 |
| 位置与路线提示 | 已实现：给模型「你在哪」「从这里走通过哪些路」，并标明是参考不是指令。 |
| 跨任务记忆和技能 | 已有独立实现，后续继续完善整合。 |
| 更广泛的兼容性及过期知识处理 | 持续验证与开发中。 |

现在可以尝试设备布局层，以及屏幕层：

```sh
iphone twin scan
iphone twin show
iphone twin rebuild
iphone twin report
```

**scan 会一页页走过主屏并记录，全程只看不点，并不是扫描整台手机或所有 App。** 任务中的后续观察也可以刷新布局信息。屏幕层由重放你自己的留档建出来，所以一开始是空的。

详细设计见[数字孪生文档](docs/digital-twin.md)。如果你遇到了布局识别错误、页面变化后失效、打开错误 App，这些都可以成为下一步改进的入口。

## 看屏幕：自己选取舍

每次观察都会跑 OCR。整屏解析（交给视觉模型看整张图）才能读出无字图标和 OCR 读不到的列表项 —— 它同时也是一步里最慢的一环。

```sh
IPHONE_SCREEN_PARSE=on_demand iphone run "你的任务"
```

`always`（默认）、`on_demand`、`off` 决定整屏解析什么时候跑。`on_demand` 下，模型主动要看、或某一步要放弃之前兜底一次，才做整屏解析；另有一次很短的标注调用让孪生继续长。一次单机冒烟对照把每步中位耗时从 34.0s 降到 18.8s —— 每格只跑一遍、always 在前，只能算一个方向，不是测量结论。**在有更好的证据之前，默认值仍然是 `always`。** 见[整屏解析说明](docs/screen-parsing.md)（英文）。

## 教会它一个 App，把经验分享出去

一次有用的发现，不应该只停留在一次运行里。

App 知识描述“这个 App 是什么、有什么结构”；技能描述“怎样完成一件事”。它们以文件存在，区分共享内容和个人内容：

```text
knowledge/apps/<app-id>/APP.md         共享的 App 知识
skills/<skill-name>/SKILL.md           共享的任务技能
.iphone/knowledge/ 与 .iphone/skills/  个人知识与草稿
```

看看自己的工作区里有什么：

```sh
iphone skill list
iphone skill show <app-id>
iphone skill export <app-id> ./skill-export
```

`<app-id>` 用列表显示的 ID 替换。导出需要个人层已有该 App 的知识，刚安装时列表可能为空。导出会移除部分来源记录，但文本和文件仍需要人工检查后再分享。

我们提供了一个[最小技能样例](examples/shared-skill/skills/read-ios-version/SKILL.md)和[分享指南](docs/sharing.md)。样例是草稿，不是已经真机认证的技能包。社区技能的整理、验证和分发流程，也欢迎一起建设。

## 模型由你选择

代码包含 Alibaba/Qwen、OpenAI、Anthropic、DeepSeek、Moonshot/Kimi、Zhipu/GLM 和 OpenRouter 的配置预设，也支持自定义 OpenAI-compatible 端点。

预设解决配置入口，不代表该服务商所有模型都已验证。图片输入、工具调用和坐标行为要看具体模型与端点。目前注册表对 `alibaba:qwen3.7-plus` 声明了坐标标定，其余模型默认使用更保守的坐标控制。

可在网页设置或 `.iphone/config.toml` 中配置。更多说明见[启动指南](docs/getting-started.md)和[兼容性限制](docs/limitations.md)。

## 它是怎么工作的

```text
                    CLI / 本机网页
                          |
                       任务循环
          观察 → 模型 → 校验 → 执行 → 检查结果
                          |
        driver · perceive · model · memory · skills · twin
                          |
               macOS iPhone 镜像 → 真实 iPhone
```

任务循环显式组织每一步：驱动负责抓帧与输入，感知负责把画面变成可用信息，记忆与技能提供历史经验，孪生层开始描述设备。

本次导出在独立的 macOS / Python 3.13 环境中通过了 **1,784 个自动化测试**。这是回归测试结果，不是真机任务成功率。[验证说明](docs/evaluation.md) · [架构设计](docs/architecture.md)

## 一起做，边做边学

我们正在探索一个问题：**Agent 能不能通过经验，越来越会用一台具体的手机？**

我们希望把实验、失败、推翻过的假设和设计思路都分享出来。你不需要先理解全部架构，也可以带来有价值的贡献。

| 你对什么感兴趣 | 可以从哪里开始 |
|---|---|
| 真机体验 | 记录一次成功或失败，说明系统、语言、模型和复现步骤。 |
| Agent、记忆与数字孪生 | 提出布局身份、过期知识或屏幕转移建模方案。 |
| App 工作流 | 整理一个有前提条件和验证方法的技能。 |
| 模型评测 | 验证图片、工具调用和坐标行为。 |
| Python 或前端 | 修复问题、补测试、改善本机交互界面。 |
| 写作与教学 | 分享教程、翻译、脱敏后的演示。 |

从[路线图](ROADMAP.md)、[贡献指南](CONTRIBUTING.md)和 [Issues](https://github.com/wangzan101/iphone-agent/issues) 开始。[公开开发笔记](docs/build-in-public.md)记录当前方向，也提供分享实验的方法。

本地开发：

```sh
python -m pip install -e '.[dev,repl]'
python -m pip install ruff
python -m pytest -q
python -m ruff check iphone_agent tests scripts
```

Fork 仓库，在分支上做一项明确改动，附上验证结果再提交 PR。公开仓库包含完整产品实现和独立测试，不需要拿到维护者的个人设备资料。

**如果你喜欢这个方向，可以点一个 Star，在自己的手机上试一次，或分享带有项目链接的运行案例。** 一份说得清楚的失败报告，和一次漂亮的成功演示同样有价值。

## 常见问题

**这是一个 iPhone App 吗？**  
不是。程序运行在 Mac 上，通过 iPhone 镜像操作手机。

**要越狱或者在手机上装 Agent 吗？**  
不需要。需要先完成镜像配置，并授予诊断提示的 Mac 权限。

**运行时还能用电脑吗？**  
后台输入是设计的一部分，常规路径尽量不占用你的光标和焦点。连接或恢复问题仍可能需要处理。

**所有内容都只在本地吗？**  
程序执行与运行记录在本地，屏幕、OCR 和任务上下文可能发给你选择的模型端点。使用本地模型可以改变数据去向，但仍需验证其兼容性。

**任何 App 都一定能成功吗？**  
不能保证。它通过像素与界面行为工作，目前是 Alpha，建议先尝试小任务并报告失败案例。

**数字孪生已经完成了吗？**  
还没有。第一期设备布局层已经实现，更完整的屏幕与转移建模仍在路线图中。

## 开源协议

采用 [Apache License 2.0](LICENSE)。详见 [NOTICE](NOTICE)。除另有明确约定外，提交贡献采用项目相同的许可证。

iPhone 是 Apple Inc. 的商标。本项目与 Apple 无关联，亦未获其背书。凭据、截图和问题报告的处理说明见 [SECURITY.md](SECURITY.md)。
