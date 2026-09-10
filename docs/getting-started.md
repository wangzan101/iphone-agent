# Getting started / 启动指南

Go from a clean checkout to one supervised task. Start small: reading an iOS version is easier to inspect than a long multi-app workflow.

这份指南的目标是跑通一次可以观察、可以核对的小任务，而不是一开始就让手机无人值守地工作。

## 1. Connect the phone first

Use a Mac and iPhone that meet [Apple's iPhone Mirroring requirements](https://support.apple.com/en-us/120421): macOS Sequoia 15 or later on supported hardware, iOS 18 or later, and the required account, connectivity and regional conditions.

Open iPhone Mirroring manually and confirm that you can see and interact with your phone. Follow Apple's setup instructions before diagnosing the agent. Keep the phone nearby and locked while mirroring; unlocking it can interrupt the connection.

先在 Mac 上手动打开「iPhone 镜像」，确认能看见并操作手机。镜像本身还没有连接成功时，先解决连接问题。条件以[苹果中文说明](https://support.apple.com/zh-cn/120421)为准。

## 2. Install in a project environment

Python 3.12+ is required. Check the interpreter before creating the environment:

```sh
python3 --version
git clone https://github.com/wangzan101/iphone-agent.git
cd iphone-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[repl]'
```

If the directory already exists, inspect it before cloning or updating; do not overwrite your work. If `python3` is older than 3.12, use an installed newer interpreter to create the environment. The macOS driver depends on Apple frameworks; Linux and Windows are not supported execution hosts.

Run commands from the project directory. By default it is the workspace root: local configuration, personal knowledge and run records belong there. `IPHONE_WORKSPACE` can override that root. Launching from another directory can make existing settings appear missing.

请使用项目虚拟环境。后续每次新开终端，先进入项目目录并激活 `.venv`，避免装到了一个 Python、却用另一个 Python 运行。

## 3. Check the Mac environment

```sh
iphone doctor
```

Follow the specific checks and fixes it reports. macOS permissions may need to be granted to the terminal or application hosting Python, then that process restarted. Do not disable system protections to bypass a failed check.

A successful diagnostic is not proof that every input primitive works on your machine. The optional `scripts/driver_gate.py` is an interactive device check that can change phone state; inspect it and use a suitable test context before running it.

诊断提示缺什么就处理什么。需要手动授权、配对时由你完成。诊断通过不等于已经验收所有操作，更不等于任何任务都能成功。

## 4. Start the local interface

```sh
iphone serve
```

Open [the local app](http://127.0.0.1:8765). Keep this terminal running; use another activated terminal for CLI commands.

The interface follows your saved language preference or browser language, with English as the fallback. Choose English / 简体中文 in the sidebar or setup dialog. See [language support](languages.md); phone-language compatibility is evaluated separately.

Use the local interface only on a trusted machine. Do not expose this phone-control service to the internet through a tunnel or public reverse proxy without a separately designed authentication and security layer.

## Configure a model

### Recommended: the Settings page

Choose a provider and model, enter your API key locally, and return to the task interface. The UI writes workspace configuration. Keep credentials out of issues, screenshots and messages to a coding assistant.

You need API access to the endpoint you select. A consumer chat subscription is not necessarily API access. Screen images, OCR and task context may be sent to that endpoint; review what is visible on your phone before running a task.

The repository's presets are configuration conveniences, not a model compatibility matrix. Check that the specific endpoint accepts the needed image and tool inputs.

推荐在网页设置中填写。不要把密钥粘贴到公开 Issue、录屏或聊天记录里。模型能聊天，不代表它支持本项目所需的图片、工具调用和坐标行为。

### Alternative: local TOML

Use [config.example.toml](../examples/config.example.toml) as a starting point. Create `.iphone/config.toml` in your workspace with a local editor; do not overwrite an existing configuration.

For the bundled Alibaba profile, the minimal model selection is:

```toml
model = "alibaba:qwen3.7-plus"
```

Provide the key through the local Settings page or the provider's environment variable; the Alibaba preset recognizes `DASHSCOPE_API_KEY`. The example intentionally contains no credential.

Restrict the local configuration file's permissions:

```sh
chmod 600 .iphone/config.toml
```

The loader rejects group/other-readable configuration files, even when you currently store only the model name. Environment settings can override file settings; check for an old exported model or provider setting if a change seems ineffective.

Custom providers are declared in the configuration's provider table. Follow the fields validated in [model/config.py](../iphone_agent/model/config.py) rather than assuming every option from another SDK is supported.

配置文件属于个人工作区，不应提交。文件权限必须限制为当前用户可读写；环境变量还可能覆盖文件里的配置。不要把整个 `.iphone/` 打包分享。

## 5. Run one small task

From the web task interface, ask:

> 打开设置，进通用，进关于本机，告诉我 iOS 版本号

Or use an activated terminal:

```sh
iphone run "打开设置，进通用，进关于本机，告诉我 iOS 版本号"
```

Adjust the prompt to the device language. Watch the operation and compare the answer with the visible iOS version. The About screen can also contain device identifiers: do not share an unredacted recording.

If the agent gets lost, stop and inspect the failure. In the terminal, Ctrl-C requests a stop; an in-flight action may complete first. Do not start with payments, deletion, account changes or sending real messages.

第一次任务最好只读取信息。亲眼核对答案；失败时记录停在哪一步，不要为了演示效果不断追加高风险操作。「关于本机」也可能显示设备标识，录屏前后都需要检查。

## 6. Inspect and explore

Run records are created under `runs/` in the workspace. Use an actual run directory:

```sh
iphone replay runs/<run-id>
```

Replay displays recorded evidence; it does not repeat the phone actions. Records can contain private screens and text.

To explore the first digital-twin layer:

```sh
iphone twin scan
iphone twin show
```

Scanning changes the phone's visible state by returning to the first home-screen page. It records that page, not every page or every app. See [digital-twin.md](digital-twin.md).

Once you have something worth sharing, follow [the sharing checklist](sharing.md), not a wholesale upload of the workspace.

## Troubleshooting

| Symptom | Check next |
|---|---|
| Dependency installation fails outside macOS | Run on a supported Mac; the driver uses Apple frameworks. |
| Python version error | Create a new virtual environment with Python 3.12+; activating an old environment does not upgrade it. |
| `iphone` command not found | Activate this checkout's virtual environment and confirm the editable install completed. |
| Mirror missing, paused or disconnected | Open iPhone Mirroring manually and restore the connection before retrying. |
| Capture or input permissions fail | Follow `iphone doctor`; verify which host application needs permission and restart it if needed. |
| Configuration permissions rejected | Run `chmod 600 .iphone/config.toml` on the intended workspace's config. |
| API authentication or model error | Recheck endpoint, model ID and API access locally; do not post the key. |
| Settings disappear between commands | Check the current directory and `IPHONE_WORKSPACE`. |
| Port 8765 is in use | Inspect the existing process, or run `iphone serve --port 8766` and open that port. |
| Wrong app opens or an action misses | Record OS, UI language, model and a minimal reproduction; see [limitations](limitations.md). |

For help, open a [bug report](https://github.com/wangzan101/iphone-agent/issues) with the command, expected result, actual result and redacted diagnostics. State whether you reproduced it once or repeatedly.

排障报告里最有用的是：环境、最短复现步骤、预期、实际发生了什么，以及是否重复出现。不要贴密钥或整份运行目录。
