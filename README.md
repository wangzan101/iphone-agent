<p align="center">
  <img src="assets/logo/mark.svg" width="88" alt="iPhone Agent">
</p>

<h1 align="center">iPhone Agent</h1>
<p align="center"><strong>Give your iPhone an agent of its own.</strong></p>
<p align="center">Operate a real iPhone today. Help build an agent that learns its way around it.</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/platform-macOS-black" alt="macOS">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/status-Alpha-orange" alt="Alpha">
</p>

<p align="center">
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="https://iphone-agent.com">Website</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#a-digital-twin-of-your-phone">Digital twin</a> ·
  <a href="#build-with-us">Build with us</a>
</p>

Plenty of agents can write your code. **This one can use your phone.**

iPhone Agent sees your iPhone through macOS iPhone Mirroring, chooses actions, taps and types, checks what happened, and keeps a record you can inspect. Give it a task through a local web app or your terminal, using your own model endpoint.

**No jailbreak. No app installed on your iPhone. Background input on your Mac.**

Our goal goes further: an agent that remembers useful routes, notices when its knowledge is stale, and turns experience into reusable skills. We are building that in public, starting with home-screen layout learning.

> **Early alpha, actively evolving.** The task loop, local UI, memory, skills and first device-layout layer are implemented. Device and model compatibility are still being expanded. See [current scope](docs/digital-twin.md) and [known limitations](docs/limitations.md).

## See the idea

Explore the [animated product walkthrough on the website](https://iphone-agent.com). Its phone screens are illustrative mockups, not a recorded agent run.

A simple first task is:

```sh
iphone run "Open Settings, go to General > About, and tell me the iOS version."
```

The sequence below illustrates the intended loop; it is not a captured execution trace:

```text
You:       Tell me the iOS version.
Agent:     Observe → open Settings → navigate → read → check → answer
Your Mac:  Runs the agent and hosts the local interface.
Your phone: The task happens in the actual Settings app.
Afterward: Inspect the recorded actions and observations.
```

We welcome short, redacted real-device demonstrations. [Share a run or skill](docs/sharing.md) to help make the examples more representative.

## Why we're building it

A phone holds apps and workflows that do not always have a convenient API. Pixel-based interaction opens a path to them. But finding and pressing a button is only part of the problem.

Where am I? Did the intended app open? Have I done this before? Is yesterday's route still valid?

Those are the questions behind iPhone Agent. We want successful runs to leave useful knowledge, and failed runs to leave enough evidence to improve the next attempt.

| What you get | How it helps |
|---|---|
| **Your real iPhone** | Use the apps already on the device through iPhone Mirroring. |
| **Background input** | Normal input paths aim to leave your Mac cursor and focus available; connection and recovery may still need attention. |
| **A visible task loop** | Inspect what the agent observed, attempted and checked. |
| **Memory and app skills** | Keep useful context across tasks and review reusable knowledge as files. |
| **Your own model endpoint** | Configure providers or a compatible endpoint without changing the task interface. |
| **A growing device model** | Start with home-screen layout and contribute to the next layers of the digital twin. |

## What to try

Start with one small task and watch it run. These are prompts to explore, not a promise that every device or app version will behave identically.

| Try asking… | What you're exploring |
|---|---|
| “Open Settings and tell me the iOS version.” | App opening, navigation and reading. |
| “What apps can you see on the current home screen?” | Screen understanding. |
| “Read the text on this screen and summarize it.” | OCR and model interpretation. |

Adapt prompts to your phone's language. After a successful run, inspect the record and try a related task. For write operations, use a test context and stay present.

## Quick start

### Before you begin

- A Mac and iPhone with **iPhone Mirroring working already**. See [Apple's requirements](https://support.apple.com/en-us/120421), including OS, hardware, account and regional availability.
- Python **3.12 or newer**.
- A model endpoint and your own API credentials. API usage may be billed separately from a chat subscription.

The driver runs on macOS. Nothing needs to be installed on the iPhone by this project.

### Option A: ask your coding agent to help install

Paste this into Codex, Claude Code or another coding assistant that can work with local files and a terminal:

```text
Help me set up https://github.com/wangzan101/iphone-agent on this Mac.
Read README.md and docs/getting-started.md first. Check the existing
directory before cloning, create a project virtual environment, install
the dependencies, and run iphone doctor. Guide me through any manual
iPhone Mirroring or macOS permission steps. Let me enter API credentials
locally, then help me start iphone serve. Wait for me to choose a first
phone task.
```

This is an installation prompt, not a bundled MCP integration. Your coding assistant helps with setup; iPhone Agent runs phone tasks.

### Option B: install yourself

```sh
git clone https://github.com/wangzan101/iphone-agent.git
cd iphone-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[repl]'
iphone doctor
```

Follow the reported setup checks, then start the local app:

```sh
iphone serve
```

The web interface supports **English and 简体中文**. Switch languages in the sidebar or first-run setup dialog; your phone’s language and original task records remain unchanged. See [language support](docs/languages.md).

Open **http://127.0.0.1:8765**, configure your provider, model and API key in Settings, and return to the task interface. For file-based configuration, see [the example](examples/config.example.toml) and [setup guide](docs/getting-started.md).

Keep the mirror connected and try the Settings task above. If setup fails, [start with the troubleshooting table](docs/getting-started.md#troubleshooting).

### Pick your interface

| Interface | Command | Best for |
|---|---|---|
| Local web app | `iphone serve` | Watching the phone, entering tasks and managing settings. |
| Interactive terminal | `iphone` | A continuing conversation from your terminal. |
| One task | `iphone run "your task"` | A focused request from the command line. |

In the terminal, Ctrl-C requests a stop; an in-flight action may finish first. For CLI inspection of a completed run:

```sh
iphone replay runs/<run-id>
```

Replace `<run-id>` with an actual directory from your own run. Replay inspects the record; it does not re-execute those actions.

## A digital twin of your phone

**What if your agent didn't have to rediscover your phone on every task?**

Our vision is a revisable model of the device: where apps are, what screens mean, which paths have worked, and when that knowledge needs checking again.

| Layer | Current state |
|---|---|
| Home-screen layout | Implemented: schema, grid inference, persistence and scanning. |
| Layout-assisted app opening | Implemented: the home screen is tried first, identity is verified, and it falls back when the layout is wrong. |
| Screen knowledge | Implemented: screens named by a vision label, rebuildable from stored runs, with wrong-merge and wrong-owner merge gates. |
| Location and route hints | Implemented: "where you are" and "routes that worked from here", marked as reference rather than instruction. |
| Cross-task memory and skills | Implemented separately; broader integration is evolving. |
| Broad compatibility and robust stale-state handling | Active validation and development work. |

Try the device-layout layer, then the screen layer:

```sh
iphone twin scan
iphone twin show
iphone twin rebuild
iphone twin report
```

**The scan walks the home-screen pages and records them without tapping. It is not a full-phone crawl.** Later observations can refresh layout information as tasks run. The screen layer is built by replaying your own finished runs, so it starts empty.

Read [the design and boundaries](docs/digital-twin.md). Good contributions include a reproducible layout failure, a stale-page case or a better way to verify app identity.

## Reading the screen: pick your tradeoff

Every observation runs OCR. A whole-screen parse by the vision model is what finds icon-only controls and rows OCR cannot read — and it is also the slowest step in a task.

```sh
IPHONE_SCREEN_PARSE=on_demand iphone run "your task"
```

`always` (the default), `on_demand` and `off` decide when that parse runs. In `on_demand` the parse happens when the model asks for it or as a one-shot fallback before a step gives up, and a short label call keeps the twin learning. A smoke comparison on one device moved the median per-step time from 34.0s to 18.8s — one run per cell, `always` first, so a direction rather than a measurement. **The default stays `always` until there is better evidence.** See [screen parsing](docs/screen-parsing.md).

## Teach it an app. Share what works.

A useful discovery should be able to outlive one run.

App knowledge describes an app; a skill describes how to complete a task. They are readable files, with separate shared and personal layers:

```text
knowledge/apps/<app-id>/APP.md       Shared app knowledge
skills/<skill-name>/SKILL.md         Shared task knowledge
.iphone/knowledge/ and .iphone/skills/  Personal knowledge and drafts
```

Explore what your own workspace has learned:

```sh
iphone skill list
iphone skill show <app-id>
iphone skill export <app-id> ./skill-export
```

Replace `<app-id>` with an ID shown by the list. Export requires an app in your personal layer; a fresh workspace may have none. It removes selected provenance fields, but the resulting text and files still need review before sharing.

Start from [a small example skill](examples/shared-skill/skills/read-ios-version/SKILL.md), then follow [the sharing guide](docs/sharing.md). The example is a draft, not a validated app pack. Community distribution and review conventions are still growing.

## Models: bring your own

The code has presets for Alibaba/Qwen, OpenAI, Anthropic, DeepSeek, Moonshot/Kimi, Zhipu/GLM and OpenRouter, plus configuration for custom OpenAI-compatible endpoints.

A preset supplies configuration; it does not certify every model or API. Image input, tool calls and coordinate behavior depend on the endpoint. The current registry contains a coordinate-calibrated profile for `alibaba:qwen3.7-plus`; other models use more conservative coordinate controls.

Configure from the web Settings page or `.iphone/config.toml`. See [setup](docs/getting-started.md#configure-a-model) for credentials and [limitations](docs/limitations.md) for compatibility boundaries.

## Under the hood

```text
                   CLI / local web app
                           |
                      Task harness
         observe → model → validate → execute → inspect
                           |
        driver · perceive · model · memory · skills · twin
                           |
              macOS iPhone Mirroring → real iPhone
```

The harness is an explicit task loop. The driver handles capture and input; perception turns pixels into usable observations; memory and skills carry context across runs; the twin layer begins to model the device.

This export passes **1,784 automated tests** in an independent macOS/Python 3.13 environment. These are regression tests, not a real-device task-success benchmark. See [evaluation](docs/evaluation.md) and [architecture](docs/architecture.md).

## Build with us

We are building this to explore a question: **can an agent get better at using a particular phone through experience?**

We want to share the experiments, mistaken assumptions and design decisions along the way. You do not need to solve the whole system to make a useful contribution.

| You enjoy… | A useful contribution |
|---|---|
| Trying things on real devices | A reproducible success or failure with OS, language and model details. |
| Agent systems and memory | Layout identity, stale knowledge or transition-model proposals. |
| App workflows | A reviewed skill with preconditions and a way to check the result. |
| Model evaluation | Image/tool compatibility reports and coordinate calibration. |
| Python or frontend development | A focused fix, test or local-UI improvement. |
| Writing and teaching | A walkthrough, translation or redacted demo. |

Start with [the roadmap](ROADMAP.md), [contribution guide](CONTRIBUTING.md) and [Issues](https://github.com/wangzan101/iphone-agent/issues). [Build notes](docs/build-in-public.md) explain what we are working on and how to share an experiment.

To develop locally:

```sh
python -m pip install -e '.[dev,repl]'
python -m pip install ruff
python -m pytest -q
python -m ruff check iphone_agent tests scripts
```

Fork the repository, make a focused change on a branch, and open a PR with its validation. The public checkout contains the product implementation and independent tests; private device records are not required.

**If the direction resonates, star the project, try it on your phone, or share a run with a link back.** A well-described failure can be just as useful as a polished demo.

## FAQ

**Is this an iPhone app?**  
No. It runs on your Mac and controls the mirrored iPhone.

**Do I need to jailbreak or install a phone-side agent?**  
No. You do need a working iPhone Mirroring setup and the Mac permissions reported by diagnostics.

**Can I keep working on my Mac?**  
Background input is part of the design. Normal operation aims to avoid taking your cursor/focus, but connection problems or recovery can still need attention.

**Does everything stay local?**  
Execution and records are local. Screenshots, OCR and task context may be sent to the model endpoint you select. A locally hosted compatible model changes that data path; it does not by itself prove model compatibility.

**Will it operate every app reliably?**  
No. This is an alpha working through pixels and UI behavior. Start with small supervised tasks and help report where it fails.

**Is the digital twin finished?**  
No. The first device-layout layer is implemented; broader screen and transition modeling is part of the roadmap.

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE). Contributions are submitted under the project license unless explicitly agreed otherwise.

iPhone is a trademark of Apple Inc. This project is not affiliated with or endorsed by Apple. For handling credentials, screenshots and reports, see [SECURITY.md](SECURITY.md).
