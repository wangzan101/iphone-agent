# Building iPhone Agent in public / 公开开发

**Can an agent get better at using a particular phone through experience?**

That is the question behind this project. The goal is not just to execute a task once, but to make useful experience inspectable, revisable and reusable.

我们想公开的不只是最后的代码，还有中间的思考：为什么这样设计、哪里失败了、什么证据让我们改变判断。边做边学，也让后来加入的人有办法接上这段探索。

## Where we are starting

The public implementation has a task harness, a macOS mirroring driver, perception, model configuration, local UI, memory, skills and an initial device-layout layer.

The first public candidate passed 1,310 automated regression tests. That establishes a software baseline, not broad real-device reliability. The next useful evidence comes from supervised runs on different layouts, languages, devices and models.

See [architecture](architecture.md), [evaluation](evaluation.md) and [the roadmap](../ROADMAP.md).

## Current focus: the digital twin

We are starting with a concrete layer: home-screen layout and app opening. A larger device model should earn its complexity through observable improvements.

| Question | Useful evidence | Possible contribution |
|---|---|---|
| Did we identify the right app? | A wrong-app case with a minimal reproduction. | Identity checks and fallback tests. |
| Is a remembered layout still valid? | A changed arrangement and the resulting behavior. | Stale-state handling and diagnostics. |
| Can a route be reused safely? | A repeated task with clear start/end conditions. | A reviewed skill or transition proposal. |
| Does a change actually help? | Comparable before/after cases. | Evaluation fixtures and measured results. |

这些问题就是当前的参与入口，不是已经全部解决的能力。优先把主屏布局、App 身份和过期知识处理做扎实，再用真实案例推动更完整的屏幕与路径建模。

## An experiment note

A small note can use this structure:

```text
Title: One question, not a broad claim

Problem — what did I want the agent to do?
Hypothesis — why did I think this change would help?
Setup — OS, UI language, model, starting state.
Attempt — the smallest meaningful change or experiment.
Evidence — what happened, how many tries, what I inspected.
Failure — what did not work or remains uncertain?
Decision — keep, revise, or discard the idea; why?
Next test — what observation would change my mind?
Links — related issue, PR, or redacted demo.
```

Use synthetic examples where possible. Follow [sharing.md](sharing.md) before publishing any device material.

中文也可以直接按“问题 → 假设 → 实验 → 证据 → 失败 → 决定 → 下一个验证”来写。不必每篇都宣布突破，一次被证据推翻的假设也值得记录。

## Ways to join

- **Try a task:** report the smallest success or failure you can explain.
- **Own a focused problem:** propose a fix or experiment in an issue before a large implementation.
- **Teach an app:** write a skill with visible prerequisites and a checkable result.
- **Improve onboarding:** document what blocked a clean installation.
- **Explain the system:** publish a walkthrough or translate a design note.
- **Share the project:** link to the repository with a concrete reason it interested you.

Start in [Issues](https://github.com/wangzan101/iphone-agent/issues) or [CONTRIBUTING.md](../CONTRIBUTING.md). These are invitations, not claims that a staffed community, scheduled event or a set of labeled starter issues already exists.

## What we will keep visible

The intended habit is to connect meaningful changes with their reasoning, evidence and remaining limitations. Public engineering notes should link to relevant issues and PRs so readers can follow the decision, not just the announcement.

We are not promising a daily release or a fixed delivery date. A public update should contain something useful to inspect, try or discuss.

完整公开实现与测试放在公共仓库；个人设备数据和原始运行记录不应该为了“公开开发”而公开。我们分享的是能帮助别人理解和复现的证据，而不是全部私人工作区。

## First public milestone

The initial milestone is a contributor-friendly baseline: clear setup, a truthful capability description, a runnable example format and a focused digital-twin roadmap. Broader compatibility and real-device demonstrations are work to contribute, not boxes silently marked complete.
