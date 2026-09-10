# Share a run, a skill, or an idea / 分享指南

A contribution can start with one useful observation. You do not need a polished framework or a flawless demo.

你可以分享一次成功，也可以分享一次说得清楚的失败。关键是让别人知道：你尝试了什么，在哪里有效，又在哪里失效。

## Share a run

Use a short description or redacted clip:

```text
Task:
Environment: macOS / iOS / UI language / model ID
Starting screen:
Expected result:
Observed result:
What I checked manually:
Attempts: succeeded __ / tried __
Known limitations:
Project: https://github.com/wangzan101/iphone-agent
```

Prefer a small, inspectable task. Label sped-up or edited footage. Do not present a staged UI or animation as agent execution. A single successful run is a case study, not a general success-rate claim.

Screenshots and recordings may reveal notifications, account names, addresses, chats, balances and device identifiers. Review every frame and any audio. Retake the demo with test content if redaction would be unreliable.

录屏可以剪辑、加速，但要说明。一次成功就是一次案例，不要写成“全场景可用”。遮住姓名还不够，通知、聊天、订单、设备标识和声音也可能泄露信息。

## Understand the knowledge layers

| File or directory | Role |
|---|---|
| `knowledge/apps/<app-id>/APP.md` | Shared app identity, structure and general guidance. |
| `skills/<skill-name>/SKILL.md` | Shared guidance for completing a task. |
| `.iphone/knowledge/` | Personal app knowledge and local discoveries. |
| `.iphone/skills/` | Personal skills, including drafts. |
| `runs/` | Execution records; not a shareable skill library. |

The app profile and skill have different jobs. “This is Settings” belongs in app knowledge. “Read the iOS version and check the row label” belongs in a skill.

The repository includes a [draft app profile](../examples/shared-skill/knowledge/apps/settings/APP.md) and [draft skill](../examples/shared-skill/skills/read-ios-version/SKILL.md). They live under `examples/`, so they are not installed into the active shared knowledge layer.

## Export something you have learned

Inspect your workspace first:

```sh
iphone skill list
iphone skill show <app-id>
iphone skill export <app-id> ./skill-export
```

Replace `<app-id>` with a real ID from your list. A fresh installation may have no personal app knowledge. Export requires that app in the personal layer; it is not a command to download community skills.

The export creates app knowledge under `knowledge/apps/<app-id>/` and includes personal skills that reference that app. The destination app directory must not already exist. Existing destination skills can be skipped, so use a fresh export destination when you want a complete reviewable bundle.

Export removes selected provenance: screen-map source/sample fields and procedure run references. **It does not guarantee removal of private content from prose, labels or other files.**

导出是整理文件，不是自动脱敏服务。不要把导出成功当成可以直接上传；输出里的每一份内容仍然需要检查。

## Review before publishing

- Remove names, contacts, messages, addresses, amounts, identifiers, account-specific labels and embedded credentials.
- Replace personal examples with synthetic values and generic app states.
- Check all files, not just `SKILL.md`; do not include original run logs or screenshots.
- State language, app version if known, starting state and prerequisites.
- Explain the success check and when the agent should stop rather than guess.
- Separate observed behavior from a proposed procedure.
- Keep risk classification honest: reading, writing and irreversible actions are different.
- Share only material you have the right to contribute.

Do not edit an original private run to manufacture a clean historical record. Prepare a separate public example or record a new run with test data.

## Start with the draft example

Read the [example bundle](../examples/shared-skill/README.md). Its files match the current parsers, but both are marked `draft` and have no real-device validation claim.

To adapt it, work on a copy, translate visible labels to your device language and add observed preconditions and a success check. After review, place your app profile and skill in the corresponding shared paths if you want them in a development checkout. Check for existing files before copying.

A draft skill is not automatically routable. The current model distinguishes authored `manual` skills from learned `draft`, `verified` and `stale` skills, and proposed scenarios. App profiles have their own `draft`/`verified` status. Do not change a status merely to make a demo appear validated; document the review and evidence behind it.

These are project-specific file conventions, not a guarantee that an arbitrary skill from another agent framework can be installed unchanged.

样例先教你文件怎么组织，不承诺复制进去就能自动执行。状态字段关系到系统是否使用它；“能被解析”和“已验证可用”是两回事。

## Submit a contribution

For a first app skill or a larger workflow, open an [issue](https://github.com/wangzan101/iphone-agent/issues) with the draft and evidence so the scope can be reviewed. For an agreed change, fork the public repository, use a focused branch and open a PR.

Include:

1. What problem the contribution solves.
2. What was tested, on which environment, and how often.
3. What remains a hypothesis or fails.
4. What files were reviewed for private data.
5. Any source attribution or license requirements.

Follow [CONTRIBUTING.md](../CONTRIBUTING.md). Shared contributions use the project license unless explicitly agreed otherwise. A community marketplace or one-command distribution service is not currently part of this repository.

## Share an engineering note

Use [the build-in-public format](build-in-public.md#an-experiment-note). An explanation of why a layout heuristic failed can be more reusable than an unannotated recording of success.

欢迎在自己的博客或社交平台写思路、贴项目链接，再把相关链接发到 Issue。代码、实验、文档和复盘都可以成为参与方式。
