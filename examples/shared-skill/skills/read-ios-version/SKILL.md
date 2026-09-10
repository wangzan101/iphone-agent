---
name: read-ios-version
description: 在设置中读取 iOS 版本号，并核对对应字段
apps: ["settings"]
risk: read
status: draft
---

# Read the iOS version / 读取 iOS 版本

Human-authored draft. No successful device run is claimed.

## Preconditions

- iPhone Mirroring is connected.
- The user requested the iOS version.
- Visible labels are in Simplified Chinese, or the procedure has been adapted to the device language.

## Proposed procedure

1. Open 设置 and verify the app identity from the visible screen.
2. Find 通用 using current observations; do not rely on fixed coordinates.
3. Open 关于本机.
4. Locate the iOS 版本 field and read its associated value.
5. Recheck the field label and report only the version requested.

## Success check

The answer must come from a visible version field on the About screen, not the model's prior knowledge or an earlier run. Do not invent a version when the value cannot be read.

## Stop conditions

Stop and explain if the app identity is uncertain, a required label cannot be found, the mirror disconnects or the value is unreadable. Do not change settings to complete this read-only task.

## Sharing

The About screen may contain serial numbers and other identifiers. Share only a redacted excerpt or a synthetic example. Record actual test results separately before claiming validation.
