# Changelog

Public exports from the maintainers' private Lab. Dates are export dates, not release dates.

## 2026-09-24

### Added

- **Canonical app ids and shared app knowledge.** An app id used to be the pinyin of whatever the phone displayed, so the same app was `she-zhi` on a Chinese device and `settings` on an English one and the shared knowledge in this repo only worked for one of them. Apple's system apps now have hand-written English slugs, and every other name an app answers to lives in that app's `aliases:` frontmatter. `knowledge/apps/` ships thirteen packs (settings, weather, notes, reminders, calendar, clock, photos, calculator, health, maps, files, app-store, shortcuts). `iphone skill migrate-ids` moves an older workspace across, dry run by default.
- **A hard task set.** `evalset/tasks/hard/` holds twelve read-only tasks aimed at what the daily four miss: third-party apps, Chinese input, deep entry points, media libraries and apps that open behind a popup. `iphone eval tasks --set <subdir>` (and `iphone eval verify ... --set`) run a named set; the daily numbers are unaffected. See [docs/evaluation.md](docs/evaluation.md).
- **An outcome for every action.** A step used to be either "it worked" or "it didn't". A third value, **unknown**, now exists and is written for any screen-changing action nobody actually verified. `iphone_agent/harness/verdict.py` is the single entry point for reading and writing it, including for older records. Tap, scroll, key, erase and `open_app` all go through it; exits that escape it are marked rather than assumed fine.
- **The result carries the system's conclusion.** Each tool result is preceded by one plain sentence — "this step was not verified…", "the input box now reads…" — and the prompt tells the model to act on that sentence first. Prompt version 25.
- **`level` on a screen label.** `home` / `top` / `inner`: where a screen sits inside its app, stored as a vote count so one misread cannot flip a screen's identity. Left empty when a popup covers the page. Recorded this release; used by navigation later.
- **`open_app` reports where it landed.** Opening an app resumes wherever it was left, so a successful open is not the same as being on the app's first page. The result now says which screen it reached and whether that is the home page.
- **Screens you have been to.** When the task does not name a destination literally, the model is handed the app's map — screens visited, how often, and how to walk to each from here — instead of the code guessing which screen the task meant. The existing route hint still wins when it fires.
- **Offline outcome statistics.** `python -m iphone_agent.eval.taps <runs dir>` breaks results down per action into met / not met / unverified, and counts a directory as a run only when at least one action executed.

### Changed

- **Procedure tools are off by default** (`IPHONE_SKILL_TOOLS=on` to enable). On real devices, handing learned procedures to the model reliably wasted a step and then marked the procedure stale. Extraction from successful runs keeps going — that is the raw material for rebuilding fingerprints. The prompt no longer describes procedure tools when none are in the request.
- **Procedure risk comes from the procedure's own steps**, not from the app it belongs to, which unblocks auto-verification for procedures split out of draft apps. The danger-word table moved to `iphone_agent/riskwords.py` so the safety gate, exploration and the skill layer share one copy.
- **Fingerprints only accept evidence that survives.** Status-bar rows change every minute and vision-only labels are worded differently every call (and absent entirely under on-demand parsing); both are now filtered out when a fingerprint is built and when one is checked, so procedures learned under the old rule heal instead of failing on their first step.
- **App knowledge is visible when it has content**, whether or not its risk has been confirmed. `iphone skill list` says which of those two is missing instead of reporting a single "draft".
- **A run refuses to start when the mirror is completely covered**, ending with `mirror_hidden`, and `iphone doctor` applies the same rule through the same code. Scrolling and home-screen page flips need a sliver of the window; taps and typing do not, and are not gated on it.
- **An unverified step no longer counts as progress** and no longer resets the no-progress counter, so the fuse still trips; after three in a row the harness looks at the whole screen once.
- **Tool calls are checked against the `required` list in their own schema**: the first omission is refused with a readable message, a second identical omission is let through and marked unverified.
- **Grading stopped failing answers that were read off the screen.** A run that only opened an app and answered is no longer an automatic failure when the checked answer value was actually read from a screen it visited. Without an answer check, an answer that merely echoes a screen row becomes "needs review" — never a pass, and never enough to soften a real failure.

### Notes

- This export has not had supervised real-device acceptance.
- The system prompt in this repo uses a generic account placeholder, so its `prompt_hash` and the two measured `EXACT_SEGMENT_TOKENS` keys differ from the upstream values by design. Character counts are identical and the token numbers are carried over as estimates — see [docs/export-scrub.md](docs/export-scrub.md).

## 2026-09-16

### Added

- **Screen-level digital twin.** Screens are named by a vision label and stored per app, with the anchors that identify them and the transitions observed out of them. The record is rebuilt by replaying your own finished runs, so it can be discarded and rebuilt at any time. New commands: `iphone twin rebuild`, `iphone twin report`, `iphone twin bench`. See [docs/digital-twin.md](docs/digital-twin.md).
- **Location and route hints.** When the current screen is recognised, the model is given a short "where you are" and "routes that worked from here", both marked as stale-able reference rather than instruction.
- **Merge gates for the twin.** Wrong merge (two different screens recognised as one) and wrong owner (an event attributed to the wrong app) must both be zero. Labels are made by hand from real recordings and are not shipped.
- **Selectable screen parsing.** `IPHONE_SCREEN_PARSE` chooses `always`, `on_demand` or `off`. In `on_demand`, observations run OCR only; a whole-screen parse happens when the model asks for it or as a one-shot fallback before a step gives up, and a short label call keeps the twin learning. **The default stays `always`.** See [docs/screen-parsing.md](docs/screen-parsing.md).
- **Per-step phase timing** in run records: `capture`, `settle`, `ocr`, `parse`, `label`, `zoom`, `judge`, plus a count of why each parse ran.
- **`iphone screen`** prints the current parsing mode, the numbered elements and a marked-up image of the current frame.
- **Tap expectation checks.** A tap must say what it expects to happen; the outcome is checked against that and recorded.
- **Offline tap statistics** via `python -m iphone_agent.eval.taps <runs dir>`.
- **Web UI: new chat.** A button starts a fresh conversation; it refuses while a task is still running. Replay now starts from the first frame.

### Changed

- **Element numbering is OCR-first.** OCR boxes fix the numbering and parsed elements merge into that order, so an element keeps its number across both sources on one frame.
- **`open_app` tries the home screen first.** When the stored layout knows which page an app is on, it goes there directly instead of starting from search, verifies the screen it landed on really is that app, and falls back when the layout is wrong or empty.
- **Coordinate taps are safety-classified** by the element box they land in. A tap landing outside every box cannot be classified: it is recorded as unclassified, counted, and flagged for review instead of being treated as safe.
- Coordinate conversion, including the region passed to zoom, now goes through a single entry point.

### Notes

- A smoke comparison of `always` against `on_demand` on one device moved the median per-step time from 34.0s to 18.8s: one run per cell, `always` first in the order. That is a direction, not a benchmark, and it is not a general speed claim. The default is unchanged pending better evidence.
- This export has not had supervised real-device acceptance.

## 2026-09-10

- Initial public alpha release.
