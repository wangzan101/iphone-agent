# Changelog

Public exports from the maintainers' private Lab. Dates are export dates, not release dates.

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
