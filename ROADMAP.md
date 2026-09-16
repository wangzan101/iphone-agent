# Roadmap

This roadmap describes direction, not delivery dates.

## Implemented, with validation still expanding

- Pixel-based observation, OCR, model calls and action execution.
- CLI, local web UI, run inspection and replay.
- File memory, skill drafts and routing.
- Device-layout schema, home-screen scanning and layout-assisted app opening.
- Screen-level twin: screens named by a vision label, rebuildable from stored runs, with location and route hints and two blocking merge gates.
- Selectable screen-parsing modes (`always`, `on_demand`, `off`) with per-step phase timing.
- Coordinate-tap safety classification and tap expectation checks.

## Current focus: digital twin

- Validate scanning and opening on more real devices and home-screen arrangements.
- Improve app identity verification and recover from wrong destinations.
- Explain uncertain or stale layout information in diagnostics.
- Extend screen and transition knowledge, and handle stale or conflicting labels.
- Gather enough evidence to decide whether `on_demand` screen parsing should become the default. Today it is not: the only comparison we have is a one-run-per-cell smoke test on a single device, and we have not checked whether the twin learns as well from short label calls as from full parses.

## Contribution opportunities

- **Testing:** produce a synthetic case where layout inference groups icons incorrectly.
- **Device validation:** document mirror size, UI language and primitive behavior without private screenshots.
- **Models:** document image/tool compatibility and coordinate calibration for one model.
- **App skills:** propose a portable skill with explicit preconditions and verification.
- **Developer experience:** reproduce a clean-install or diagnostic failure.

Detailed acceptance criteria should be agreed in an issue. These are opportunities, not claims that issues have already been created.

## Later

Broader device-state modeling, reusable transition knowledge and stronger action supervision. Changes should be driven by reproducible failures and measured improvement.
