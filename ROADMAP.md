# Roadmap

This roadmap describes direction, not delivery dates.

## Implemented, with validation still expanding

- Pixel-based observation, OCR, model calls and action execution.
- CLI, local web UI, run inspection and replay.
- File memory, skill drafts and routing.
- Device-layout schema, home-screen scanning and layout-assisted app opening.

## Current focus: digital twin

- Validate scanning and opening on more real devices and home-screen arrangements.
- Improve app identity verification and recover from wrong destinations.
- Explain uncertain or stale layout information in diagnostics.
- Develop reliable screen and transition representations with evidence and revision handling.

## Contribution opportunities

- **Testing:** produce a synthetic case where layout inference groups icons incorrectly.
- **Device validation:** document mirror size, UI language and primitive behavior without private screenshots.
- **Models:** document image/tool compatibility and coordinate calibration for one model.
- **App skills:** propose a portable skill with explicit preconditions and verification.
- **Developer experience:** reproduce a clean-install or diagnostic failure.

Detailed acceptance criteria should be agreed in an issue. These are opportunities, not claims that issues have already been created.

## Later

Broader device-state modeling, reusable transition knowledge and stronger action supervision. Changes should be driven by reproducible failures and measured improvement.
