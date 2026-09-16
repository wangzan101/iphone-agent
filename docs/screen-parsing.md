# Screen parsing: always, on demand, or off

The agent reads a screen from two sources.

- **OCR** runs locally on every observation. It is fast and cheap, and it returns text boxes.
- **Whole-screen parsing** sends the frame to the vision model and asks it to describe every element. It is the only source for things OCR cannot read: icon-only controls, list rows whose label is rendered as an image, values that OCR drops. It is also the slowest step in a task, and on some screens it dominates the time per step.

Element numbering is OCR-first: OCR boxes fix the numbers, and parsed elements are merged into that ordering rather than renumbering it. The same element keeps the same number across the two sources on one frame.

## The three modes

`IPHONE_SCREEN_PARSE` selects when whole-screen parsing runs.

| Mode | What happens on each observation |
|---|---|
| `always` | OCR, then a whole-screen parse. **This is the code default.** |
| `on_demand` | OCR only. A whole-screen parse runs when the model asks for one, or as a one-shot fallback. A short label call names the screen so the twin keeps growing. |
| `off` | OCR only. No parsing, no labelling; the twin does not learn new screens. |

```sh
IPHONE_SCREEN_PARSE=on_demand iphone run "your task"
```

Legacy values still work: `on`, `1` and `true` mean `always`; `0`, `off` and `false` mean `off`. Anything else is an error rather than a silent fallback, so a typo is visible immediately.

In `on_demand`, three things can still trigger a parse:

1. **The model asks.** The `observe` tool requests a full parse of the current frame when the model decides OCR is not enough.
2. **A mechanical fallback.** If a step is about to end in failure or a stop, the agent parses the frame once per task and lets the model judge again with the fuller evidence. Its purpose is to avoid giving up on a screen the agent never actually looked at; it is deliberately a one-shot.
3. **A short label call.** A small vision call names the screen and picks among candidates the twin already knows. It is much cheaper than a full parse and exists so that running in `on_demand` does not stop the digital twin from learning.

`iphone screen` prints the current mode and says whether the frame it shows was fully parsed or OCR-only.

## Per-step timing

Run records carry a `phases` block per step — `capture`, `settle`, `ocr`, `parse`, `label`, `zoom`, `judge` — plus `parse_by`, which counts why each parse happened (`model`, `always`, `fallback`). Phases can nest (settle contains its own captures), so read them side by side; adding them into one total does not mean anything.

## What we actually measured

We ran a smoke comparison of `always` against `on_demand` on one device: one run per cell, `always` first in the order, a small task set. Median per-step time went from 34.0s to 18.8s.

That is one ordered comparison on one phone, not a benchmark and not a general speed claim. Ordering effects, device state and task mix are all uncontrolled at this sample size, and we have not yet checked that the twin learns as well from label calls as it does from full parses.

**So the default stays `always`.** `on_demand` is available for anyone who wants the tradeoff now, and changing the default is a separate decision that needs more evidence than we have.
