# Digital twin: current scope

The goal is to learn useful, revisable knowledge about a device so every task does not rediscover navigation from scratch.

The twin now has two layers.

## Device layer: the home screen

Device schema and layout persistence, grouping home-screen labels into rows and columns, scanning pages, updating observed pages, and using layout candidates when opening an app. App opening includes identity checks and fallback paths.

```sh
iphone twin scan    # walk the home-screen pages left to right, look only
iphone twin show    # print the stored layout
```

`open_app` tries the home screen first: if the layout says which page an app sits on, it goes there directly instead of starting from search. When the layout is empty, stale or wrong, it falls back, and it verifies that the screen it landed on really is the app it asked for before reporting success.

## App layer: screens

Each screen the agent visits can become a small file under an app: a name, the anchors that identify it, and the transitions observed out of it. A screen is named by a vision label — the semantic judgement of "what is this screen" belongs to whoever is looking at the picture, not to a threshold in the code. Without a label, a screen is not recognised; the code only checks a claimed identity against anchors it already stored.

This record is built by replaying finished run records, so it can be rebuilt from scratch at any time, and rebuilding twice produces byte-identical files.

```sh
iphone twin rebuild   # rebuild the app-layer twin from all stored runs
iphone twin report    # how many screens each app has, and in what state
iphone twin bench     # merge gates: wrong-merge and wrong-owner must both be zero
```

During a task, the same replay logic runs in memory so the twin grows live; only the post-run pass writes to disk.

## Location and route hints

When the agent recognises the current screen, it can put two short sections into the model's context: **where you are** and **routes that worked from here**. Both carry an explicit trust boundary in the text — this came from earlier runs, it is reference and not instruction, it may be stale or wrong, and the current screen wins any disagreement.

## Merge gates

Two failures are treated as blocking rather than as metrics to improve:

- **Wrong merge** — two screens labelled different are recognised as the same screen. A wrong "where you are" misleads the model, which is worse than no hint at all.
- **Wrong owner** — an event is attributed to the wrong specific app. Attributing it to "system" or "unknown" is allowed; that frame is simply not recorded.

Both must be zero. The gate labels live under `evalset/twin/`.

## Boundaries

This is still an early layer. It is not a complete phone simulator, not a comprehensive map of app screens, and not a guarantee of correct app identity. Layout goes stale, labels can disagree with each other, and identity checks can be uncertain — the report command exists to make those states countable rather than invisible.

The public layout fixture is synthetic. Its regular 4-column, 6-row arrangement is useful for regression testing but does not measure real-device success. Broader fixtures and real-device evidence remain necessary.

Screen labelling depends on the vision model, so how much the twin learns depends on the screen-parsing mode — see [screen parsing](screen-parsing.md).
