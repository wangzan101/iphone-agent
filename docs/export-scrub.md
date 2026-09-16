# Export scrub notes

This repo is exported from a private Lab checkout. The Lab checkout keeps its
own wording (real app/bank names, real run paths); every export to this
public repo must repeat the substitutions below. This file is the checklist —
update it whenever a new export introduces another Lab-only detail.

## 1. System-prompt example (`iphone_agent/harness/prompt.py`)

Lab's `_METHOD` section uses a real bank name in the "account shows X"
example. Public uses a generic placeholder instead: the account label reads
as "工资账户" ("salary account"), not any real bank.

Because the example text differs, `PROMPT_VERSION`'s hash differs too:

- `EXACT_SEGMENT_TOKENS` in `iphone_agent/harness/tokens.py` is keyed by
  `("system", <prompt_hash>)`. The public key is `42680c245e3a` (Lab's
  measured hash is different).
- The token count recorded against that key (`3051`) is **not** a fresh
  measurement — it's carried over from Lab's `e9b0ed49927c` measurement,
  since swapping one 4-character example word for another does not change
  the segment's character count. Treat it as an estimate until someone
  re-runs `scripts/calibrate_tokens.py --probe segments` against the public
  prompt text.

## 2. Bookkeeping-app references

Lab tests and comments reference a specific bookkeeping app by name. Public
replaces the app name with generic wording:

- `tests/test_candidates.py`: the IME candidate-bar fixture used the app's
  pinyin name as a typed-text token; public uses a neutral pinyin token
  (`ceshi`) instead. The fixture's behavior (candidate-bar parsing) is
  unaffected — the token is typed text, not an assertion target.
- `iphone_agent/eval/ab.py`: the ⚠ comment explaining why the hard-gate
  write task was swapped for a read-only one no longer names the app; it
  describes the failure generically (the write task's entry page keeps
  leftover state from a previous run, and under unattended operation the
  danger-word gate refuses delete/backspace, so the model has no way to
  clear it).

## 3. Hard-gate task ids

`HARD` in `iphone_agent/eval/ab.py` uses the neutral id
`ledger-account-list`. The Lab id names the bookkeeping app the task runs
against; rename it on every export. Do not write the Lab id here — this
file is public.

These hard-gate task files (and the write-task variant referenced in the
comment) are **not shipped** in the public repo — `evalset/` here only
carries the curated public task set. `iphone eval ab` still references the
`ledger-account-list` / `weather-city-list` / `settings-camera-grid` ids, but
running the hard-gate group requires the operator to supply their own task
files under `tasks/vision/`.

## 4. Run paths and internal design-doc paths

Lab comments sometimes cite concrete run directories (e.g. an
`evalset/.../off/runs/...` path) as evidence for a bug writeup. Run paths do
not ship — keep the technical explanation (what went wrong, why) and drop the
path. See `iphone_agent/model/coords.py` for the pattern: the
norm1000-vs-pixel conversion failure modes are kept, the Lab run-directory
citations are not.

Internal design-doc paths (`docs/superpowers/...`) are a different case: they
carry no private content, and they mark where a rule came from, so comments
and test docstrings keep them. Be aware they are dead links in this repo —
those documents stay in Lab.

## General rule

When exporting from Lab to public: keep the technical substance of every
comment (what broke, why, what the fix does), and only strip the parts that
name a real person, a real app/bank/institution, a device identifier, a
filesystem path under the maintainer's home directory, or a Lab-only run/log
path. If a substitution changes a measured constant (like the prompt hash
above), say so explicitly rather than silently re-using the old number.
