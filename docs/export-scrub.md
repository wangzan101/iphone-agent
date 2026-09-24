# Export scrub notes

This repo is exported from a private Lab checkout. The Lab checkout keeps its
own wording (real app/bank names, real run paths); every export to this
public repo must repeat the substitutions below. This file is the checklist —
update it whenever a new export introduces another Lab-only detail.

## 1. System-prompt example (`iphone_agent/harness/prompt.py`)

Lab's `_METHOD` section uses a real bank name in the "account shows X"
example. Public uses a generic placeholder instead: the account label reads
as "工资账户" ("salary account"), not any real bank.

Because the example text differs, `PROMPT_VERSION`'s hash differs too. Recompute
it on **every** export that touches the prompt text:

    .venv/bin/python -c "from iphone_agent.harness.prompt import prompt_hash; print(prompt_hash(True))"

- `tests/test_prompt.py::test_hash_with_coords_equals_pinned_value` pins
  `prompt_hash(True)`. Public value at prompt version 25: `8c2632b81c13`.
- `EXACT_SEGMENT_TOKENS` in `iphone_agent/harness/tokens.py` is keyed by
  `("system", <prompt_hash>)`. Since 2026-09-17 there are **two** measured
  system keys, one per `IPHONE_SKILL_TOOLS` state. Public keys at prompt
  version 25: `1ab31c31d48e` (`system_prompt(True, True, False)`, the
  production default) and `09e7210e56bf` (`system_prompt(True, True, True)`).
  Both are computed from the public prompt text; Lab's measured hashes differ.
- The token counts recorded against those keys (`3061` and `3130`) are **not**
  fresh measurements — they're carried over from Lab's measurement, since
  swapping one 4-character example word for another does not change the
  segment's character count (verified: the public prompt renders 5454 / 5579
  characters, the same as Lab's). Treat them as estimates until someone
  re-runs `scripts/calibrate_tokens.py --probe segments` against the public
  prompt text. The code comment beside the keys says so too.

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
- App ids derived from that name follow it. Tests that assert on the derived
  id (`tests/test_appids.py`, `tests/test_screenmap.py`,
  `tests/test_appmigrate.py`) use the slug of the public placeholder, not of
  the real name. `tests/test_appids.py` also uses a third-party display name as
  "a name not in the canonical table"; Lab's is a real bank, public's is a
  generic one.

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

## 5. Places, people and devices in fixtures and tasks

Test fixtures and evaluation tasks are copied from real runs, so they carry
whatever was on the screen that day. Substitutions applied so far:

- `tests/fixtures/verdict/`: the two step records captured for the
  "unknown is not a pass" work name a real city in a search query and two real
  shops in its results, and a real person in a social feed. Public uses a
  generic query (`咖啡探店`), `咖啡馆 A` / `咖啡馆 B`, and a placeholder post
  title. The fixtures are also **renamed**: Lab names them after the run id and
  step (`<runid>_step<N>.json`), public names them after what they show
  (`tap_no_expect.json`, `type_partial_landing.json`). `tests/test_verdict.py`
  loads them by name, so rename both together.
- `evalset/tasks/hard/`: two tasks searched for landmarks in the maintainer's
  city and one searched for the maintainer's own phone model. Public uses a
  well-known public landmark, a generic search term, and a product that is not
  a device identifier. Adapt the ids' `device_note` too: it cited a Lab-only
  numbered design doc, which does not ship.
- `knowledge/apps/files/APP.md`: the truncated-filename example was a real file
  from the maintainer's iCloud Drive; public uses an invented filename.
- Device model strings (`iPhone 16e`) are device identifiers. Existing public
  text uses either a bare `iPhone` (`iphone_agent/health.py`) or a different
  model number where a concrete-looking example reads better
  (`tests/test_bench.py`). Either is fine; do not reintroduce the real one.
- Third-party app names that merely appear as home-screen labels or as apps a
  task drives (微博, 小红书, Reddit, YouTube, Chrome …) are not private and stay.
  The bookkeeping app in §2 and §3 is the exception, because it is the one the
  maintainer's own ledger lives in.

## 6. Run paths in newly written comments

Lab writes `runs/<timestamp>-<id>` and `evalset/results/tasks-<timestamp>` into
⚠ comments as evidence. Those are reminder-tier for the gate script, but §4
says they do not ship: when copying a comment that cites one, keep the
technical explanation and replace the path with a neutral phrase
("一次真机运行", "一次真机运行的第 N 步", "一次真机 tasks 跑"). The 2026-09-24
export did this for every comment it touched, including ones already shipped in
files it was updating anyway (`iphone_agent/twin/events.py`,
`iphone_agent/eval/verify.py`). Older files it did not touch still carry them;
that is a difference in age, not a rule.

## 7. Tests that reference fixtures the public repo does not ship

Do not delete such a test silently — adapt it, so the rule it guards keeps a
guard. `tests/test_verify.py` had a test asserting the Lab hard-gate vision
tasks load and stay out of the daily set. Those task files do not ship, so
public rewrites it against `evalset/tasks/hard/`, which does
(`test_hard_set_loads_and_stays_out_of_the_daily_set`), keeping the actual
invariant: a named set loads on its own and never mixes into the top-level set.

## General rule

When exporting from Lab to public: keep the technical substance of every
comment (what broke, why, what the fix does), and only strip the parts that
name a real person, a real app/bank/institution, a device identifier, a
filesystem path under the maintainer's home directory, or a Lab-only run/log
path. If a substitution changes a measured constant (like the prompt hash
above), say so explicitly rather than silently re-using the old number.
