# Evaluation

Automated regression tests exercise mocked device behavior, transport handling, layout inference, memory and execution logic. Passing these tests does not establish a task success rate on real phones.

## Public fixtures

- `tests/data/wire_*_golden.json`: synthetic protocol messages.
- `evalset/labels/00-current.json`: synthetic home-screen geometry at a historical compatibility path. It is not a recording of a user's installed apps. Legacy test names/comments referring to real data describe their origin, not this public fixture.
- `evalset/tasks/settings-ios-version.json`: illustrative read task. Replace its expected version and UI text for your own test phone.

No raw runs, OCR captures, private notes or historical performance datasets are shipped.

For real-device results, report device/OS, UI language, model, task, number of attempts, success criteria and failures. Redact screenshots and logs. Until a new candidate is checked on a real device, do not describe its offline result as real-device acceptance.
