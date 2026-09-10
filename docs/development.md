# Development

Use macOS and Python 3.12+.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,repl]'
python -m pip install ruff build
python -m pytest -q
python -m ruff check iphone_agent tests scripts
python -m build
```

The automated tests use fake devices, generated images and local fixtures. They must not require a model key or an attached phone. Real-device operation is a separate, supervised activity.

`scripts/driver_gate.py` checks input primitives interactively and uses `scripts/_screen.py`. It changes device state; read the prompts and use a suitable test device.

`scripts/calibrate_tokens.py` is retained because tests exercise its pure calculations. Running its model probes can make billable requests; it is not run as a probe in CI.

Do not use a private checkout on PYTHONPATH to make public tests pass. Fix or supply the actual public dependency.

Source text is primarily Chinese at present; public overview documentation is provided in English and Chinese. Translations and clearer explanations are welcome.

## Web language checks

The UI remains framework-free and ships its own local message catalog. Node.js runs the dependency-free language tests; without Node, pytest reports that check as skipped. An optional Playwright smoke test checks the actual page using fully mocked phone/model APIs. See [languages.md](languages.md) for commands and boundaries.
