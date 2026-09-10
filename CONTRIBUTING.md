# Contributing

Start with a reproducible observation. For a significant behavior or architecture change, open an issue before implementing so ongoing work can be coordinated.

## Development

Follow [development.md](docs/development.md). Keep changes focused, include a regression test when behavior changes, and describe what you actually verified. Mock tests and real-device acceptance are different evidence.

Useful contributions include app-opening failures, home-screen layout cases, device compatibility reports, model calibration and portable app skills. See [ROADMAP.md](ROADMAP.md).

## Pull requests

Describe the problem, resulting behavior, validation and remaining limitations. Include system versions, UI language and model identifier where relevant, but omit account identifiers, secrets and private screen content.

Tests should construct their own state and use synthetic fixtures. Do not commit `.iphone/`, `runs/`, credentials or personal screenshots. Skills intended for sharing must use generic examples.

Maintainers develop in a private Lab containing personal device data and experiments. This repository contains the complete public implementation and tests. Accepted contributions are imported into Lab with their authorship and PR reference retained before the next public export. If integration conflicts with ongoing work, maintainers discuss the adaptation instead of silently overwriting it.

## License and attribution

This project uses [Apache License 2.0](LICENSE). Unless explicitly agreed otherwise, contributions intentionally submitted for inclusion are under that license. Submit only material you have the right to contribute, retain required third-party notices, and identify external sources in your PR.

For a first skill or demonstration, follow [the sharing guide](docs/sharing.md). For an experiment or design explanation, use [the build-in-public format](docs/build-in-public.md#an-experiment-note).
