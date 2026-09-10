# Architecture

The CLI and local web UI share the session and task harness. The harness observes the phone, constructs model context, validates actions, executes them and checks the result.

| Module | Responsibility |
|---|---|
| driver | Mirror window discovery, capture, coordinates and input injection |
| perceive | OCR, model-assisted vision, element extraction and image-change checks |
| model | Provider/model configuration, reply parsing and compatible transport |
| harness | Task loop, history, action validation, execution, recovery and logging |
| memory | File-based memory and reconstructed screen maps |
| skills | Skill storage, extraction, routing and execution support |
| twin | Device-layout schema, inference, scanning and updates |
| cli / web | User interfaces sharing the underlying execution components |
| eval | Replay and task-verification utilities |

Input operates through macOS iPhone Mirroring. No app is installed on the phone. The driver uses macOS-specific mechanisms, including private system interfaces; OS changes may require adaptation.

A Workspace places configuration and knowledge under `.iphone/` and run artifacts under `runs/`. `IPHONE_WORKSPACE` can select a different workspace root. These directories are user data, not distributable examples.

The dependency graph is not perfectly layered: model code currently refers to action vocabulary under harness. This is a documented tradeoff, not a claim of strict dependency isolation.

Some source comments refer to numbered internal research notes. Those historical references are not required to build or test this checkout. The public documentation describes current behavior; original device recordings are not included.
