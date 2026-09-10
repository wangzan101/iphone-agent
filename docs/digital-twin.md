# Digital twin: current scope

The goal is to learn useful, revisable knowledge about a device so every task does not rediscover navigation from scratch.

The current implementation includes device schema and layout persistence, grouping home-screen labels into rows/columns, scanning, updating observed pages and using layout candidates when opening an app. App opening includes identity checks and fallback paths.

This is an early device-layout layer. It is not yet a complete phone simulator, a comprehensive map of app screens, or a guarantee of correct app identity. Layout can become stale; observations and identity checks can be uncertain.

The next work is to measure scanning and opening across devices, handle stale or ambiguous knowledge, and build evidence-backed screen/transition representations.

The public layout fixture is synthetic. Its regular 4-column, 6-row arrangement is useful for regression testing but does not measure real-device success. Broader fixtures and real-device evidence remain necessary.
