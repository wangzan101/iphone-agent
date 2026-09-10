# Security and privacy

This alpha can operate real apps and change real data. Supervise initial runs and do not rely on action classification as an authorization boundary for sensitive operations.

Screenshots, OCR text, task descriptions and context may be transmitted to your configured model endpoint. Local run logs and learned memory may contain private information. Use an isolated workspace and review material before sharing.

Keep credentials in your own local configuration or provider environment variables. Do not place them in issues, fixtures or shell examples containing real values. The local web interface should remain on loopback.

For a suspected vulnerability, do not publish an exploit containing credentials or private data. Use GitHub private vulnerability reporting if enabled. If it is unavailable, open a minimal issue asking for a private reporting channel without disclosing sensitive details. No dedicated security contact or response SLA is currently established.
