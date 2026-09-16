# Known limitations

- Requires a working macOS iPhone Mirroring setup.
- Pixel observations lack a general app identity API or reliable accessibility tree.
- OCR, timing and app layouts vary; actions and identity checks can fail.
- Chinese entry relies on input-method behavior and may need device-specific validation.
- Provider presets and OpenAI-compatible transports do not guarantee every model's image/tool support.
- Digital-twin layout learning is an early implementation, not a complete device model.
- Safety classification and verification are not complete protection against harmful actions.
- Safety classification reads the text of the tap target. A tap given as raw coordinates is classified by whichever element box it lands in; a tap that lands outside every box (a bare icon) cannot be classified, so it is recorded as unclassified, counted in the run record, and flagged for review rather than silently treated as safe.
- A tap must state what it expects to happen, and the result is checked against that. The check is evidence for the model and for review; it does not stop a wrong tap from having happened.
- Memory and run logs can retain sensitive content.
- Public synthetic fixtures do not reproduce the full private real-device evaluation set.
- Supervised real-device acceptance of the first public candidate remains pending.
