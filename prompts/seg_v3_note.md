# About seg_v3 (Opus on Inanna's Descent)

The 10 Opus runs on Inanna's Descent (29–30 June 2026) used an earlier prompt, `seg_v3`.
Its exact text was not kept: the file was edited after these runs.

From the model output we can see two differences from `seg_v4`:

- the line range was written inside the scene name (`text_en`), e.g. `"Resolve to descend (strokes 1-3)"`;
- each segment had one extra field, `exchange_channel`.

The segment definition and the seven function states were the same as in `seg_v4`.
A script moved the line ranges into `line_start` / `line_end`.
No number in the paper uses `text_en` or `exchange_channel`.
