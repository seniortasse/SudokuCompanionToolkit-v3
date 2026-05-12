# Puzzle catalog

Wave 3 establishes the canonical puzzle-record pipeline.

This package now owns:

- technique profile normalization
- pattern linking from the canonical pattern registry
- difficulty classification from weight
- metadata enrichment for app and book use
- canonical `PuzzleRecord` building
- record persistence
- an MVP generator bridge via JSONL

## Important Wave 3 design choice

Wave 3 intentionally uses a normalized JSONL generator bridge first.

That lets the canonical content pipeline stabilize before we patch
directly into the current generator internals. This is lower risk and
keeps the publishing architecture clean.

A later wave can replace or supplement the JSONL bridge with a direct
adapter to the existing generator modules.