# Book builder

Wave 5 establishes the declarative book engine.

This package now owns:

- book spec loading
- section allocation
- puzzle selection by criteria
- section ordering
- reuse / dedupe filtering
- final book manifest construction

Current Wave 5 scope:
- build one book from one spec
- section manifests are written separately
- assigned puzzle payloads are copied into the built book folder
- ordering is currently weight-first with configurable tie-breakers