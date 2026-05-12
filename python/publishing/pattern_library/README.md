# Pattern library

Wave 2 establishes the first-class pattern subsystem.

This package now owns:

- Excel pattern ingestion
- pattern normalization
- pattern metadata enrichment
- strict pattern validation
- canonical registry writing

Current Wave 2 scope:
- classic 9x9 only
- workbook sheets interpreted as 9x9 clue masks
- one mask per sheet, default anchored at A1 unless overridden

Future waves can add:
- richer Excel source specifications
- auto-discovery of multiple patterns per sheet
- pattern preview rendering
- pattern generation tools