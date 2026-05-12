# PDF renderer

Wave 6 establishes the first printable PDF renderer stack.

This package now owns:

- loading a built book package into render models
- layout profiles for 2-up and 4-up letter-sized pages
- title page rendering
- section divider rendering
- puzzle page rendering
- solution page rendering
- final book PDF export

Current Wave 6 scope:
- supports classic 9x9 built book packages
- supports `classic_four_up` and `classic_two_up`
- outputs a print-friendly interior PDF
- does not yet include advanced typography, cover rendering, or preview thumbnails