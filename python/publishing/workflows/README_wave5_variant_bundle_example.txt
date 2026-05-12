Wave 5 — export a French 4-up bundle in one command:
python -m python.publishing.workflows.export_publication_variant_bundle `
  --book-dir "datasets/sudoku_books/classic9/books/BK-CL9-DW-B01" `
  --base-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --locale fr `
  --layout 4up `
  --output-publications-dir "datasets/sudoku_books/classic9/publications" `
  --output-bundles-dir "exports/sudoku_books/bundles"

Wave 5 — export a German 6-up bundle in one command:
python -m python.publishing.workflows.export_publication_variant_bundle `
  --book-dir "datasets/sudoku_books/classic9/books/BK-CL9-DW-B01" `
  --base-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --locale de `
  --layout 6up `
  --output-publications-dir "datasets/sudoku_books/classic9/publications" `
  --output-bundles-dir "exports/sudoku_books/bundles"

Wave 5 — export an English 1-up bundle in one command:
python -m python.publishing.workflows.export_publication_variant_bundle `
  --book-dir "datasets/sudoku_books/classic9/books/BK-CL9-DW-B01" `
  --base-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --locale en `
  --layout 1up `
  --output-publications-dir "datasets/sudoku_books/classic9/publications" `
  --output-bundles-dir "exports/sudoku_books/bundles"