Wave 6 — export a locale × layout matrix in one command:

python -m python.publishing.workflows.export_publication_variant_matrix `
  --book-dir "datasets/sudoku_books/classic9/books/BK-CL9-DW-B01" `
  --base-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --locales de fr it es `
  --layouts 2up 4up 6up 12up `
  --output-publications-dir "datasets/sudoku_books/classic9/publications" `
  --output-bundles-dir "exports/sudoku_books/bundles"

Wave 6 — include English and 1-up too:

python -m python.publishing.workflows.export_publication_variant_matrix `
  --book-dir "datasets/sudoku_books/classic9/books/BK-CL9-DW-B01" `
  --base-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --locales en de fr `
  --layouts 1up 4up 6up `
  --output-publications-dir "datasets/sudoku_books/classic9/publications" `
  --output-bundles-dir "exports/sudoku_books/bundles"

Wave 6 — stop on first failure:

python -m python.publishing.workflows.export_publication_variant_matrix `
  --book-dir "datasets/sudoku_books/classic9/books/BK-CL9-DW-B01" `
  --base-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --locales de fr it es `
  --layouts 2up 4up 6up 12up `
  --output-publications-dir "datasets/sudoku_books/classic9/publications" `
  --output-bundles-dir "exports/sudoku_books/bundles" `
  --stop-on-first-failure