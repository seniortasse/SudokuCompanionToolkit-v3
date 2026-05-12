Wave 0 — build the approved base publication package:
python -m python.publishing.workflows.build_publication_package `
  --book-dir "datasets/sudoku_books/classic9/books/BK-CL9-DW-B01" `
  --publication-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --output-publications-dir "datasets/sudoku_books/classic9/publications"

Wave 0 — validate the approved gold publication package:
python -m python.publishing.workflows.check_base_publication_gold `
  --publication-dir "datasets/sudoku_books/classic9/publications/BK-CL9-DW-B01__PUB-BK-CL9-DW-B01-KDP-6UP-8511-BW" `
  --gold-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --expect-estimated-page-count 188

Wave 3 — compile an English 1-up resolved spec:
python -m python.publishing.workflows.compile_publication_variant `
  --base-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --output-spec "datasets/sudoku_books/classic9/publication_specs_compiled/BK-CL9-DW-B01.kdp.en.1up.8_5x11.json" `
  --locale en `
  --layout 1up

Wave 3 — compile an English 2-up resolved spec:
python -m python.publishing.workflows.compile_publication_variant `
  --base-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --output-spec "datasets/sudoku_books/classic9/publication_specs_compiled/BK-CL9-DW-B01.kdp.en.2up.8_5x11.json" `
  --locale en `
  --layout 2up

Wave 3 — compile an English 4-up resolved spec:
python -m python.publishing.workflows.compile_publication_variant `
  --base-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --output-spec "datasets/sudoku_books/classic9/publication_specs_compiled/BK-CL9-DW-B01.kdp.en.4up.8_5x11.json" `
  --locale en `
  --layout 4up

Wave 3 — compile an English 6-up resolved spec:
python -m python.publishing.workflows.compile_publication_variant `
  --base-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --output-spec "datasets/sudoku_books/classic9/publication_specs_compiled/BK-CL9-DW-B01.kdp.en.6up.8_5x11.json" `
  --locale en `
  --layout 6up

Wave 3 — compile an English 12-up resolved spec:
python -m python.publishing.workflows.compile_publication_variant `
  --base-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.6up.8_5x11.json" `
  --output-spec "datasets/sudoku_books/classic9/publication_specs_compiled/BK-CL9-DW-B01.kdp.en.12up.8_5x11.json" `
  --locale en `
  --layout 12up

Build a publication package from a compiled spec:
python -m python.publishing.workflows.build_publication_package `
  --book-dir "datasets/sudoku_books/classic9/books/BK-CL9-DW-B01" `
  --publication-spec "datasets/sudoku_books/classic9/publication_specs_compiled/BK-CL9-DW-B01.kdp.en.1up.8_5x11.json" `
  --output-publications-dir "datasets/sudoku_books/classic9/publications"