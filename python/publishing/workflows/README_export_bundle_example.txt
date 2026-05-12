Build a publication package:
python -m python.publishing.workflows.build_publication_package --book-dir "datasets/sudoku_books/classic9/books/BK-CL9-DW-B01" --publication-spec "datasets/sudoku_books/classic9/publication_specs/BK-CL9-DW-B01.kdp.8_5x11.json"

Export the interior:
python -m python.publishing.workflows.export_book_interior --publication-dir "datasets/sudoku_books/classic9/publications/BK-CL9-DW-B01__PUB-BK-CL9-DW-B01-KDP-8511-BW" --output-pdf "exports/sudoku_books/pdf/BK-CL9-DW-B01-interior.pdf"

Export the cover:
python -m python.publishing.workflows.export_book_cover --publication-dir "datasets/sudoku_books/classic9/publications/BK-CL9-DW-B01__PUB-BK-CL9-DW-B01-KDP-8511-BW" --output-pdf "exports/sudoku_books/pdf/BK-CL9-DW-B01-cover.pdf"

Export the full publication bundle:
python -m python.publishing.workflows.export_publication_bundle --publication-dir "datasets/sudoku_books/classic9/publications/BK-CL9-DW-B01__PUB-BK-CL9-DW-B01-KDP-8511-BW" --output-dir "exports/sudoku_books/bundles/BK-CL9-DW-B01__PUB-BK-CL9-DW-B01-KDP-8511-BW"