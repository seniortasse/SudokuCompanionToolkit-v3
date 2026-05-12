List print formats:
python -m python.publishing.workflows.list_print_formats

Preview a publication plan:
python -m python.publishing.workflows.preview_publication_plan --publication-dir "datasets/sudoku_books/classic9/publications/BK-CL9-DW-B01__PUB-BK-CL9-DW-B01-KDP-8511-BW"

Validate a publication:
python -m python.publishing.workflows.validate_publication --publication-dir "datasets/sudoku_books/classic9/publications/BK-CL9-DW-B01__PUB-BK-CL9-DW-B01-KDP-8511-BW"

Export preview PNGs:
python -m python.publishing.workflows.export_publication_previews --publication-dir "datasets/sudoku_books/classic9/publications/BK-CL9-DW-B01__PUB-BK-CL9-DW-B01-KDP-8511-BW" --output-dir "exports/sudoku_books/previews/BK-CL9-DW-B01"

Export the full bundle:
python -m python.publishing.workflows.export_publication_bundle --publication-dir "datasets/sudoku_books/classic9/publications/BK-CL9-DW-B01__PUB-BK-CL9-DW-B01-KDP-8511-BW" --output-dir "exports/sudoku_books/bundles/BK-CL9-DW-B01__PUB-BK-CL9-DW-B01-KDP-8511-BW"