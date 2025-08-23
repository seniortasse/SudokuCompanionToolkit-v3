# Docstring Annotations — How To

This annotator inserts readable docstrings into the most important modules and functions so newcomers (and future-you) can understand the pipeline quickly.

## Run (dry run first)
```powershell
# From your project root and with your venv active:
python tools/annotate_docstrings.py --dry-run
type docs\annotation_report.json
```

- It lists which files were found, which functions got a docstring, and which were skipped (not found or already documented).

## Apply
```powershell
python tools/annotate_docstrings.py --apply
```

- Edits are **in-place** and only add docstrings if missing. Existing docstrings are preserved.

## If your file layout differs
- Edit `tools/annotate_docstrings.py` → update the `TARGETS` mapping to match your paths and function names.

## Undo (if needed)
- If you use Git: `git diff` to review; `git restore <file>` to revert before commit.
