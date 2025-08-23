# Sudoku Companion — Toolkit


## 🔤 Add type hints (selected functions) — optional
```powershell
# Preview the planned changes
python tools/annotate_types.py --dry-run
type docs\type_annotation_report.json

# Apply in-place
python tools/annotate_types.py --apply
```

Type aliases live in `types_sudoku.py`:
- `Grid = list[list[int]]` (0 = empty)
- `Candidates = dict[str, list[int]]`
- `Move` (TypedDict): fields for placement/elimination & UI hints

## 📚 Build HTML docs (pdoc)
```powershell
pip install -r requirements-docs.txt
python tools/build_docs.py
# Open docs\site\index.html in your browser
```
