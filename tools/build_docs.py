# tools/build_docs.py
"""
Build API docs (HTML) into docs/site using pdoc.
- Requires: pip install -r requirements-docs.txt
- Usage: python tools/build_docs.py
"""
import subprocess, sys, shutil, os
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "docs" / "site"
    out.mkdir(parents=True, exist_ok=True)
    # Build docs for our top-level packages
    pkgs = ["apps", "solver", "vision"]
    cmd = [sys.executable, "-m", "pdoc", "-o", str(out)] + pkgs
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Docs built → {out} (open index.html)")

if __name__ == "__main__":
    main()
