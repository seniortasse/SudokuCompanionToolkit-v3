"""
Split JSONL files into N parts, writing outputs alongside the input file.

Key behavior:
- Produces exactly N output files.
- Parts are as equal in size as possible.
- By default skips empty lines (configurable).
- Includes progress logs during counting + writing so it never feels "stuck".
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


def ts() -> str:
    # Local time timestamp for logs
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"[{ts()}] {msg}", flush=True)


@dataclass
class ProgressConfig:
    log_every_lines: int
    log_every_secs: float


class ProgressPrinter:
    def __init__(self, cfg: ProgressConfig, *, quiet: bool = False) -> None:
        self.cfg = cfg
        self.quiet = quiet
        self._t0 = time.time()
        self._t_last = self._t0

    def maybe_print(self, lines_done: int, extra: str = "") -> None:
        now = time.time()
        should_by_lines = self.cfg.log_every_lines > 0 and (lines_done % self.cfg.log_every_lines == 0)
        should_by_time = (now - self._t_last) >= self.cfg.log_every_secs

        if (should_by_lines or should_by_time) and not self.quiet:
            elapsed = now - self._t0
            rate = (lines_done / elapsed) if elapsed > 0 else 0.0
            msg = f"progress: lines={lines_done:,}, elapsed={elapsed:,.1f}s, rate={rate:,.0f} lines/s"
            if extra:
                msg += f", {extra}"
            log(msg, quiet=self.quiet)
            self._t_last = now


def count_lines(
    path: Path,
    *,
    skip_empty: bool,
    prog: ProgressPrinter,
    quiet: bool,
) -> int:
    log(f"Counting lines: {path}", quiet=quiet)
    n = 0
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if skip_empty and not line.strip():
                skipped += 1
                continue
            n += 1
            prog.maybe_print(n, extra="phase=count")
    log(f"Count done: total_nonempty_lines={n:,}" + (f", skipped_empty={skipped:,}" if skipped else ""), quiet=quiet)
    return n


def compute_part_sizes(total: int, parts: int) -> List[int]:
    if parts <= 0:
        raise ValueError("parts must be >= 1")
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def split_one(
    in_path: Path,
    *,
    parts: int,
    overwrite: bool,
    skip_empty: bool,
    dry_run: bool,
    quiet: bool,
    prog_cfg: ProgressConfig,
) -> List[Path]:
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    prog_count = ProgressPrinter(prog_cfg, quiet=quiet)
    total = count_lines(in_path, skip_empty=skip_empty, prog=prog_count, quiet=quiet)

    sizes = compute_part_sizes(total, parts)

    out_dir = in_path.parent
    stem = in_path.stem
    suffix = in_path.suffix
    out_paths = [out_dir / f"{stem}_part_{i+1}{suffix}" for i in range(parts)]

    log(f"Will create {parts} parts next to input folder: {out_dir}", quiet=quiet)
    for i, (p, sz) in enumerate(zip(out_paths, sizes), start=1):
        log(f"  part {i:02d}: {p.name}  ({sz:,} lines)", quiet=quiet)

    if not overwrite:
        existing = [p for p in out_paths if p.exists()]
        if existing:
            msg = "Refusing to overwrite existing output files:\n" + "\n".join(str(p) for p in existing)
            raise FileExistsError(msg)

    if dry_run:
        log("Dry-run enabled: no files written.", quiet=quiet)
        return out_paths

    # Open outputs
    outs = []
    try:
        for p in out_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            outs.append(p.open("w", encoding="utf-8", newline="\n"))

        log("Splitting (writing outputs)...", quiet=quiet)
        prog_write = ProgressPrinter(prog_cfg, quiet=quiet)

        idx_part = 0
        remaining_in_part = sizes[idx_part] if parts > 0 else 0
        written_total = 0
        skipped = 0

        with in_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if skip_empty and not line.strip():
                    skipped += 1
                    continue

                # advance if current part filled
                while idx_part < parts and remaining_in_part == 0:
                    idx_part += 1
                    if idx_part < parts:
                        remaining_in_part = sizes[idx_part]
                        log(f"Now writing part {idx_part+1}/{parts}...", quiet=quiet)

                if idx_part >= parts:
                    break

                outs[idx_part].write(line if line.endswith("\n") else (line + "\n"))
                remaining_in_part -= 1
                written_total += 1

                # progress log w/ part info
                prog_write.maybe_print(
                    written_total,
                    extra=f"phase=write, part={idx_part+1}/{parts}",
                )

        log(
            f"Write done: written_nonempty_lines={written_total:,}"
            + (f", skipped_empty={skipped:,}" if skipped else ""),
            quiet=quiet,
        )
        return out_paths

    finally:
        for f in outs:
            try:
                f.close()
            except Exception:
                pass


def expand_inputs(patterns: Iterable[str], *, quiet: bool) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        matched = glob.glob(pat, recursive=True)
        if matched:
            log(f"Matched {len(matched)} file(s) for pattern: {pat}", quiet=quiet)
            paths.extend(Path(m).resolve() for m in matched)
        else:
            # treat as literal path
            p = Path(pat).resolve()
            log(f"No glob matches for '{pat}', treating as literal path: {p}", quiet=quiet)
            paths.append(p)

    # dedupe preserving order
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Split JSONL files into N near-equal parts, writing alongside the input."
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Input .jsonl file path(s) or glob(s). Example: debug/telemetry/*.jsonl",
    )
    ap.add_argument("--parts", type=int, default=10, help="Number of output parts to create (default: 10).")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing *_part_*.jsonl outputs.")
    ap.add_argument(
        "--keep-empty-lines",
        action="store_true",
        help="Do not skip empty lines (by default empty lines are skipped).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print what would happen without writing files.")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress/status logs.")

    # Progress controls
    ap.add_argument(
        "--log-every-lines",
        type=int,
        default=200_000,
        help="Emit a progress log every N processed lines (default: 200000). Use 0 to disable.",
    )
    ap.add_argument(
        "--log-every-secs",
        type=float,
        default=2.0,
        help="Emit a progress log at least every N seconds (default: 2.0).",
    )

    args = ap.parse_args(argv)

    quiet = bool(args.quiet)
    skip_empty = not bool(args.keep_empty_lines)
    prog_cfg = ProgressConfig(log_every_lines=args.log_every_lines, log_every_secs=args.log_every_secs)

    log("split_jsonl starting...", quiet=quiet)
    log(f"Python: {sys.version.split()[0]}", quiet=quiet)

    in_paths = expand_inputs(args.inputs, quiet=quiet)
    if not in_paths:
        log("No input files found.", quiet=quiet)
        return 2

    # Validate parts
    if args.parts <= 0:
        print("[error] --parts must be >= 1", file=sys.stderr)
        return 2

    for p in in_paths:
        try:
            log(f"Processing file: {p}", quiet=quiet)
            outs = split_one(
                p,
                parts=args.parts,
                overwrite=bool(args.overwrite),
                skip_empty=skip_empty,
                dry_run=bool(args.dry_run),
                quiet=quiet,
                prog_cfg=prog_cfg,
            )
            if not args.dry_run:
                log(f"[ok] {p.name} -> {len(outs)} files", quiet=quiet)
        except Exception as e:
            print(f"[error] {p}: {e}", file=sys.stderr)
            return 1

    log("All done.", quiet=quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))