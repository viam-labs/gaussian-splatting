"""export-splat stage: collect what 3DGRUT wrote (USDZ, PLY) into `<work>/<run>/splat/` with a report."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger("splat-build")

EXPORT_EXTENSIONS = (".usdz", ".ply")
NOTE_FILE = "NOTE.txt"


def find_exports(out_dir: Path, experiment_name: str) -> dict[str, Path]:
    """Newest file per extension under `<out_dir>/<experiment_name>` (fallback: anywhere under out_dir)."""
    roots = [out_dir / experiment_name, out_dir]
    found: dict[str, Path] = {}
    for ext in EXPORT_EXTENSIONS:
        for root in roots:
            if not root.exists():
                continue
            candidates = [p for p in root.rglob(f"*{ext}") if p.is_file()]
            if candidates:
                found[ext] = max(candidates, key=lambda p: p.stat().st_mtime)
                break
    return found


def export_splat(
    train_out_dir: Path,
    experiment_name: str,
    splat_dir: Path,
    *,
    colmap_report: dict[str, Any] | None,
    trained: bool,
) -> dict[str, Any] | None:
    """Copy the splat files into `splat_dir`. Returns the report when a splat was found, else None."""
    splat_dir.mkdir(parents=True, exist_ok=True)
    if not trained:
        note = (
            "Training has not run yet. Run train/command.sh in the 3DGRUT environment (or re-run "
            "splat-build with --run-train --threedgrut-dir PATH), then re-run splat-build to collect it.\n"
        )
        (splat_dir / NOTE_FILE).write_text(note)
        log.info("no training output yet; wrote %s", splat_dir / NOTE_FILE)
        return None
    found = find_exports(train_out_dir, experiment_name)
    if not found:
        (splat_dir / NOTE_FILE).write_text(
            f"Training ran but no {EXPORT_EXTENSIONS} file was found under {train_out_dir}. "
            "Check train/train.log.\n"
        )
        log.warning("no splat export found under %s", train_out_dir)
        return None
    sources: dict[str, str] = {}
    for ext, src in found.items():
        dst = splat_dir / f"scene{ext}"
        shutil.copy2(src, dst)
        sources[dst.name] = str(src)
        log.info("copied %s -> %s", src, dst)
    report: dict[str, Any] = {"experiment_name": experiment_name, "sources": sources, "colmap": colmap_report}
    (splat_dir / "report.json").write_text(json.dumps(report, indent=2))
    stale_note = splat_dir / NOTE_FILE
    if stale_note.exists():
        stale_note.unlink()
    return report
