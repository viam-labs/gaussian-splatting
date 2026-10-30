"""Stage cache: each stage owns a folder, is done when `<folder>/.done` exists, and `--force` invalidates it
and everything after it.

Kept apart from the typer app so the skip/force logic is unit-testable without running any real stage.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("splat-build")

DONE_MARKER = ".done"

STAGES: tuple[str, ...] = ("export", "join", "colmap", "train", "export-splat", "finalize")


class StageError(RuntimeError):
    """A stage could not do its job; the CLI prints the message and exits non-zero."""


def stage_index(name: str) -> int:
    if name not in STAGES:
        raise StageError(f"unknown stage {name!r}; stages are {', '.join(STAGES)}")
    return STAGES.index(name)


def is_done(stage_dir: Path) -> bool:
    return (stage_dir / DONE_MARKER).exists()


def mark_done(stage_dir: Path, info: dict[str, Any] | None = None) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {"finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(info or {})}
    (stage_dir / DONE_MARKER).write_text(json.dumps(payload, indent=2))


def archive_stage(stage_dir: Path) -> None:
    """Move a stage folder aside (never delete) so `--force` keeps the previous attempt for comparison."""
    if not stage_dir.exists():
        return
    archive_root = stage_dir.parent / "_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / f"{stage_dir.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    shutil.move(str(stage_dir), str(target))
    log.info("archived previous %s to %s", stage_dir.name, target)


@dataclass
class StagePlan:
    """Which stages to run given --until / --force."""

    until: str | None = None
    force: str | None = None

    def should_run(self, stage: str, stage_dir: Path) -> bool:
        if self.until is not None and stage_index(stage) > stage_index(self.until):
            return False
        if self.is_forced(stage):
            return True
        return not is_done(stage_dir)

    def is_forced(self, stage: str) -> bool:
        return self.force is not None and stage_index(stage) >= stage_index(self.force)

    def stops_after(self, stage: str) -> bool:
        return self.until == stage


def run_stage(
    plan: StagePlan,
    stage: str,
    stage_dir: Path,
    fn: Callable[[], dict[str, Any] | None],
    *,
    mark: bool = True,
) -> bool:
    """Run `fn` for `stage` unless cached. Returns True when the stage ran.

    `fn` returns a dict recorded in the done marker, or None to run without marking the stage done (used by
    stages that only produced a command or a note and should re-run next time).
    """
    if not plan.should_run(stage, stage_dir):
        if plan.until is not None and stage_index(stage) > stage_index(plan.until):
            return False
        log.info("[%s] cached at %s (use --force %s to redo)", stage, stage_dir, stage)
        return False
    if plan.is_forced(stage) and is_done(stage_dir):
        archive_stage(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    log.info("[%s] running -> %s", stage, stage_dir)
    t0 = time.monotonic()
    info = fn()
    elapsed = time.monotonic() - t0
    if mark and info is not None:
        mark_done(stage_dir, {"elapsed_s": round(elapsed, 2), **info})
        log.info("[%s] done in %.1f s", stage, elapsed)
    else:
        log.info("[%s] finished in %.1f s (not marked done; will run again next time)", stage, elapsed)
    return True
