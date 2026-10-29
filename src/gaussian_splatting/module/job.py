"""The scan job: the asyncio task behind `start_scan` / `upload_scan`, and the status it reports.

Owning the loop in a plain class (not the resource) keeps the state machine testable with fakes and keeps
`scanner.py` down to config plumbing and DoCommand dispatch.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from gaussian_splatting import schema
from gaussian_splatting.module.capture import CaptureDeps, CaptureSettings, capture_viewpoint, with_retry
from gaussian_splatting.module.config import ScanRequest
from gaussian_splatting.module.planner import Viewpoint
from gaussian_splatting.module.uploader import DataClientHandle, Uploader, UploadReport, UploadTarget

TERMINAL_STATES = frozenset({"done", "failed", "cancelled"})


class ScanJob:
    def __init__(
        self,
        *,
        request: ScanRequest,
        scan_dir: Path,
        manifest: schema.Manifest,
        viewpoints: list[Viewpoint],
        deps: CaptureDeps,
        settings: CaptureSettings,
        camera_lock_command: Mapping[str, Any],
        upload_target: UploadTarget | None,
        open_client: Callable[[], Awaitable[DataClientHandle]] | None,
        upload_note: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.scan_dir = scan_dir
        self.manifest = manifest
        self.viewpoints = viewpoints
        self.deps = deps
        self.settings = settings
        self.camera_lock_command = camera_lock_command
        self.upload_target = upload_target
        self.open_client = open_client
        self.upload_note = upload_note  # why upload is off although requested, surfaced as last_error
        self.logger = logger or logging.getLogger(__name__)

        existing = schema.read_frames(scan_dir)
        self.state = "planning"  # never the manifest's state: a resumed "done" scan would look finished
        self.poses_done = len({f.viewpoint for f in existing})
        self.poses_skipped = len(manifest.skipped_viewpoints)
        self.images_saved = len(existing)
        self.uploads_done = 0
        self.uploads_pending = sum(
            1 for f in existing if not (f.uploaded.binary_id and f.uploaded.tabular_ok)
        )
        self.sequence_id = manifest.sequence_id
        self.last_error: str | None = upload_note
        self._next_index = max((f.index for f in existing), default=0) + 1
        self._done_viewpoints = {f.viewpoint for f in existing}
        self._cancel_requested = False
        self._t_start = time.monotonic()
        self._t_end: float | None = None

    # -- status --

    @property
    def finished(self) -> bool:
        return self.state in TERMINAL_STATES

    def cancel(self) -> None:
        """Stop after the current viewpoint; the upload phase still runs for what was captured."""
        self._cancel_requested = True

    def status(self) -> dict[str, Any]:
        elapsed = (self._t_end or time.monotonic()) - self._t_start
        eta: float | None = None
        if self.state == "capturing" and self.poses_done > 0:
            eta = elapsed / self.poses_done * max(self.request.num_poses - self.poses_done, 0)
        return {
            "scan_id": self.request.scan_id,
            "state": self.state,
            "poses_done": self.poses_done,
            "poses_planned": self.request.num_poses,
            "candidates": len(self.viewpoints),
            "poses_skipped": self.poses_skipped,
            "images_saved": self.images_saved,
            "uploads_done": self.uploads_done,
            "uploads_pending": self.uploads_pending,
            "sequence_id": self.sequence_id,
            "elapsed_s": round(elapsed, 1),
            "eta_s": None if eta is None else round(eta, 1),
            "last_error": self.last_error,
            "scan_dir": str(self.scan_dir),
        }

    # -- phases --

    async def run_scan(self) -> None:
        try:
            await self._capture_phase()
            await self._upload_phase()
            self._finish("cancelled" if self._cancel_requested else "done")
        except asyncio.CancelledError:
            self._finish("cancelled")
            raise
        except Exception as e:
            self.logger.exception("scan %s failed", self.request.scan_id)
            self.last_error = f"{type(e).__name__}: {e}"
            self._finish("failed")

    async def run_upload(self) -> None:
        try:
            await self._upload_phase()
            self._finish("done")
        except asyncio.CancelledError:
            self._finish("cancelled")
            raise
        except Exception as e:
            self.logger.exception("upload of %s failed", self.request.scan_id)
            self.last_error = f"{type(e).__name__}: {e}"
            self._finish("failed")

    def _set_state(self, state: str) -> None:
        self.state = state
        self.manifest.state = state
        schema.write_manifest(self.scan_dir, self.manifest)

    def _finish(self, state: str) -> None:
        self._t_end = time.monotonic()
        self.manifest.finished_at = schema.format_rfc3339(self.deps.clock())
        self.manifest.last_error = self.last_error
        self._set_state(state)

    async def _capture_phase(self) -> None:
        self._set_state("capturing")
        if self.camera_lock_command:
            try:
                await with_retry(
                    self.logger,
                    "camera lock command",
                    lambda: self.deps.camera.do_command(self.camera_lock_command),
                    sleep=self.deps.sleep,
                )
            except Exception as e:
                self.logger.warning(
                    "camera_lock_command failed, continuing with the camera's own settings: %s", e
                )

        for vp in self.viewpoints:
            if self._cancel_requested or self.poses_done >= self.request.num_poses:
                break
            if vp.index in self._done_viewpoints:
                continue
            outcome = await capture_viewpoint(self.deps, self.settings, vp, self.scan_dir, self._next_index)
            if outcome.skipped_reason is not None:
                self.poses_skipped += 1
                self.manifest.skipped_viewpoints.append({"index": vp.index, "reason": outcome.skipped_reason})
                schema.write_manifest(self.scan_dir, self.manifest)
                self.logger.info("viewpoint %d skipped: %s", vp.index, outcome.skipped_reason)
                continue
            self.poses_done += 1
            self.images_saved += len(outcome.frames)
            self.uploads_pending += len(outcome.frames)
            self._next_index += len(outcome.frames)
            self._done_viewpoints.add(vp.index)
            self.logger.info(
                "viewpoint %d captured (%d/%d), sharpness %.1f",
                vp.index,
                self.poses_done,
                self.request.num_poses,
                outcome.frames[0].sharpness,
            )
        if self.poses_done < self.request.num_poses and not self._cancel_requested:
            self.logger.warning(
                "candidates exhausted: %d of %d poses captured, %d skipped",
                self.poses_done,
                self.request.num_poses,
                self.poses_skipped,
            )

    async def _upload_phase(self) -> None:
        if self.upload_target is None or self.open_client is None:
            return
        if self.images_saved == 0:
            self.logger.info("nothing captured, skipping upload")
            return
        self._set_state("uploading")
        handle = await self.open_client()
        try:
            uploader = Uploader(
                handle.client,
                self.upload_target,
                logger=self.logger,
                sleep=self.deps.sleep,
                on_progress=self._on_upload_progress,
            )
            report = await uploader.upload_scan(self.scan_dir)
        finally:
            handle.close()
        self._set_state("finalizing")
        self._on_upload_progress(report)
        self.sequence_id = report.sequence_id
        # The uploader worked on its own copy of the manifest read from disk; take its results over before the
        # job's final manifest write, or that write would clobber them with stale values.
        self.manifest.sequence_id = report.sequence_id
        self.manifest.dataset_added = report.dataset_added
        if report.dataset_added and self.manifest.dataset_id is None:
            self.manifest.dataset_id = self.request.dataset_id
        if report.dataset_error is not None:
            self.last_error = f"sequence created but not added to dataset: {report.dataset_error}"

    def _on_upload_progress(self, report: UploadReport) -> None:
        self.uploads_done = report.uploads_done + report.uploads_skipped
        self.uploads_pending = report.uploads_pending
