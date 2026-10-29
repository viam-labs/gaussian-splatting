"""Phase 2 (spec section 6.4): push a scan directory to the Viam cloud and wrap it in a sequence.

Driven entirely by `frames.jsonl`: every frame's upload state is written back after each call, so a re-run
after a network failure skips what already landed. The data client is an injected Protocol so tests use a fake
and nothing here ever needs credentials.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from grpclib import GRPCError, Status
from viam.app.viam_client import ViamClient
from viam.proto.app.data import SequenceResourceFilter
from viam.rpc.dial import DialOptions

from gaussian_splatting import schema

UPLOAD_ATTEMPTS = 3
RETRY_DELAYS_S: tuple[float, ...] = (2.0, 5.0)
CAMERA_COMPONENT_TYPE = "rdk:component:camera"
SENSOR_COMPONENT_TYPE = "rdk:component:sensor"
CAMERA_METHOD = "GetImages"
SENSOR_METHOD = "Readings"
SEQUENCE_PAD = timedelta(seconds=1)
SPLAT_TAG = "splat"


class UploadError(RuntimeError):
    """A cloud call still failed after the retries. The scan directory is intact; `upload_scan` can re-run."""


class DataClientLike(Protocol):
    """The four DataClient calls the uploader makes, keyword arguments only."""

    async def binary_data_capture_upload(
        self,
        binary_data: bytes,
        part_id: str,
        component_type: str,
        component_name: str,
        method_name: str,
        file_extension: str,
        method_parameters: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
        dataset_ids: list[str] | None = None,
        data_request_times: tuple[datetime, datetime] | None = None,
        mime_type: str | None = None,
    ) -> str: ...

    async def tabular_data_capture_upload(
        self,
        tabular_data: list[Mapping[str, Any]],
        part_id: str,
        component_type: str,
        component_name: str,
        method_name: str,
        data_request_times: list[tuple[datetime, datetime]],
        method_parameters: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> str: ...

    async def create_sequence(
        self,
        part_id: str,
        resources: list[SequenceResourceFilter],
        start_time: datetime,
        end_time: datetime,
        sequence_tags: list[str] | None = None,
        timeout: float | None = None,
    ) -> str: ...

    async def add_sequences_to_dataset(
        self, dataset_id: str, sequence_ids: list[str], timeout: float | None = None
    ) -> None: ...


@dataclass(frozen=True)
class CloudCredentials:
    api_key: str
    api_key_id: str
    part_id: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> CloudCredentials | None:
        """viam-server sets these for every module, `.env` for local runs. None when any is missing."""
        env = os.environ if env is None else env
        key, key_id, part = (env.get(k) for k in ("VIAM_API_KEY", "VIAM_API_KEY_ID", "VIAM_MACHINE_PART_ID"))
        if not key or not key_id or not part:
            return None
        return cls(api_key=key, api_key_id=key_id, part_id=part)


@dataclass
class DataClientHandle:
    client: DataClientLike
    close: Callable[[], None]


async def open_data_client(creds: CloudCredentials, app_url: str | None = None) -> DataClientHandle:
    viam_client = await ViamClient.create_from_dial_options(
        DialOptions.with_api_key(api_key=creds.api_key, api_key_id=creds.api_key_id), app_url=app_url
    )
    return DataClientHandle(client=viam_client.data_client, close=viam_client.close)


@dataclass(frozen=True)
class UploadTarget:
    part_id: str
    camera_name: str
    scan_state_name: str
    dataset_id: str | None
    tags: tuple[str, ...] = ()

    def all_tags(self, scan_id: str) -> list[str]:
        ordered: list[str] = []
        for t in (scan_id, SPLAT_TAG, *self.tags):
            if t not in ordered:
                ordered.append(t)
        return ordered


@dataclass
class UploadReport:
    uploads_done: int = 0
    uploads_skipped: int = 0
    uploads_pending: int = 0
    sequence_id: str | None = None
    dataset_added: bool = False
    dataset_error: str | None = None


class Uploader:
    def __init__(
        self,
        client: DataClientLike,
        target: UploadTarget,
        *,
        logger: logging.Logger | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        retry_delays_s: Sequence[float] = RETRY_DELAYS_S,
        on_progress: Callable[[UploadReport], None] | None = None,
    ) -> None:
        self.client = client
        self.target = target
        self.logger = logger or logging.getLogger(__name__)
        self.sleep = sleep
        self.retry_delays_s = list(retry_delays_s)
        self.on_progress = on_progress

    async def _retry[T](self, what: str, fn: Callable[[], Awaitable[T]]) -> T:
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            try:
                return await fn()
            except GRPCError as e:
                if e.status == Status.PERMISSION_DENIED:
                    raise  # not transient: retrying a permission error only wastes time
                if attempt >= UPLOAD_ATTEMPTS:
                    raise UploadError(f"{what} failed after {attempt} attempts: {e}") from e
                self.logger.warning("%s failed (attempt %d/%d): %s", what, attempt, UPLOAD_ATTEMPTS, e)
            except Exception as e:
                if attempt >= UPLOAD_ATTEMPTS:
                    raise UploadError(f"{what} failed after {attempt} attempts: {e}") from e
                self.logger.warning("%s failed (attempt %d/%d): %s", what, attempt, UPLOAD_ATTEMPTS, e)
            await self.sleep(self.retry_delays_s[min(attempt - 1, len(self.retry_delays_s) - 1)])
        raise AssertionError("unreachable")

    async def upload_scan(
        self, scan_dir: Path, *, is_cancelled: Callable[[], bool] = lambda: False
    ) -> UploadReport:
        manifest = schema.read_manifest(scan_dir)
        frames = schema.read_frames(scan_dir)
        tags = self.target.all_tags(manifest.scan_id)
        extras = schema.frame_extras(manifest)
        report = UploadReport(uploads_pending=sum(1 for f in frames if not self._is_uploaded(f)))

        for frame in frames:
            if self._is_uploaded(frame):
                report.uploads_skipped += 1
                continue
            if is_cancelled():
                self.logger.info("upload cancelled with %d frames pending", report.uploads_pending)
                return report
            await self._upload_frame(scan_dir, frames, frame, tags, extras)
            report.uploads_done += 1
            report.uploads_pending -= 1
            if self.on_progress is not None:
                self.on_progress(report)

        if report.uploads_pending > 0 or not frames:
            return report

        if manifest.sequence_id is None:
            manifest.sequence_id = await self._retry(
                "create_sequence", lambda: self._create_sequence(frames, tags)
            )
            schema.write_manifest(scan_dir, manifest)
        report.sequence_id = manifest.sequence_id

        if self.target.dataset_id and not manifest.dataset_added:
            manifest.dataset_id = self.target.dataset_id
            try:
                await self._retry(
                    "add_sequences_to_dataset",
                    lambda: self.client.add_sequences_to_dataset(
                        dataset_id=str(self.target.dataset_id), sequence_ids=[str(manifest.sequence_id)]
                    ),
                )
                manifest.dataset_added = True
            except (GRPCError, UploadError) as e:
                # Typically the module key lacks organization scope. The sequence exists, so splat-build can
                # finish the dataset add with an org key (`--finalize`): a documented fallback, not a failure.
                report.dataset_error = str(e)
                self.logger.warning("sequence %s not added to dataset: %s", manifest.sequence_id, e)
            schema.write_manifest(scan_dir, manifest)
        report.dataset_added = manifest.dataset_added
        return report

    @staticmethod
    def _is_uploaded(frame: schema.FrameRecord) -> bool:
        return frame.uploaded.binary_id is not None and frame.uploaded.tabular_ok

    async def _upload_frame(
        self,
        scan_dir: Path,
        frames: list[schema.FrameRecord],
        frame: schema.FrameRecord,
        tags: list[str],
        extras: Mapping[str, Any],
    ) -> None:
        times = (schema.parse_rfc3339(frame.t_request), schema.parse_rfc3339(frame.t_capture))
        if frame.uploaded.binary_id is None:
            png = (scan_dir / frame.file).read_bytes()
            frame.uploaded.binary_id = await self._retry(
                f"binary upload {frame.file}",
                lambda: self.client.binary_data_capture_upload(
                    binary_data=png,
                    part_id=self.target.part_id,
                    component_type=CAMERA_COMPONENT_TYPE,
                    component_name=self.target.camera_name,
                    method_name=CAMERA_METHOD,
                    file_extension=".png",
                    data_request_times=times,
                    tags=tags,
                    mime_type="image/png",
                ),
            )
            schema.rewrite_frames(scan_dir, frames)
        if not frame.uploaded.tabular_ok:
            payload = schema.frame_payload(frame, extras)
            await self._retry(
                f"tabular upload {frame.file}",
                lambda: self.client.tabular_data_capture_upload(
                    tabular_data=[{"readings": payload}],
                    part_id=self.target.part_id,
                    component_type=SENSOR_COMPONENT_TYPE,
                    component_name=self.target.scan_state_name,
                    method_name=SENSOR_METHOD,
                    data_request_times=[times],
                    tags=tags,
                ),
            )
            frame.uploaded.tabular_ok = True
            schema.rewrite_frames(scan_dir, frames)

    async def _create_sequence(self, frames: list[schema.FrameRecord], tags: list[str]) -> str:
        start = min(schema.parse_rfc3339(f.t_request) for f in frames) - SEQUENCE_PAD
        end = max(schema.parse_rfc3339(f.t_capture) for f in frames) + SEQUENCE_PAD
        return await self.client.create_sequence(
            part_id=self.target.part_id,
            resources=[
                SequenceResourceFilter(resource_name=self.target.camera_name, method_name=CAMERA_METHOD),
                SequenceResourceFilter(resource_name=self.target.scan_state_name, method_name=SENSOR_METHOD),
            ],
            start_time=start,
            end_time=end,
            sequence_tags=tags,
        )
