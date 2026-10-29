"""join stage: one sequence's rows from the parquet export -> the robot's scan directory, rebuilt.

Pairs each image row (binary_data.parquet) with the scan-state reading (tabular_data.parquet) that carries the
same capture timestamp. Output is byte-compatible with the module's scan dir (schema.py), so the colmap stage
does not care whether its input came from the cloud or from `--scan-dir`.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from gaussian_splatting import schema as s
from gaussian_splatting.build.export import ExportLayout
from gaussian_splatting.build.stages import StageError

log = logging.getLogger("splat-build")

EXACT_TOLERANCE_US = 1_000
NEAREST_TOLERANCE_US = 200_000

# Column names documented at docs.viam.com/train/sequence-dataset-format, plus aliases to be defensive.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "sequence_id": ("sequence_id", "sequenceId"),
    "timestamp": ("timestamp", "time_received", "timeReceived", "time_requested"),
    "part_id": ("part_id", "partId"),
    "component_name": ("component_name", "componentName"),
    "method_name": ("method_name", "methodName"),
    "path": ("path", "file_path", "filePath"),
    "payload": ("payload", "data", "readings"),
}


@dataclass
class JoinReport:
    sequence_id: str
    num_binary_rows: int
    num_tabular_rows: int
    num_paired: int
    num_nearest_fallback: int
    unpaired_images: list[str] = field(default_factory=list)
    camera_model_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "num_binary_rows": self.num_binary_rows,
            "num_tabular_rows": self.num_tabular_rows,
            "num_paired": self.num_paired,
            "num_nearest_fallback": self.num_nearest_fallback,
            "unpaired_images": self.unpaired_images,
            "camera_model_source": self.camera_model_source,
        }


def _column(table: pa.Table, logical: str) -> pa.ChunkedArray[Any]:
    for name in COLUMN_ALIASES[logical]:
        if name in table.column_names:
            return table[name]
    raise StageError(
        f"no column for {logical!r} (tried {COLUMN_ALIASES[logical]}); table has {table.column_names}"
    )


def _timestamps_us(col: pa.ChunkedArray[Any]) -> list[int]:
    """Any timestamp representation -> integer microseconds since epoch (UTC)."""
    values: Sequence[object]
    if pa.types.is_timestamp(col.type):
        values = pc.cast(pc.cast(col, pa.timestamp("us", tz="UTC")), pa.int64()).to_pylist()
    else:
        values = col.to_pylist()
    out: list[int] = []
    for v in values:
        if v is None:
            raise StageError("null timestamp in the export")
        if isinstance(v, int):
            out.append(v)
        else:
            dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            out.append(int(dt.timestamp() * 1_000_000))
    return out


def _filter_sequence(table: pa.Table, sequence_id: str) -> pa.Table:
    col = _column(table, "sequence_id")
    mask = pc.equal(col, pa.scalar(sequence_id))
    return table.filter(mask)


def _parse_payload(raw: object) -> dict[str, Any]:
    """The tabular payload is the method response as JSON: {"readings": {...frame record...}}."""
    data: object = raw
    if isinstance(raw, str | bytes):
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise StageError(f"tabular payload is not a JSON object: {type(raw).__name__}")
    readings = data.get("readings", data)
    if not isinstance(readings, dict):
        raise StageError("tabular payload has a non-object 'readings'")
    return readings


def _frame_from_readings(readings: dict[str, Any]) -> s.FrameRecord:
    try:
        return s.FrameRecord.from_dict(readings)
    except (KeyError, TypeError) as e:
        raise StageError(
            f"scan-state reading is not a FrameRecord ({e!r}); keys present: {sorted(readings.keys())}"
        ) from e


def camera_model_from_readings(readings: dict[str, Any]) -> s.CameraModel | None:
    cm = readings.get("camera_model")
    if not isinstance(cm, dict):
        return None
    known = {f for f in s.CameraModel.__dataclass_fields__}
    try:
        return s.CameraModel(**{k: v for k, v in cm.items() if k in known})
    except TypeError as e:
        raise StageError(f"camera_model in the scan-state reading is incomplete: {e}") from e


def _pair_rows(binary_ts: list[int], tabular_ts: list[int]) -> tuple[dict[int, int], int, list[int]]:
    """Binary row -> tabular row by timestamp. Exact within 1 ms, else nearest within 200 ms (warned)."""
    pairs: dict[int, int] = {}
    used: set[int] = set()
    fallbacks = 0
    unpaired: list[int] = []
    order = sorted(range(len(tabular_ts)), key=lambda j: tabular_ts[j])
    for i, ts in enumerate(binary_ts):
        best_j: int | None = None
        best_dt = NEAREST_TOLERANCE_US + 1
        for j in order:
            if j in used:
                continue
            dt = abs(tabular_ts[j] - ts)
            if dt < best_dt:
                best_dt, best_j = dt, j
        if best_j is None or best_dt > NEAREST_TOLERANCE_US:
            unpaired.append(i)
            continue
        if best_dt > EXACT_TOLERANCE_US:
            fallbacks += 1
            log.warning(
                "image at %d us paired to reading %.1f ms away (nearest-neighbour fallback)",
                ts,
                best_dt / 1000,
            )
        pairs[i] = best_j
        used.add(best_j)
    return pairs, fallbacks, unpaired


def _write_image_as_png(src: Path, dst: Path) -> None:
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
        return
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise StageError(f"could not decode image {src}")
    log.warning(
        "%s is %s, re-encoding to PNG (the source was already lossy)",
        src.name,
        src.suffix,
    )
    cv2.imwrite(str(dst), img)


def join_sequence(
    layout: ExportLayout,
    sequence_id: str,
    out_dir: Path,
    *,
    dataset_id: str | None = None,
    camera_override: s.CameraModel | None = None,
) -> JoinReport:
    binary = _filter_sequence(pq.read_table(layout.parquet_dir / "binary_data.parquet"), sequence_id)
    tabular = _filter_sequence(pq.read_table(layout.parquet_dir / "tabular_data.parquet"), sequence_id)
    if binary.num_rows == 0:
        raise StageError(f"sequence {sequence_id} has no images in binary_data.parquet")
    if tabular.num_rows == 0:
        raise StageError(f"sequence {sequence_id} has no readings in tabular_data.parquet")
    log.info("sequence %s: %d images, %d readings", sequence_id, binary.num_rows, tabular.num_rows)

    binary_ts = _timestamps_us(_column(binary, "timestamp"))
    tabular_ts = _timestamps_us(_column(tabular, "timestamp"))
    paths = [str(v) for v in _column(binary, "path").to_pylist()]
    payloads = _column(tabular, "payload").to_pylist()
    pairs, fallbacks, unpaired = _pair_rows(binary_ts, tabular_ts)
    if not pairs:
        raise StageError("no image could be paired with a scan-state reading by timestamp")
    for i in unpaired:
        log.warning("image %s has no reading within %d ms; dropped", paths[i], NEAREST_TOLERANCE_US // 1000)

    first_readings = _parse_payload(payloads[pairs[min(pairs)]])
    camera = camera_model_from_readings(first_readings)
    camera_source = "scan-state reading"
    if camera is None:
        camera = camera_override
        camera_source = "cli flags"
    if camera is None:
        raise StageError(
            "no camera model: the scan-state readings carry no 'camera_model' and no "
            "--fx/--fy/--cx/--cy/--width/--height were given"
        )

    images_dir = out_dir / s.IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / s.FRAMES_NAME
    if frames_path.exists():
        frames_path.unlink()
    frames: list[s.FrameRecord] = []
    for i in sorted(pairs, key=lambda k: binary_ts[k]):
        readings = _parse_payload(payloads[pairs[i]])
        frame = _frame_from_readings(readings)
        frame.file = f"{s.IMAGES_DIR}/{frame.index:04d}.png"
        src = layout.root / paths[i]
        if not src.exists():
            raise StageError(f"image file missing from the export: {src}")
        _write_image_as_png(src, out_dir / frame.file)
        frames.append(frame)
    frames.sort(key=lambda f: f.index)
    s.rewrite_frames(out_dir, frames)

    part_ids = (
        {str(v) for v in _column(binary, "part_id").to_pylist()}
        if any(n in binary.column_names for n in COLUMN_ALIASES["part_id"])
        else set()
    )
    components = (
        {str(v) for v in _column(binary, "component_name").to_pylist()}
        if any(n in binary.column_names for n in COLUMN_ALIASES["component_name"])
        else set()
    )
    manifest = s.Manifest(
        scan_id=str(first_readings.get("scan_id", sequence_id)),
        part_id=next(iter(part_ids)) if len(part_ids) == 1 else None,
        arm=str(first_readings.get("arm", "")),
        camera=next(iter(components)) if len(components) == 1 else "",
        camera_model=camera,
        state="done",
        sequence_id=sequence_id,
        dataset_id=dataset_id,
        resolved={k: first_readings[k] for k in ("num_poses", "min_sharpness") if k in first_readings},
    )
    s.write_manifest(out_dir, manifest)
    report = JoinReport(
        sequence_id=sequence_id,
        num_binary_rows=binary.num_rows,
        num_tabular_rows=tabular.num_rows,
        num_paired=len(frames),
        num_nearest_fallback=fallbacks,
        unpaired_images=[paths[i] for i in unpaired],
        camera_model_source=camera_source,
    )
    (out_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
    log.info("wrote %d frames to %s (camera model from %s)", len(frames), out_dir, camera_source)
    return report


def copy_scan_dir(scan_dir: Path, out_dir: Path, camera_override: s.CameraModel | None = None) -> int:
    """`--scan-dir`: use a local scan directory as the join output. Returns the number of frames."""
    frames = s.read_frames(scan_dir)
    if not frames:
        raise StageError(f"{scan_dir} has no {s.FRAMES_NAME}")
    if not (scan_dir / s.MANIFEST_NAME).exists():
        raise StageError(f"{scan_dir} has no {s.MANIFEST_NAME}")
    manifest = s.read_manifest(scan_dir)
    if manifest.camera_model is None:
        if camera_override is None:
            raise StageError(f"{scan_dir} manifest has no camera_model and no intrinsics given on the CLI")
        manifest.camera_model = camera_override
    (out_dir / s.IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    from gaussian_splatting.build.colmap_stage import link_or_copy

    for fr in frames:
        link_or_copy(scan_dir / fr.file, out_dir / fr.file)
    s.rewrite_frames(out_dir, frames)
    s.write_manifest(out_dir, manifest)
    log.info("copied %d frames from %s to %s", len(frames), scan_dir, out_dir)
    return len(frames)
