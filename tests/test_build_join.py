"""join stage against a synthetic parquet export (documented columns) built from the synthetic scene."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gaussian_splatting import schema as s
from gaussian_splatting.build import join
from gaussian_splatting.build.export import ExportLayout, unpack_parquet
from gaussian_splatting.build.stages import StageError
from tests.synthetic_scene import SyntheticScene, build_scene

SEQ = "seq-123"
OTHER_SEQ = "seq-other"
PART = "part-abc"


def _us(rfc3339: str) -> int:
    return int(datetime.fromisoformat(rfc3339.replace("Z", "+00:00")).timestamp() * 1_000_000)


def write_export(
    scene: SyntheticScene,
    export_root: Path,
    *,
    with_camera_model: bool = True,
    tabular_offset_us: int = 0,
    as_zip: bool = False,
) -> ExportLayout:
    """A sequence-dataset export as `viam dataset export` lays it out: parquet tables + binary_data/<id>."""
    export_root.mkdir(parents=True)
    bin_rows: list[dict[str, object]] = []
    tab_rows: list[dict[str, object]] = []
    for fr in scene.frames:
        blob_id = f"org/loc/{fr.index:04d}-blob"
        rel = Path("binary_data") / f"{blob_id}.png"
        (export_root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scene.scan_dir / fr.file, export_root / rel)
        ts = _us(fr.t_capture)
        bin_rows.append(
            {
                "sequence_id": SEQ,
                "timestamp": ts,
                "part_id": PART,
                "component_name": "wrist",
                "method_name": "GetImages",
                "path": str(rel),
            }
        )
        readings = fr.to_dict()
        readings["scan_id"] = scene.scan_dir.name
        readings["num_poses"] = len(scene.frames)
        if with_camera_model:
            readings["camera_model"] = s.CameraModel.__dataclass_fields__ and {
                k: getattr(scene.camera, k) for k in s.CameraModel.__dataclass_fields__
            }
        tab_rows.append(
            {
                "sequence_id": SEQ,
                "timestamp": ts + tabular_offset_us,
                "part_id": PART,
                "component_name": "scan-state",
                "method_name": "Readings",
                "payload": json.dumps({"readings": readings}),
            }
        )
    # A second sequence that must be ignored.
    bin_rows.append({**bin_rows[0], "sequence_id": OTHER_SEQ, "timestamp": 1})
    tab_rows.append({**tab_rows[0], "sequence_id": OTHER_SEQ, "timestamp": 1})

    ts_type = pa.timestamp("us", tz="UTC")
    binary = pa.table(
        {
            "sequence_id": pa.array([r["sequence_id"] for r in bin_rows], pa.string()),
            "timestamp": pa.array([r["timestamp"] for r in bin_rows], pa.int64()).cast(ts_type),
            "part_id": pa.array([r["part_id"] for r in bin_rows], pa.string()),
            "component_name": pa.array([r["component_name"] for r in bin_rows], pa.string()),
            "method_name": pa.array([r["method_name"] for r in bin_rows], pa.string()),
            "path": pa.array([r["path"] for r in bin_rows], pa.string()),
            "classifications": pa.array(
                [[] for _ in bin_rows], pa.list_(pa.struct([("label", pa.string())]))
            ),
        }
    )
    tabular = pa.table(
        {
            "sequence_id": pa.array([r["sequence_id"] for r in tab_rows], pa.string()),
            "timestamp": pa.array([r["timestamp"] for r in tab_rows], pa.int64()).cast(ts_type),
            "part_id": pa.array([r["part_id"] for r in tab_rows], pa.string()),
            "component_name": pa.array([r["component_name"] for r in tab_rows], pa.string()),
            "method_name": pa.array([r["method_name"] for r in tab_rows], pa.string()),
            "payload": pa.array([r["payload"] for r in tab_rows], pa.string()),
        }
    )
    sequences = pa.table(
        {
            "sequence_id": pa.array([SEQ, OTHER_SEQ], pa.string()),
            "tags": pa.array([["splat"], []], pa.list_(pa.string())),
            "start_at": pa.array([0, 0], pa.int64()).cast(ts_type),
            "end_at": pa.array([10**15, 1], pa.int64()).cast(ts_type),
        }
    )
    if as_zip:
        import zipfile

        tmp = export_root / "_tables"
        tmp.mkdir()
        for name, table in (("binary_data", binary), ("tabular_data", tabular), ("sequences", sequences)):
            pq.write_table(table, tmp / f"{name}.parquet")
        with zipfile.ZipFile(export_root / "dataset-xyz.zip", "w") as zf:
            for p in tmp.iterdir():
                zf.write(p, arcname=p.name)
        shutil.rmtree(tmp)
    else:
        pq.write_table(binary, export_root / "binary_data.parquet")
        pq.write_table(tabular, export_root / "tabular_data.parquet")
        pq.write_table(sequences, export_root / "sequences.parquet")
    return ExportLayout(root=export_root, parquet_dir=unpack_parquet(export_root))


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return build_scene(tmp_path_factory.mktemp("scene") / "scan", num_views=6, width=160, height=120)


def test_join_rebuilds_the_scan_dir(scene: SyntheticScene, tmp_path: Path) -> None:
    layout = write_export(scene, tmp_path / "export", as_zip=True)
    out = tmp_path / "join"
    report = join.join_sequence(layout, SEQ, out, dataset_id="ds-1")
    assert report.num_paired == len(scene.frames) and report.num_nearest_fallback == 0
    assert report.unpaired_images == []
    assert report.camera_model_source == "scan-state reading"

    rebuilt = s.read_frames(out)
    original = s.read_frames(scene.scan_dir)
    assert [f.to_dict() for f in rebuilt] == [f.to_dict() for f in original]
    for fr in original:
        assert (out / fr.file).read_bytes() == (scene.scan_dir / fr.file).read_bytes()
    manifest = s.read_manifest(out)
    assert manifest.camera_model == scene.camera
    assert manifest.sequence_id == SEQ and manifest.dataset_id == "ds-1"
    assert manifest.part_id == PART and manifest.camera == "wrist"
    assert manifest.scan_id == scene.scan_dir.name
    assert manifest.resolved["num_poses"] == len(scene.frames)


def test_join_nearest_fallback_and_camera_override(scene: SyntheticScene, tmp_path: Path) -> None:
    layout = write_export(scene, tmp_path / "export", with_camera_model=False, tabular_offset_us=50_000)
    with pytest.raises(StageError, match="no camera model"):
        join.join_sequence(layout, SEQ, tmp_path / "join-fail")
    report = join.join_sequence(layout, SEQ, tmp_path / "join", camera_override=scene.camera)
    assert report.num_paired == len(scene.frames)
    assert report.num_nearest_fallback == len(scene.frames)
    assert report.camera_model_source == "cli flags"
    assert s.read_manifest(tmp_path / "join").camera_model == scene.camera


def test_join_drops_images_without_a_reading(scene: SyntheticScene, tmp_path: Path) -> None:
    layout = write_export(scene, tmp_path / "export", tabular_offset_us=5_000_000)
    with pytest.raises(StageError, match="no image could be paired"):
        join.join_sequence(layout, SEQ, tmp_path / "join")


def test_join_unknown_sequence(scene: SyntheticScene, tmp_path: Path) -> None:
    layout = write_export(scene, tmp_path / "export")
    with pytest.raises(StageError, match="has no images"):
        join.join_sequence(layout, "nope", tmp_path / "join")


def test_copy_scan_dir(scene: SyntheticScene, tmp_path: Path) -> None:
    out = tmp_path / "join"
    n = join.copy_scan_dir(scene.scan_dir, out)
    assert n == len(scene.frames)
    assert [f.to_dict() for f in s.read_frames(out)] == [f.to_dict() for f in scene.frames]
    assert s.read_manifest(out).camera_model == scene.camera
