from datetime import UTC, datetime
from pathlib import Path

import pytest
from grpclib import GRPCError, Status

from gaussian_splatting import schema
from gaussian_splatting.module import uploader as up
from tests.fakes import FakeDataClient

SLEEPS: list[float] = []


async def no_sleep(s: float) -> None:
    SLEEPS.append(s)


def make_scan(tmp_path: Path, n: int = 3) -> Path:
    (tmp_path / "images").mkdir()
    schema.write_manifest(
        tmp_path,
        schema.Manifest(
            scan_id="scan-a",
            state="done",
            arm="lite6",
            camera_model=schema.CameraModel(640, 480, 820.0, 820.0, 320.0, 240.0),
            resolved={"num_poses": 3, "min_sharpness": 60.0},
        ),
    )
    for i in range(1, n + 1):
        (tmp_path / f"images/{i:04d}.png").write_bytes(b"png" + bytes([i]))
        schema.append_frame(
            tmp_path,
            schema.FrameRecord(
                index=i,
                file=f"images/{i:04d}.png",
                viewpoint=i - 1,
                burst=0,
                t_request=f"2026-11-03T14:02:{10 + i:02d}.000Z",
                t_capture=f"2026-11-03T14:02:{10 + i:02d}.100Z",
                T_base_cam=schema.Transform([0.3, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0]),
                arm_end_pose=schema.ArmEndPose(1, 2, 3, 0, 0, 1, 0),
                joints_deg=[0.0] * 6,
                sharpness=100.0,
                settle_s=0.5,
                width=640,
                height=480,
            ),
        )
    return tmp_path


def target(dataset_id: str | None = "ds-1") -> up.UploadTarget:
    return up.UploadTarget(
        part_id="part-1",
        camera_name="camera",
        scan_state_name="scan-state",
        dataset_id=dataset_id,
        tags=("cell-v1",),
    )


async def test_full_upload_creates_sequence_and_adds_to_dataset(tmp_path: Path) -> None:
    scan = make_scan(tmp_path)
    client = FakeDataClient()
    report = await up.Uploader(client, target(), sleep=no_sleep).upload_scan(scan)

    assert report.uploads_done == 3 and report.uploads_pending == 0
    assert report.sequence_id == "seq-7" and report.dataset_added and report.dataset_error is None
    assert client.count("binary") == 3 and client.count("tabular") == 3

    _, b = client.calls[0]
    assert b["component_type"] == "rdk:component:camera" and b["method_name"] == "GetImages"
    assert b["component_name"] == "camera" and b["file_extension"] == ".png" and b["mime_type"] == "image/png"
    assert b["binary_data"] == b"png\x01" and b["tags"] == ["scan-a", "splat", "cell-v1"]
    assert b["data_request_times"] == (
        datetime(2026, 11, 3, 14, 2, 11, tzinfo=UTC),
        datetime(2026, 11, 3, 14, 2, 11, 100000, tzinfo=UTC),
    )
    _, t = client.calls[1]
    assert t["component_type"] == "rdk:component:sensor" and t["method_name"] == "Readings"
    assert t["component_name"] == "scan-state" and t["data_request_times"] == [b["data_request_times"]]
    readings = t["tabular_data"][0]["readings"]
    assert readings["index"] == 1 and "uploaded" not in readings
    assert {"camera_model", "scan_id", "arm", "num_poses", "min_sharpness"} <= set(readings)
    assert readings["scan_id"] == "scan-a" and readings["arm"] == "lite6"
    assert readings["num_poses"] == 3 and readings["min_sharpness"] == 60.0
    assert readings["camera_model"]["fx"] == 820.0 and readings["camera_model"]["source"] == "camera"
    assert schema.FrameRecord.from_dict(readings).index == 1

    seq = next(kw for k, kw in client.calls if k == "sequence")
    assert seq["part_id"] == "part-1" and seq["sequence_tags"] == ["scan-a", "splat", "cell-v1"]
    assert [(r.resource_name, r.method_name) for r in seq["resources"]] == [
        ("camera", "GetImages"),
        ("scan-state", "Readings"),
    ]
    assert seq["start_time"] == datetime(2026, 11, 3, 14, 2, 10, tzinfo=UTC)
    assert seq["end_time"] == datetime(2026, 11, 3, 14, 2, 14, 100000, tzinfo=UTC)
    ds = next(kw for k, kw in client.calls if k == "dataset")
    assert ds == {"dataset_id": "ds-1", "sequence_ids": ["seq-7"]}

    frames = schema.read_frames(scan)
    assert all(f.uploaded.binary_id and f.uploaded.tabular_ok for f in frames)
    manifest = schema.read_manifest(scan)
    assert manifest.sequence_id == "seq-7" and manifest.dataset_added and manifest.dataset_id == "ds-1"


async def test_rerun_is_idempotent(tmp_path: Path) -> None:
    scan = make_scan(tmp_path)
    await up.Uploader(FakeDataClient(), target(), sleep=no_sleep).upload_scan(scan)
    client = FakeDataClient()
    report = await up.Uploader(client, target(), sleep=no_sleep).upload_scan(scan)
    assert client.calls == []
    assert report.uploads_done == 0 and report.uploads_skipped == 3 and report.sequence_id == "seq-7"


async def test_retries_with_backoff_then_persists(tmp_path: Path) -> None:
    SLEEPS.clear()
    scan = make_scan(tmp_path, n=1)
    client = FakeDataClient(fail={"binary": 2})
    report = await up.Uploader(client, target(None), sleep=no_sleep).upload_scan(scan)
    assert client.count("binary") == 3 and report.uploads_done == 1
    assert SLEEPS == [2.0, 5.0]
    assert report.sequence_id is not None and not report.dataset_added


async def test_gives_up_after_three_attempts_and_keeps_ledger_clean(tmp_path: Path) -> None:
    scan = make_scan(tmp_path, n=2)
    client = FakeDataClient(fail={"tabular": 3})
    with pytest.raises(up.UploadError):
        await up.Uploader(client, target(), sleep=no_sleep).upload_scan(scan)
    frames = schema.read_frames(scan)
    assert frames[0].uploaded.binary_id == "bin-1" and not frames[0].uploaded.tabular_ok
    assert frames[1].uploaded.binary_id is None
    # second run resumes: binary of frame 1 is not re-sent
    client2 = FakeDataClient()
    report = await up.Uploader(client2, target(), sleep=no_sleep).upload_scan(scan)
    assert client2.count("binary") == 1 and client2.count("tabular") == 2 and report.sequence_id is not None


async def test_permission_denied_on_dataset_keeps_sequence(tmp_path: Path) -> None:
    scan = make_scan(tmp_path, n=1)
    client = FakeDataClient(deny_dataset=True)
    report = await up.Uploader(client, target(), sleep=no_sleep).upload_scan(scan)
    assert report.sequence_id == "seq-3" and not report.dataset_added
    assert report.dataset_error is not None and "organization scope" in report.dataset_error
    assert client.count("dataset") == 1, "permission errors are not retried"
    manifest = schema.read_manifest(scan)
    assert manifest.sequence_id == "seq-3" and manifest.dataset_id == "ds-1" and not manifest.dataset_added


async def test_cancel_drains_current_frame_only(tmp_path: Path) -> None:
    scan = make_scan(tmp_path, n=3)
    client = FakeDataClient()
    uploader = up.Uploader(client, target(), sleep=no_sleep)
    report = await uploader.upload_scan(scan, is_cancelled=lambda: client.count("binary") >= 1)
    assert report.uploads_done == 1 and report.uploads_pending == 2 and report.sequence_id is None
    assert client.count("sequence") == 0


def test_credentials_from_env() -> None:
    assert up.CloudCredentials.from_env({}) is None
    assert up.CloudCredentials.from_env({"VIAM_API_KEY": "k", "VIAM_API_KEY_ID": "i"}) is None
    creds = up.CloudCredentials.from_env(
        {"VIAM_API_KEY": "k", "VIAM_API_KEY_ID": "i", "VIAM_MACHINE_PART_ID": "p"}
    )
    assert creds == up.CloudCredentials("k", "i", "p")


def test_permission_error_is_grpc_status() -> None:
    e = GRPCError(Status.PERMISSION_DENIED, "x")
    assert e.status == Status.PERMISSION_DENIED
