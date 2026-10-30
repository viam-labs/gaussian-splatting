import asyncio
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from viam.proto.common import PoseInFrame

from gaussian_splatting import schema
from gaussian_splatting.module import config as c
from gaussian_splatting.module.scanner import Scanner, ScanRefused
from tests.fakes import FakeArm, FakeCamera, FakeMotion, FakeScanState

ATTRS: dict[str, Any] = {
    "arm": "arm",
    "camera": "camera",
    "motion": "builtin",
    "scan_state": "scan-state",
    "region_mm": {"center": [250.0, 0.0, 0.0], "size": [300.0, 300.0, 50.0]},
    "standoff_mm": [150.0, 300.0],
    "elevation_deg": [30.0, 80.0],
    "azimuth_deg": [-120.0, 120.0],
    "dwell_s": 0.0,
    "min_sharpness": 1.0,
    "camera_lock_command": {"set_exposure_auto": False},
}


def make_scanner(
    tmp_path: Path, attrs: dict[str, Any] | None = None, camera: FakeCamera | None = None
) -> Scanner:
    arm = FakeArm(moving_polls=0)
    scanner = Scanner("scanner")
    scanner.bind(
        c.parse_scanner_config(attrs or ATTRS),
        arm=arm,
        camera=camera or FakeCamera(),
        motion=FakeMotion(arm),
        scan_state=FakeScanState(),
    )
    scanner.scans_root = tmp_path / "scans"
    return scanner


async def call(scanner: Scanner, command: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], dict(await scanner.do_command(command)))


async def wait_done(scanner: Scanner, timeout_s: float = 10.0) -> dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        st = await call(scanner, {"scan_status": {}})
        if st["state"] in ("done", "failed", "cancelled"):
            return st
        assert asyncio.get_event_loop().time() < deadline, st
        await asyncio.sleep(0.01)


async def test_start_status_done(tmp_path: Path) -> None:
    scanner = make_scanner(tmp_path)
    assert await scanner.do_command({"scan_status": {}}) == {"state": "idle"}

    resp = await call(scanner, {"start_scan": {"scan_id": "s1", "num_poses": 3}})
    assert resp["state"] in ("planning", "capturing") and resp["planned"] == 3 and resp["candidates"] >= 3
    scan_dir = Path(str(resp["scan_dir"]))
    assert scan_dir == tmp_path / "scans" / "s1"
    with pytest.raises(ValueError, match="one scan at a time"):
        await scanner.do_command({"start_scan": {"scan_id": "s2"}})

    st = await wait_done(scanner)
    assert (
        st["state"] == "done"
        and st["poses_done"] == 3
        and st["images_saved"] == 3
        and st["poses_skipped"] == 0
    )
    assert st["uploads_pending"] == 3 and "sequence_id" not in st and "last_error" not in st
    frames = schema.read_frames(scan_dir)
    assert [f.index for f in frames] == [1, 2, 3] and [f.viewpoint for f in frames] == [0, 1, 2]
    manifest = schema.read_manifest(scan_dir)
    assert manifest.state == "done" and manifest.finished_at and len(manifest.planned_viewpoints) >= 3
    assert manifest.camera_model is not None and manifest.camera_model.source == "camera"
    assert (
        manifest.camera_model.k1 == 0.1
        and manifest.camera_model.k3 == 0.3
        and manifest.camera_model.p1 == 0.001
    )
    assert manifest.camera_in_arm_source == "frame_system" and manifest.camera_in_arm is not None
    assert np.allclose(manifest.camera_in_arm.t_m, [0, 0, 0.05], atol=1e-9)
    camera = scanner.camera
    assert isinstance(camera, FakeCamera) and camera.commands == [{"set_exposure_auto": False}]
    state = scanner.scan_state
    assert isinstance(state, FakeScanState) and len(state.frames) == 3
    payload = state.frames[-1]
    assert {"camera_model", "scan_id", "arm", "num_poses", "min_sharpness"} <= set(payload)
    assert "uploaded" not in payload
    assert schema.FrameRecord.from_dict(dict(payload)).index == 3
    assert payload["scan_id"] == "s1" and payload["arm"] == "arm" and payload["num_poses"] == 3
    assert payload["min_sharpness"] == 1.0 and payload["camera_model"]["source"] == "camera"
    await scanner.close()


async def test_dry_run_moves_nothing(tmp_path: Path) -> None:
    scanner = make_scanner(tmp_path)
    resp = await call(scanner, {"start_scan": {"num_poses": 5, "dry_run": True}})
    assert resp["dry_run"] is True and resp["planned"] >= 5 and len(resp["viewpoints"]) == resp["planned"]
    assert "top_face_coverage" in resp["stats"]
    assert not (tmp_path / "scans").exists()
    assert (
        scanner.motion is not None and isinstance(scanner.motion, FakeMotion) and scanner.motion.moves == []
    )
    assert await scanner.do_command({"scan_status": {}}) == {"state": "idle"}


async def test_cancel_stops_after_current_viewpoint(tmp_path: Path) -> None:
    scanner = make_scanner(tmp_path)
    arm = scanner.arm
    assert isinstance(arm, FakeArm)
    arm.moving_polls = 3  # a few polls per move so the loop yields and cancel lands mid-scan
    await scanner.do_command({"start_scan": {"scan_id": "s", "num_poses": 50}})
    await asyncio.sleep(0.05)
    resp = await call(scanner, {"cancel_scan": {}})
    assert resp["cancelled"] is True
    st = await wait_done(scanner)
    assert st["state"] == "cancelled" and 0 < st["poses_done"] < 50
    assert schema.read_manifest(tmp_path / "scans" / "s").state == "cancelled"
    assert (await call(scanner, {"cancel_scan": {}}))["cancelled"] is False


async def test_resume_skips_captured_viewpoints(tmp_path: Path) -> None:
    scanner = make_scanner(tmp_path)
    resp = await call(scanner, {"start_scan": {"scan_id": "r", "num_poses": 2, "seed": 9}})
    await wait_done(scanner)
    planned = schema.read_manifest(tmp_path / "scans" / "r").planned_viewpoints
    # resume never resamples, so the target is capped by the candidates the original plan had
    target = min(4, int(resp["candidates"]))
    assert target > 2

    with pytest.raises(ValueError, match="resume"):
        await scanner.do_command({"start_scan": {"scan_id": "r"}})
    await scanner.do_command({"start_scan": {"scan_id": "r", "num_poses": target, "resume": True}})
    st = await wait_done(scanner)
    assert st["poses_done"] == target and st["images_saved"] == target
    frames = schema.read_frames(tmp_path / "scans" / "r")
    assert [f.viewpoint for f in frames] == list(range(target))
    assert [f.index for f in frames] == list(range(1, target + 1))
    assert schema.read_manifest(tmp_path / "scans" / "r").planned_viewpoints == planned


async def test_upload_requested_without_credentials_is_a_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k in ("VIAM_API_KEY", "VIAM_API_KEY_ID", "VIAM_MACHINE_PART_ID"):
        monkeypatch.delenv(k, raising=False)
    scanner = make_scanner(tmp_path)
    resp = await call(scanner, {"start_scan": {"num_poses": 1, "upload": True}})
    assert resp["upload"] is False and "VIAM_API_KEY" in str(resp["note"])
    st = await wait_done(scanner)
    assert st["state"] == "done" and "upload disabled" in str(st["last_error"])


async def test_refuses_without_intrinsics_or_camera_frame(tmp_path: Path) -> None:
    scanner = make_scanner(tmp_path, camera=FakeCamera(with_intrinsics=False))
    with pytest.raises(ScanRefused, match="camera_intrinsics"):
        await scanner.do_command({"start_scan": {"dry_run": True}})

    attrs = {**ATTRS, "camera_intrinsics": {"K": {"fx": 800, "fy": 800, "cx": 320, "cy": 240}}}
    scanner = make_scanner(tmp_path, attrs, camera=FakeCamera(with_intrinsics=False))
    with pytest.raises(ScanRefused, match="width and height"):
        await scanner.do_command({"start_scan": {"dry_run": True}})

    attrs["camera_intrinsics"] = {**attrs["camera_intrinsics"], "width": 640, "height": 480}
    scanner = make_scanner(tmp_path, attrs, camera=FakeCamera(with_intrinsics=False))
    resp = await call(scanner, {"start_scan": {"dry_run": True}})
    assert resp["camera_model"]["source"] == "config"

    class NoFrame(FakeMotion):
        async def get_pose(self, component_name: str, destination_frame: str) -> PoseInFrame:
            raise RuntimeError("frame not found")

    scanner = make_scanner(tmp_path)
    assert isinstance(scanner.arm, FakeArm)
    scanner.motion = NoFrame(scanner.arm)
    with pytest.raises(ScanRefused, match="camera_in_arm"):
        await scanner.do_command({"start_scan": {"dry_run": True}})

    attrs = {**ATTRS, "camera_in_arm": {"translation": {"x": 0, "y": 0, "z": 50}}}
    scanner = make_scanner(tmp_path, attrs)
    scanner.motion = NoFrame(scanner.arm)  # type: ignore[arg-type]
    resp = await call(scanner, {"start_scan": {"dry_run": True}})
    assert resp["camera_in_arm_source"] == "config"


async def test_unknown_command(tmp_path: Path) -> None:
    scanner = make_scanner(tmp_path)
    with pytest.raises(ValueError, match="unknown command"):
        await scanner.do_command({"fly": {}})


async def test_prune_removes_old_finished_scans(tmp_path: Path) -> None:
    scanner = make_scanner(tmp_path)
    root = tmp_path / "scans"
    for name, finished in (("old", "2020-01-01T00:00:00.000Z"), ("new", "2999-01-01T00:00:00.000Z")):
        (root / name).mkdir(parents=True)
        schema.write_manifest(root / name, schema.Manifest(scan_id=name, state="done", finished_at=finished))
    resp = await call(scanner, {"prune": {"days": 7}})
    assert resp["removed"] == ["old"] and not (root / "old").exists() and (root / "new").exists()


async def test_upload_results_persist_in_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: the uploader edits its own manifest copy; the job's final write must not clobber it.
    from gaussian_splatting.module import scanner as scanner_module
    from gaussian_splatting.module.uploader import DataClientHandle
    from tests.fakes import FakeDataClient

    for k, v in (("VIAM_API_KEY", "k"), ("VIAM_API_KEY_ID", "id"), ("VIAM_MACHINE_PART_ID", "part")):
        monkeypatch.setenv(k, v)
    fake = FakeDataClient()

    async def opener(creds: object) -> DataClientHandle:
        return DataClientHandle(client=fake, close=lambda: None)

    monkeypatch.setattr(scanner_module, "open_data_client", opener)
    scanner = make_scanner(tmp_path)
    await call(
        scanner, {"start_scan": {"num_poses": 2, "upload": True, "dataset_id": "ds-1", "scan_id": "u"}}
    )
    st = await wait_done(scanner)
    assert st["state"] == "done" and st["sequence_id"]
    manifest = schema.read_manifest(tmp_path / "scans" / "u")
    assert manifest.sequence_id == st["sequence_id"]
    assert manifest.dataset_added is True and manifest.dataset_id == "ds-1"
