"""Smoke test against a local viam-server running dev/local-machine.json (fake Lite 6, fake camera).

viam-server -config dev/local-machine.json &
uv run pytest -q -m integration
"""

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from viam.components.sensor import Sensor
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.services.generic import Generic

from gaussian_splatting import geometry, schema

pytestmark = pytest.mark.integration
ADDRESS = "localhost:8080"


async def connect() -> RobotClient:
    opts = RobotClient.Options(dial_options=DialOptions(insecure=True, disable_webrtc=True))
    return await RobotClient.at_address(ADDRESS, opts)


async def call(scanner: Generic, command: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], dict(await scanner.do_command(command)))


async def wait_done(scanner: Generic, timeout_s: float = 120.0) -> dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        st = await call(scanner, {"scan_status": {}})
        if st["state"] in ("done", "failed", "cancelled"):
            return st
        assert asyncio.get_event_loop().time() < deadline, st
        await asyncio.sleep(0.5)


async def test_dry_run_and_three_pose_scan() -> None:
    machine = await connect()
    try:
        scanner = Generic.from_robot(machine, "scanner")
        sensor = Sensor.from_robot(machine, "scan-state")

        dry = await call(scanner, {"start_scan": {"num_poses": 5, "dry_run": True}})
        assert dry["dry_run"] is True and dry["planned"] >= 5
        assert dry["camera_model"]["source"] == "camera" and dry["camera_in_arm_source"] == "frame_system"
        stats = dry["stats"]
        assert 0.0 < stats["reachable_window_fraction"] <= 1.0 and stats["top_face_coverage"] > 0.0

        scan_id = f"it-{int(asyncio.get_event_loop().time() * 1000)}"
        resp = await call(
            scanner, {"start_scan": {"scan_id": scan_id, "num_poses": 3, "upload": False, "dwell_s": 0.1}}
        )
        assert resp["state"] in ("planning", "capturing")
        st = await wait_done(scanner)
        assert st["state"] == "done", st
        assert st["poses_done"] == 3 and st["images_saved"] == 3, st

        scan_dir = Path(str(st["scan_dir"]))
        frames = schema.read_frames(scan_dir)
        manifest = schema.read_manifest(scan_dir)
        assert len(frames) == 3 and manifest.state == "done"
        assert manifest.camera_model is not None and manifest.camera_model.colmap_model == "FULL_OPENCV"
        planned = {int(v["index"]): v for v in manifest.planned_viewpoints}
        for fr in frames:
            assert (scan_dir / fr.file).stat().st_size > 0
            assert fr.width == 640 and fr.height == 480 and fr.sharpness >= 0.5
            assert fr.t_capture >= fr.t_request[:19]  # captured_at comes from the fake camera
            vp = planned[fr.viewpoint]
            T_planned = geometry.make_T(
                geometry.quat_wxyz_to_rotation(np.array(vp["q_wxyz"])), np.array(vp["t_m"])
            )
            T_got = geometry.make_T(
                geometry.quat_wxyz_to_rotation(np.array(fr.T_base_cam.q_wxyz)), np.array(fr.T_base_cam.t_m)
            )
            assert np.linalg.norm(T_planned[:3, 3] - T_got[:3, 3]) < 0.001, "camera within 1 mm of the plan"
            assert np.dot(T_planned[:3, 2], T_got[:3, 2]) > np.cos(np.radians(0.5)), (
                "view direction within 0.5 deg"
            )

        readings = cast(dict[str, Any], dict(await sensor.get_readings()))
        assert int(readings["index"]) == 3 and readings["scan_id"] == scan_id
        assert readings["camera_model"]["source"] == "camera" and int(readings["num_poses"]) == 3
        assert schema.FrameRecord.from_dict(readings).index == 3
        print(json.dumps(st, indent=2))
    finally:
        await machine.close()
