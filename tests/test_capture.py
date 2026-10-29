import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from viam.media.video import CameraMimeType, NamedImage

from gaussian_splatting import geometry, schema
from gaussian_splatting.module import capture as cap
from gaussian_splatting.module import planner as pl
from tests.fakes import (
    CAMERA_MODEL,
    T_FLANGE_CAM,
    FakeArm,
    FakeCamera,
    FakeMotion,
    FakeScanState,
    flat_image,
    sharp_image,
)

SLEEPS: list[float] = []


async def no_sleep(s: float) -> None:
    SLEEPS.append(s)


def viewpoint() -> pl.Viewpoint:
    plan = pl.plan_viewpoints(
        pl.PlannerParams(
            (250.0, 0.0, 0.0), (300.0, 300.0, 50.0), (150.0, 300.0), (30.0, 80.0), (-120.0, 120.0), 5
        ),
        CAMERA_MODEL,
        T_FLANGE_CAM,
    )
    return plan.viewpoints[0]


def make_deps(
    camera: FakeCamera | None = None, motion: FakeMotion | None = None, arm: FakeArm | None = None
) -> tuple[cap.CaptureDeps, FakeArm, FakeCamera, FakeMotion, FakeScanState]:
    arm = arm or FakeArm()
    camera = camera or FakeCamera(captured_at=datetime(2026, 11, 3, 14, 2, 11, 187000, tzinfo=UTC))
    motion = motion or FakeMotion(arm)
    state = FakeScanState()
    deps = cap.CaptureDeps(
        arm=arm,
        camera=camera,
        motion=motion,
        pose_reader=cap.FrameSystemPoseReader(motion, arm, camera.name),
        scan_state=state,
        logger=logging.getLogger("test"),
        sleep=no_sleep,
        clock=lambda: datetime(2026, 11, 3, 14, 2, 11, 120000, tzinfo=UTC),
    )
    return deps, arm, camera, motion, state


async def test_happy_path_writes_frame_and_updates_sensor(tmp_path: Path) -> None:
    deps, arm, camera, motion, state = make_deps()
    vp = viewpoint()
    out = await cap.capture_viewpoint(
        deps, cap.CaptureSettings(dwell_s=0.3, min_sharpness=10.0), vp, tmp_path, 7
    )

    assert out.skipped_reason is None and out.roll_offset_deg == 0.0
    assert len(out.frames) == 1
    fr = out.frames[0]
    assert fr.index == 7 and fr.file == "images/0007.png" and fr.viewpoint == vp.index and fr.burst == 0
    assert (tmp_path / fr.file).exists()
    assert fr.width == 640 and fr.height == 480
    assert fr.joints_deg == arm.joints
    assert fr.t_request == "2026-11-03T14:02:11.120Z" and fr.t_capture == "2026-11-03T14:02:11.187Z"
    assert fr.sharpness > 10.0
    assert len(motion.moves) == 1 and motion.moves[0].reference_frame == "arm_origin"
    # the pose written is the planned pose: the fake arm goes exactly where motion sends it
    T = geometry.make_T(
        geometry.quat_wxyz_to_rotation(np.array(fr.T_base_cam.q_wxyz)), np.array(fr.T_base_cam.t_m)
    )
    assert np.allclose(T, vp.T_base_cam(), atol=1e-6)
    assert schema.read_frames(tmp_path)[0] == fr
    assert state.frames[0]["index"] == 7
    assert 0.3 in SLEEPS  # dwell


async def test_roll_retry_then_skip(tmp_path: Path) -> None:
    arm = FakeArm()
    deps, _, _, motion, _ = make_deps(motion=FakeMotion(arm, fail_first=2), arm=arm)
    out = await cap.capture_viewpoint(deps, cap.CaptureSettings(min_sharpness=0.0), viewpoint(), tmp_path, 1)
    assert out.roll_offset_deg == -45.0 and len(motion.moves) == 3
    planned_z = viewpoint().T_base_cam()[:3, 2]
    achieved_z = geometry.pose_to_T(motion.moves[-1].pose)[:3, 2]
    assert np.allclose(planned_z, achieved_z), "roll retries keep the view direction"

    deps, _, _, motion, _ = make_deps(motion=FakeMotion(arm, always_fail=True), arm=arm)
    out = await cap.capture_viewpoint(deps, cap.CaptureSettings(), viewpoint(), tmp_path, 1)
    assert out.skipped_reason == "unreachable" and out.frames == [] and len(motion.moves) == 5


async def test_sharpness_retake_keeps_sharpest(tmp_path: Path) -> None:
    camera = FakeCamera(images=[flat_image(), flat_image(), sharp_image()])
    deps, _, camera, _, _ = make_deps(camera=camera)
    out = await cap.capture_viewpoint(deps, cap.CaptureSettings(min_sharpness=50.0), viewpoint(), tmp_path, 1)
    assert camera.calls == 3 and out.frames[0].sharpness > 50.0

    camera = FakeCamera(images=[flat_image()])
    deps, _, camera, _, _ = make_deps(camera=camera)
    out = await cap.capture_viewpoint(deps, cap.CaptureSettings(min_sharpness=50.0), viewpoint(), tmp_path, 1)
    assert camera.calls == 3, "gives up after 2 retakes and keeps the best frame"
    assert out.frames[0].sharpness < 1.0


async def test_burst_keep_all(tmp_path: Path) -> None:
    deps, *_ = make_deps()
    settings = cap.CaptureSettings(min_sharpness=0.0, images_per_pose=2, keep_all=True)
    out = await cap.capture_viewpoint(deps, settings, viewpoint(), tmp_path, 3)
    assert [(f.index, f.burst) for f in out.frames] == [(3, 0), (4, 1)]
    assert len(schema.read_frames(tmp_path)) == 2


async def test_camera_retry_wrapper(tmp_path: Path) -> None:
    deps, _, camera, _, _ = make_deps(camera=FakeCamera(fail_first=2))
    out = await cap.capture_viewpoint(deps, cap.CaptureSettings(min_sharpness=0.0), viewpoint(), tmp_path, 1)
    assert camera.calls == 3 and len(out.frames) == 1

    deps, _, camera, _, _ = make_deps(camera=FakeCamera(fail_first=3))
    with pytest.raises(ConnectionError):
        await cap.capture_viewpoint(deps, cap.CaptureSettings(), viewpoint(), tmp_path, 1)


async def test_pose_drift_triggers_one_retry(tmp_path: Path) -> None:
    class DriftingReader:
        def __init__(self, inner: cap.CameraPoseReader) -> None:
            self.inner = inner
            self.reads = 0

        async def read(self) -> tuple[geometry.Mat, object]:
            self.reads += 1
            T, pose = await self.inner.read()
            if self.reads == 2:  # second read of the first attempt: 1 mm off
                T = T.copy()
                T[0, 3] += 0.001
            return T, pose

    deps, *_ = make_deps()
    reader = DriftingReader(deps.pose_reader)
    deps.pose_reader = reader  # type: ignore[assignment]
    out = await cap.capture_viewpoint(deps, cap.CaptureSettings(min_sharpness=0.0), viewpoint(), tmp_path, 1)
    assert reader.reads == 4 and len(out.frames) == 1


def test_decode_png_jpeg_and_rgba() -> None:
    img = sharp_image(8, 6)
    png = NamedImage(
        "c", __import__("tests.fakes", fromlist=["encode_png"]).encode_png(img), CameraMimeType.PNG
    )
    assert np.array_equal(cap.decode_image(png), img)  # type: ignore[arg-type]
    rgba_bytes = b"RGBA" + (8).to_bytes(4, "big") + (6).to_bytes(4, "big") + img[..., ::-1].tobytes()
    assert (
        cap.decode_image(NamedImage("c", rgba_bytes, CameraMimeType.VIAM_RGBA)) is None
    )  # 3 channels, not 4
    import cv2

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).tobytes()
    decoded = cap.decode_image(
        NamedImage(
            "c", b"RGBA" + (8).to_bytes(4, "big") + (6).to_bytes(4, "big") + rgba, CameraMimeType.VIAM_RGBA
        )
    )
    assert decoded is not None and np.array_equal(decoded, img)
    assert cap.decode_image(NamedImage("d", b"\x00\x01", CameraMimeType.VIAM_RAW_DEPTH)) is None
    assert cap.decode_image(NamedImage("x", b"not an image", CameraMimeType.JPEG)) is None


def test_fallback_pose_reader_composes_camera_in_arm() -> None:
    arm = FakeArm()
    reader = cap.FallbackPoseReader(arm, T_FLANGE_CAM)
    import asyncio

    T, pose = asyncio.run(reader.read())
    assert np.allclose(T, arm.T_base_flange @ T_FLANGE_CAM)
    assert pose.z == pytest.approx(154.0)
