"""Capture at one viewpoint (spec section 8): move, settle, pose-image-pose, sharpness check, write the frame.

Dependencies are typed as small Protocols rather than the SDK's abstract classes so the whole sequence runs
in unit tests against fakes with a dozen lines each; the real `Arm`, `Camera`, `Motion` and `Sensor` clients
satisfy them structurally.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.common import Pose, PoseInFrame, ResponseMetadata
from viam.proto.component.arm import JointPositions

from gaussian_splatting import geometry, schema
from gaussian_splatting.geometry import Mat
from gaussian_splatting.module.planner import Viewpoint

RGBA_MAGIC = b"RGBA"
RGBA_HEADER_LENGTH = 12


class ArmLike(Protocol):
    name: str

    async def is_moving(self) -> bool: ...
    async def get_end_position(self) -> Pose: ...
    async def get_joint_positions(self) -> JointPositions: ...


class CameraLike(Protocol):
    name: str

    async def get_images(self) -> tuple[Sequence[NamedImage], ResponseMetadata]: ...
    async def do_command(self, command: Mapping[str, Any]) -> Mapping[str, Any]: ...


class MotionLike(Protocol):
    async def move(self, component_name: str, destination: PoseInFrame) -> bool: ...
    async def get_pose(self, component_name: str, destination_frame: str) -> PoseInFrame: ...


class FrameSink(Protocol):
    """The scan-state sensor, or anything else that accepts `set_frame`."""

    async def do_command(self, command: Mapping[str, Any]) -> Mapping[str, Any]: ...


class CameraPoseReader(Protocol):
    async def read(self) -> tuple[Mat, Pose]:
        """Returns (T_base_cam in metres, arm end pose as reported by the arm)."""
        ...


class FrameSystemPoseReader:
    """Camera pose straight from the frame system: `get_pose(camera, "<arm>_origin")`."""

    def __init__(self, motion: MotionLike, arm: ArmLike, camera_name: str) -> None:
        self.motion = motion
        self.arm = arm
        self.camera_name = camera_name

    async def read(self) -> tuple[Mat, Pose]:
        pif = await self.motion.get_pose(
            component_name=self.camera_name, destination_frame=f"{self.arm.name}_origin"
        )
        end = await self.arm.get_end_position()
        return geometry.pose_to_T(pif.pose), end


class FallbackPoseReader:
    """No camera frame on the machine: compose the configured `camera_in_arm` with the arm's end pose."""

    def __init__(self, arm: ArmLike, T_flange_cam: Mat) -> None:
        self.arm = arm
        self.T_flange_cam = T_flange_cam

    async def read(self) -> tuple[Mat, Pose]:
        end = await self.arm.get_end_position()
        return geometry.pose_to_T(end) @ self.T_flange_cam, end


@dataclass
class CaptureSettings:
    dwell_s: float = 0.5
    min_sharpness: float = 60.0
    images_per_pose: int = 1
    keep_all: bool = False
    sharpness_retakes: int = 2
    settle_timeout_s: float = 60.0
    poll_s: float = 0.05
    camera_retry_attempts: int = 3
    camera_retry_sleep_s: float = 0.5
    pose_tolerance_mm: float = 0.2
    pose_tolerance_deg: float = 0.05
    roll_retries_deg: tuple[float, ...] = (45.0, -45.0, 90.0, -90.0)


@dataclass
class CaptureDeps:
    arm: ArmLike
    camera: CameraLike
    motion: MotionLike
    pose_reader: CameraPoseReader
    scan_state: FrameSink | None = None
    frame_extras: Mapping[str, Any] = field(default_factory=dict)  # schema.frame_extras(manifest)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)


@dataclass
class Shot:
    image_bgr: NDArray[np.uint8]
    sharpness: float
    t_request: str
    t_capture: str


@dataclass
class CaptureOutcome:
    frames: list[schema.FrameRecord]
    skipped_reason: str | None = None
    roll_offset_deg: float = 0.0


async def with_retry[T](
    logger: logging.Logger,
    what: str,
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    sleep_s: float = 0.5,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """USB cameras drop out during arm motion; a few retries beat failing the viewpoint."""
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as e:
            if attempt >= attempts:
                raise
            logger.warning("%s failed (attempt %d/%d): %s; retrying", what, attempt, attempts, e)
            await sleep(sleep_s)
    raise AssertionError("unreachable")


def decode_image(img: NamedImage) -> NDArray[np.uint8] | None:
    """JPEG/PNG via OpenCV; Viam raw RGBA by hand (magic, width, height, big-endian). None if undecodable."""
    mime = CameraMimeType.from_string(str(img.mime_type))
    data = img.data
    if mime == CameraMimeType.VIAM_RGBA or data[:4] == RGBA_MAGIC:
        if len(data) < RGBA_HEADER_LENGTH:
            return None
        width = int.from_bytes(data[4:8], "big")
        height = int.from_bytes(data[8:12], "big")
        pixels = np.frombuffer(data, dtype=np.uint8, offset=RGBA_HEADER_LENGTH)
        if pixels.size != width * height * 4:
            return None
        rgba = pixels.reshape(height, width, 4)
        return np.asarray(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), dtype=np.uint8)
    if mime in (CameraMimeType.PCD, CameraMimeType.VIAM_RAW_DEPTH):
        return None
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        return None
    return np.asarray(decoded, dtype=np.uint8)


def sharpness(image_bgr: NDArray[np.uint8]) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def transform_from_T(T: Mat) -> schema.Transform:
    return schema.Transform(
        t_m=[float(v) for v in T[:3, 3]],
        q_wxyz=[float(v) for v in geometry.rotation_to_quat_wxyz(T[:3, :3])],
    )


def arm_end_pose_record(pose: Pose) -> schema.ArmEndPose:
    return schema.ArmEndPose(
        x_mm=pose.x, y_mm=pose.y, z_mm=pose.z, o_x=pose.o_x, o_y=pose.o_y, o_z=pose.o_z, theta_deg=pose.theta
    )


def poses_agree(T_a: Mat, T_b: Mat, settings: CaptureSettings) -> bool:
    dt_mm = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])) * 1000.0
    dr_deg = geometry.rotation_angle_deg(T_a[:3, :3], T_b[:3, :3])
    return dt_mm <= settings.pose_tolerance_mm and dr_deg <= settings.pose_tolerance_deg


def rolled(T_base_cam: Mat, roll_deg: float) -> Mat:
    """Rotate the camera about its own optical axis; the view direction is unchanged."""
    return T_base_cam @ geometry.make_T(geometry.rot_z(math.radians(roll_deg)), np.zeros(3))


async def move_to_viewpoint(deps: CaptureDeps, settings: CaptureSettings, vp: Viewpoint) -> float | None:
    """Roll offset (deg) that planned, or None when every roll failed. Roll is free for a splat."""
    base = vp.T_base_cam()
    frame = f"{deps.arm.name}_origin"
    for roll_deg in (0.0, *settings.roll_retries_deg):
        destination = PoseInFrame(reference_frame=frame, pose=geometry.T_to_pose(rolled(base, roll_deg)))
        try:
            ok = await deps.motion.move(component_name=deps.camera.name, destination=destination)
        except Exception as e:
            deps.logger.warning("viewpoint %d: move failed at roll %+.0f: %s", vp.index, roll_deg, e)
            ok = False
        if ok:
            return roll_deg
    return None


async def wait_until_still(deps: CaptureDeps, settings: CaptureSettings) -> None:
    deadline = time.monotonic() + settings.settle_timeout_s
    while await deps.arm.is_moving():
        if time.monotonic() > deadline:
            raise TimeoutError(f"arm still moving after {settings.settle_timeout_s} s")
        await deps.sleep(settings.poll_s)
    await deps.sleep(settings.dwell_s)


async def take_shot(deps: CaptureDeps, settings: CaptureSettings) -> Shot:
    t_request = deps.clock()
    images, metadata = await with_retry(
        deps.logger,
        "camera get_images",
        deps.camera.get_images,
        attempts=settings.camera_retry_attempts,
        sleep_s=settings.camera_retry_sleep_s,
        sleep=deps.sleep,
    )
    t_after = deps.clock()
    decoded = next((d for d in (decode_image(img) for img in images) if d is not None), None)
    if decoded is None:
        raise ValueError(f"no decodable colour image among {[str(i.mime_type) for i in images]}")
    if metadata.HasField("captured_at") and metadata.captured_at.seconds > 0:
        t_capture = metadata.captured_at.ToDatetime(tzinfo=UTC)
    else:
        t_capture = t_after
    return Shot(
        image_bgr=decoded,
        sharpness=sharpness(decoded),
        t_request=schema.format_rfc3339(t_request),
        t_capture=schema.format_rfc3339(t_capture),
    )


async def take_sharp_shot(deps: CaptureDeps, settings: CaptureSettings) -> Shot:
    best = await take_shot(deps, settings)
    for _ in range(settings.sharpness_retakes):
        if best.sharpness >= settings.min_sharpness:
            break
        deps.logger.info("sharpness %.1f below %.1f, retaking", best.sharpness, settings.min_sharpness)
        shot = await take_shot(deps, settings)
        if shot.sharpness > best.sharpness:
            best = shot
    return best


async def capture_viewpoint(
    deps: CaptureDeps,
    settings: CaptureSettings,
    vp: Viewpoint,
    scan_dir: Path,
    next_index: int,
) -> CaptureOutcome:
    roll = await move_to_viewpoint(deps, settings, vp)
    if roll is None:
        return CaptureOutcome(frames=[], skipped_reason="unreachable")
    t_arrived = time.monotonic()
    await wait_until_still(deps, settings)

    for attempt in range(2):
        T_before, end_pose = await deps.pose_reader.read()
        joints = await deps.arm.get_joint_positions()
        settle_s = time.monotonic() - t_arrived
        shots = [await take_sharp_shot(deps, settings) for _ in range(settings.images_per_pose)]
        T_after, _ = await deps.pose_reader.read()
        if poses_agree(T_before, T_after, settings):
            break
        if attempt == 0:
            deps.logger.info(
                "viewpoint %d: arm still settling, waiting %.2f s and retrying", vp.index, settings.dwell_s
            )
            await deps.sleep(settings.dwell_s)
        else:
            deps.logger.warning("viewpoint %d: pose still drifting after retry, keeping the frame", vp.index)

    kept = shots if settings.keep_all else [max(shots, key=lambda s: s.sharpness)]
    (scan_dir / schema.IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    frames: list[schema.FrameRecord] = []
    for burst, shot in enumerate(kept):
        index = next_index + len(frames)
        file = f"{schema.IMAGES_DIR}/{index:04d}.png"
        if not cv2.imwrite(str(scan_dir / file), shot.image_bgr):
            raise OSError(f"could not write {scan_dir / file}")
        record = schema.FrameRecord(
            index=index,
            file=file,
            viewpoint=vp.index,
            burst=burst,
            t_request=shot.t_request,
            t_capture=shot.t_capture,
            T_base_cam=transform_from_T(T_before),
            arm_end_pose=arm_end_pose_record(end_pose),
            joints_deg=[float(v) for v in joints.values],
            sharpness=shot.sharpness,
            settle_s=settle_s,
            width=int(shot.image_bgr.shape[1]),
            height=int(shot.image_bgr.shape[0]),
        )
        schema.append_frame(scan_dir, record)
        frames.append(record)
        if deps.scan_state is not None:
            try:
                await deps.scan_state.do_command(
                    {"set_frame": schema.frame_payload(record, deps.frame_extras)}
                )
            except Exception as e:
                deps.logger.warning("scan-state update failed: %s", e)
    return CaptureOutcome(frames=frames, roll_offset_deg=roll)
