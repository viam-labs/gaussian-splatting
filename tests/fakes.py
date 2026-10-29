"""In-memory stand-ins for the arm, camera, motion service, scan-state sensor and data client.

They implement the Protocols in `module/capture.py` and `module/uploader.py`, nothing more. The fake motion
service keeps the fake arm's end pose consistent with the camera pose it reports, so pose round trips are
exact.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np
from grpclib import GRPCError, Status
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.data import SequenceResourceFilter
from viam.proto.common import Pose, PoseInFrame, ResponseMetadata
from viam.proto.component.arm import JointPositions
from viam.proto.component.camera import DistortionParameters, GetPropertiesResponse, IntrinsicParameters

from gaussian_splatting import geometry, schema
from gaussian_splatting.geometry import Mat

T_FLANGE_CAM: Mat = geometry.make_T(np.eye(3), np.array([0.0, 0.0, 0.05]))  # camera 50 mm along the tool z
CAMERA_MODEL = schema.CameraModel(width=640, height=480, fx=820.0, fy=820.0, cx=320.0, cy=240.0)


def sharp_image(width: int = 640, height: int = 480, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def flat_image(width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return bytes(buf.tobytes())


class FakeArm:
    def __init__(self, name: str = "arm", moving_polls: int = 2) -> None:
        self.name = name
        self.T_base_flange: Mat = geometry.make_T(geometry.rot_y(np.pi), np.array([0.087, 0.0, 0.154]))
        self.moving_polls = moving_polls
        self._polls_left = 0
        self.joints = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    def start_motion(self) -> None:
        self._polls_left = self.moving_polls

    async def is_moving(self) -> bool:
        if self._polls_left > 0:
            self._polls_left -= 1
            return True
        return False

    async def get_end_position(self) -> Pose:
        return geometry.T_to_pose(self.T_base_flange)

    async def get_joint_positions(self) -> JointPositions:
        return JointPositions(values=self.joints)


class FakeMotion:
    def __init__(self, arm: FakeArm, *, fail_first: int = 0, always_fail: bool = False) -> None:
        self.arm = arm
        self.fail_first = fail_first
        self.always_fail = always_fail
        self.moves: list[PoseInFrame] = []

    async def move(self, component_name: str, destination: PoseInFrame) -> bool:
        self.moves.append(destination)
        if self.always_fail or len(self.moves) <= self.fail_first:
            return False
        assert destination.reference_frame == f"{self.arm.name}_origin"
        self.arm.T_base_flange = geometry.pose_to_T(destination.pose) @ geometry.invert_T(T_FLANGE_CAM)
        self.arm.start_motion()
        return True

    async def get_pose(self, component_name: str, destination_frame: str) -> PoseInFrame:
        assert destination_frame == f"{self.arm.name}_origin"
        return PoseInFrame(
            reference_frame=destination_frame, pose=geometry.T_to_pose(self.arm.T_base_flange @ T_FLANGE_CAM)
        )


class FakeCamera:
    def __init__(
        self,
        name: str = "camera",
        *,
        images: list[np.ndarray] | None = None,
        mime: CameraMimeType = CameraMimeType.PNG,
        with_intrinsics: bool = True,
        fail_first: int = 0,
        captured_at: datetime | None = None,
    ) -> None:
        self.name = name
        self.images = images or [sharp_image()]
        self.mime = mime
        self.with_intrinsics = with_intrinsics
        self.fail_first = fail_first
        self.captured_at = captured_at
        self.calls = 0
        self.commands: list[Mapping[str, Any]] = []

    async def get_images(self) -> tuple[list[NamedImage], ResponseMetadata]:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ConnectionError("usb dropout")
        image = self.images[min(self.calls - 1, len(self.images) - 1)]
        if self.mime == CameraMimeType.VIAM_RGBA:
            rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            h, w = rgba.shape[:2]
            data = b"RGBA" + w.to_bytes(4, "big") + h.to_bytes(4, "big") + rgba.tobytes()
        else:
            data = encode_png(image)
        md = ResponseMetadata()
        if self.captured_at is not None:
            md.captured_at.FromDatetime(self.captured_at.astimezone(UTC).replace(tzinfo=None))
        return [NamedImage("color", data, self.mime)], md

    async def get_properties(self) -> GetPropertiesResponse:
        if not self.with_intrinsics:
            return GetPropertiesResponse()
        return GetPropertiesResponse(
            intrinsic_parameters=IntrinsicParameters(
                width_px=640,
                height_px=480,
                focal_x_px=820.0,
                focal_y_px=820.0,
                center_x_px=320.0,
                center_y_px=240.0,
            ),
            distortion_parameters=DistortionParameters(
                model="brown_conrady", parameters=[0.1, -0.2, 0.001, 0.002, 0.3]
            ),
        )

    async def do_command(self, command: Mapping[str, Any]) -> Mapping[str, Any]:
        self.commands.append(dict(command))
        return {"ok": True}


class FakeScanState:
    def __init__(self) -> None:
        self.frames: list[Mapping[str, Any]] = []

    async def do_command(self, command: Mapping[str, Any]) -> Mapping[str, Any]:
        self.frames.append(dict(command["set_frame"]))
        return {"ok": True}


class FakeDataClient:
    """Records every call. `fail` maps a method name to how many times it should fail before succeeding."""

    def __init__(self, *, fail: dict[str, int] | None = None, deny_dataset: bool = False) -> None:
        self.fail = dict(fail or {})
        self.deny_dataset = deny_dataset
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._ids = 0

    def _maybe_fail(self, method: str) -> None:
        if self.fail.get(method, 0) > 0:
            self.fail[method] -= 1
            raise GRPCError(Status.UNAVAILABLE, f"{method} flaked")

    def _next_id(self, prefix: str) -> str:
        self._ids += 1
        return f"{prefix}-{self._ids}"

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
    ) -> str:
        self.calls.append(("binary", {k: v for k, v in locals().items() if k != "self"}))
        self._maybe_fail("binary")
        return self._next_id("bin")

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
    ) -> str:
        self.calls.append(("tabular", {k: v for k, v in locals().items() if k != "self"}))
        self._maybe_fail("tabular")
        return self._next_id("tab")

    async def create_sequence(
        self,
        part_id: str,
        resources: list[SequenceResourceFilter],
        start_time: datetime,
        end_time: datetime,
        sequence_tags: list[str] | None = None,
        timeout: float | None = None,
    ) -> str:
        self.calls.append(("sequence", {k: v for k, v in locals().items() if k != "self"}))
        self._maybe_fail("sequence")
        return self._next_id("seq")

    async def add_sequences_to_dataset(
        self, dataset_id: str, sequence_ids: list[str], timeout: float | None = None
    ) -> None:
        self.calls.append(("dataset", {"dataset_id": dataset_id, "sequence_ids": sequence_ids}))
        if self.deny_dataset:
            raise GRPCError(Status.PERMISSION_DENIED, "organization scope required")
        self._maybe_fail("dataset")

    def count(self, kind: str) -> int:
        return sum(1 for k, _ in self.calls if k == kind)
