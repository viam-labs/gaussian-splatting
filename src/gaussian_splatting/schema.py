"""On-disk ledger of a scan: manifest.json and frames.jsonl.

Shared by the module (writer) and splat-build (reader).

Kept as plain dataclasses with explicit to/from dict so the JSON is the contract, not Python pickling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
FRAMES_NAME = "frames.jsonl"
IMAGES_DIR = "images"


@dataclass
class Transform:
    """Rigid transform: translation in metres, quaternion (w, x, y, z)."""

    t_m: list[float]
    q_wxyz: list[float]


@dataclass
class ArmEndPose:
    x_mm: float
    y_mm: float
    z_mm: float
    o_x: float
    o_y: float
    o_z: float
    theta_deg: float


@dataclass
class CameraModel:
    """Pinhole + OpenCV distortion, pixels. `source` says where it came from (camera | config)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    source: str = "camera"

    @property
    def colmap_model(self) -> str:
        if all(v == 0.0 for v in (self.k1, self.k2, self.p1, self.p2, self.k3)):
            return "PINHOLE"
        return "FULL_OPENCV" if self.k3 != 0.0 else "OPENCV"

    @property
    def colmap_params(self) -> list[float]:
        base = [self.fx, self.fy, self.cx, self.cy]
        if self.colmap_model == "PINHOLE":
            return base
        if self.colmap_model == "OPENCV":
            return base + [self.k1, self.k2, self.p1, self.p2]
        return base + [self.k1, self.k2, self.p1, self.p2, self.k3, 0.0, 0.0, 0.0]


@dataclass
class UploadState:
    binary_id: str | None = None
    tabular_ok: bool = False


@dataclass
class FrameRecord:
    """One saved image. `T_base_cam` is camera-to-arm-base. Times are RFC 3339 UTC strings."""

    index: int
    file: str
    viewpoint: int
    burst: int
    t_request: str
    t_capture: str
    T_base_cam: Transform
    arm_end_pose: ArmEndPose
    joints_deg: list[float]
    sharpness: float
    settle_s: float
    width: int
    height: int
    uploaded: UploadState = field(default_factory=UploadState)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FrameRecord:
        return cls(
            index=int(d["index"]),
            file=str(d["file"]),
            viewpoint=int(d["viewpoint"]),
            burst=int(d["burst"]),
            t_request=str(d["t_request"]),
            t_capture=str(d["t_capture"]),
            T_base_cam=Transform(**d["T_base_cam"]),
            arm_end_pose=ArmEndPose(**d["arm_end_pose"]),
            joints_deg=[float(v) for v in d["joints_deg"]],
            sharpness=float(d["sharpness"]),
            settle_s=float(d["settle_s"]),
            width=int(d["width"]),
            height=int(d["height"]),
            uploaded=UploadState(**d.get("uploaded", {})),
        )


@dataclass
class Manifest:
    scan_id: str
    schema_version: int = SCHEMA_VERSION
    request: dict[str, Any] = field(default_factory=dict)
    resolved: dict[str, Any] = field(default_factory=dict)
    machine_id: str | None = None
    part_id: str | None = None
    arm: str = ""
    camera: str = ""
    camera_model: CameraModel | None = None
    camera_in_arm: Transform | None = None
    camera_in_arm_source: str = "frame_system"
    frame_system_hash: str | None = None
    module_version: str = ""
    viam_server_version: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    state: str = "planning"
    sequence_id: str | None = None
    dataset_id: str | None = None
    planned_viewpoints: list[dict[str, Any]] = field(default_factory=list)
    skipped_viewpoints: list[dict[str, Any]] = field(default_factory=list)
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        d = dict(d)
        cm = d.get("camera_model")
        cia = d.get("camera_in_arm")
        d["camera_model"] = CameraModel(**cm) if cm else None
        d["camera_in_arm"] = Transform(**cia) if cia else None
        return cls(**d)


def write_manifest(scan_dir: Path, manifest: Manifest) -> None:
    tmp = scan_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2))
    tmp.replace(scan_dir / MANIFEST_NAME)  # atomic: a crash never leaves a half-written manifest


def read_manifest(scan_dir: Path) -> Manifest:
    return Manifest.from_dict(json.loads((scan_dir / MANIFEST_NAME).read_text()))


def append_frame(scan_dir: Path, frame: FrameRecord) -> None:
    with (scan_dir / FRAMES_NAME).open("a") as f:
        f.write(json.dumps(frame.to_dict()) + "\n")


def read_frames(scan_dir: Path) -> list[FrameRecord]:
    path = scan_dir / FRAMES_NAME
    if not path.exists():
        return []
    return [FrameRecord.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def rewrite_frames(scan_dir: Path, frames: list[FrameRecord]) -> None:
    """Replace frames.jsonl atomically (used to persist upload state)."""
    tmp = scan_dir / (FRAMES_NAME + ".tmp")
    tmp.write_text("".join(json.dumps(fr.to_dict()) + "\n" for fr in frames))
    tmp.replace(scan_dir / FRAMES_NAME)
