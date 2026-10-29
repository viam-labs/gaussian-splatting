"""Scanner attributes and start_scan requests: one parser for validate_config, reconfigure and do_command.

Parsing lives apart from the resource class so the rules are unit-testable without viam-server and so
`validate_config` (class method, runs before any dependency exists) and `reconfigure` cannot drift apart.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from viam.proto.common import Pose

from gaussian_splatting import geometry
from gaussian_splatting.geometry import Mat

DEFAULT_ARM_REACH_MM = 440.0  # UFactory Lite 6; override with `arm_reach_mm` for other arms
DEFAULT_STANDOFF_MM = (200.0, 450.0)
DEFAULT_ELEVATION_DEG = (25.0, 80.0)
DEFAULT_AZIMUTH_DEG = (-150.0, 150.0)
DEFAULT_NUM_POSES = 150


class ConfigError(ValueError):
    """An attribute or request field is missing or malformed. The message names the field."""


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in the arm base frame, millimetres."""

    center_mm: tuple[float, float, float]
    size_mm: tuple[float, float, float]

    def to_dict(self) -> dict[str, list[float]]:
        return {"center": list(self.center_mm), "size": list(self.size_mm)}


@dataclass(frozen=True)
class Intrinsics:
    """`camera_intrinsics` fallback attribute. Width and height are optional when the camera reports them."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    width: int | None = None
    height: int | None = None


@dataclass
class ScannerConfig:
    arm: str
    camera: str
    motion: str
    region: Box
    scan_state: str | None = None
    standoff_mm: tuple[float, float] = DEFAULT_STANDOFF_MM
    elevation_deg: tuple[float, float] = DEFAULT_ELEVATION_DEG
    azimuth_deg: tuple[float, float] = DEFAULT_AZIMUTH_DEG
    dwell_s: float = 0.5
    min_sharpness: float = 60.0
    static_mask: str | None = None
    camera_lock_command: dict[str, Any] = field(default_factory=dict)
    dataset_id: str | None = None
    upload: bool = False
    camera_intrinsics: Intrinsics | None = None
    camera_in_arm: Mat | None = None  # T_flange_cam, metres
    arm_reach_mm: float = DEFAULT_ARM_REACH_MM
    num_poses: int = DEFAULT_NUM_POSES
    images_per_pose: int = 1
    seed: int = 0

    def required_dependencies(self) -> list[str]:
        deps = [self.arm, self.camera, self.motion]
        if self.scan_state:
            deps.append(self.scan_state)
        return deps


def _number(attrs: Mapping[str, Any], key: str, default: float | None = None) -> float:
    v = attrs.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        raise ConfigError(f"'{key}' must be a finite number, got {v!r}")
    return float(v)


def _string(attrs: Mapping[str, Any], key: str, required: bool) -> str | None:
    v = attrs.get(key)
    if v is None:
        if required:
            raise ConfigError(f"'{key}' is required")
        return None
    if not isinstance(v, str) or not v:
        raise ConfigError(f"'{key}' must be a non-empty string")
    return v


def _bool(attrs: Mapping[str, Any], key: str, default: bool) -> bool:
    v = attrs.get(key, default)
    if not isinstance(v, bool):
        raise ConfigError(f"'{key}' must be true or false")
    return v


def _vec3(value: object, key: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 3:
        raise ConfigError(f"'{key}' must be a list of 3 numbers")
    out = tuple(_number({"v": v}, "v") for v in value)
    return out[0], out[1], out[2]


def _range(
    attrs: Mapping[str, Any], key: str, default: tuple[float, float], lo: float, hi: float
) -> tuple[float, float]:
    value = attrs.get(key)
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ConfigError(f"'{key}' must be [min, max]")
    a, b = (_number({"v": v}, "v") for v in value)
    if not (lo <= a <= b <= hi):
        raise ConfigError(f"'{key}' must satisfy {lo} <= min <= max <= {hi}, got {value!r}")
    return a, b


def parse_region(value: object, key: str = "region_mm") -> Box:
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{key}' must be {{center: [x,y,z], size: [sx,sy,sz]}}")
    size = _vec3(value.get("size"), f"{key}.size")
    if any(s <= 0 for s in size):
        raise ConfigError(f"'{key}.size' must be positive")
    return Box(center_mm=_vec3(value.get("center"), f"{key}.center"), size_mm=size)


def parse_intrinsics(value: object) -> Intrinsics:
    """`{"K": {fx, fy, cx, cy}, "dist": {k1, k2, k3, p1, p2}, "width"?, "height"?}`, the viam:opencv shape."""
    if not isinstance(value, Mapping) or not isinstance(value.get("K"), Mapping):
        raise ConfigError("'camera_intrinsics' must be {K: {fx, fy, cx, cy}, dist: {k1, k2, k3, p1, p2}}")
    k = value["K"]
    dist = value.get("dist") or {}
    if not isinstance(dist, Mapping):
        raise ConfigError("'camera_intrinsics.dist' must be a mapping")
    width = value.get("width")
    height = value.get("height")
    return Intrinsics(
        fx=_number(k, "fx"),
        fy=_number(k, "fy"),
        cx=_number(k, "cx"),
        cy=_number(k, "cy"),
        k1=_number(dist, "k1", 0.0),
        k2=_number(dist, "k2", 0.0),
        k3=_number(dist, "k3", 0.0),
        p1=_number(dist, "p1", 0.0),
        p2=_number(dist, "p2", 0.0),
        width=int(_number(value, "width")) if width is not None else None,
        height=int(_number(value, "height")) if height is not None else None,
    )


def parse_frame_transform(value: object, key: str = "camera_in_arm") -> Mat:
    """A Viam `frame` block (translation mm, orientation vector in degrees) to a 4x4 transform in metres."""
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{key}' must be {{translation: {{x,y,z}}, orientation: {{type, value}}}}")
    tr = value.get("translation") or {}
    orient = value.get("orientation") or {}
    if not isinstance(tr, Mapping) or not isinstance(orient, Mapping):
        raise ConfigError(f"'{key}' translation and orientation must be mappings")
    ov = orient.get("value") or {"x": 0.0, "y": 0.0, "z": 1.0, "th": 0.0}
    if orient.get("type", "ov_degrees") != "ov_degrees" or not isinstance(ov, Mapping):
        raise ConfigError(f"'{key}.orientation' must be of type ov_degrees")
    pose = Pose(
        x=_number(tr, "x", 0.0),
        y=_number(tr, "y", 0.0),
        z=_number(tr, "z", 0.0),
        o_x=_number(ov, "x", 0.0),
        o_y=_number(ov, "y", 0.0),
        o_z=_number(ov, "z", 1.0),
        theta=_number(ov, "th", 0.0),
    )
    return geometry.pose_to_T(pose)


def parse_scanner_config(attrs: Mapping[str, Any]) -> ScannerConfig:
    arm = _string(attrs, "arm", required=True)
    camera = _string(attrs, "camera", required=True)
    motion = _string(attrs, "motion", required=True)
    assert arm is not None and camera is not None and motion is not None
    upload = _bool(attrs, "upload", False)
    scan_state = _string(attrs, "scan_state", required=False)
    if upload and scan_state is None:
        raise ConfigError("'scan_state' is required when 'upload' is true")
    lock = attrs.get("camera_lock_command") or {}
    if not isinstance(lock, Mapping):
        raise ConfigError("'camera_lock_command' must be a mapping")
    num_poses = int(_number(attrs, "num_poses", DEFAULT_NUM_POSES))
    images_per_pose = int(_number(attrs, "images_per_pose", 1))
    if num_poses < 1 or images_per_pose < 1:
        raise ConfigError("'num_poses' and 'images_per_pose' must be >= 1")
    reach = _number(attrs, "arm_reach_mm", DEFAULT_ARM_REACH_MM)
    if reach <= 0:
        raise ConfigError("'arm_reach_mm' must be positive")
    intrinsics = attrs.get("camera_intrinsics")
    camera_in_arm = attrs.get("camera_in_arm")
    return ScannerConfig(
        arm=arm,
        camera=camera,
        motion=motion,
        scan_state=scan_state,
        region=parse_region(attrs.get("region_mm")),
        standoff_mm=_range(attrs, "standoff_mm", DEFAULT_STANDOFF_MM, 1.0, 10_000.0),
        elevation_deg=_range(attrs, "elevation_deg", DEFAULT_ELEVATION_DEG, 0.0, 90.0),
        azimuth_deg=_range(attrs, "azimuth_deg", DEFAULT_AZIMUTH_DEG, -180.0, 180.0),
        dwell_s=_number(attrs, "dwell_s", 0.5),
        min_sharpness=_number(attrs, "min_sharpness", 60.0),
        static_mask=_string(attrs, "static_mask", required=False),
        camera_lock_command=dict(lock),
        dataset_id=_string(attrs, "dataset_id", required=False),
        upload=upload,
        camera_intrinsics=parse_intrinsics(intrinsics) if intrinsics is not None else None,
        camera_in_arm=parse_frame_transform(camera_in_arm) if camera_in_arm is not None else None,
        arm_reach_mm=reach,
        num_poses=num_poses,
        images_per_pose=images_per_pose,
        seed=int(_number(attrs, "seed", 0)),
    )


@dataclass
class ScanRequest:
    """A `start_scan` command with the config defaults filled in."""

    scan_id: str
    num_poses: int
    images_per_pose: int
    keep_all: bool
    region: Box
    standoff_mm: tuple[float, float]
    elevation_deg: tuple[float, float]
    azimuth_deg: tuple[float, float]
    seed: int
    dwell_s: float
    min_sharpness: float
    upload: bool
    dataset_id: str | None
    tags: list[str]
    resume: bool
    dry_run: bool

    @classmethod
    def from_command(
        cls, cmd: Mapping[str, Any], cfg: ScannerConfig, now: datetime | None = None
    ) -> ScanRequest:
        now = now or datetime.now(UTC)
        region = cmd.get("region_mm")
        tags = cmd.get("tags") or []
        if (
            not isinstance(tags, Sequence)
            or isinstance(tags, str)
            or not all(isinstance(t, str) for t in tags)
        ):
            raise ConfigError("'tags' must be a list of strings")
        num_poses = int(_number(cmd, "num_poses", cfg.num_poses))
        images_per_pose = int(_number(cmd, "images_per_pose", cfg.images_per_pose))
        if num_poses < 1 or images_per_pose < 1:
            raise ConfigError("'num_poses' and 'images_per_pose' must be >= 1")
        return cls(
            scan_id=_string(cmd, "scan_id", required=False) or now.strftime("scan-%Y%m%d-%H%M%S"),
            num_poses=num_poses,
            images_per_pose=images_per_pose,
            keep_all=_bool(cmd, "keep_all", False),
            region=parse_region(region) if region is not None else cfg.region,
            standoff_mm=_range(cmd, "standoff_mm", cfg.standoff_mm, 1.0, 10_000.0),
            elevation_deg=_range(cmd, "elevation_deg", cfg.elevation_deg, 0.0, 90.0),
            azimuth_deg=_range(cmd, "azimuth_deg", cfg.azimuth_deg, -180.0, 180.0),
            seed=int(_number(cmd, "seed", cfg.seed)),
            dwell_s=_number(cmd, "dwell_s", cfg.dwell_s),
            min_sharpness=_number(cmd, "min_sharpness", cfg.min_sharpness),
            upload=_bool(cmd, "upload", cfg.upload),
            dataset_id=_string(cmd, "dataset_id", required=False) or cfg.dataset_id,
            tags=list(tags),
            resume=_bool(cmd, "resume", False),
            dry_run=_bool(cmd, "dry_run", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "num_poses": self.num_poses,
            "images_per_pose": self.images_per_pose,
            "keep_all": self.keep_all,
            "region_mm": self.region.to_dict(),
            "standoff_mm": list(self.standoff_mm),
            "elevation_deg": list(self.elevation_deg),
            "azimuth_deg": list(self.azimuth_deg),
            "seed": self.seed,
            "dwell_s": self.dwell_s,
            "min_sharpness": self.min_sharpness,
            "upload": self.upload,
            "dataset_id": self.dataset_id,
            "tags": list(self.tags),
            "resume": self.resume,
            "dry_run": self.dry_run,
        }
