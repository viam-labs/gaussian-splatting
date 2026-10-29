"""Synthetic posed scan of a textured plane, written as a scan directory via schema.py.

A 1.2 m x 1.2 m textured plane at z=0 in the "arm base" frame is imaged by a pinhole camera from known
poses on a cap above it. Each image is rendered with a homography (cv2.warpPerspective), so the only geometry
involved is the pinhole projection of a plane: if splat-build's world-to-camera convention or metre scale were
wrong, COLMAP would not triangulate points on z=0 inside the plane. This is also the only end-to-end input
available today, since the robot test uses a textureless fake camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from gaussian_splatting import geometry as g
from gaussian_splatting import schema as s

PLANE_HALF_M = 0.6  # plane spans [-0.6, 0.6] in x and y, at z = 0
TEXTURE_PX_PER_M = 1000  # 1 mm per texel

Mat = NDArray[np.float64]


@dataclass
class SyntheticScene:
    scan_dir: Path
    camera: s.CameraModel
    frames: list[s.FrameRecord]
    poses_T_base_cam: list[Mat]  # same order as frames


def make_texture(rng: np.random.Generator, size_px: int) -> NDArray[np.uint8]:
    """Multi-scale coloured noise: SIFT needs blobs at several scales, not white noise alone."""
    acc = np.zeros((size_px, size_px, 3), dtype=np.float64)
    for scale, weight in ((4, 0.2), (16, 0.3), (64, 0.5)):
        low = rng.uniform(0.0, 1.0, size=(size_px // scale, size_px // scale, 3))
        up = cv2.resize(low, (size_px, size_px), interpolation=cv2.INTER_CUBIC)
        acc += weight * up
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    return (acc * 255.0).astype(np.uint8)


def _texture_to_plane() -> Mat:
    """Texture pixel (u, v) -> plane metres (x, y): x = u / px_per_m - half."""
    inv = 1.0 / TEXTURE_PX_PER_M
    return np.array([[inv, 0.0, -PLANE_HALF_M], [0.0, inv, -PLANE_HALF_M], [0.0, 0.0, 1.0]])


def intrinsics_matrix(cam: s.CameraModel) -> Mat:
    return np.array([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]])


def render_plane(texture: NDArray[np.uint8], cam: s.CameraModel, t_base_cam: Mat) -> NDArray[np.uint8]:
    """Render the plane z=0 seen from the camera-to-base pose `t_base_cam` (metres)."""
    t_cam_base = g.invert_T(t_base_cam)
    r = t_cam_base[:3, :3]
    t = t_cam_base[:3, 3]
    # Points (x, y, 0): K [r1 r2 t] [x y 1]^T
    h_plane_to_img = intrinsics_matrix(cam) @ np.column_stack([r[:, 0], r[:, 1], t])
    h = h_plane_to_img @ _texture_to_plane()
    warped = cv2.warpPerspective(
        texture, h, (cam.width, cam.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return np.asarray(warped, dtype=np.uint8)


def sample_poses(rng: np.random.Generator, n: int) -> list[Mat]:
    """Cameras on a cap above the plane looking at random points on it, image up near base +Z."""
    poses: list[Mat] = []
    for _ in range(n):
        target = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), 0.0])
        elevation = math.radians(rng.uniform(45.0, 80.0))
        azimuth = rng.uniform(-math.pi, math.pi)
        distance = rng.uniform(0.7, 1.0)
        direction = np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )
        position = target + distance * direction
        roll = math.radians(rng.uniform(-10.0, 10.0))
        poses.append(g.make_T(g.look_at_rotation(position, target, roll), position))
    return poses


def _frame_record(index: int, t_base_cam: Mat, cam: s.CameraModel) -> s.FrameRecord:
    q = g.rotation_to_quat_wxyz(t_base_cam[:3, :3])
    pose = g.T_to_pose(t_base_cam)
    t_ms = 1_000 + 250 * index
    stamp = f"2026-11-03T14:02:{(t_ms // 1000) % 60:02d}.{t_ms % 1000:03d}Z"
    return s.FrameRecord(
        index=index,
        file=f"{s.IMAGES_DIR}/{index:04d}.png",
        viewpoint=index,
        burst=0,
        t_request=stamp,
        t_capture=stamp,
        T_base_cam=s.Transform(t_m=[float(v) for v in t_base_cam[:3, 3]], q_wxyz=[float(v) for v in q]),
        arm_end_pose=s.ArmEndPose(pose.x, pose.y, pose.z, pose.o_x, pose.o_y, pose.o_z, pose.theta),
        joints_deg=[0.0] * 6,
        sharpness=200.0,
        settle_s=0.6,
        width=cam.width,
        height=cam.height,
    )


def build_scene(
    scan_dir: Path, num_views: int = 25, width: int = 640, height: int = 480, seed: int = 0
) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    cam = s.CameraModel(
        width=width,
        height=height,
        fx=0.9 * width,
        fy=0.9 * width,
        cx=width / 2.0,
        cy=height / 2.0,
        source="config",
    )
    texture = make_texture(rng, int(2 * PLANE_HALF_M * TEXTURE_PX_PER_M))
    poses = sample_poses(rng, num_views)

    (scan_dir / s.IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    frames: list[s.FrameRecord] = []
    for i, t_base_cam in enumerate(poses, start=1):
        frame = _frame_record(i, t_base_cam, cam)
        img = render_plane(texture, cam, t_base_cam)
        cv2.imwrite(str(scan_dir / frame.file), img)
        s.append_frame(scan_dir, frame)
        frames.append(frame)

    manifest = s.Manifest(
        scan_id="synthetic-plane",
        request={"num_poses": num_views},
        resolved={"num_poses": num_views, "min_sharpness": 60.0},
        arm="fake-arm",
        camera="fake-camera",
        camera_model=cam,
        state="done",
    )
    s.write_manifest(scan_dir, manifest)
    return SyntheticScene(scan_dir=scan_dir, camera=cam, frames=frames, poses_T_base_cam=poses)
