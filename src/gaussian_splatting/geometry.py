"""Pose math shared by the module and splat-build.

Viam represents rotations as an orientation vector (o_x, o_y, o_z, theta): (o_x, o_y, o_z) is where the
frame's +Z axis points, theta is the roll about it. The conversion below is a port of RDK `spatialmath`
(`OrientationVector.Quaternion` and `QuatToOV`): R = Rz(lon) Ry(lat) Rz(theta) with lon = atan2(o_y, o_x),
lat = acos(o_z), and a special case within POLE_RADIUS of +/-Z where theta is measured as if lon were zero.
Getting this wrong rotates every camera pose in the splat, so it is tested against RDK's own fixtures.

Conventions used everywhere in this package:
- T is a 4x4 homogeneous transform, translation in metres.
- Quaternions are (w, x, y, z).
- `T_a_b` maps points in frame b to frame a (b-to-a). `T_base_cam` is camera-to-arm-base.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from viam.proto.common import Pose

POLE_RADIUS = 0.0001  # RDK orientationVectorPoleRadius: 1 - |o_z| <= this means "pole math"

Mat = NDArray[np.float64]


def rot_z(a: float) -> Mat:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> Mat:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(a: float) -> Mat:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _in_pole_radius(oz_normalized: float) -> bool:
    return 1.0 - abs(oz_normalized) <= POLE_RADIUS


def ov_to_rotation(ox: float, oy: float, oz: float, theta_rad: float) -> Mat:
    """Orientation vector (radians) to rotation matrix, exactly as RDK decodes it."""
    norm = math.sqrt(ox * ox + oy * oy + oz * oz)
    if norm == 0.0:
        raise ValueError("orientation vector direction is zero")
    ox, oy, oz = ox / norm, oy / norm, oz / norm
    oz = max(-1.0, min(1.0, oz))
    lat = math.acos(oz)
    lon = math.atan2(oy, ox)
    theta = theta_rad
    if _in_pole_radius(oz):
        theta = theta - lon if oz > 0 else theta + lon
    return rot_z(lon) @ rot_y(lat) @ rot_z(theta)


def rotation_to_ov(r: Mat) -> tuple[float, float, float, float]:
    """Rotation matrix to orientation vector (radians), the inverse of `ov_to_rotation`.

    Solves R = Rz(lon) Ry(lat) Rz(theta) exactly for theta instead of RDK's plane-angle construction; the two
    agree (RDK's forward decode is this very product), and this form has no sign-test branch.
    """
    ox, oy, oz = (float(v) for v in r[:, 2])
    oz_c = max(-1.0, min(1.0, oz))
    lat = math.acos(oz_c)
    lon = math.atan2(oy, ox)
    twist = rot_y(-lat) @ rot_z(-lon) @ r  # remaining Rz(theta')
    theta = math.atan2(twist[1, 0], twist[0, 0])
    if _in_pole_radius(oz_c):
        theta = theta + lon if oz_c > 0 else theta - lon
        theta = math.remainder(theta, 2.0 * math.pi)
    if theta == 0.0:
        theta = 0.0  # fold -0.0
    return ox, oy, oz, theta


def rotation_to_quat_wxyz(r: Mat) -> Mat:
    """Rotation matrix to unit quaternion (w, x, y, z), w >= 0."""
    m = r
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def quat_wxyz_to_rotation(q: Mat) -> Mat:
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64) / np.linalg.norm(q))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_T(r: Mat, t_m: Mat) -> Mat:
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = np.asarray(t_m, dtype=np.float64).reshape(3)
    return t


def invert_T(t: Mat) -> Mat:
    r = t[:3, :3]
    out = np.eye(4)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t[:3, 3]
    return out


def pose_to_T(pose: Pose) -> Mat:
    """Viam Pose (mm, orientation vector with theta in degrees) to a 4x4 transform in metres."""
    r = ov_to_rotation(pose.o_x, pose.o_y, pose.o_z, math.radians(pose.theta))
    return make_T(r, np.array([pose.x, pose.y, pose.z]) / 1000.0)


def T_to_pose(t: Mat) -> Pose:
    """4x4 transform in metres to a Viam Pose (mm, theta in degrees)."""
    ox, oy, oz, theta = rotation_to_ov(t[:3, :3])
    x, y, z = (float(v) * 1000.0 for v in t[:3, 3])
    return Pose(x=x, y=y, z=z, o_x=ox, o_y=oy, o_z=oz, theta=math.degrees(theta))


def look_at_rotation(position: Mat, target: Mat, roll_rad: float = 0.0) -> Mat:
    """Camera rotation whose +Z points from `position` to `target`, image up closest to world up, then rolled.

    Columns are the camera axes in the parent frame. Camera convention is Viam's and COLMAP's: x right,
    y down, z forward. "Image up closest to world up" means camera -y is as close to world +Z as the view allows.
    """
    z = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    n = float(np.linalg.norm(z))
    if n < 1e-9:
        raise ValueError("position coincides with target")
    z = z / n
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z, world_up))) > 0.999:
        world_up = np.array([1.0, 0.0, 0.0])  # looking straight down or up: pick x as the tie-break
    x = np.cross(-world_up, z)  # y_cam = -up projected, x = y × z  → x = cross(-up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = math.cos(roll_rad), math.sin(roll_rad)
    xr = c * x + s * y
    yr = -s * x + c * y
    return np.column_stack([xr, yr, z])


def colmap_world_to_cam(t_world_cam: Mat) -> tuple[Mat, Mat]:
    """Camera-to-world transform to COLMAP's world-to-camera (q_wxyz, t), translation in metres."""
    t_cam_world = invert_T(t_world_cam)
    return rotation_to_quat_wxyz(t_cam_world[:3, :3]), t_cam_world[:3, 3].copy()


def rotation_angle_deg(ra: Mat, rb: Mat) -> float:
    """Angle in degrees between two rotations."""
    c = (float(np.trace(ra.T @ rb)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))
