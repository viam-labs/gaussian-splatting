import math

import numpy as np
import pytest
from viam.proto.common import Pose

from gaussian_splatting import geometry as g


def test_identity() -> None:
    r = g.ov_to_rotation(0, 0, 1, 0)
    assert np.allclose(r, np.eye(3))
    assert g.rotation_to_ov(np.eye(3)) == (0.0, 0.0, 1.0, 0.0)


def test_rdk_fixture_roll_45_about_x() -> None:
    # RDK spatialmath/orientation_test.go: ov45x = {Theta: pi/2, OX: 0, OY: -sqrt2/2, OZ: sqrt2/2} <-> Rx(pi/4)
    th = math.pi / 4
    r = g.ov_to_rotation(0.0, -math.sqrt(2) / 2, math.sqrt(2) / 2, math.pi / 2)
    assert np.allclose(r, g.rot_x(th), atol=1e-12)
    ox, oy, oz, theta = g.rotation_to_ov(g.rot_x(th))
    assert np.allclose(
        [ox, oy, oz, theta], [0.0, -math.sqrt(2) / 2, math.sqrt(2) / 2, math.pi / 2], atol=1e-12
    )


@pytest.mark.parametrize("theta_deg", [0, 30, -90, 179])
def test_pole_pointing_up_is_rotation_about_z(theta_deg: float) -> None:
    th = math.radians(theta_deg)
    assert np.allclose(g.ov_to_rotation(0, 0, 1, th), g.rot_z(th), atol=1e-12)
    ov = g.rotation_to_ov(g.rot_z(th))
    assert np.allclose(ov, (0.0, 0.0, 1.0, th), atol=1e-12)


def test_pole_pointing_down() -> None:
    # Pointing straight down (camera looking at the table) is the common scan orientation.
    r = g.ov_to_rotation(0, 0, -1, 0)
    assert np.allclose(r[:, 2], [0, 0, -1])
    ov = g.rotation_to_ov(r)
    assert np.allclose(ov[:3], (0.0, 0.0, -1.0), atol=1e-12)
    assert np.allclose(g.ov_to_rotation(*ov), r, atol=1e-12)


def test_near_pole_round_trip_matches_rdk_pole_convention() -> None:
    # Inside the pole radius theta is measured as if lon were 0 (RDK issue 6627 regression).
    tilt = math.radians(0.5)
    for az_deg in (0, 45, 135, -120):
        az = math.radians(az_deg)
        ov_in = (
            math.sin(tilt) * math.cos(az),
            math.sin(tilt) * math.sin(az),
            math.cos(tilt),
            math.radians(20),
        )
        r = g.ov_to_rotation(*ov_in)
        assert np.allclose(r[:, 2], ov_in[:3], atol=1e-12)
        ov_out = g.rotation_to_ov(r)
        assert np.allclose(ov_out, ov_in, atol=1e-9)


def test_random_round_trips() -> None:
    rng = np.random.default_rng(0)
    for _ in range(500):
        q = rng.normal(size=4)
        r = g.quat_wxyz_to_rotation(q / np.linalg.norm(q))
        ov = g.rotation_to_ov(r)
        assert np.allclose(g.ov_to_rotation(*ov), r, atol=1e-9)
        assert np.allclose(g.quat_wxyz_to_rotation(g.rotation_to_quat_wxyz(r)), r, atol=1e-9)


def test_pose_mm_to_metres_and_back() -> None:
    p = Pose(x=312.0, y=-105.0, z=248.0, o_x=0.0, o_y=0.0, o_z=-1.0, theta=90.0)
    t = g.pose_to_T(p)
    assert np.allclose(t[:3, 3], [0.312, -0.105, 0.248])
    back = g.T_to_pose(t)
    assert np.allclose([back.x, back.y, back.z], [312.0, -105.0, 248.0])
    assert np.allclose([back.o_x, back.o_y, back.o_z], [0.0, 0.0, -1.0], atol=1e-12)
    assert math.isclose(back.theta, 90.0, abs_tol=1e-9)


def test_colmap_world_to_cam_inverts() -> None:
    r = g.look_at_rotation(np.array([0.3, -0.1, 0.4]), np.array([0.3, 0.0, 0.0]), roll_rad=0.2)
    t_wc = g.make_T(r, np.array([0.3, -0.1, 0.4]))
    q, t = g.colmap_world_to_cam(t_wc)
    r_cw = g.quat_wxyz_to_rotation(q)
    assert np.allclose(r_cw, r.T, atol=1e-12)
    assert np.allclose(t, -r.T @ np.array([0.3, -0.1, 0.4]), atol=1e-12)
    # A world point on the optical axis projects to the image centre: p_cam = R_cw p + t has x=y=0, z>0.
    p_cam = r_cw @ np.array([0.3, 0.0, 0.0]) + t
    assert abs(p_cam[0]) < 1e-9 and abs(p_cam[1]) < 1e-9 and p_cam[2] > 0


def test_look_at_points_z_at_target_and_keeps_image_up() -> None:
    pos = np.array([0.0, -0.3, 0.3])
    tgt = np.array([0.0, 0.0, 0.0])
    r = g.look_at_rotation(pos, tgt)
    z = (tgt - pos) / np.linalg.norm(tgt - pos)
    assert np.allclose(r[:, 2], z)
    assert np.isclose(np.linalg.det(r), 1.0)
    # camera -y (image up) should have a positive world-z component
    assert -r[2, 1] > 0
    # x is horizontal when looking obliquely
    assert abs(r[2, 0]) < 1e-9
