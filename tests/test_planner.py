import math

import numpy as np
import pytest

from gaussian_splatting import geometry
from gaussian_splatting.module import planner as pl
from tests.fakes import CAMERA_MODEL, T_FLANGE_CAM


def params(**overrides: object) -> pl.PlannerParams:
    base: dict[str, object] = dict(
        region_center_mm=(250.0, 0.0, 0.0),
        region_size_mm=(300.0, 300.0, 50.0),
        standoff_mm=(150.0, 300.0),
        elevation_deg=(30.0, 80.0),
        azimuth_deg=(-120.0, 120.0),
        num_poses=40,
        seed=0,
    )
    base.update(overrides)
    return pl.PlannerParams(**base)  # type: ignore[arg-type]


def test_deterministic_by_seed() -> None:
    a = pl.plan_viewpoints(params(seed=3), CAMERA_MODEL, T_FLANGE_CAM)
    b = pl.plan_viewpoints(params(seed=3), CAMERA_MODEL, T_FLANGE_CAM)
    c = pl.plan_viewpoints(params(seed=4), CAMERA_MODEL, T_FLANGE_CAM)
    assert [v.to_dict() for v in a.viewpoints] == [v.to_dict() for v in b.viewpoints]
    assert a.stats.to_dict() == b.stats.to_dict()
    assert [v.t_m for v in a.viewpoints] != [v.t_m for v in c.viewpoints]


def test_candidates_respect_window_and_look_at_top_face() -> None:
    p = params()
    plan = pl.plan_viewpoints(p, CAMERA_MODEL, T_FLANGE_CAM)
    assert plan.stats.candidates == 3 * p.num_poses
    assert len(plan.viewpoints) >= p.num_poses
    top_z = (p.region_center_mm[2] + p.region_size_mm[2] / 2) / 1000.0
    for v in plan.viewpoints:
        assert p.elevation_deg[0] <= v.elevation_deg <= p.elevation_deg[1]
        assert p.azimuth_deg[0] <= v.azimuth_deg <= p.azimuth_deg[1]
        assert abs(v.roll_deg) <= p.roll_jitter_deg
        assert v.look_at_m[2] == pytest.approx(top_z)
        assert abs(v.look_at_m[0] - 0.25) <= 0.15 and abs(v.look_at_m[1]) <= 0.15
        T = v.T_base_cam()
        to_target = np.array(v.look_at_m) - T[:3, 3]
        assert np.linalg.norm(to_target) == pytest.approx(v.distance_mm / 1000.0)
        assert np.dot(T[:3, 2], to_target / np.linalg.norm(to_target)) == pytest.approx(1.0, abs=1e-9)
        assert pl.reject_reason(T, p, CAMERA_MODEL, T_FLANGE_CAM) is None
    assert [v.index for v in plan.viewpoints] == list(range(len(plan.viewpoints)))


def test_shell_count_and_interleaving() -> None:
    two = pl.plan_viewpoints(params(standoff_mm=(150.0, 300.0)), CAMERA_MODEL, T_FLANGE_CAM)
    three = pl.plan_viewpoints(params(standoff_mm=(200.0, 450.0)), CAMERA_MODEL, T_FLANGE_CAM)
    assert two.stats.shells_mm == [150.0, 300.0]
    assert three.stats.shells_mm == [200.0, 325.0, 450.0]
    shells = [v.shell for v in two.viewpoints[:4]]
    assert shells[0] == 0 and shells[1] == 1, "inner shell first, then alternating"


def test_filter_rules() -> None:
    p = params()
    look = np.array([0.25, 0.0, 0.025])
    good = geometry.make_T(
        geometry.look_at_rotation(np.array([0.25, 0.0, 0.25]), look), np.array([0.25, 0.0, 0.25])
    )
    assert pl.reject_reason(good, p, CAMERA_MODEL, T_FLANGE_CAM) is None

    far = good.copy()
    far[:3, 3] = [0.6, 0.0, 0.25]
    assert pl.reject_reason(far, p, CAMERA_MODEL, T_FLANGE_CAM) == "reach"

    pos = np.array([0.25, 0.0, -0.1])
    low = geometry.make_T(geometry.look_at_rotation(pos, np.array([0.25, 0.0, 0.1])), pos)
    assert pl.reject_reason(low, p, CAMERA_MODEL, T_FLANGE_CAM) == "below_region"

    pos = np.array([0.3, 0.0, 0.1])
    at_base = geometry.make_T(geometry.look_at_rotation(pos, np.array([0.0, 0.0, 0.05])), pos)
    assert pl.reject_reason(at_base, p, CAMERA_MODEL, T_FLANGE_CAM) == "base_in_frustum"


def test_stats_are_reported() -> None:
    plan = pl.plan_viewpoints(params(num_poses=60), CAMERA_MODEL, T_FLANGE_CAM)
    s = plan.stats
    assert 0.0 < s.reachable_window_fraction <= 1.0
    assert s.top_face_coverage > 0.5
    assert 2.0 < s.nn_direction_gap_deg_median < 40.0
    assert s.nn_direction_gap_deg_max >= s.nn_direction_gap_deg_median
    assert s.accepted + sum(s.rejected.values()) == s.candidates


def test_unreachable_window_rejects_everything() -> None:
    plan = pl.plan_viewpoints(params(arm_reach_mm=100.0), CAMERA_MODEL, T_FLANGE_CAM)
    assert plan.viewpoints == []
    assert plan.stats.rejected.get("reach") == plan.stats.candidates
    assert plan.stats.top_face_coverage == 0.0


def test_viewpoint_round_trip() -> None:
    plan = pl.plan_viewpoints(params(num_poses=5), CAMERA_MODEL, T_FLANGE_CAM)
    v = plan.viewpoints[0]
    back = pl.Viewpoint.from_dict(v.to_dict())
    assert back == v
    assert np.allclose(back.T_base_cam(), v.T_base_cam())


def test_halton_is_low_discrepancy() -> None:
    xs = [pl.halton(i, 2) for i in range(1, 9)]
    assert xs == [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625]
    assert math.isclose(pl.halton(1, 3), 1 / 3)
