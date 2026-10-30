"""Viewpoint planning (spec section 7). Pure functions, no Viam calls.

Look-at points are spread over the top face of the region box (Halton), view directions come from a
Fibonacci lattice cut to the elevation/azimuth window, and each candidate sits on one of a few discrete
standoff shells. A geometric filter rejects what the arm cannot reach or what would put the arm's own
base in frame; the survivors are ordered with a greedy nearest-neighbour tour per shell, shells
interleaved, so a cancelled scan still covers the region at every distance. Everything is seeded.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from gaussian_splatting import geometry
from gaussian_splatting.geometry import Mat
from gaussian_splatting.schema import CameraModel

COVERAGE_GRID = 10
COVERAGE_MIN_VIEWS = 3
SHELL_SPLIT_MM = 150.0  # standoff ranges wider than this get a third shell


@dataclass(frozen=True)
class PlannerParams:
    region_center_mm: tuple[float, float, float]
    region_size_mm: tuple[float, float, float]
    standoff_mm: tuple[float, float]
    elevation_deg: tuple[float, float]
    azimuth_deg: tuple[float, float]
    num_poses: int
    seed: int = 0
    arm_reach_mm: float = 440.0
    reach_fraction: float = 0.9
    roll_jitter_deg: float = 10.0
    candidate_factor: int = 3
    base_height_mm: float = (
        200.0  # the arm's base column, sampled from z=0 up to here, must stay out of frame
    )


@dataclass
class Viewpoint:
    """One planned camera pose, camera-to-base, metres. `index` is the position in the ordered plan."""

    index: int
    shell: int
    distance_mm: float
    elevation_deg: float
    azimuth_deg: float
    roll_deg: float
    look_at_m: list[float]
    t_m: list[float]
    q_wxyz: list[float]

    def T_base_cam(self) -> Mat:
        return geometry.make_T(geometry.quat_wxyz_to_rotation(np.array(self.q_wxyz)), np.array(self.t_m))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Viewpoint:
        return cls(
            index=int(d["index"]),
            shell=int(d["shell"]),
            distance_mm=float(d["distance_mm"]),
            elevation_deg=float(d["elevation_deg"]),
            azimuth_deg=float(d["azimuth_deg"]),
            roll_deg=float(d["roll_deg"]),
            look_at_m=[float(v) for v in d["look_at_m"]],
            t_m=[float(v) for v in d["t_m"]],
            q_wxyz=[float(v) for v in d["q_wxyz"]],
        )


@dataclass
class PlanStats:
    candidates: int
    accepted: int
    rejected: dict[str, int]
    reachable_window_fraction: float
    top_face_coverage: float
    nn_direction_gap_deg_median: float
    nn_direction_gap_deg_max: float
    shells_mm: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    """Accepted candidates in capture order (more than `num_poses` on purpose: motion may still skip some)."""

    viewpoints: list[Viewpoint]
    stats: PlanStats


def halton(index: int, base: int) -> float:
    """Radical inverse of `index` in `base`; index starts at 1."""
    f, r = 1.0, 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r


def _look_at_points_m(p: PlannerParams, rng: np.random.Generator) -> Mat:
    cx, cy, cz = p.region_center_mm
    sx, sy, sz = p.region_size_mm
    shift = rng.random(
        2
    )  # Cranley-Patterson rotation: same low-discrepancy pattern, seed-dependent placement
    pts = np.empty((p.num_poses, 3))
    for i in range(p.num_poses):
        u = (halton(i + 1, 2) + shift[0]) % 1.0
        v = (halton(i + 1, 3) + shift[1]) % 1.0
        pts[i] = (cx + (u - 0.5) * sx, cy + (v - 0.5) * sy, cz + sz / 2.0)
    return pts / 1000.0


def _directions_deg(p: PlannerParams, rng: np.random.Generator, count: int) -> list[tuple[float, float]]:
    """`count` (elevation, azimuth) pairs in degrees, near-uniform over the window."""
    e0, e1 = (math.radians(v) for v in p.elevation_deg)
    a0, a1 = (math.radians(v) for v in p.azimuth_deg)
    window_fraction = ((a1 - a0) * (math.sin(e1) - math.sin(e0))) / (4.0 * math.pi)
    if window_fraction <= 0.0:
        return []
    lattice_size = min(int(math.ceil(count / window_fraction)) * 2, 2_000_000)
    offset = rng.uniform(0.0, 2.0 * math.pi)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    inside: list[tuple[float, float]] = []
    for i in range(lattice_size):
        z = 1.0 - 2.0 * (i + 0.5) / lattice_size
        elev = math.asin(z)
        if not (e0 <= elev <= e1):
            continue
        phi = i * golden + offset
        az = math.atan2(math.sin(phi), math.cos(phi))
        if not (a0 <= az <= a1):
            continue
        inside.append((math.degrees(elev), math.degrees(az)))
    if len(inside) <= count:
        return inside
    # Even subsampling keeps the lattice's spread; truncating would bias toward high elevations.
    picks = np.linspace(0, len(inside) - 1, count).round().astype(int)
    return [inside[int(k)] for k in picks]


def _shells_mm(p: PlannerParams) -> list[float]:
    lo, hi = p.standoff_mm
    if hi - lo < 1e-9:
        return [lo]
    n = 3 if hi - lo > SHELL_SPLIT_MM else 2
    return [float(v) for v in np.linspace(lo, hi, n)]


def _direction_vector(elevation_deg: float, azimuth_deg: float) -> Mat:
    e, a = math.radians(elevation_deg), math.radians(azimuth_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def project_inside(T_base_cam: Mat, cam: CameraModel, point_m: Mat) -> bool:
    """True when `point_m` (base frame) is in front of the camera and lands inside the image."""
    p = geometry.invert_T(T_base_cam) @ np.append(point_m, 1.0)
    if p[2] <= 1e-9:
        return False
    u = cam.fx * p[0] / p[2] + cam.cx
    v = cam.fy * p[1] / p[2] + cam.cy
    return bool(0.0 <= u < cam.width and 0.0 <= v < cam.height)


def reject_reason(T_base_cam: Mat, p: PlannerParams, cam: CameraModel, T_flange_cam: Mat) -> str | None:
    """Geometric filter. Returns None when the candidate passes, else a short reason key."""
    T_base_flange = T_base_cam @ geometry.invert_T(T_flange_cam)
    if float(np.linalg.norm(T_base_flange[:3, 3])) > p.reach_fraction * p.arm_reach_mm / 1000.0:
        return "reach"
    region_bottom_m = (p.region_center_mm[2] - p.region_size_mm[2] / 2.0) / 1000.0
    if T_base_cam[2, 3] < region_bottom_m or T_base_flange[2, 3] < region_bottom_m:
        return "below_region"
    for z in np.linspace(0.0, p.base_height_mm / 1000.0, 5):
        if project_inside(T_base_cam, cam, np.array([0.0, 0.0, z])):
            return "base_in_frustum"
    return None


def _nearest_neighbour_order(points: Mat) -> list[int]:
    n = len(points)
    if n == 0:
        return []
    remaining = list(range(1, n))
    order = [0]
    while remaining:
        last = points[order[-1]]
        d = np.linalg.norm(points[remaining] - last, axis=1)
        k = int(np.argmin(d))
        order.append(remaining.pop(k))
    return order


def _interleave_shells(accepted: list[Viewpoint], num_shells: int) -> list[Viewpoint]:
    tours: list[list[Viewpoint]] = []
    for shell in range(num_shells):
        members = [v for v in accepted if v.shell == shell]
        pts = np.array([v.t_m for v in members]) if members else np.empty((0, 3))
        tours.append([members[i] for i in _nearest_neighbour_order(pts)])
    ordered: list[Viewpoint] = []
    while any(tours):
        for tour in tours:
            if tour:
                ordered.append(tour.pop(0))
    return ordered


def _nn_direction_gaps_deg(dirs: Mat) -> tuple[float, float]:
    if len(dirs) < 2:
        return 0.0, 180.0
    cos = np.clip(dirs @ dirs.T, -1.0, 1.0)
    np.fill_diagonal(cos, -1.0)
    gaps = np.degrees(np.arccos(cos.max(axis=1)))
    return float(np.median(gaps)), float(gaps.max())


def _top_face_coverage(views: list[Viewpoint], p: PlannerParams, cam: CameraModel) -> float:
    cx, cy, cz = p.region_center_mm
    sx, sy, sz = p.region_size_mm
    xs = cx + (np.arange(COVERAGE_GRID) + 0.5) / COVERAGE_GRID * sx - sx / 2.0
    ys = cy + (np.arange(COVERAGE_GRID) + 0.5) / COVERAGE_GRID * sy - sy / 2.0
    transforms = [v.T_base_cam() for v in views]
    covered = 0
    for x in xs:
        for y in ys:
            pt = np.array([x, y, cz + sz / 2.0]) / 1000.0
            seen = sum(1 for T in transforms if project_inside(T, cam, pt))
            covered += int(seen >= COVERAGE_MIN_VIEWS)
    return covered / float(COVERAGE_GRID * COVERAGE_GRID)


def plan_viewpoints(p: PlannerParams, cam: CameraModel, T_flange_cam: Mat) -> Plan:
    rng = np.random.default_rng(p.seed)
    look_at = _look_at_points_m(p, rng)
    count = p.candidate_factor * p.num_poses
    directions = _directions_deg(p, rng, count)
    shells = _shells_mm(p)
    rolls = rng.uniform(-p.roll_jitter_deg, p.roll_jitter_deg, size=len(directions))

    accepted: list[Viewpoint] = []
    rejected: dict[str, int] = {}
    reachable = 0
    for i, (elev, az) in enumerate(directions):
        target = look_at[i % p.num_poses]
        shell = i % len(shells)
        position = target + shells[shell] / 1000.0 * _direction_vector(elev, az)
        rotation = geometry.look_at_rotation(position, target, math.radians(float(rolls[i])))
        T = geometry.make_T(rotation, position)
        reason = reject_reason(T, p, cam, T_flange_cam)
        if reason not in ("reach", "below_region"):
            reachable += 1
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        accepted.append(
            Viewpoint(
                index=-1,
                shell=shell,
                distance_mm=shells[shell],
                elevation_deg=elev,
                azimuth_deg=az,
                roll_deg=float(rolls[i]),
                look_at_m=[float(v) for v in target],
                t_m=[float(v) for v in position],
                q_wxyz=[float(v) for v in geometry.rotation_to_quat_wxyz(rotation)],
            )
        )

    ordered = _interleave_shells(accepted, len(shells))
    for k, v in enumerate(ordered):
        v.index = k
    first = ordered[: p.num_poses]
    dirs = (
        np.array([_direction_vector(v.elevation_deg, v.azimuth_deg) for v in first])
        if first
        else np.empty((0, 3))
    )
    gap_median, gap_max = _nn_direction_gaps_deg(dirs)
    stats = PlanStats(
        candidates=len(directions),
        accepted=len(ordered),
        rejected=rejected,
        reachable_window_fraction=reachable / len(directions) if directions else 0.0,
        top_face_coverage=_top_face_coverage(first, p, cam),
        nn_direction_gap_deg_median=gap_median,
        nn_direction_gap_deg_max=gap_max,
        shells_mm=shells,
    )
    return Plan(viewpoints=ordered, stats=stats)
