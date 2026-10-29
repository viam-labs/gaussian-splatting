"""colmap stage: scan dir (images + known poses) -> 3DGRUT-ready COLMAP scene with pycolmap.

Follows COLMAP's FAQ "Reconstruct sparse model from known camera poses": extract features with a single camera
of the known model, write a reconstruction holding the known world-to-camera poses and no points, match
exhaustively, triangulate with the poses fixed. Optionally bundle-adjust with intrinsics fixed into a separate
`sparse/0_ba` as a diagnostic of pose quality; training always uses the unrefined poses so the splat stays
rigidly attached to the arm base.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from gaussian_splatting import geometry as g
from gaussian_splatting import schema as s
from gaussian_splatting.build.stages import StageError

log = logging.getLogger("splat-build")

SPARSE_DIR = Path("sparse") / "0"
SPARSE_BA_DIR = Path("sparse") / "0_ba"
SPARSE_KNOWN_DIR = Path("sparse_known")
MASKS_DIR = "masks"
DATABASE_NAME = "database.db"
REPORT_NAME = "report.json"

MIN_OBSERVATIONS_PER_IMAGE = 500
MIN_POINTS_FOR_COLMAP_INIT = (
    100  # below this the sparse model is useless as a seed; train with initialization=random
)


@dataclass
class AcceptanceRow:
    check: str
    threshold: str
    value: str
    passed: bool | None  # None: not evaluated (data not available)


@dataclass
class PoseCorrection:
    median_mm: float
    max_mm: float
    median_deg: float
    max_deg: float
    num_images: int
    gauge_scale: float = 1.0  # Sim3 scale that aligned the BA result back onto the known poses


@dataclass
class ColmapReport:
    num_frames: int
    num_images_registered: int
    num_images_with_min_observations: int
    fraction_images_with_min_observations: float
    mean_reprojection_error_px: float
    num_points3d: int
    init_random: bool
    camera_model: str
    camera_params: list[float]
    pose_correction: PoseCorrection | None = None
    acceptance: list[AcceptanceRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard link when the filesystem allows it (no duplicate image bytes), copy otherwise."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _stage_images(scan_dir: Path, colmap_dir: Path, frames: list[s.FrameRecord]) -> list[str]:
    images_dir = colmap_dir / s.IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for fr in frames:
        src = scan_dir / fr.file
        if not src.exists():
            raise StageError(f"frame {fr.index} refers to a missing image: {src}")
        name = Path(fr.file).name
        link_or_copy(src, images_dir / name)
        names.append(name)
    return names


def _stage_masks(colmap_dir: Path, static_mask: Path, names: list[str]) -> Path:
    """COLMAP wants `<mask_path>/<image_name>.png`, 3DGRUT `images/<stem>_mask.png`: same bytes, two names."""
    if not static_mask.exists():
        raise StageError(f"static mask not found: {static_mask}")
    masks_dir = colmap_dir / MASKS_DIR
    masks_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        link_or_copy(static_mask, masks_dir / f"{name}.png")
        link_or_copy(static_mask, colmap_dir / s.IMAGES_DIR / f"{Path(name).stem}_mask.png")
    return masks_dir


def _make_camera(cam: s.CameraModel, camera_id: int = 1) -> pycolmap.Camera:
    camera = pycolmap.Camera()
    camera.camera_id = camera_id
    camera.model = pycolmap.CameraModelId(cam.colmap_model)
    camera.width = cam.width
    camera.height = cam.height
    camera.params = [float(v) for v in cam.colmap_params]
    return camera


def _frame_cam_from_world(fr: s.FrameRecord) -> pycolmap.Rigid3d:
    r_base_cam = g.quat_wxyz_to_rotation(np.asarray(fr.T_base_cam.q_wxyz, dtype=np.float64))
    t_base_cam = g.make_T(r_base_cam, np.asarray(fr.T_base_cam.t_m, dtype=np.float64))
    q_wxyz, t_m = g.colmap_world_to_cam(t_base_cam)
    rotation = pycolmap.Rotation3d(
        np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    )  # pycolmap takes xyzw
    return pycolmap.Rigid3d(rotation, t_m)


def write_known_poses(
    database_path: Path, cam: s.CameraModel, frames: list[s.FrameRecord], output_dir: Path
) -> pycolmap.Reconstruction:
    """Reconstruction with the database's image ids/names, the known camera and poses, and no points."""
    db = pycolmap.Database.open(str(database_path))
    try:
        db_images = {img.name: img for img in db.read_all_images()}
        db_cameras = db.read_all_cameras()
    finally:
        db.close()
    if len(db_cameras) != 1:
        raise StageError(f"expected one camera in the database (single_camera), found {len(db_cameras)}")
    camera_id = int(db_cameras[0].camera_id)

    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(_make_camera(cam, camera_id))
    missing: list[str] = []
    for fr in frames:
        name = Path(fr.file).name
        db_img = db_images.get(name)
        if db_img is None:
            missing.append(name)
            continue
        image = pycolmap.Image(name=name, camera_id=camera_id, image_id=int(db_img.image_id))
        rec.add_image_with_trivial_frame(image, _frame_cam_from_world(fr))
    if missing:
        raise StageError(
            f"{len(missing)} images missing from the COLMAP database after extraction: {missing[:5]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(output_dir))
    return rec


def _camera_centres(rec: pycolmap.Reconstruction, image_ids: list[int]) -> np.ndarray:
    centres = []
    for image_id in image_ids:
        pose = rec.image(image_id).cam_from_world()
        centres.append(-(pose.rotation.matrix().T @ pose.translation))
    return np.asarray(centres, dtype=np.float64).reshape(-1, 3)


def align_to(before: pycolmap.Reconstruction, after: pycolmap.Reconstruction) -> float:
    """Remove the 7-DoF gauge freedom of bundle adjustment: Sim3-align `after` onto `before` (in place) using
    camera centres. Returns the fitted scale. Without this, a drift of the whole scene (which costs nothing in
    reprojection) would be reported as a per-image pose correction."""
    ids = [i for i in after.reg_image_ids() if before.exists_image(i)]
    if len(ids) < 3:
        return 1.0
    sim3 = pycolmap.estimate_sim3d(_camera_centres(after, ids), _camera_centres(before, ids))
    if sim3 is None:
        log.warning("Sim3 alignment of the refined model failed; reporting raw pose corrections")
        return 1.0
    after.transform(sim3)
    return float(sim3.scale)


def _pose_correction(before: pycolmap.Reconstruction, after: pycolmap.Reconstruction) -> PoseCorrection:
    d_mm: list[float] = []
    d_deg: list[float] = []
    for image_id in after.reg_image_ids():
        if not before.exists_image(image_id):
            continue
        a = before.image(image_id).cam_from_world()
        b = after.image(image_id).cam_from_world()
        ca = -(a.rotation.matrix().T @ a.translation)  # camera centre in world
        cb = -(b.rotation.matrix().T @ b.translation)
        d_mm.append(float(np.linalg.norm(ca - cb)) * 1000.0)
        d_deg.append(g.rotation_angle_deg(a.rotation.matrix(), b.rotation.matrix()))
    if not d_mm:
        return PoseCorrection(0.0, 0.0, 0.0, 0.0, 0)
    return PoseCorrection(
        median_mm=statistics.median(d_mm),
        max_mm=max(d_mm),
        median_deg=statistics.median(d_deg),
        max_deg=max(d_deg),
        num_images=len(d_mm),
    )


def refine_extrinsics(
    rec: pycolmap.Reconstruction, output_dir: Path
) -> tuple[pycolmap.Reconstruction, PoseCorrection]:
    """Bundle-adjust poses and points with intrinsics fixed. Diagnostic only; returns the refined copy."""
    refined = pycolmap.Reconstruction(rec)
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_sensor_from_rig = False
    options.refine_rig_from_world = True
    options.refine_points3D = True
    options.print_summary = False
    pycolmap.bundle_adjustment(refined, options)
    gauge_scale = align_to(rec, refined)
    log.info("BA gauge removed by Sim3 alignment (scale %.6f)", gauge_scale)
    output_dir.mkdir(parents=True, exist_ok=True)
    refined.write(str(output_dir))
    correction = _pose_correction(rec, refined)
    correction.gauge_scale = gauge_scale
    return refined, correction


def _fmt(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}"


def acceptance_table(
    report: ColmapReport, frames: list[s.FrameRecord], manifest: s.Manifest | None
) -> list[AcceptanceRow]:
    rows: list[AcceptanceRow] = []
    num_poses = None
    min_sharpness = None
    if manifest is not None:
        num_poses = manifest.resolved.get("num_poses", manifest.request.get("num_poses"))
        min_sharpness = manifest.resolved.get("min_sharpness")
    if num_poses:
        rows.append(
            AcceptanceRow(
                "Poses captured",
                f">= 0.8 x {num_poses}",
                f"{len(frames)}",
                len(frames) >= 0.8 * float(num_poses),
            )
        )
    else:
        rows.append(
            AcceptanceRow("Poses captured", ">= 0.8 x num_poses", f"{len(frames)} (num_poses unknown)", None)
        )
    sharpness = [fr.sharpness for fr in frames]
    median_sharp = statistics.median(sharpness) if sharpness else 0.0
    if min_sharpness is not None and sharpness:
        rows.append(
            AcceptanceRow(
                "Sharpness",
                f"every frame >= {min_sharpness}",
                f"min {_fmt(min(sharpness), 1)}, median {_fmt(median_sharp, 1)}",
                min(sharpness) >= float(min_sharpness),
            )
        )
    else:
        rows.append(
            AcceptanceRow(
                "Sharpness", "every frame >= min_sharpness", f"median {_fmt(median_sharp, 1)}", None
            )
        )
    rows.append(
        AcceptanceRow("Coverage", "no gap > 25 deg; >= 90 % of top face by >= 3 views", "needs planner", None)
    )
    rows.append(
        AcceptanceRow(
            "COLMAP registration",
            f">= 95 % images with >= {MIN_OBSERVATIONS_PER_IMAGE} observations",
            f"{100.0 * report.fraction_images_with_min_observations:.1f} % "
            f"({report.num_images_with_min_observations}/{report.num_frames})",
            report.fraction_images_with_min_observations >= 0.95,
        )
    )
    rows.append(
        AcceptanceRow(
            "Reprojection error",
            "mean < 1.0 px",
            f"{_fmt(report.mean_reprojection_error_px, 3)} px",
            report.mean_reprojection_error_px < 1.0 and report.num_points3d > 0,
        )
    )
    if report.pose_correction is not None and report.pose_correction.num_images > 0:
        pc = report.pose_correction
        rows.append(
            AcceptanceRow(
                "Pose correction (BA)",
                "median < 2 mm, < 0.3 deg; max < 3 mm, < 0.5 deg",
                f"median {_fmt(pc.median_mm)} mm / {_fmt(pc.median_deg, 3)} deg; "
                f"max {_fmt(pc.max_mm)} mm / {_fmt(pc.max_deg, 3)} deg",
                pc.median_mm < 2.0 and pc.median_deg < 0.3 and pc.max_mm < 3.0 and pc.max_deg < 0.5,
            )
        )
    else:
        rows.append(AcceptanceRow("Pose correction (BA)", "median < 2 mm, < 0.3 deg", "not run", None))
    rows.append(AcceptanceRow("Scale (fiducial)", "PnP vs frame-system residual < 3 mm", "no fiducial", None))
    return rows


def format_acceptance(rows: list[AcceptanceRow]) -> str:
    status = {True: "PASS", False: "FAIL", None: "n/a"}
    widths = (
        max(len(r.check) for r in rows),
        max(len(r.threshold) for r in rows),
        max(len(r.value) for r in rows),
    )
    lines = [
        f"{'Check':<{widths[0]}}  {'Threshold':<{widths[1]}}  {'Value':<{widths[2]}}  Result",
        "-" * (sum(widths) + 12),
    ]
    for r in rows:
        lines.append(
            f"{r.check:<{widths[0]}}  {r.threshold:<{widths[1]}}  {r.value:<{widths[2]}}  {status[r.passed]}"
        )
    return "\n".join(lines)


def _empty_points(rec: pycolmap.Reconstruction) -> None:
    for point3d_id in list(rec.point3D_ids()):
        rec.delete_point3D(point3d_id)


def run_colmap(
    scan_dir: Path,
    colmap_dir: Path,
    *,
    static_mask: Path | None = None,
    refine: bool = False,
    seed: int = 0,
    num_threads: int = -1,
) -> ColmapReport:
    frames = s.read_frames(scan_dir)
    if not frames:
        raise StageError(f"no frames in {scan_dir / s.FRAMES_NAME}")
    manifest = s.read_manifest(scan_dir) if (scan_dir / s.MANIFEST_NAME).exists() else None
    if manifest is None or manifest.camera_model is None:
        raise StageError(f"{scan_dir / s.MANIFEST_NAME} has no camera_model; splat-build needs intrinsics")
    cam = manifest.camera_model

    colmap_dir.mkdir(parents=True, exist_ok=True)
    names = _stage_images(scan_dir, colmap_dir, frames)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = cam.colmap_model
    reader.camera_params = ",".join(repr(float(v)) for v in cam.colmap_params)
    if static_mask is not None:
        reader.mask_path = str(_stage_masks(colmap_dir, static_mask, names))
        log.info("static mask applied to %d images", len(names))

    database_path = colmap_dir / DATABASE_NAME
    if database_path.exists():
        database_path.unlink()  # stage folder is ours; a stale database would carry old features
    pycolmap.set_random_seed(seed)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = num_threads
    log.info(
        "extracting SIFT features for %d images (%s %s)", len(names), cam.colmap_model, cam.colmap_params
    )
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(colmap_dir / s.IMAGES_DIR),
        image_names=names,  # keeps `<stem>_mask.png` files in images/ out of the database
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.cpu,
    )

    known = write_known_poses(database_path, cam, frames, colmap_dir / SPARSE_KNOWN_DIR)
    log.info("wrote %d known poses to %s", known.num_images(), colmap_dir / SPARSE_KNOWN_DIR)

    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = num_threads
    log.info("exhaustive matching")
    pycolmap.match_exhaustive(
        database_path=str(database_path), matching_options=matching, device=pycolmap.Device.cpu
    )

    pipeline = pycolmap.IncrementalPipelineOptions()
    pipeline.num_threads = num_threads
    pipeline.random_seed = seed
    sparse_dir = colmap_dir / SPARSE_DIR
    sparse_dir.mkdir(parents=True, exist_ok=True)
    log.info("triangulating with fixed poses")
    rec = pycolmap.triangulate_points(
        reconstruction=known,
        database_path=str(database_path),
        image_path=str(colmap_dir / s.IMAGES_DIR),
        output_path=str(sparse_dir),
        clear_points=True,
        options=pipeline,
        refine_intrinsics=False,
    )

    notes: list[str] = []
    num_points = int(rec.num_points3D())
    init_random = num_points < MIN_POINTS_FOR_COLMAP_INIT
    if init_random:
        notes.append(
            f"only {num_points} points triangulated (< {MIN_POINTS_FOR_COLMAP_INIT}); "
            "wrote an empty points3D and flagged initialization=random for 3DGRUT"
        )
        log.warning("%s", notes[-1])
        _empty_points(rec)
        rec.write(str(sparse_dir))

    obs_per_image = [int(rec.image(i).num_points3D) for i in rec.reg_image_ids()]
    with_min = sum(1 for n in obs_per_image if n >= MIN_OBSERVATIONS_PER_IMAGE)
    report = ColmapReport(
        num_frames=len(frames),
        num_images_registered=int(rec.num_reg_images()),
        num_images_with_min_observations=with_min,
        fraction_images_with_min_observations=with_min / len(frames),
        mean_reprojection_error_px=float(rec.compute_mean_reprojection_error()) if num_points else 0.0,
        num_points3d=int(rec.num_points3D()),
        init_random=init_random,
        camera_model=cam.colmap_model,
        camera_params=cam.colmap_params,
        notes=notes,
    )

    if refine and not init_random:
        log.info("bundle adjusting poses with intrinsics fixed (diagnostic)")
        _, report.pose_correction = refine_extrinsics(rec, colmap_dir / SPARSE_BA_DIR)
    elif refine:
        notes.append("--refine-extrinsics skipped: no points to adjust against")

    report.acceptance = acceptance_table(report, frames, manifest)
    (colmap_dir / REPORT_NAME).write_text(json.dumps(report.to_dict(), indent=2))
    return report
