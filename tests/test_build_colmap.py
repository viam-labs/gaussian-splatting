"""End-to-end COLMAP stage on the synthetic textured plane.

The assertions on the triangulated points are the convention test: if the world-to-camera conversion, the
quaternion order or the metre scale were wrong, points would not land on z=0 inside the 1.2 m plane.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from gaussian_splatting import schema as s
from gaussian_splatting.build import colmap_stage as c
from tests.synthetic_scene import PLANE_HALF_M, SyntheticScene, build_scene


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return build_scene(tmp_path_factory.mktemp("scene") / "scan", num_views=25)


@pytest.fixture(scope="module")
def colmap_run(
    scene: SyntheticScene, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, c.ColmapReport]:
    colmap_dir = tmp_path_factory.mktemp("colmap")
    report = c.run_colmap(scene.scan_dir, colmap_dir, refine=True)
    print(c.format_acceptance(report.acceptance))
    return colmap_dir, report


def test_registration_and_reprojection(colmap_run: tuple[Path, c.ColmapReport]) -> None:
    _, report = colmap_run
    assert report.num_images_registered == report.num_frames == 25
    assert report.fraction_images_with_min_observations >= 0.9
    assert report.mean_reprojection_error_px < 1.5
    assert not report.init_random
    assert report.num_points3d > 1000


def test_points_lie_on_the_plane_in_metres(colmap_run: tuple[Path, c.ColmapReport]) -> None:
    colmap_dir, _ = colmap_run
    rec = pycolmap.Reconstruction(str(colmap_dir / c.SPARSE_DIR))
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    median_abs_z_m = float(np.median(np.abs(xyz[:, 2])))
    assert median_abs_z_m < 0.01, f"median |z| = {median_abs_z_m * 1000:.1f} mm"
    inside = np.all(np.abs(xyz[:, :2]) <= PLANE_HALF_M + 0.02, axis=1)
    assert inside.mean() > 0.99


def test_poses_written_match_the_frames(
    scene: SyntheticScene, colmap_run: tuple[Path, c.ColmapReport]
) -> None:
    colmap_dir, _ = colmap_run
    rec = pycolmap.Reconstruction(str(colmap_dir / c.SPARSE_DIR))
    for fr, t_base_cam in zip(scene.frames, scene.poses_T_base_cam, strict=True):
        img = rec.find_image_with_name(Path(fr.file).name)
        assert img is not None
        cam_from_world = img.cam_from_world()
        centre = -(cam_from_world.rotation.matrix().T @ cam_from_world.translation)
        assert np.allclose(centre, t_base_cam[:3, 3], atol=1e-6)
        assert np.allclose(cam_from_world.rotation.matrix(), t_base_cam[:3, :3].T, atol=1e-6)
    cam = rec.cameras[next(iter(rec.cameras))]
    assert cam.model.name == "PINHOLE"
    assert list(cam.params) == scene.camera.colmap_params


def test_bundle_adjustment_barely_moves_perfect_poses(colmap_run: tuple[Path, c.ColmapReport]) -> None:
    colmap_dir, report = colmap_run
    assert report.pose_correction is not None
    assert report.pose_correction.num_images == 25
    assert report.pose_correction.median_mm < 2.0
    assert report.pose_correction.median_deg < 0.3
    assert (colmap_dir / c.SPARSE_BA_DIR / "images.bin").exists()
    assert (colmap_dir / c.SPARSE_DIR / "images.bin").exists()  # training input is the unrefined one


def test_acceptance_rows_and_report_file(colmap_run: tuple[Path, c.ColmapReport]) -> None:
    colmap_dir, report = colmap_run
    by_check = {row.check: row for row in report.acceptance}
    assert by_check["COLMAP registration"].passed is True
    assert by_check["Reprojection error"].passed is True
    assert by_check["Pose correction (BA)"].passed is True
    assert by_check["Coverage"].passed is None
    assert (colmap_dir / c.REPORT_NAME).exists()
    assert (colmap_dir / c.SPARSE_KNOWN_DIR / "cameras.bin").exists()
    assert not (colmap_dir / s.IMAGES_DIR / "0001_mask.png").exists()


def test_static_mask_is_written_in_both_namings(scene: SyntheticScene, tmp_path: Path) -> None:
    import cv2

    mask = np.full((scene.camera.height, scene.camera.width), 255, dtype=np.uint8)
    mask[: scene.camera.height // 3, :] = 0  # ignore the top third
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)
    colmap_dir = tmp_path / "colmap"
    report = c.run_colmap(scene.scan_dir, colmap_dir, static_mask=mask_path)
    assert (colmap_dir / c.MASKS_DIR / "0001.png.png").exists()
    assert (colmap_dir / s.IMAGES_DIR / "0001_mask.png").exists()
    assert report.num_images_registered == 25
    # Masked-out keypoints never enter the database: no observation in the top third of any image.
    rec = pycolmap.Reconstruction(str(colmap_dir / c.SPARSE_DIR))
    for image_id in rec.reg_image_ids():
        ys = [p.xy[1] for p in rec.image(image_id).points2D if p.has_point3D()]
        assert ys and min(ys) >= scene.camera.height // 3 - 1


def test_textureless_images_do_not_crash(tmp_path: Path) -> None:
    import cv2

    scan_dir = tmp_path / "scan"
    scene = build_scene(scan_dir, num_views=5)
    for fr in scene.frames:
        cv2.imwrite(
            str(scan_dir / fr.file), np.full((scene.camera.height, scene.camera.width, 3), 90, np.uint8)
        )
    report = c.run_colmap(scan_dir, tmp_path / "colmap", refine=True)
    assert report.init_random is True
    assert report.num_points3d == 0
    rec = pycolmap.Reconstruction(str(tmp_path / "colmap" / c.SPARSE_DIR))
    assert rec.num_points3D() == 0 and rec.num_reg_images() == 5
