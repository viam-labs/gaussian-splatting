from pathlib import Path

from gaussian_splatting import schema as s


def _frame(i: int) -> s.FrameRecord:
    return s.FrameRecord(
        index=i,
        file=f"images/{i:04d}.png",
        viewpoint=i,
        burst=0,
        t_request="2026-11-03T14:02:11.120Z",
        t_capture="2026-11-03T14:02:11.187Z",
        T_base_cam=s.Transform(t_m=[0.3, -0.1, 0.25], q_wxyz=[0.71, 0.0, 0.70, 0.0]),
        arm_end_pose=s.ArmEndPose(1, 2, 3, 0, 0, 1, 0),
        joints_deg=[0.0] * 6,
        sharpness=142.3,
        settle_s=0.6,
        width=1280,
        height=720,
    )


def test_frames_round_trip(tmp_path: Path) -> None:
    s.append_frame(tmp_path, _frame(1))
    s.append_frame(tmp_path, _frame(2))
    frames = s.read_frames(tmp_path)
    assert [f.index for f in frames] == [1, 2]
    frames[0].uploaded = s.UploadState(binary_id="abc", tabular_ok=True)
    s.rewrite_frames(tmp_path, frames)
    again = s.read_frames(tmp_path)
    assert again[0].uploaded.binary_id == "abc" and again[1].uploaded.binary_id is None


def test_manifest_round_trip(tmp_path: Path) -> None:
    m = s.Manifest(
        scan_id="x",
        camera_model=s.CameraModel(1280, 720, 900.0, 900.0, 640.0, 360.0, k1=-0.1),
        camera_in_arm=s.Transform([0, 0, 0.05], [1, 0, 0, 0]),
    )
    s.write_manifest(tmp_path, m)
    back = s.read_manifest(tmp_path)
    assert back.camera_model is not None and back.camera_model.colmap_model == "OPENCV"
    assert back.camera_model.colmap_params == [900.0, 900.0, 640.0, 360.0, -0.1, 0.0, 0.0, 0.0]
    assert back.camera_in_arm is not None and back.camera_in_arm.t_m == [0, 0, 0.05]


def test_camera_model_variants() -> None:
    assert s.CameraModel(1, 1, 1, 1, 0, 0).colmap_model == "PINHOLE"
    assert s.CameraModel(1, 1, 1, 1, 0, 0, k3=0.01).colmap_model == "FULL_OPENCV"
    assert len(s.CameraModel(1, 1, 1, 1, 0, 0, k3=0.01).colmap_params) == 12
