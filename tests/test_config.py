from typing import Any

import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from gaussian_splatting.module import config as c
from gaussian_splatting.module.scanner import Scanner

BASE: dict[str, Any] = {
    "arm": "lite6",
    "camera": "wrist",
    "motion": "builtin",
    "region_mm": {"center": [300.0, 0.0, 0.0], "size": [500.0, 500.0, 100.0]},
}


def test_defaults_and_dependencies() -> None:
    cfg = c.parse_scanner_config(BASE)
    assert cfg.required_dependencies() == ["lite6", "wrist", "builtin"]
    assert cfg.standoff_mm == (200.0, 450.0) and cfg.elevation_deg == (25.0, 80.0)
    assert cfg.upload is False and cfg.camera_intrinsics is None and cfg.camera_in_arm is None

    cfg = c.parse_scanner_config({**BASE, "scan_state": "scan-state", "upload": True, "dataset_id": "d"})
    assert cfg.required_dependencies() == ["lite6", "wrist", "builtin", "scan-state"]


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"motion": None}, "'motion' is required"),
        ({"upload": True}, "'scan_state' is required"),
        ({"region_mm": {"center": [0, 0, 0], "size": [1, 0, 1]}}, "positive"),
        ({"elevation_deg": [80, 30]}, "elevation_deg"),
        ({"azimuth_deg": [-200, 10]}, "azimuth_deg"),
        ({"standoff_mm": [100]}, "[min, max]"),
        ({"dwell_s": "fast"}, "dwell_s"),
        ({"camera_intrinsics": {"dist": {}}}, "camera_intrinsics"),
    ],
)
def test_rejections(bad: dict[str, object], message: str) -> None:
    attrs = {**BASE, **bad}
    attrs = {k: v for k, v in attrs.items() if v is not None}
    with pytest.raises(c.ConfigError, match=message):
        c.parse_scanner_config(attrs)


def test_fallback_attributes_parse() -> None:
    cfg = c.parse_scanner_config(
        {
            **BASE,
            "camera_intrinsics": {"K": {"fx": 900, "fy": 901, "cx": 640, "cy": 360}, "dist": {"k1": -0.1}},
            "camera_in_arm": {
                "translation": {"x": 0, "y": 0, "z": 50},
                "orientation": {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 1, "th": 90}},
            },
        }
    )
    assert (
        cfg.camera_intrinsics is not None
        and cfg.camera_intrinsics.fx == 900.0
        and cfg.camera_intrinsics.k1 == -0.1
    )
    assert cfg.camera_in_arm is not None
    assert np.allclose(cfg.camera_in_arm[:3, 3], [0, 0, 0.05])
    assert np.allclose(cfg.camera_in_arm[:3, 0], [0, 1, 0])  # 90 deg about z


def test_scan_request_overrides_config() -> None:
    cfg = c.parse_scanner_config({**BASE, "num_poses": 10, "seed": 5})
    req = c.ScanRequest.from_command({"num_poses": 3, "tags": ["a"], "dry_run": True}, cfg)
    assert req.num_poses == 3 and req.seed == 5 and req.tags == ["a"] and req.dry_run and not req.upload
    assert req.scan_id.startswith("scan-")
    assert req.to_dict()["region_mm"] == {"center": [300.0, 0.0, 0.0], "size": [500.0, 500.0, 100.0]}
    with pytest.raises(c.ConfigError):
        c.ScanRequest.from_command({"tags": "a"}, cfg)


def test_validate_config_through_proto() -> None:
    config = ComponentConfig(name="scanner", attributes=dict_to_struct({**BASE, "scan_state": "s"}))
    required, optional = Scanner.validate_config(config)
    assert list(required) == ["lite6", "wrist", "builtin", "s"] and list(optional) == []
    with pytest.raises(c.ConfigError):
        Scanner.validate_config(ComponentConfig(name="scanner", attributes=dict_to_struct({"arm": "a"})))
