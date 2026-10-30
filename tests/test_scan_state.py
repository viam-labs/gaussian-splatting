from typing import Any

import pytest

from gaussian_splatting.module.scan_state import ScanState


async def test_readings_follow_set_frame() -> None:
    sensor = ScanState("scan-state")
    assert await sensor.get_readings() == {"state": "idle"}

    record: dict[str, Any] = {
        "index": 3,
        "viewpoint": 2,
        "T_base_cam": {"t_m": [0.1, 0.2, 0.3], "q_wxyz": [1, 0, 0, 0]},
    }
    assert await sensor.do_command({"set_frame": record}) == {"ok": True}
    readings = await sensor.get_readings()
    assert dict(readings) == record, "readings are exactly the pushed payload"

    await sensor.do_command({"clear": True})
    assert await sensor.get_readings() == {"state": "idle"}


async def test_rejects_bad_commands() -> None:
    sensor = ScanState("scan-state")
    with pytest.raises(ValueError):
        await sensor.do_command({"set_frame": 5})
    with pytest.raises(ValueError):
        await sensor.do_command({"bogus": {}})
