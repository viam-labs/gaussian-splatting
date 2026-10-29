"""`viam:gaussian-splatting:scan-state`: a sensor whose readings are the running scan's last frame record.

It exists so the sequence filter has a real tabular resource (`<scan_state>`, `Readings`) and so the
capture-control fallback path (data manager polling this sensor) carries the same payload the uploader sends.
The scanner pushes records through `do_command({"set_frame": ...})`, which works across module processes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from viam.components.sensor import Sensor
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import SensorReading, ValueTypes

IDLE_READING: dict[str, SensorReading] = {"state": "idle"}


class ScanState(Sensor, EasyResource):  # type: ignore[misc]  # API is Final on Sensor, ClassVar on EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "gaussian-splatting"), "scan-state")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._last_frame: dict[str, Any] | None = None

    async def get_readings(
        self,
        *,
        extra: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        **kwargs: object,
    ) -> Mapping[str, SensorReading]:
        if self._last_frame is None:
            return dict(IDLE_READING)
        return dict(self._last_frame)  # exactly the pushed payload: FrameRecord fields plus frame_extras

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs: object,
    ) -> Mapping[str, ValueTypes]:
        if "set_frame" in command:
            frame = command["set_frame"]
            if not isinstance(frame, Mapping):
                raise ValueError("set_frame expects a frame record mapping")
            self._last_frame = dict(frame)
            return {"ok": True}
        if "clear" in command:
            self._last_frame = None
            return {"ok": True}
        raise ValueError(f"unknown command, expected set_frame or clear: {sorted(command)}")
