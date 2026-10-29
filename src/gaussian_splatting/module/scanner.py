"""`viam:gaussian-splatting:scanner`: the generic service that owns planning, capture, the ledger and upload.

Integration pattern (spec section 4): depend on APIs, read intrinsics and the camera pose through the standard
interfaces, and fall back to config overrides only when a resource cannot provide them.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import os
import shutil
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self, cast

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.sensor import Sensor
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.proto.component.camera import GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import Motion
from viam.utils import ValueTypes, struct_to_dict

from gaussian_splatting import geometry, schema
from gaussian_splatting.geometry import Mat
from gaussian_splatting.module import capture
from gaussian_splatting.module.config import ScannerConfig, ScanRequest, parse_scanner_config
from gaussian_splatting.module.job import ScanJob
from gaussian_splatting.module.planner import Plan, PlannerParams, Viewpoint, plan_viewpoints
from gaussian_splatting.module.uploader import (
    CloudCredentials,
    DataClientHandle,
    UploadTarget,
    open_data_client,
)

BROWN_CONRADY = "brown_conrady"  # RDK order: k1, k2, p1, p2, k3


class ScanRefused(RuntimeError):
    """The machine lacks something a scan needs (intrinsics, camera frame); config is fine."""


def default_scans_root() -> Path:
    base = os.environ.get("VIAM_MODULE_DATA") or str(Path.home() / ".viam" / "gaussian-splatting")
    return Path(base) / "scans"


def module_version() -> str:
    try:
        return importlib.metadata.version("gaussian-splatting")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


class CameraWithProperties(capture.CameraLike, Protocol):
    async def get_properties(self) -> GetPropertiesResponse: ...


class Scanner(Generic, EasyResource):  # type: ignore[misc]  # API is Final on Generic, ClassVar on EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "gaussian-splatting"), "scanner")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.cfg: ScannerConfig | None = None
        self.arm: capture.ArmLike | None = None
        self.camera: CameraWithProperties | None = None
        self.motion: capture.MotionLike | None = None
        self.scan_state: capture.FrameSink | None = None
        self.scans_root: Path = default_scans_root()
        self._job: ScanJob | None = None
        self._task: asyncio.Task[None] | None = None

    # -- lifecycle --

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        # EasyResource.new only sets the name; viam-server does not call reconfigure after creation.
        scanner = cls(config.name)
        scanner.reconfigure(config, dependencies)
        return scanner

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        cfg = parse_scanner_config(struct_to_dict(config.attributes))
        return cfg.required_dependencies(), []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        cfg = parse_scanner_config(struct_to_dict(config.attributes))

        def dep(name: ResourceName) -> ResourceBase:
            if name not in dependencies:
                raise ValueError(f"dependency {name.name} ({name.subtype}) was not provided")
            return dependencies[name]

        self.bind(
            cfg,
            arm=cast(capture.ArmLike, dep(Arm.get_resource_name(cfg.arm))),
            camera=cast(CameraWithProperties, dep(Camera.get_resource_name(cfg.camera))),
            motion=cast(capture.MotionLike, dep(Motion.get_resource_name(cfg.motion))),
            scan_state=cast(capture.FrameSink, dep(Sensor.get_resource_name(cfg.scan_state)))
            if cfg.scan_state
            else None,
        )

    def bind(
        self,
        cfg: ScannerConfig,
        *,
        arm: capture.ArmLike,
        camera: CameraWithProperties,
        motion: capture.MotionLike,
        scan_state: capture.FrameSink | None,
    ) -> None:
        self.cfg = cfg
        self.arm = arm
        self.camera = camera
        self.motion = motion
        self.scan_state = scan_state

    async def close(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    # -- DoCommand --

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs: object,
    ) -> Mapping[str, ValueTypes]:
        if "start_scan" in command:
            return _drop_none(await self.start_scan(_as_mapping(command["start_scan"])))
        if "scan_status" in command:
            return _drop_none(self.scan_status())
        if "cancel_scan" in command:
            return _drop_none(self.cancel_scan())
        if "upload_scan" in command:
            return _drop_none(await self.upload_scan(_as_mapping(command["upload_scan"])))
        if "prune" in command:
            return _drop_none(self.prune(_as_mapping(command["prune"])))
        raise ValueError(
            f"unknown command {sorted(command)}; "
            "expected start_scan, scan_status, cancel_scan, upload_scan or prune"
        )

    def scan_status(self) -> dict[str, Any]:
        if self._job is None:
            return {"state": "idle"}
        return self._job.status()

    def cancel_scan(self) -> dict[str, Any]:
        if self._job is None or self._job.finished:
            return {"state": "idle" if self._job is None else self._job.state, "cancelled": False}
        self._job.cancel()
        return {"scan_id": self._job.request.scan_id, "state": self._job.state, "cancelled": True}

    async def start_scan(self, cmd: Mapping[str, Any]) -> dict[str, Any]:
        cfg = self._require_config()
        self._require_idle()
        request = ScanRequest.from_command(cmd, cfg)
        camera_model = await self._resolve_camera_model(cfg)
        T_flange_cam, source, pose_reader = await self._resolve_camera_in_arm(cfg)
        scan_dir = self.scans_root / request.scan_id

        if request.resume and (scan_dir / schema.MANIFEST_NAME).exists():
            manifest = schema.read_manifest(scan_dir)
            viewpoints = [Viewpoint.from_dict(d) for d in manifest.planned_viewpoints]
            manifest.request = dict(cmd)
            plan_stats = manifest.resolved.get("plan_stats", {})
        else:
            if (scan_dir / schema.FRAMES_NAME).exists():
                raise ValueError(
                    f"scan {request.scan_id} already has frames; pass resume=true or a new scan_id"
                )
            plan = self._plan(request, cfg, camera_model, T_flange_cam)
            viewpoints = plan.viewpoints
            plan_stats = plan.stats.to_dict()
            if request.dry_run:
                return {
                    "scan_id": request.scan_id,
                    "state": "done",
                    "dry_run": True,
                    "planned": len(viewpoints),
                    "stats": plan_stats,
                    "camera_model": camera_model.__dict__,
                    "camera_in_arm_source": source,
                    "viewpoints": [v.to_dict() for v in viewpoints],
                }
            manifest = self._new_manifest(request, cmd, cfg, camera_model, T_flange_cam, source, plan)
            scan_dir.mkdir(parents=True, exist_ok=True)
            (scan_dir / schema.IMAGES_DIR).mkdir(exist_ok=True)
            schema.write_manifest(scan_dir, manifest)

        upload_target, open_client, upload_note = self._upload_setup(request, cfg)
        job = ScanJob(
            request=request,
            scan_dir=scan_dir,
            manifest=manifest,
            viewpoints=viewpoints,
            deps=self._capture_deps(cfg, pose_reader, manifest),
            settings=self._capture_settings(request),
            camera_lock_command=cfg.camera_lock_command,
            upload_target=upload_target,
            open_client=open_client,
            upload_note=upload_note,
            logger=self.logger,
        )
        self._job = job
        self._task = asyncio.create_task(job.run_scan(), name=f"scan-{request.scan_id}")
        return {
            "scan_id": request.scan_id,
            "state": job.state,
            "planned": request.num_poses,
            "candidates": len(viewpoints),
            "scan_dir": str(scan_dir),
            "stats": plan_stats,
            "upload": upload_target is not None,
            "note": upload_note,
        }

    async def upload_scan(self, cmd: Mapping[str, Any]) -> dict[str, Any]:
        cfg = self._require_config()
        self._require_idle()
        scan_id = cmd.get("scan_id")
        if not isinstance(scan_id, str) or not scan_id:
            raise ValueError("upload_scan needs a scan_id")
        scan_dir = self.scans_root / scan_id
        if not (scan_dir / schema.MANIFEST_NAME).exists():
            raise ValueError(f"no scan {scan_id} under {self.scans_root}")
        manifest = schema.read_manifest(scan_dir)
        request = ScanRequest.from_command({**manifest.resolved, "upload": True, **cmd}, cfg)
        upload_target, open_client, upload_note = self._upload_setup(request, cfg)
        if upload_target is None:
            raise ValueError(upload_note or "upload is not possible")
        _, _, pose_reader = await self._resolve_camera_in_arm(cfg)
        job = ScanJob(
            request=request,
            scan_dir=scan_dir,
            manifest=manifest,
            viewpoints=[Viewpoint.from_dict(d) for d in manifest.planned_viewpoints],
            deps=self._capture_deps(cfg, pose_reader, manifest),
            settings=self._capture_settings(request),
            camera_lock_command={},
            upload_target=upload_target,
            open_client=open_client,
            logger=self.logger,
        )
        self._job = job
        self._task = asyncio.create_task(job.run_upload(), name=f"upload-{scan_id}")
        return {"scan_id": scan_id, "state": "uploading", "uploads_pending": job.uploads_pending}

    def prune(self, cmd: Mapping[str, Any]) -> dict[str, Any]:
        """Remove finished scans older than `days` (default 30). Files stay after upload otherwise."""
        days = float(cmd.get("days", 30))
        cutoff = datetime.now(UTC) - timedelta(days=days)
        removed: list[str] = []
        running = self._job.request.scan_id if self._job is not None and not self._job.finished else None
        for scan_dir in sorted(p for p in self.scans_root.glob("*") if p.is_dir()):
            if scan_dir.name == running or not (scan_dir / schema.MANIFEST_NAME).exists():
                continue
            manifest = schema.read_manifest(scan_dir)
            stamp = manifest.finished_at or manifest.started_at
            if stamp is None or schema.parse_rfc3339(stamp) > cutoff:
                continue
            shutil.rmtree(scan_dir)
            removed.append(scan_dir.name)
        return {"removed": removed, "days": days}

    # -- resolution helpers --

    def _require_config(self) -> ScannerConfig:
        if self.cfg is None or self.arm is None or self.camera is None or self.motion is None:
            raise RuntimeError("scanner is not configured")
        return self.cfg

    def _require_idle(self) -> None:
        if self._job is not None and not self._job.finished:
            raise ValueError(f"scan {self._job.request.scan_id} is {self._job.state}; one scan at a time")

    async def _resolve_camera_model(self, cfg: ScannerConfig) -> schema.CameraModel:
        assert self.camera is not None
        props = await capture.with_retry(self.logger, "camera get_properties", self.camera.get_properties)
        ip = props.intrinsic_parameters
        if ip.focal_x_px > 0 and ip.width_px > 0 and ip.height_px > 0:
            dist = props.distortion_parameters
            params = [float(v) for v in dist.parameters]
            if dist.model and dist.model != BROWN_CONRADY:
                self.logger.warning(
                    "unknown distortion model %r, assuming %s order", dist.model, BROWN_CONRADY
                )
            k1, k2, p1, p2, k3 = (params + [0.0] * 5)[:5]
            return schema.CameraModel(
                width=int(ip.width_px),
                height=int(ip.height_px),
                fx=float(ip.focal_x_px),
                fy=float(ip.focal_y_px),
                cx=float(ip.center_x_px),
                cy=float(ip.center_y_px),
                k1=k1,
                k2=k2,
                p1=p1,
                p2=p2,
                k3=k3,
                source="camera",
            )
        ci = cfg.camera_intrinsics
        if ci is None:
            raise ScanRefused(
                f"camera {cfg.camera} reports no intrinsics and 'camera_intrinsics' is not configured"
            )
        width = ci.width or int(ip.width_px)
        height = ci.height or int(ip.height_px)
        if width <= 0 or height <= 0:
            raise ScanRefused("'camera_intrinsics' needs width and height because the camera reports neither")
        return schema.CameraModel(
            width=width, height=height, fx=ci.fx, fy=ci.fy, cx=ci.cx, cy=ci.cy,
            k1=ci.k1, k2=ci.k2, p1=ci.p1, p2=ci.p2, k3=ci.k3, source="config",
        )  # fmt: skip

    async def _resolve_camera_in_arm(self, cfg: ScannerConfig) -> tuple[Mat, str, capture.CameraPoseReader]:
        """T_flange_cam from the frame system (camera pose against arm end pose) or from config."""
        assert self.arm is not None and self.motion is not None
        try:
            pif = await self.motion.get_pose(component_name=cfg.camera, destination_frame=f"{cfg.arm}_origin")
            end = await self.arm.get_end_position()
        except Exception as e:
            if cfg.camera_in_arm is None:
                raise ScanRefused(
                    f"frame system has no pose for camera {cfg.camera} in {cfg.arm}_origin ({e}) "
                    "and 'camera_in_arm' is not configured"
                ) from e
            self.logger.warning(
                "camera %s not in the frame system (%s); using camera_in_arm from config", cfg.camera, e
            )
            return cfg.camera_in_arm, "config", capture.FallbackPoseReader(self.arm, cfg.camera_in_arm)
        T_flange_cam = geometry.invert_T(geometry.pose_to_T(end)) @ geometry.pose_to_T(pif.pose)
        return T_flange_cam, "frame_system", capture.FrameSystemPoseReader(self.motion, self.arm, cfg.camera)

    def _plan(
        self, request: ScanRequest, cfg: ScannerConfig, cam: schema.CameraModel, T_flange_cam: Mat
    ) -> Plan:
        params = PlannerParams(
            region_center_mm=request.region.center_mm,
            region_size_mm=request.region.size_mm,
            standoff_mm=request.standoff_mm,
            elevation_deg=request.elevation_deg,
            azimuth_deg=request.azimuth_deg,
            num_poses=request.num_poses,
            seed=request.seed,
            arm_reach_mm=cfg.arm_reach_mm,
        )
        plan = plan_viewpoints(params, cam, T_flange_cam)
        if not plan.viewpoints:
            raise ScanRefused(f"no viewpoint passed the geometric filter: {plan.stats.rejected}")
        return plan

    def _new_manifest(
        self,
        request: ScanRequest,
        cmd: Mapping[str, Any],
        cfg: ScannerConfig,
        cam: schema.CameraModel,
        T_flange_cam: Mat,
        source: str,
        plan: Plan,
    ) -> schema.Manifest:
        resolved = request.to_dict()
        resolved["plan_stats"] = plan.stats.to_dict()
        resolved["arm_reach_mm"] = cfg.arm_reach_mm
        return schema.Manifest(
            scan_id=request.scan_id,
            request=dict(cmd),
            resolved=resolved,
            machine_id=os.environ.get("VIAM_MACHINE_ID"),
            part_id=os.environ.get("VIAM_MACHINE_PART_ID"),
            arm=cfg.arm,
            camera=cfg.camera,
            camera_model=cam,
            camera_in_arm=capture.transform_from_T(T_flange_cam),
            camera_in_arm_source=source,
            module_version=module_version(),
            started_at=schema.format_rfc3339(datetime.now(UTC)),
            state="planning",
            dataset_id=request.dataset_id if request.upload else None,
            planned_viewpoints=[v.to_dict() for v in plan.viewpoints],
        )

    def _capture_deps(
        self, cfg: ScannerConfig, pose_reader: capture.CameraPoseReader, manifest: schema.Manifest
    ) -> capture.CaptureDeps:
        assert self.arm is not None and self.camera is not None and self.motion is not None
        return capture.CaptureDeps(
            arm=self.arm,
            camera=self.camera,
            motion=self.motion,
            pose_reader=pose_reader,
            scan_state=self.scan_state,
            frame_extras=schema.frame_extras(manifest),
            logger=self.logger,
        )

    @staticmethod
    def _capture_settings(request: ScanRequest) -> capture.CaptureSettings:
        return capture.CaptureSettings(
            dwell_s=request.dwell_s,
            min_sharpness=request.min_sharpness,
            images_per_pose=request.images_per_pose,
            keep_all=request.keep_all,
        )

    def _upload_setup(
        self, request: ScanRequest, cfg: ScannerConfig
    ) -> tuple[UploadTarget | None, Callable[[], Awaitable[DataClientHandle]] | None, str | None]:
        """(target, client opener, note). A note means upload was requested but cannot run."""
        if not request.upload:
            return None, None, None
        if cfg.scan_state is None:
            return None, None, "upload disabled: 'scan_state' is not configured"
        creds = CloudCredentials.from_env()
        if creds is None:
            return (
                None,
                None,
                "upload disabled: VIAM_API_KEY, VIAM_API_KEY_ID and VIAM_MACHINE_PART_ID are not all set",
            )
        target = UploadTarget(
            part_id=creds.part_id,
            camera_name=cfg.camera,
            scan_state_name=cfg.scan_state,
            dataset_id=request.dataset_id,
            tags=tuple(request.tags),
        )

        async def opener() -> DataClientHandle:
            return await open_data_client(creds)

        return target, opener, None


def _drop_none(response: Mapping[str, Any]) -> dict[str, Any]:
    """Protobuf null decodes as 0 on the SDK client side, so absent fields are omitted, not sent as null."""
    return {k: v for k, v in response.items() if v is not None}


def _as_mapping(value: ValueTypes) -> Mapping[str, Any]:
    if value is None or value is True:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"command payload must be a mapping, got {type(value).__name__}")
    return value
