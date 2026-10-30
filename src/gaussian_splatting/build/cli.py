"""`splat-build`: sequence dataset -> COLMAP -> 3DGRUT -> splat, as cached stages under a work dir.

    splat-build --dataset-id <id> --sequence-id <id> --work runs/ [--until <stage>] [--force <stage>]

Stage folders (spec §10): `<work>/<dataset_id>/export`, then `<work>/<run>/{join,colmap,train,splat,finalize}`
where `<run>` is the sequence id (or the scan id with `--scan-dir`). A stage is skipped when `<folder>/.done`
exists; `--force <stage>` archives that folder and redoes it and everything after.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from gaussian_splatting import schema as s
from gaussian_splatting.build import colmap_stage, export, export_splat, finalize, join, train
from gaussian_splatting.build.stages import STAGES, StageError, StagePlan, is_done, run_stage, stage_index

log = logging.getLogger("splat-build")

app = typer.Typer(add_completion=False, help=__doc__, rich_markup_mode=None)


@dataclass
class BuildOptions:
    dataset_id: str | None
    sequence_id: str | None
    work: Path
    until: str | None = None
    force: str | None = None
    export_dir: Path | None = None
    scan_dir: Path | None = None
    threedgrut_dir: Path | None = None
    threedgrut_python: str | None = None
    run_train: bool = False
    finalize: bool = False
    static_mask: Path | None = None
    refine_extrinsics: bool = False
    camera_override: s.CameraModel | None = None
    viam_bin: str = "viam"
    seed: int = 0


def _camera_override(
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    width: int | None,
    height: int | None,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    k3: float,
) -> s.CameraModel | None:
    given = [fx, fy, cx, cy, width, height]
    if all(v is None for v in given):
        return None
    if any(v is None for v in given):
        raise StageError("camera override needs all of --fx --fy --cx --cy --width --height")
    assert fx is not None and fy is not None and cx is not None and cy is not None
    assert width is not None and height is not None
    return s.CameraModel(width, height, fx, fy, cx, cy, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3, source="config")


def _run_id(opts: BuildOptions) -> str:
    if opts.sequence_id:
        return opts.sequence_id
    if opts.scan_dir is not None and (opts.scan_dir / s.MANIFEST_NAME).exists():
        return s.read_manifest(opts.scan_dir).scan_id
    raise StageError("need --sequence-id (or --scan-dir with a manifest.json to name the run)")


def _print_acceptance(colmap_dir: Path) -> None:
    report_path = colmap_dir / colmap_stage.REPORT_NAME
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text())
    rows = [colmap_stage.AcceptanceRow(**r) for r in report.get("acceptance", [])]
    if rows:
        typer.echo(colmap_stage.format_acceptance(rows))
        typer.echo(f"(details in {report_path})")


def run_build(opts: BuildOptions) -> None:
    if opts.until is not None:
        stage_index(opts.until)
    if opts.force is not None:
        stage_index(opts.force)
    plan = StagePlan(until=opts.until, force=opts.force)
    run_id = _run_id(opts)
    run_root = opts.work / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    log.info("run %s under %s", run_id, run_root)

    # ---- export (per dataset) ----
    layout: export.ExportLayout | None = None
    if opts.scan_dir is None:
        if opts.export_dir is not None:
            layout = export.resolve_export(None, opts.work, opts.export_dir, viam_bin=opts.viam_bin)
            log.info("[export] using existing export at %s", opts.export_dir)
        else:
            if opts.dataset_id is None:
                raise StageError("need --dataset-id, or --export-dir / --scan-dir for an offline run")
            export_root = opts.work / opts.dataset_id / "export"
            dataset_id = opts.dataset_id

            def do_export() -> dict[str, Any]:
                layout_now = export.resolve_export(dataset_id, opts.work, None, viam_bin=opts.viam_bin)
                return {"parquet_dir": str(layout_now.parquet_dir)}

            run_stage(plan, "export", export_root, do_export)
            if is_done(export_root):
                layout = export.ExportLayout(root=export_root, parquet_dir=export.unpack_parquet(export_root))
    else:
        log.info("[export] skipped: --scan-dir %s", opts.scan_dir)
    if plan.stops_after("export"):
        return

    # ---- join ----
    join_dir = run_root / "join"

    def do_join() -> dict[str, Any]:
        if opts.scan_dir is not None:
            n = join.copy_scan_dir(opts.scan_dir, join_dir, opts.camera_override)
            return {"source": str(opts.scan_dir), "num_frames": n}
        assert layout is not None and opts.sequence_id is not None
        report = join.join_sequence(
            layout,
            opts.sequence_id,
            join_dir,
            dataset_id=opts.dataset_id,
            camera_override=opts.camera_override,
        )
        return report.to_dict()

    run_stage(plan, "join", join_dir, do_join)
    if plan.stops_after("join"):
        return

    # ---- colmap ----
    colmap_dir = run_root / "colmap"

    def do_colmap() -> dict[str, Any]:
        report = colmap_stage.run_colmap(
            join_dir, colmap_dir, static_mask=opts.static_mask, refine=opts.refine_extrinsics, seed=opts.seed
        )
        return {
            "num_points3d": report.num_points3d,
            "fraction_images_with_min_observations": report.fraction_images_with_min_observations,
            "mean_reprojection_error_px": report.mean_reprojection_error_px,
            "init_random": report.init_random,
        }

    run_stage(plan, "colmap", colmap_dir, do_colmap)
    _print_acceptance(colmap_dir)
    if plan.stops_after("colmap"):
        return

    # ---- train ----
    train_dir = run_root / "train"
    colmap_report: dict[str, Any] | None = None
    report_path = colmap_dir / colmap_stage.REPORT_NAME
    if report_path.exists():
        colmap_report = json.loads(report_path.read_text())
    init_random = bool(colmap_report.get("init_random", False)) if colmap_report else False
    threedgrut_dir = train.resolve_threedgrut_dir(opts.threedgrut_dir)
    train_out = train_dir / train.OUT_SUBDIR

    def do_train() -> dict[str, Any] | None:
        cmd = train.build_train_command(
            colmap_dir,
            train_out,
            run_id,
            init_random=init_random,
            python=train.resolve_python(opts.threedgrut_python),
        )
        script = train.write_command(train_dir, cmd, threedgrut_dir)
        typer.echo("3DGRUT command:\n  " + " ".join(cmd))
        log.info("wrote %s", script)
        if not opts.run_train:
            log.info(
                "[train] not running (pass --run-train with --threedgrut-dir or %s)", train.ENV_THREEDGRUT_DIR
            )
            return None
        if threedgrut_dir is None:
            raise StageError(f"--run-train needs --threedgrut-dir or {train.ENV_THREEDGRUT_DIR}")
        train.run_training(cmd, threedgrut_dir, train_dir / train.LOG_FILE)
        return {"command": cmd, "threedgrut_dir": str(threedgrut_dir), "init_random": init_random}

    run_stage(plan, "train", train_dir, do_train)
    if plan.stops_after("train"):
        return

    # ---- export-splat ----
    splat_dir = run_root / "splat"

    def do_export_splat() -> dict[str, Any] | None:
        return export_splat.export_splat(
            train_out, run_id, splat_dir, colmap_report=colmap_report, trained=is_done(train_dir)
        )

    run_stage(plan, "export-splat", splat_dir, do_export_splat)
    if plan.stops_after("export-splat"):
        return

    # ---- finalize (opt-in) ----
    if opts.finalize:
        finalize_dir = run_root / "finalize"
        if not opts.dataset_id or not opts.sequence_id:
            raise StageError("--finalize needs --dataset-id and --sequence-id")
        dataset_id_f, sequence_id_f = opts.dataset_id, opts.sequence_id

        def do_finalize() -> dict[str, Any]:
            finalize.run_finalize(dataset_id_f, sequence_id_f)
            return {"dataset_id": dataset_id_f, "sequence_id": sequence_id_f}

        run_stage(plan, "finalize", finalize_dir, do_finalize)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    try:
        import pycolmap

        pycolmap.logging.minloglevel = 0 if verbose else 2  # COLMAP's own glog: show only warnings unless -v
    except ImportError:  # pragma: no cover - build extra not installed
        pass


StageArg = Annotated[str | None, typer.Option(help=f"one of {', '.join(STAGES)}")]


@app.command()
def main(
    dataset_id: Annotated[str | None, typer.Option("--dataset-id", help="Viam sequence dataset id")] = None,
    sequence_id: Annotated[str | None, typer.Option("--sequence-id", help="sequence to build")] = None,
    work: Annotated[Path, typer.Option("--work", help="work dir holding all stage folders")] = Path("runs"),
    until: StageArg = None,
    force: StageArg = None,
    export_dir: Annotated[
        Path | None, typer.Option("--export-dir", help="existing `viam dataset export` dir (skips the CLI)")
    ] = None,
    scan_dir: Annotated[
        Path | None, typer.Option("--scan-dir", help="local scan dir (skips export and join)")
    ] = None,
    threedgrut_dir: Annotated[
        Path | None,
        typer.Option("--threedgrut-dir", help=f"3DGRUT checkout (or ${train.ENV_THREEDGRUT_DIR})"),
    ] = None,
    threedgrut_python: Annotated[
        str | None,
        typer.Option("--threedgrut-python", help=f"python for 3DGRUT (or ${train.ENV_THREEDGRUT_PYTHON})"),
    ] = None,
    run_train: Annotated[
        bool, typer.Option("--run-train", help="run 3DGRUT instead of only writing the command")
    ] = False,
    finalize_flag: Annotated[
        bool,
        typer.Option("--finalize", help="add the sequence to the dataset with an org key (VIAM_API_KEY*)"),
    ] = False,
    static_mask: Annotated[
        Path | None, typer.Option("--static-mask", help="grayscale PNG, 0 = ignore")
    ] = None,
    refine_extrinsics: Annotated[
        bool, typer.Option("--refine-extrinsics", help="diagnostic bundle adjustment into sparse/0_ba")
    ] = False,
    fx: Annotated[float | None, typer.Option(help="intrinsics override (px)")] = None,
    fy: Annotated[float | None, typer.Option(help="intrinsics override (px)")] = None,
    cx: Annotated[float | None, typer.Option(help="intrinsics override (px)")] = None,
    cy: Annotated[float | None, typer.Option(help="intrinsics override (px)")] = None,
    width: Annotated[int | None, typer.Option(help="image width for the override")] = None,
    height: Annotated[int | None, typer.Option(help="image height for the override")] = None,
    k1: Annotated[float, typer.Option(help="distortion override")] = 0.0,
    k2: Annotated[float, typer.Option(help="distortion override")] = 0.0,
    p1: Annotated[float, typer.Option(help="distortion override")] = 0.0,
    p2: Annotated[float, typer.Option(help="distortion override")] = 0.0,
    k3: Annotated[float, typer.Option(help="distortion override")] = 0.0,
    viam_bin: Annotated[str, typer.Option("--viam-bin", help="viam CLI executable")] = "viam",
    seed: Annotated[int, typer.Option(help="COLMAP random seed")] = 0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Build a Gaussian splat from one Viam sequence (or a local scan dir)."""
    _configure_logging(verbose)
    try:
        opts = BuildOptions(
            dataset_id=dataset_id,
            sequence_id=sequence_id,
            work=work,
            until=until,
            force=force,
            export_dir=export_dir,
            scan_dir=scan_dir,
            threedgrut_dir=threedgrut_dir,
            threedgrut_python=threedgrut_python,
            run_train=run_train,
            finalize=finalize_flag,
            static_mask=static_mask,
            refine_extrinsics=refine_extrinsics,
            camera_override=_camera_override(fx, fy, cx, cy, width, height, k1, k2, p1, p2, k3),
            viam_bin=viam_bin,
            seed=seed,
        )
        run_build(opts)
    except StageError as e:
        log.error("%s", e)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
