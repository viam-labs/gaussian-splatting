"""export stage: `viam dataset export` into `<work>/<dataset_id>/export/`, parquet zip unpacked next to it.

The only stage that talks to the Viam cloud (through the viam CLI's own login). Tests and offline runs bypass
it with `--export-dir`, which points at an existing export of the same layout.
"""

from __future__ import annotations

import logging
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

from gaussian_splatting.build.stages import StageError

log = logging.getLogger("splat-build")

PARQUET_SUBDIR = "parquet"
BINARY_SUBDIR = "binary_data"
PARQUET_FILES = ("binary_data.parquet", "tabular_data.parquet", "sequences.parquet")


@dataclass
class ExportLayout:
    """Where the three parquet tables and the downloaded blobs live."""

    root: Path  # paths in binary_data.parquet are relative to this
    parquet_dir: Path


def run_viam_export(dataset_id: str, destination: Path, viam_bin: str = "viam") -> None:
    destination.mkdir(parents=True, exist_ok=True)
    cmd = [viam_bin, "dataset", "export", f"--dataset-id={dataset_id}", f"--destination={destination}"]
    log.info("running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise StageError(f"viam CLI not found ({viam_bin}); install it or pass --export-dir") from e
    except subprocess.CalledProcessError as e:
        raise StageError(f"viam dataset export failed with exit code {e.returncode}") from e


def unpack_parquet(export_root: Path) -> Path:
    """Find the parquet tables under an export root, unzipping `<dataset-id>.zip` if needed."""
    direct = export_root / PARQUET_SUBDIR
    if all((direct / f).exists() for f in PARQUET_FILES):
        return direct
    if all((export_root / f).exists() for f in PARQUET_FILES):
        return export_root
    zips = sorted(export_root.glob("*.zip"))
    if not zips:
        raise StageError(
            f"no parquet tables or zip under {export_root}; expected {PARQUET_FILES} or a <dataset-id>.zip"
        )
    if len(zips) > 1:
        log.warning("several zips in %s, using the newest: %s", export_root, zips[-1].name)
    zip_path = max(zips, key=lambda p: p.stat().st_mtime)
    direct.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Flatten: some exports nest the tables one folder deep. Refuse anything that escapes the target.
            target = direct / Path(member).name
            if member.endswith("/") or not member.endswith(".parquet"):
                continue
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
    missing = [f for f in PARQUET_FILES if not (direct / f).exists()]
    if missing:
        raise StageError(
            f"{zip_path.name} did not contain {missing}; it holds {zipfile.ZipFile(zip_path).namelist()}"
        )
    log.info("unpacked %s -> %s", zip_path.name, direct)
    return direct


def resolve_export(
    dataset_id: str | None, work: Path, export_dir: Path | None, viam_bin: str = "viam"
) -> ExportLayout:
    """Use `--export-dir` when given, else export the dataset under the work dir (cached by the stage)."""
    if export_dir is not None:
        if not export_dir.exists():
            raise StageError(f"--export-dir {export_dir} does not exist")
        return ExportLayout(root=export_dir, parquet_dir=unpack_parquet(export_dir))
    if dataset_id is None:
        raise StageError("need --dataset-id (to export) or --export-dir / --scan-dir (offline)")
    root = work / dataset_id / "export"
    run_viam_export(dataset_id, root, viam_bin=viam_bin)
    return ExportLayout(root=root, parquet_dir=unpack_parquet(root))
