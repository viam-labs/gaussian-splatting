"""train stage: compose (and optionally run) the 3DGRUT training command. No other trainer, ever.

3DGRUT lives in its own environment, so it is a subprocess with cwd = the 3DGRUT checkout. By default the
stage only writes `train/command.sh`; `--run-train` executes it and streams output to `train/train.log`.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path

from gaussian_splatting.build.stages import StageError

log = logging.getLogger("splat-build")

CONFIG_NAME = "apps/colmap_3dgut.yaml"
COMMAND_FILE = "command.sh"
LOG_FILE = "train.log"
OUT_SUBDIR = "out"
ENV_THREEDGRUT_DIR = "THREEDGRUT_DIR"
ENV_THREEDGRUT_PYTHON = "THREEDGRUT_PYTHON"


def build_train_command(
    colmap_dir: Path,
    out_dir: Path,
    experiment_name: str,
    *,
    init_random: bool = False,
    python: str = "python",
) -> list[str]:
    """Spec §10 step 4, as an argv list. Hydra overrides are `key=value`."""
    cmd = [
        python,
        "train.py",
        "--config-name",
        CONFIG_NAME,
        f"path={colmap_dir.resolve()}",
        f"out_dir={out_dir.resolve()}",
        f"experiment_name={experiment_name}",
        "export_usd.enabled=true",
        "export_usd.format=nurec",
        "export_usd.apply_normalizing_transform=false",
    ]
    if init_random:
        cmd.append("initialization=random")
    return cmd


def write_command(train_dir: Path, cmd: list[str], threedgrut_dir: Path | None) -> Path:
    train_dir.mkdir(parents=True, exist_ok=True)
    cwd_line = f"cd {shlex.quote(str(threedgrut_dir))}" if threedgrut_dir else 'cd "$THREEDGRUT_DIR"'
    script = "#!/bin/sh\n# 3DGRUT training command written by splat-build\nset -e\n" + cwd_line + "\n"
    script += shlex.join(cmd) + "\n"
    path = train_dir / COMMAND_FILE
    path.write_text(script)
    path.chmod(0o755)
    return path


def resolve_threedgrut_dir(flag: Path | None) -> Path | None:
    if flag is not None:
        return flag
    env = os.environ.get(ENV_THREEDGRUT_DIR)
    return Path(env) if env else None


def resolve_python(flag: str | None) -> str:
    return flag or os.environ.get(ENV_THREEDGRUT_PYTHON) or "python"


def run_training(cmd: list[str], threedgrut_dir: Path, log_path: Path) -> int:
    """Run the command with cwd = 3DGRUT checkout, tee-ing stdout/stderr to `log_path`. Returns exit code."""
    if not (threedgrut_dir / "train.py").exists():
        raise StageError(f"{threedgrut_dir} has no train.py; is this the 3DGRUT checkout?")
    log.info("running 3DGRUT in %s: %s", threedgrut_dir, shlex.join(cmd))
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(threedgrut_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            print(line, end="")
        code = proc.wait()
    if code != 0:
        raise StageError(f"3DGRUT exited with code {code}; see {log_path}")
    return code
