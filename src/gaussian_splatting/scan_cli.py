"""`splat-scan`: a thin wrapper around the scanner's DoCommands, run from a laptop.

Reads VIAM_API_KEY, VIAM_API_KEY_ID and VIAM_MACHINE_ADDRESS from the environment or a local .env.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.services.generic import Generic

TERMINAL_STATES = {"done", "failed", "cancelled", "idle"}

app = typer.Typer(no_args_is_help=True, help="Drive the viam:gaussian-splatting:scanner service.")

AddressOpt = Annotated[
    str | None,
    typer.Option(
        "--machine-address", envvar="VIAM_MACHINE_ADDRESS", help="Machine address, e.g. <part>.viam.cloud"
    ),
]
ScannerOpt = Annotated[str, typer.Option("--scanner", help="Name of the scanner service")]
InsecureOpt = Annotated[bool, typer.Option("--insecure", help="Plain gRPC without auth (local viam-server)")]


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader: KEY=VALUE lines, no quoting, existing env wins."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


async def _dial(address: str | None, insecure: bool) -> RobotClient:
    if address is None:
        raise typer.BadParameter("--machine-address or VIAM_MACHINE_ADDRESS is required")
    if insecure:
        opts = RobotClient.Options(dial_options=DialOptions(insecure=True, disable_webrtc=True))
        return await RobotClient.at_address(address, opts)
    api_key = os.environ.get("VIAM_API_KEY")
    api_key_id = os.environ.get("VIAM_API_KEY_ID")
    if not api_key or not api_key_id:
        raise typer.BadParameter(
            "set VIAM_API_KEY and VIAM_API_KEY_ID (see .env.example), or pass --insecure"
        )
    return await RobotClient.at_address(address, RobotClient.Options.with_api_key(api_key, api_key_id))


async def _command(
    address: str | None, insecure: bool, scanner: str, command: Mapping[str, Any]
) -> Mapping[str, Any]:
    machine = await _dial(address, insecure)
    try:
        return await Generic.from_robot(machine, scanner).do_command(command)
    finally:
        await machine.close()


def _count(st: Mapping[str, Any], key: str) -> int:
    value = st.get(key, 0)
    return int(value) if isinstance(value, int | float) else 0


def _progress_line(st: Mapping[str, Any]) -> str:
    eta = st.get("eta_s")
    eta_txt = f" eta {eta:.0f}s" if isinstance(eta, int | float) else ""
    line = (
        f"[{st.get('state')}] poses {_count(st, 'poses_done')}/{_count(st, 'poses_planned')}"
        f" skipped {_count(st, 'poses_skipped')} images {_count(st, 'images_saved')}"
        f" uploads {_count(st, 'uploads_done')} done/{_count(st, 'uploads_pending')} pending"
        f" elapsed {st.get('elapsed_s')}s{eta_txt}"
    )
    if st.get("last_error"):
        line += f"  ({st['last_error']})"
    return line


async def _poll_until_done(
    address: str | None, insecure: bool, scanner: str, poll_s: float
) -> Mapping[str, Any]:
    machine = await _dial(address, insecure)
    try:
        service = Generic.from_robot(machine, scanner)
        last = ""
        while True:
            st = await service.do_command({"scan_status": {}})
            line = _progress_line(st)
            if line != last:
                typer.echo(line)
                last = line
            if st.get("state") in TERMINAL_STATES:
                return st
            await asyncio.sleep(poll_s)
    finally:
        await machine.close()


@app.callback()
def _init() -> None:
    _load_dotenv()


@app.command()
def start(
    machine_address: AddressOpt = None,
    scanner: ScannerOpt = "scanner",
    insecure: InsecureOpt = False,
    num_poses: Annotated[int | None, typer.Option("--num-poses")] = None,
    scan_id: Annotated[str | None, typer.Option("--scan-id")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    tags: Annotated[list[str] | None, typer.Option("--tags", help="Repeatable")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Plan only, move nothing")] = False,
    no_upload: Annotated[bool, typer.Option("--no-upload")] = False,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    poll_s: Annotated[float, typer.Option("--poll")] = 2.0,
) -> None:
    """Start a scan and print progress until it finishes."""
    request: dict[str, Any] = {"dry_run": dry_run, "resume": resume}
    if num_poses is not None:
        request["num_poses"] = num_poses
    if scan_id is not None:
        request["scan_id"] = scan_id
    if seed is not None:
        request["seed"] = seed
    if tags:
        request["tags"] = tags
    if no_upload:
        request["upload"] = False

    async def run() -> None:
        resp = await _command(machine_address, insecure, scanner, {"start_scan": request})
        if dry_run:
            resp = dict(resp)
            viewpoints = resp.pop("viewpoints", [])
            typer.echo(json.dumps(resp, indent=2, default=str))
            typer.echo(f"{len(viewpoints)} candidate viewpoints")
            return
        typer.echo(f"started {resp.get('scan_id')} -> {resp.get('scan_dir')}")
        if resp.get("note"):
            typer.echo(f"note: {resp['note']}")
        final = await _poll_until_done(machine_address, insecure, scanner, poll_s)
        if final.get("sequence_id"):
            typer.echo(f"sequence_id: {final['sequence_id']}")
        if final.get("state") != "done":
            raise typer.Exit(code=1)

    asyncio.run(run())


@app.command()
def status(
    machine_address: AddressOpt = None, scanner: ScannerOpt = "scanner", insecure: InsecureOpt = False
) -> None:
    """Print the current scan status."""
    st = asyncio.run(_command(machine_address, insecure, scanner, {"scan_status": {}}))
    typer.echo(json.dumps(dict(st), indent=2, default=str))


@app.command()
def cancel(
    machine_address: AddressOpt = None, scanner: ScannerOpt = "scanner", insecure: InsecureOpt = False
) -> None:
    """Stop after the current viewpoint; captured frames are still uploaded."""
    resp = asyncio.run(_command(machine_address, insecure, scanner, {"cancel_scan": {}}))
    typer.echo(json.dumps(dict(resp), indent=2, default=str))


@app.command()
def upload(
    scan_id: Annotated[str, typer.Option("--scan-id")],
    machine_address: AddressOpt = None,
    scanner: ScannerOpt = "scanner",
    insecure: InsecureOpt = False,
    poll_s: Annotated[float, typer.Option("--poll")] = 2.0,
) -> None:
    """Re-run the upload phase for a scan on the robot's disk."""

    async def run() -> None:
        resp = await _command(machine_address, insecure, scanner, {"upload_scan": {"scan_id": scan_id}})
        typer.echo(f"uploading {resp.get('scan_id')}: {resp.get('uploads_pending')} frames pending")
        final = await _poll_until_done(machine_address, insecure, scanner, poll_s)
        if final.get("sequence_id"):
            typer.echo(f"sequence_id: {final['sequence_id']}")
        if final.get("state") != "done":
            raise typer.Exit(code=1)

    asyncio.run(run())


if __name__ == "__main__":
    app()
