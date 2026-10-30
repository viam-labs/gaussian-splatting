"""Stage cache, train command composition, export-splat collection, and the CLI with --until / --force."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gaussian_splatting.build import export_splat, train
from gaussian_splatting.build.cli import app
from gaussian_splatting.build.stages import StageError, StagePlan, is_done, mark_done, run_stage
from tests.synthetic_scene import SyntheticScene, build_scene

runner = CliRunner()


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> SyntheticScene:
    return build_scene(tmp_path_factory.mktemp("scene") / "scan", num_views=12, width=320, height=240)


# ---------------------------------------------------------------- stage cache


def test_stage_plan_skip_force_until(tmp_path: Path) -> None:
    plan = StagePlan()
    assert plan.should_run("join", tmp_path / "join")
    mark_done(tmp_path / "join", {"x": 1})
    assert not plan.should_run("join", tmp_path / "join")
    assert json.loads((tmp_path / "join" / ".done").read_text())["x"] == 1

    assert StagePlan(force="join").should_run("join", tmp_path / "join")
    assert StagePlan(force="join").should_run("colmap", tmp_path / "colmap")
    mark_done(tmp_path / "export")
    assert not StagePlan(force="join").should_run("export", tmp_path / "export")

    assert not StagePlan(until="join").should_run("colmap", tmp_path / "colmap")
    assert StagePlan(until="join").stops_after("join")
    with pytest.raises(StageError):
        StagePlan(until="nope").should_run("join", tmp_path / "x")


def test_run_stage_archives_on_force_and_skips_marking_on_none(tmp_path: Path) -> None:
    calls: list[str] = []
    stage_dir = tmp_path / "train"

    def only_command() -> None:
        calls.append("cmd")
        return None

    assert run_stage(StagePlan(), "train", stage_dir, only_command)
    assert not is_done(stage_dir)  # None => not done, runs again next time
    assert run_stage(StagePlan(), "train", stage_dir, lambda: {"ok": True})
    assert is_done(stage_dir)
    assert not run_stage(StagePlan(), "train", stage_dir, lambda: {"ok": True})
    (stage_dir / "marker.txt").write_text("old")
    assert run_stage(StagePlan(force="train"), "train", stage_dir, lambda: {"ok": 2})
    archived = list((tmp_path / "_archive").glob("train-*"))
    assert len(archived) == 1 and (archived[0] / "marker.txt").read_text() == "old"
    assert not (stage_dir / "marker.txt").exists()


# ---------------------------------------------------------------- train command


def test_train_command_matches_spec(tmp_path: Path) -> None:
    cmd = train.build_train_command(tmp_path / "colmap", tmp_path / "out", "seq-1")
    assert cmd[:4] == ["python", "train.py", "--config-name", "apps/colmap_3dgut.yaml"]
    assert f"path={(tmp_path / 'colmap').resolve()}" in cmd
    assert f"out_dir={(tmp_path / 'out').resolve()}" in cmd
    assert "experiment_name=seq-1" in cmd
    assert cmd[-3:] == [
        "export_usd.enabled=true",
        "export_usd.format=nurec",
        "export_usd.apply_normalizing_transform=false",
    ]
    assert "initialization=random" not in cmd
    assert train.build_train_command(tmp_path, tmp_path, "s", init_random=True)[-1] == "initialization=random"
    script = train.write_command(tmp_path / "train", cmd, Path("/opt/3dgrut"))
    text = script.read_text()
    assert "cd /opt/3dgrut" in text and "train.py --config-name apps/colmap_3dgut.yaml" in text
    assert os.access(script, os.X_OK)


def _fake_threedgrut(root: Path) -> Path:
    """A stand-in train.py that writes the files 3DGRUT would; exercises the real subprocess path."""
    root.mkdir(parents=True)
    (root / "train.py").write_text(
        "import sys, pathlib\n"
        "args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)\n"
        "run = pathlib.Path(args['out_dir']) / args['experiment_name'] / '2026-01-01'\n"
        "run.mkdir(parents=True)\n"
        "(run / 'export_last.usdz').write_bytes(b'usdz')\n"
        "(run / 'export_last.ply').write_bytes(b'ply')\n"
        "print('fake 3dgrut trained', args['path'])\n"
    )
    return root


def test_run_training_streams_log_and_export_splat_collects(tmp_path: Path) -> None:
    tdg = _fake_threedgrut(tmp_path / "3dgrut")
    out = tmp_path / "train" / "out"
    cmd = train.build_train_command(tmp_path / "colmap", out, "seq-9", python=sys.executable)
    (tmp_path / "train").mkdir()
    train.run_training(cmd, tdg, tmp_path / "train" / "train.log")
    assert "fake 3dgrut trained" in (tmp_path / "train" / "train.log").read_text()

    found = export_splat.find_exports(out, "seq-9")
    assert set(found) == {".usdz", ".ply"}
    report = export_splat.export_splat(out, "seq-9", tmp_path / "splat", colmap_report={"k": 1}, trained=True)
    assert report is not None
    assert (tmp_path / "splat" / "scene.usdz").read_bytes() == b"usdz"
    assert (tmp_path / "splat" / "scene.ply").read_bytes() == b"ply"
    assert json.loads((tmp_path / "splat" / "report.json").read_text())["colmap"] == {"k": 1}


def test_export_splat_without_training_writes_a_note(tmp_path: Path) -> None:
    assert (
        export_splat.export_splat(
            tmp_path / "out", "x", tmp_path / "splat", colmap_report=None, trained=False
        )
        is None
    )
    assert (tmp_path / "splat" / export_splat.NOTE_FILE).exists()


def test_run_training_rejects_bad_dir(tmp_path: Path) -> None:
    with pytest.raises(StageError, match="train.py"):
        train.run_training(["python", "train.py"], tmp_path, tmp_path / "log")


# ---------------------------------------------------------------- CLI


def test_cli_scan_dir_until_force_and_run_train(scene: SyntheticScene, tmp_path: Path) -> None:
    work = tmp_path / "runs"
    run_root = work / "synthetic-plane"  # run id falls back to the scan id

    r = runner.invoke(app, ["--scan-dir", str(scene.scan_dir), "--work", str(work), "--until", "join"])
    assert r.exit_code == 0, r.output
    assert is_done(run_root / "join") and not (run_root / "colmap").exists()

    r = runner.invoke(app, ["--scan-dir", str(scene.scan_dir), "--work", str(work), "--until", "train"])
    assert r.exit_code == 0, r.output
    assert is_done(run_root / "colmap")
    assert "COLMAP registration" in r.output
    assert (run_root / "train" / "command.sh").exists()
    assert not is_done(run_root / "train")  # only the command was written
    colmap_done = (run_root / "colmap" / ".done").stat().st_mtime_ns

    # Cached: colmap is not redone.
    r = runner.invoke(app, ["--scan-dir", str(scene.scan_dir), "--work", str(work), "--until", "train"])
    assert r.exit_code == 0, r.output
    assert (run_root / "colmap" / ".done").stat().st_mtime_ns == colmap_done

    # Full run with a fake 3DGRUT.
    tdg = _fake_threedgrut(tmp_path / "3dgrut")
    r = runner.invoke(
        app,
        [
            "--scan-dir", str(scene.scan_dir), "--work", str(work),
            "--run-train", "--threedgrut-dir", str(tdg), "--threedgrut-python", sys.executable,
        ],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert is_done(run_root / "train") and is_done(run_root / "splat")
    assert (run_root / "splat" / "scene.usdz").exists()
    cmd = json.loads((run_root / "train" / ".done").read_text())["command"]
    assert "export_usd.apply_normalizing_transform=false" in cmd

    # --force join redoes join and everything after; previous folders are archived, not deleted.
    r = runner.invoke(
        app,
        ["--scan-dir", str(scene.scan_dir), "--work", str(work), "--force", "join", "--until", "join"],
    )
    assert r.exit_code == 0, r.output
    assert is_done(run_root / "join")
    assert any((run_root / "_archive").glob("join-*"))
    assert is_done(run_root / "colmap")  # untouched by --until join; redone on the next full run


def test_cli_errors_cleanly(tmp_path: Path) -> None:
    r = runner.invoke(app, ["--work", str(tmp_path)])
    assert r.exit_code == 1
    r = runner.invoke(app, ["--work", str(tmp_path), "--sequence-id", "s", "--until", "bogus"])
    assert r.exit_code == 1
    r = runner.invoke(app, ["--work", str(tmp_path), "--sequence-id", "s", "--dataset-id", "d", "--fx", "1"])
    assert r.exit_code == 1
