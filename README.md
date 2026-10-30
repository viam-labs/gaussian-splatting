# gaussian-splatting

Viam module that turns any arm-mounted camera into a Gaussian-splat scanner. The `scanner` service drives the arm
through viewpoints over a region, captures one sharp image per viewpoint with the camera pose in the arm's base
frame, and uploads the scan to the Viam cloud as a sequence in a sequence dataset. The offline `splat-build` tool
exports a sequence, reconstructs with COLMAP from the known poses, and trains a splat with
[3DGRUT](https://github.com/nv-tlabs/3dgrut). Output: a splat in metres, registered to the arm base, exported as
USDZ for Isaac Sim.

Design and rationale: [`docs/spec.md`](docs/spec.md).

## Requirements on the machine

- An arm (`rdk:component:arm`), a camera on it (`rdk:component:camera`), and the motion service. Motion is
  required: the scan sweeps a large volume and plans against the frame-system geometries, so put every obstacle
  in the scan volume into the frame system.
- The camera must be calibrated: intrinsics reported by `get_properties()` (or given in config), and its frame
  configured as a child of the arm with the hand-eye transform (or given in config). `viam:opencv` produces both.
- viam-server ≥ 0.127.0 for sequences.

## Models

| Model | API | Role |
| --- | --- | --- |
| `viam:gaussian-splatting:scanner` | `rdk:service:generic` | planning, capture, local ledger, upload, sequence creation |
| `viam:gaussian-splatting:scan-state` | `rdk:component:sensor` | last frame record; gives the sequence a tabular resource |

### Scanner configuration

```json
{
  "arm": "arm",
  "camera": "camera",
  "motion": "builtin",
  "scan_state": "scan-state",
  "region_mm": { "center": [300, 0, 0], "size": [500, 500, 100] },
  "standoff_mm": [200, 450],
  "elevation_deg": [25, 80],
  "azimuth_deg": [-150, 150],
  "dwell_s": 0.5,
  "min_sharpness": 60.0,
  "dataset_id": "<sequence dataset id>",
  "upload": true
}
```

Optional: `camera_intrinsics` (`{"K": {fx, fy, cx, cy}, "dist": {k1, k2, k3, p1, p2}}`) when the camera does
not report intrinsics; `camera_in_arm` (a frame block: translation mm + orientation) when the camera is not in
the frame system; `static_mask` (PNG, 0 = ignore) for the part of the end effector that is always in view;
`camera_lock_command` (a DoCommand payload that locks exposure and white balance); `arm_reach_mm`.

### DoCommands

| Command | Effect |
| --- | --- |
| `start_scan` | plan and start a scan in the background; returns `scan_id`, `planned`, `scan_dir` |
| `scan_status` | state, poses done/planned/skipped, uploads done/pending, `sequence_id`, eta |
| `cancel_scan` | stop after the current move, upload what was captured |
| `upload_scan` | re-run the upload phase for a scan from disk (idempotent) |

`start_scan` fields (all optional): `scan_id`, `num_poses`, `images_per_pose`, `region_mm`, `standoff_mm`,
`elevation_deg`, `azimuth_deg`, `seed`, `dwell_s`, `upload`, `dataset_id`, `tags`, `resume`, `dry_run`.

Scans are written to `<VIAM_MODULE_DATA>/scans/<scan_id>/` (`manifest.json`, `frames.jsonl`, `images/`).

## Tools

Install for development: `make setup` (uv, Python from `.python-version`). Credentials go in `.env`
(see `.env.example`).

### `splat-scan`: talk to the robot

```bash
uv run splat-scan start --num-poses 150 [--dry-run] [--tags cell-v1]   # start, then print progress until done
uv run splat-scan status
uv run splat-scan cancel
uv run splat-scan upload --scan-id <id>
uv run splat-scan create-dataset --org-id <org> --name <name>   # one-time: the sequence dataset scans go into
```

### `splat-build`: sequence → splat, never touches the robot

```bash
uv run splat-build --dataset-id <id> --sequence-id <id> --work runs/
```

Stages, each cached under `runs/<sequence_id>/<stage>/`: `export` (viam CLI), `join` (pair images with poses),
`colmap` (known-pose triangulation, acceptance report), `train` (3DGRUT command; `--run-train` with
`--threedgrut-dir` executes it), `export-splat`. `--until <stage>` stops early, `--force <stage>` redoes a stage
and everything after it. `--scan-dir <dir>` starts from a local scan directory instead of the cloud.

## Local development without hardware

`dev/local-machine.json` configures a local viam-server with a fake Lite 6, a fake camera that reports
intrinsics, the built-in motion service and this module:

```bash
viam-server -config dev/local-machine.json
uv run splat-scan --insecure --machine-address localhost:8080 start --num-poses 5 --dry-run
make integration
```

## Deploy

CI lints, typechecks and tests on every PR. Merges to `main` publish a prerelease to the Viam Registry; a
GitHub release publishes a clean version (see `.github/workflows/deploy.yml`). The module is a source tarball;
`run.sh` builds the venv on the machine on first start.
