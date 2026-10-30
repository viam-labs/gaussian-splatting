# Tech spec: `gaussian-splat`, posed arm-camera scans for 3DGRUT

Status: draft for review, revised 2026-10-29. Owner: Robin. Module: `viam:gaussian-splatting`, its own repo
(`viam-labs/gaussian-splat`); this file moves there when the repo exists. First consumer: the Jenga cell (open
question Q19, splat as the sim twin's background, and the H1 hand-eye check). Earlier drafts live in the jenga-bot repo.

## 1. Goal

A Viam module that anyone with an arm and a camera on the arm can add to their machine. Its scanner service drives
the arm through N viewpoints, captures one sharp image per viewpoint together with the camera pose in the arm's
base frame, and uploads the scan to the Viam cloud as a sequence in a sequence dataset. A separate offline tool,
`splat-build`, turns one sequence into a Gaussian splat with 3DGRUT (`apps/colmap_3dgut.yaml`). Output of the
chain: a splat in metres, registered to the arm base, that drops into a simulator without the manual
scale-and-rotate alignment the NVIDIA Marble workflow needs.

Non-goals: calibrating intrinsics or hand-eye (the module reads them; `viam:opencv` or anything else produces
them), running COLMAP or 3DGRUT on the robot, masking the arm with a kinematic render (v2), anything specific to
one robot or cell (the Jenga cell's values live in `jenga-bot` as machine config).

## 2. Inputs 3DGRUT needs

From `threedgrut/datasets/dataset_colmap.py`, `configs/dataset/colmap.yaml` and `configs/initialization/colmap.yaml`:

| Item | Requirement |
| --- | --- |
| Layout | `<scene>/images/*.png` plus `<scene>/sparse/0/{cameras,images,points3D}.{bin,txt}` (binary tried first, text fallback). `images_2/`, `images_4/` for `dataset.downsample_factor` 2 and 4. |
| Camera models | `PINHOLE`, `SIMPLE_PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `FULL_OPENCV`, `OPENCV_FISHEYE`. Distortion is used in ray generation (3DGUT), so no undistortion step. |
| Poses | COLMAP world-to-camera `qw qx qy qz tx ty tz`, camera frame right-down-front. Loader inverts to camera-to-world and applies no further convention change. |
| Points | `initialization: colmap` reads `sparse/0/points3D` to seed the Gaussians. With known poses these come from `colmap point_triangulator`. Fallback `initialization=random`. |
| Masks | Optional `images/<name>_mask.png`, grayscale, same resolution; 0 = ignored in the loss. Used for whatever part of the end effector is in view. |
| Split | `test_split_interval: 8`: every 8th image by `images.bin` order is held out for validation. |
| World frame | `export_usd.apply_normalizing_transform: true` by default re-centres and re-scales the scene. We set it to `false` so the USD stays in the arm's base frame, in metres. |
| Export | `export_usd.enabled=true export_usd.format=nurec` for Isaac Sim 5.0 (what `viam-isaac-sim` pins); `standard` (ParticleField) once the module moves to 6.x. |

Image count: 100 to 200 views for a single-focus scene (NVIDIA GTC 2026 guidance), 60 to 80 % overlap between
neighbours, consistent lighting, no motion blur. Everything that moves between images (the arm) must be out of
frame or masked.

## 3. Inputs Viam sequences need

From `docs.viam.com/data/sequences`, `/data/capture-sync/capture-on-demand`, `/train/sequence-dataset-format`:

- A sequence is a saved filter over data in the cloud: one part, a time window, a list of `(resource_name,
  method_name)`, tags. It stores no data of its own; whatever matches the filter is in the sequence. Only JPEG and
  PNG count as binary data.
- Every piece of data carries `time_requested`, `time_received`, `part_id`, `component_name`, `method_name` and
  tags. The data manager fills these at capture time. The data client's upload calls let the caller set them.
- Created either by the capture-control sensor (`start_capture` / `stop_capture` DoCommands on
  `viam:capture-control:capture-control-sensor`, data manager polls it at 10 Hz) or explicitly with
  `DataClient.create_sequence(part_id, resources, sequence_tags, start_time, end_time)`. `create_sequence` needs
  a key with access to the part; `add_sequences_to_dataset` and dataset creation need an organization key.
- Export (`viam dataset export`) gives `binary_data.parquet` (one row per image: `sequence_id`, `timestamp` µs
  UTC, `component_name`, `method_name`, `path`), `tabular_data.parquet` (one row per reading, `payload` is the
  method response as a JSON string) and `sequences.parquet`. No alignment between resources is done for us; we join
  on timestamps.
- Minimum viam-server: 0.127.0 for sequences, 0.130.0 to capture components without their own capture config.

## 4. Integration with Viam: how the module stays generic

The pattern is the one `viam-labs/opencv` uses for hand-eye calibration: depend on APIs, declare dependencies in
config, read what you need through the standard interfaces, and fall back to a config override when a resource
does not provide it.

| Need | Source | Fallback | If neither |
| --- | --- | --- | --- |
| Arm | `arm` attribute, any `rdk:component:arm` | none | refuse config |
| Camera | `camera` attribute, any `rdk:component:camera` | none | refuse config |
| Collision-aware moves | `motion` attribute, any `rdk:service:motion` (the built-in one is enough); obstacles come from the frame-system geometries already on the machine | none: the only mode is sampled poses over a large volume, so collision-aware planning is not optional | refuse config |
| Intrinsics and distortion | `camera.get_properties()` | `camera_intrinsics` attribute, same shape as `viam:opencv` uses (`K.fx fy cx cy`, `dist.k1 k2 k3 p1 p2`) | refuse scan |
| Camera pose in arm base | frame system, camera configured as a child of the arm, read with `get_pose(camera, "<arm>_origin")` | `camera_in_arm` attribute, same shape as the `frame` block `viam:opencv:hand_eye_calibration` returns, composed with `arm.get_end_position()` | refuse scan |
| Cloud credentials | viam-server gives modules `VIAM_API_KEY`, `VIAM_API_KEY_ID`, `VIAM_MACHINE_PART_ID` | none | scan runs, upload disabled |

`validate_config` returns `[arm, camera, motion]` as required dependencies, so viam-server starts them first and
injects live clients into `reconfigure`. (`viam:opencv` makes motion optional because it has a manual mode with
operator-supplied joint positions; this module has no such mode.) Everything is expressed in the arm's base frame,
`<arm>_origin`, not `world`, so the splat is registered to the arm regardless of where the arm sits in the world.

A model forces only the dependencies it declares. A machine that adds this module and configures no scanner pays
nothing.

## 5. Design

```
Phase 1, robot, module      start_scan ──► ScanJob (asyncio task owned by the service)
                                             plan viewpoints (seeded)                      §7
                                             for each viewpoint:
                                               move, wait until still, dwell              §8
                                               get_pose, get_images, get_pose again
                                               write images/NNNN.png + frames.jsonl line
                                           ──► <VIAM_MODULE_DATA>/scans/<scan_id>/

Phase 2, robot, module      upload phase, reads frames.jsonl                               §6.4
                                             binary_data_capture_upload per image
                                             tabular_data_capture_upload per frame record
                                             create_sequence over the capture window
                                             add_sequences_to_dataset

Phase 3, offline            splat-build --dataset-id --sequence-id                          §10
                                             export → join → colmap → train → export-splat
                                           ──► runs/<sequence_id>/splat/scene.usdz
```

Three decisions, with the reasoning:

- **Capture in the module, at rest, not with timed data capture.** Timed capture samples the camera on a clock
  while the arm moves, so most frames are blurred or mid-motion, and image and pose are two unsynchronised streams
  that have to be paired afterwards by nearest timestamp. The module knows exactly when the arm has stopped. It
  reads the pose, takes the image and reads the pose again in one code path a few milliseconds apart. Nothing to
  filter, nothing to pair, and the scan directory on disk is already a complete dataset before anything touches the
  network.
- **Upload is a separate phase that reads the scan directory, stamped with capture times.** Each upload call
  carries the time the image was actually taken, not the time of the upload, so the cloud indexes the data at
  capture time and a sequence over the capture window selects it. Running upload after capture, instead of
  interleaved, costs a few minutes of wall clock and buys a simple, re-runnable step (`upload_scan`) when the
  network fails. Risk: that uploaded data is selectable by a sequence the same way captured data is. The docs
  describe the filter in terms of fields the uploads carry, but this is verified in the first spike (§13).
  Fallback: the capture-control path with the scan-state sensor below, and `splat-build` keeps only frames inside
  a dwell window.
- **A small sensor component, `viam:gaussian-splatting:scan-state`, carries the per-frame record.** Its `Readings`
  return the last frame record (camera pose, joints, viewpoint index, dwell flag). The tabular uploads use this
  real resource name and `Readings` as method, so the sequence filter has a legitimate resource, the payload shows
  up in `tabular_data.parquet`, and the capture-control fallback works unchanged.

## 6. Interfaces

### 6.1 Module and models

`viam:gaussian-splatting` with two models:

- `viam:gaussian-splatting:scanner`, `rdk:service:generic`. Owns planning, capture, the local ledger, upload and
  sequence creation.
- `viam:gaussian-splatting:scan-state`, `rdk:component:sensor`. Per-frame record, see §5.

Scanner attributes (values shown are the Jenga cell's, as an example):

```json
{
  "arm": "lite6",
  "camera": "wrist",
  "motion": "builtin",
  "scan_state": "scan-state",
  "camera_intrinsics": null,
  "camera_in_arm": null,
  "region_mm": { "center": [300.0, 0.0, 0.0], "size": [500.0, 500.0, 100.0] },
  "standoff_mm": [200.0, 450.0],
  "elevation_deg": [25.0, 80.0],
  "azimuth_deg": [-150.0, 150.0],
  "dwell_s": 0.5,
  "min_sharpness": 60.0,
  "static_mask": "masks/end_effector.png",
  "camera_lock_command": { "set_exposure_auto": false, "set_white_balance_auto": false },
  "dataset_id": "<sequence dataset id>",
  "upload": true
}
```

`camera_intrinsics` and `camera_in_arm` are the fallbacks from §4 and are normally omitted. `scan_state` names
the sensor instance; if absent and `upload` is true, the config is refused. At reconfigure the scanner checks the
frame system (`get_frame_system_config`): the camera must be a child of the arm, or `camera_in_arm` must be set.
The camera frame follows Viam's camera convention (x right, y down, z forward), which is COLMAP's
right-down-front, so no axis permutation is needed. This is asserted by the sim test (§12), not assumed.

### 6.2 DoCommands on the scanner

`start_scan`. Every field optional except where noted; defaults come from the attributes above.

```json
{"start_scan": {
  "scan_id": "cell-2026-11-03a",
  "num_poses": 150,
  "images_per_pose": 1,
  "region_mm": { "center": [300, 0, 0], "size": [500, 500, 100] },
  "standoff_mm": [200, 450],
  "elevation_deg": [25, 80],
  "azimuth_deg": [-150, 150],
  "seed": 0,
  "dwell_s": 0.5,
  "upload": true,
  "dataset_id": "…",
  "tags": ["splat", "cell-v1"],
  "resume": false,
  "dry_run": false
}}
```

| Field | Meaning |
| --- | --- |
| `num_poses` | viewpoints to capture. 120 to 200 for a cell-sized scene. The planner samples 3× candidates and stops when this many succeeded or candidates run out. |
| `images_per_pose` | frames taken at each viewpoint. 1 by default. More than 1 is a burst: the sharpest is kept unless `keep_all` is set. Identical viewpoints add no parallax, so this exists for exposure bracketing and sharpness selection, not for "more data". |
| `region_mm` | the box to reconstruct, arm base frame: centre and size in mm. Look-at points are spread over its top face, see §7. |
| `standoff_mm`, `elevation_deg`, `azimuth_deg` | where the camera stands relative to its look-at point: distance range, and the direction window, see §7. |
| `seed` | makes the viewpoint list reproducible; stored in the manifest. |
| `dwell_s` | wait after `is_moving()` turns false before capture. |
| `resume` | continue a scan_id whose manifest exists: skips viewpoints already in `frames.jsonl`. |
| `dry_run` | plan only: returns the candidate list and the geometric filter statistics, moves nothing. |

Response: `{"scan_id": "...", "state": "running", "planned": 150, "scan_dir": "..."}`. A second `start_scan`
while one is running returns an error; one scan at a time.

`scan_status` → `{"scan_id", "state": planning|capturing|uploading|finalizing|done|failed|cancelled,
"poses_done", "poses_planned", "poses_skipped", "images_saved", "uploads_done", "uploads_pending",
"sequence_id", "elapsed_s", "eta_s", "last_error"}`.

`cancel_scan` → stops after the current move, runs the upload phase for what was captured (sequence covers the
captured window), state `cancelled`. The scan dir stays usable.

`upload_scan {"scan_id"}` → re-runs phase 2 from disk for a scan whose upload failed or was disabled. Idempotent
per frame: frames already marked uploaded in `frames.jsonl` are skipped.

The job runs as an asyncio task owned by the service so the DoCommand returns immediately and viam-server's
request timeout never cuts a 15-minute scan. `close()` cancels the task.

### 6.3 Scan directory (local ledger)

```
<VIAM_MODULE_DATA>/scans/<scan_id>/
  manifest.json
  frames.jsonl
  images/0001.png …            full resolution, lossless
```

`manifest.json`: `schema_version`, `scan_id`, request as received, resolved defaults, `machine_id`, `part_id`,
`arm`, `camera`, `camera_properties` (intrinsics and distortion, and whether they came from the camera or config),
the camera-in-arm transform (and its source), `frame_system_config` hash, `module_version`,
`viam_server_version`, `started_at`, `finished_at`, `state`, `sequence_id`, `dataset_id`, `planned_viewpoints`
(the full seeded list, so `resume` does not resample).

`frames.jsonl`, one line per saved image:

```json
{"index": 17, "file": "images/0017.png", "viewpoint": 17, "burst": 0,
 "t_request": "2026-11-03T14:02:11.120Z", "t_capture": "2026-11-03T14:02:11.187Z",
 "T_base_cam": {"t_m": [0.312, -0.105, 0.248], "q_wxyz": [0.71, 0.0, 0.70, 0.0]},
 "arm_end_pose": {"x_mm": …, "y_mm": …, "z_mm": …, "o_x": …, "o_y": …, "o_z": …, "theta_deg": …},
 "joints_deg": [12.1, -30.4, 55.0, 0.2, 65.3, 10.0],
 "sharpness": 142.3, "settle_s": 0.62, "width": 1920, "height": 1080,
 "uploaded": {"binary_id": null, "tabular_ok": false}}
```

`T_base_cam` is camera-to-base, metres, quaternion `(w, x, y, z)`, from `get_pose(camera, "<arm>_origin")` (SDK ≥
0.83.0) or the §4 fallback, with the orientation vector converted to a quaternion. `t_capture` is
`ResponseMetadata.captured_at` from `get_images()` when the camera fills it, else the module clock after the call.

### 6.4 Cloud: every call, what it creates

Phase 2 is driven by `frames.jsonl`. Dataset creation is not in the module: the operator creates the sequence
dataset once (UI or CLI) and puts its id in the config.

| Call | How often | Creates in the cloud | Key |
| --- | --- | --- | --- |
| `binary_data_capture_upload(binary_data=<png>, part_id, component_type="rdk:component:camera", component_name=<camera>, method_name="GetImages", file_extension=".png", data_request_times=(t_request, t_capture), tags=[scan_id, "splat", …])` | per image | one binary record, indexed at the capture time | module key |
| `tabular_data_capture_upload(tabular_data=[{"readings": <frame record>}], part_id, component_type="rdk:component:sensor", component_name=<scan_state>, method_name="Readings", data_request_times=[(t_request, t_capture)], tags=[…])` | per image | one tabular row, same timestamps, payload = frame record | module key |
| `create_sequence(part_id, resources=[(<camera>, "GetImages"), (<scan_state>, "Readings")], sequence_tags=[scan_id, "splat", …], start_time=first t_request − 1 s, end_time=last t_capture + 1 s)` | per scan | one sequence: a saved filter that selects the rows above | key with part access |
| `add_sequences_to_dataset(dataset_id, [sequence_id])` | per scan | membership of the sequence in the dataset | organization key |

End state in the cloud: N images and N pose rows indexed at capture times, one sequence covering them, inside one
dataset that accumulates one sequence per scan. If the module key lacks organization scope for the last call, the
module records `sequence_id` in the manifest and `splat-build` does the dataset add with an org key when given
`--finalize`.

Retries: 3 attempts per call, 2 s / 5 s backoff. Files stay on disk after upload; a `prune` DoCommand removes
scans older than N days.

### 6.5 Tools

Two tools, two machines, no shared input.

**`splat-scan`, talks to the robot.** A thin wrapper around the DoCommands, run from a laptop with an API key
for the machine. `start` sends `start_scan`, then polls `scan_status` and prints progress until the sequence id
appears.

```
splat-scan start --num-poses 150 [--dry-run] [--tags cell-v1]
splat-scan status
splat-scan cancel
splat-scan upload --scan-id <id>
```

**`splat-build`, never touches the robot.** One command. Input: a sequence id. Output: a Gaussian splat.

```
splat-build --dataset-id <id> --sequence-id <id> --work runs/ [--until <stage>] [--force <stage>]
```

Stages, each writing into its own folder under the work dir and skipped on re-run when its output exists, see §10.
`--until` stops after a stage, `--force` redoes a stage and everything after it.

## 7. Viewpoint planning

Look-at poses over a region, not around a point. A single look-at point makes every ray converge on one spot, so
the reconstruction is sharp there and smeared at the edges of the work surface. A box spreads the rays over the
whole surface, which is what any later viewpoint of a tabletop needs.

1. Look-at points: `num_poses` points on the top face of `region_mm` from a 2-D Halton sequence (seeded), so the
   rays sweep the surface evenly without the clustering of uniform random samples.
2. Directions: `3 × num_poses` unit vectors from a Fibonacci lattice restricted to the window
   `elevation ∈ [e_min, e_max]`, `azimuth ∈ [a_min, a_max]`. A Fibonacci lattice gives near-equal area per point
   anywhere on the sphere, which is why it is used instead of a latitude and longitude grid.
3. Positions: look-at point plus distance × direction, distance drawn from 2 or 3 discrete shells across
   `standoff_mm` (inner shell first: an arm camera works at 10 to 30 cm in task time, and the splat must be sharp
   there). Each candidate pairs one look-at point with one direction.
4. Orientation: camera z toward its look-at point, roll chosen so image up is closest to base-frame up (the
   `viam:opencv` `look_at_rotation` rule), plus a roll jitter of ±10° (seeded) so the splat sees rotations.
5. Geometric filter before any motion: implied flange position (camera pose composed with the inverse camera-in-arm
   transform) inside 0.9 × the arm's reach, above the table plus a margin, outside configured obstacles; the arm
   base must be outside the camera frustum (a dynamic object in frame corrupts the splat); the end effector's own
   footprint in the image is handled by the static mask. The planner reports the reachable part of the direction
   window and the fraction of the region's top face covered by at least three accepted views.
6. Order: greedy nearest-neighbour in Cartesian space, shells interleaved, so a cancelled scan still covers the
   region.
7. Execution: `motion.move(component_name=<camera>, destination=PoseInFrame("<arm>_origin", pose))`. On a
   planning failure, retry the same position with roll stepped by ±45° and ±90° before skipping the candidate
   (roll is free for a splat). The scan stops when `num_poses` succeeded or candidates run out, and reports both
   counts. There is no plan-only API in the motion service, so `dry_run` reports only the geometric filter.

Overlap: with a 70° horizontal field of view at 200 mm standoff, one frame covers about 280 mm of surface, so
neighbouring views of a 500 mm region share most of their content; 150 poses over a 300° × 55° direction window
at two shells is about 12° spacing in direction.

v2, not in this spec: `target_cameras`, a list of fixed cameras on the machine whose renders must look right. The
planner would add views along each camera's ray into the region, and the cameras themselves would contribute posed
images at scan start with the arm parked out of view. A handheld photo pass over the periphery, merged in
`splat-build`, is the other v2 item. Both build on the box; nothing in v1 has to be undone.

## 8. Capture at a viewpoint

1. After `move` returns, poll `arm.is_moving()` until false, then wait `dwell_s`.
2. Read the camera pose (§4), `arm.get_end_position()`, `arm.get_joint_positions()` (pose first and last; if the
   two camera poses differ by more than 0.2 mm or 0.05°, the arm was still settling: wait and retry once).
3. `camera.get_images()`; decode; sharpness = variance of the Laplacian; below `min_sharpness` retake up to 2×.
   Camera calls go through a retry wrapper (USB dropouts during arm motion are common; `viam:opencv` does the
   same).
4. With `images_per_pose > 1`, take the burst and keep the sharpest unless `keep_all`.
5. Write the PNG, append the frame line, update the scan-state sensor's last reading.

Exposure and white balance are locked once at scan start through `camera_lock_command` (a DoCommand payload the
camera module understands; empty if the camera has no such control, in which case the operator locks it in the
camera config). The arm's joint speed cap and collision sensitivity are whatever the machine runs with; the scan
adds nothing unsafe but moves through a larger volume than the task, so the frame-system obstacles must include
anything mounted in that volume.

Time budget: 150 poses × (move ≈ 4 s + settle 0.6 s + capture 0.3 s) ≈ 12 to 15 min, then upload.

## 9. Pose math

- Viam poses are millimetres plus an orientation vector `(o_x, o_y, o_z, θ)`. The module converts to metres and a
  unit quaternion with the RDK `spatialmath` definition, unit-tested against values produced by the Go RDK
  (`viam-labs/opencv/go_utils` generates the fixtures).
- `T_base_cam` (camera-to-base) comes from the frame system, which composes arm forward kinematics with the
  camera's configured frame, or from the §4 fallback. Nothing in this spec computes FK or hand-eye itself.
- COLMAP wants world-to-camera: `R_wc = R_cwᵀ`, `t_wc = −R_wc t_cw`, i.e. `q_wc = conj(q_cw)` and
  `t_wc = −R(q_wc) t_cw`, translation in metres. `splat-build` writes this.
- Camera model: `OPENCV` with `fx fy cx cy k1 k2 p1 p2` when `k3` is zero, `FULL_OPENCV` otherwise; `PINHOLE` if
  the distortion model is empty; refuse if the intrinsics are missing.

## 10. `splat-build`: sequence → COLMAP → 3DGRUT → splat

Uses `pycolmap` rather than the COLMAP CLI so it runs anywhere Python runs; 3DGRUT is invoked as a subprocess in
its own environment (`--threedgrut-dir` or `THREEDGRUT_DIR`). Stages:

1. **export**: `viam dataset export` into `runs/<dataset_id>/export/`. Cached per dataset. (If datasets grow to
   dozens of scans, export by tag instead; every upload is tagged with the scan id.)
2. **join**: select this sequence's rows, pair each image row with its pose row by capture time (exact, both were
   stamped with the same `t_capture`), write `runs/<sequence_id>/join/{images/, frames.jsonl, manifest.json}`.
   This is the robot's scan directory rebuilt from the cloud; it is an internal stage output.
3. **colmap**, following COLMAP's FAQ "Reconstruct sparse model from known camera poses":
   `images/` hard-linked; the static mask as `masks/<name>.png.png` (COLMAP naming) and `images/<name>_mask.png`
   (3DGRUT naming); `feature_extractor` with `ImageReader.single_camera=1`, the camera model and params from the
   manifest, `camera_mask_path`; write `sparse_known/` with ids matching the database and world-to-camera poses
   from §9; `exhaustive_matcher`; `point_triangulator` → `sparse/0`. Optional `--refine-extrinsics` runs a
   `bundle_adjuster` with intrinsics fixed into `sparse/0_ba` and reports the per-image pose correction (median
   and max, mm and degrees). That number is the hand-eye and repeatability check (§11). Training uses the
   unrefined poses so the splat stays rigidly attached to the arm base. Writes `report.json`.
4. **train**: `python train.py --config-name apps/colmap_3dgut.yaml path=<colmap> out_dir=<runs> experiment_name=<sequence_id>
   export_usd.enabled=true export_usd.format=nurec export_usd.apply_normalizing_transform=false`.
5. **export-splat**: copy `scene.usdz`, `scene.ply` and the acceptance table into `runs/<sequence_id>/splat/`.
   For Isaac Sim, the follow-up `threedgrut.export.scripts.add_mesh_to_usdz --set_collision --set_invisible` with
   the consumer's collision proxies is the consumer's job (for the Jenga cell: `jenga/docs/research/06-gaussian-splatting.md`).

```
runs/<sequence_id>/
  join/        images + poses          (internal)
  colmap/      3DGRUT input + report   (internal, useful for debugging)
  splat/       scene.usdz, scene.ply, report.json   (the output)
```

## 11. Acceptance

A scan is good when all of these hold; `splat-build` prints the table after the colmap stage and again after
training.

| Check | Threshold |
| --- | --- |
| Poses captured | ≥ 0.8 × `num_poses`; skipped list reviewed |
| Sharpness | every kept frame ≥ `min_sharpness`; median reported |
| Coverage | no hole in the reachable direction window larger than 25° (nearest-neighbour angular gap); ≥ 90 % of the region's top face seen by ≥ 3 views |
| COLMAP registration | ≥ 95 % of images have ≥ 500 triangulated observations |
| Reprojection error | mean < 1.0 px after triangulation with fixed poses |
| Pose correction (BA, diagnostic) | median < 2 mm and < 0.3°, max < 3 mm and < 0.5° |
| Scale | optional, when a fiducial is in the scene: PnP camera pose vs frame-system pose, residual < 3 mm |
| 3DGRUT | held-out PSNR ≥ 28 dB at the inner shell, no floaters within 10 mm of the work surface when inspected in the consumer's viewport |

## 12. Testing

- Unit: viewpoint sampler (determinism by seed, filter rules, spacing and surface-coverage statistics,
  reachable-window report);
  orientation vector → quaternion against Go RDK values; COLMAP writer round-trips through
  `pycolmap.Reconstruction`; `frames.jsonl` and manifest schema validation; upload phase with a fake data client
  (retries, idempotent re-run, drain on cancel).
- Sim first: the same DoCommand against a `viam-isaac-sim` machine with the camera attached to the arm
  (`parent_prim`, frame on the arm). Isaac's camera poses are ground truth, so `splat-build` must produce
  near-zero BA corrections and 3DGRUT must train a clean splat of the sim scene. This is the convention test for
  §9 (axis convention, mm→m, quaternion order) and runs in CI against the mock sim where possible.
- Real: one 30-pose smoke scan (5 min), check the acceptance table, then the full scan.

## 13. Repo layout and work plan

`viam-labs/gaussian-splat`, one repo, three Python packages with separate dependency groups so the module tarball
never carries COLMAP or parquet libraries onto the robot:

| Package | Contents | Deps |
| --- | --- | --- |
| `splat_module/` | the two Viam models | viam-sdk, numpy, opencv (sharpness) |
| `splat_scan_cli/` | `splat-scan` | viam-sdk |
| `splat_build/` | `splat-build` | viam-sdk (export), pyarrow, pycolmap; 3DGRUT external |

Module packaging follows the `jenga-bot` scaffold (uv, source tarball, `setup.sh` builds the venv on the machine).

| Step | Est. | Notes |
| --- | --- | --- |
| Repo scaffold (copy from `jenga-bot`), registry entry, CI | 0.5 d | |
| Spike: uploaded captures appear in a `create_sequence` window; module key can `create_sequence`; dataset add needs an org key? | 0.5 d | decides the §5 risk; fallback is the capture-control path |
| Module: planner, job, capture, ledger, status/cancel/resume, scan-state sensor | 1 d | |
| Module: upload phase + sequence creation | 0.5 d | template: `viam-isaac-sim/dataset_upload.py` |
| `splat-scan` CLI | 0.25 d | |
| `splat-build`: export, join, colmap, train, acceptance table | 0.75 d | |
| Sim run end to end (scan → build → USDZ in a twin) | 0.5 d | needs a GPU box for 3DGRUT |
| Real smoke scan + full scan on the Jenga cell | 0.5 d | needs hand-eye done (`viam:opencv`) |

## 14. Open questions

1. Resolved 2026-10-29: a box region with look-at points spread over its top face replaces the single look-at
   point (§7). Fixed target cameras and a handheld periphery pass are v2. Still to decide later: whether
   `region_mm` can default to a frame-system geometry (the table) instead of explicit numbers.
2. Reach versus coverage: with the region well in front of the base, the far side of the direction window is
   unreachable and the coverage criterion is measured over the reachable window (§7 reports it). Confirm the
   consumer's view directions fall inside it.
3. Whether uploaded captures are selectable by sequences (spike). If not, the capture-control path: the module
   sends `start_capture` with `frequency_hz` ≈ 2 for the camera and 10 for the scan-state sensor, and
   `splat-build` keeps only frames whose timestamp falls inside a dwell window flagged by the sensor.
4. Which camera on the first consumer, its native resolution, and whether its module exposes exposure and
   white-balance lock through DoCommand or config. Resolution ≥ 1280 × 720 recommended; 3DGRUT downsamples if needed.
5. Masking the arm itself (lower links visible at low elevations): v1 filters viewpoints whose frustum contains the
   base; v2 renders the arm's kinematic model into the image to produce a per-frame mask and lifts the filter.

## 15. References

- 3DGRUT: https://github.com/nv-tlabs/3dgrut (`threedgrut/datasets/dataset_colmap.py`, `configs/dataset/colmap.yaml`,
  `configs/initialization/colmap.yaml`, `configs/base_gs.yaml`, `threedgrut/export/README.md`)
- COLMAP FAQ, known poses and masks: https://colmap.github.io/faq.html
- Viam sequences: https://docs.viam.com/data/sequences · capture on demand:
  https://docs.viam.com/data/capture-sync/capture-on-demand · dataset format:
  https://docs.viam.com/train/sequence-dataset-format · create a dataset: https://docs.viam.com/train/create-a-dataset
- Viam data client (`binary_data_capture_upload`, `tabular_data_capture_upload`, `create_sequence`,
  `add_sequences_to_dataset`): https://docs.viam.com/reference/apis/data-client
- Frame system API (`get_pose`, SDK ≥ 0.83.0): https://docs.viam.com/motion-planning/reference/frame-system-api
- Camera API (`get_properties`, `get_images` metadata): https://docs.viam.com/reference/apis/components/camera
- Integration pattern, hand-eye and intrinsics: https://github.com/viam-labs/opencv (`hand_eye_calibration`,
  `camera_calibration`, `active_calibration/pose_sampler.py`, `go_utils`)
- Upload template: `viam-modules/viam-isaac-sim/src/isaac_module/dataset_upload.py`
- First consumer's research note on splats in Isaac Sim: `jenga/docs/research/06-gaussian-splatting.md`
