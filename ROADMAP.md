# Roadmap

Status of the `viam:gaussian-splatting` module. Design in `docs/spec.md`.

## Done

- [x] V1 scanner service and scan-state sensor: seeded viewpoint planner over a box region, capture at rest with
      frame-system poses, local ledger, upload phase, sequence creation, dataset add (PR #1).
- [x] `splat-scan` CLI, including `create-dataset` for the one-time sequence-dataset setup.
- [x] `splat-build`: export, join, COLMAP known-pose triangulation, 3DGRUT command, export-splat, finalize.
- [x] Local end to end on a fake Lite 6 and fake camera (`dev/local-machine.json`).
- [x] Real cloud round trip (2026-10-29): scan → upload to Robin's Org → sequence in dataset
      `6ae3b52d0973c97689dcd421` → `viam dataset export` → join → COLMAP → 3DGRUT command.
- [x] COLMAP convention and metre scale proven on a synthetic textured plane (0.44 px, 1.3 mm).

## In progress

- [ ] **3DGRUT training run on the GPU box** (`ofc26gpu1`). Input: the 60-view synthetic plane scene built by
      `splat-build` locally. Blocked: the box has no compiler, CUDA toolkit or passwordless sudo, so 3DGRUT
      needs a user-space conda toolchain; `~/bootstrap_3dgrut.sh` on the box does that install and the
      permission classifier refused to run it from this session. Robin runs it (or allows the SSH command),
      then: rsync the COLMAP folder, `splat-build --run-train`, bring back `scene.usdz`.

## Next

- [x] Published `viam:gaussian-splatting` 0.0.1-rc.1 (platform any) on 2026-10-30; registry entry owned by the
      `viam-dev` org. Still to do: the repo's deploy credentials, so merges to `main` publish automatically.
- [ ] First scan on a real arm and camera: 30-pose smoke scan, acceptance table, then a full scan.
- [ ] Jenga cell config values (region, standoff, dataset id) in the jenga-bot repo's machine config.
- [ ] Spike result recorded: uploaded captures are selectable by sequences (confirmed 2026-10-29), so the
      capture-control fallback in the spec can be dropped.

## Later (v2)

- [ ] `target_cameras`: views along fixed cameras' rays plus their own posed images at scan start.
- [ ] Handheld periphery pass merged in `splat-build`.
- [ ] Per-frame arm mask from the kinematic model, lifting the base-in-frustum filter.
- [ ] `region_mm` defaulting to a frame-system geometry (the table).
