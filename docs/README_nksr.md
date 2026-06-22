# Neural Kernel Surface Reconstruction (NKSR)

This document describes how to run the NKSR surface reconstruction pipeline and its dependencies. NKSR is the **main method** of the project; the other reconstruction scripts are classical baselines.

## Overview

NKSR is a learned implicit-surface method: from oriented points it predicts compactly-supported kernels on a sparse voxel hierarchy, solves a sparse linear system for an implicit field, and extracts a dual mesh. Its memory and runtime grow with the number of points and the reconstructed volume, so a whole building cannot be reconstructed in a single call on one GPU.

The pipeline makes NKSR usable at building scale by decomposing the scene by **geometric difficulty** rather than a fixed grid:

- The scene is split into fine cells, each classified **planar** or **complex** by an SVD plane-fit residual.
- **Complex** regions (furniture, objects, people) grow first and absorb their surrounding flat context, so every object–wall transition is reconstructed with full context in one call — no stitching artifacts.
- Leftover **flat** surfaces form cheap coplanar units reconstructed at coarse resolution.
- Each unit is reconstructed at a resolution matched to its difficulty, then all per-unit meshes are stream-merged into one PLY.

Normals are read directly from the input PLY (the NavVis scanner exports them), and color is attached as an NKSR texture field. A **CUDA GPU is required**.

## Requirements

See the top-level `requirements.txt` for the base dependency list. NKSR and a matching CUDA build of `torch` are **not** installed by `requirements.txt` — see the [main README](../README.md) "Dependencies" section for how they are provided (the team's shared `nksr` conda environment).

## Configuration

All parameters are set in `configs/nksr_config.yaml`. Key entries:

| Key | Description |
|---|---|
| `paths.pointcloud_las` | Input LAS point cloud (xyz + colors). Set to `null` to load everything from the PLY |
| `paths.pointcloud_ply` | Input PLY (provides normals; also xyz/colors when no LAS is set) |
| `paths.output_dir` | Output directory |
| `paths.output_mesh` | Output mesh filename |
| `voxel.base_size` | Base voxel size in metres; per-class voxel sizes are multiples of this |
| `voxel.fine_chunk_size` | Cell size used for the planar/complex segmentation in metres |
| `planarity.residual_threshold` | SVD residual below which a cell is classified planar |
| `planarity.angle_threshold_deg` | Max angle between normals to merge two planar cells |
| `planarity.coplanar_dist_threshold` | Max mean point-to-plane distance to merge two planar cells (metres) |
| `planarity.min_points_per_chunk` | Minimum points for a cell to be considered (else ignored as noise) |
| `reconstruction.complex_detail_level` | NKSR detail level for complex units |
| `reconstruction.complex_mise_iter` | Dual-mesh extraction (MISE) iterations for complex units |
| `reconstruction.complex_voxel_factor` | Complex voxel size = `base_size` × this |
| `reconstruction.complex_max_extent_m` | Max spatial extent of a complex unit (metres) — bounds GPU memory |
| `reconstruction.complex_max_pts` | Max points per complex unit |
| `reconstruction.planar_very_flat_threshold` | Residual below which a planar unit is "very flat" (coarsest settings) |
| `reconstruction.planar_very_flat_detail` / `_mise_iter` / `_voxel_factor` | Settings for very-flat planar units |
| `reconstruction.planar_flat_detail` / `_mise_iter` / `_voxel_factor` | Settings for merely-flat planar units |
| `reconstruction.planar_overlap_m` | Overlap margin added to planar units for context, then trimmed back (metres) |
| `subsampling.planar_max_pts` | Max points per planar unit (voxel-downsampled until under this) |
| `misc.min_pts_per_unit` | Skip reconstruction units smaller than this |
| `misc.gpu_device` | CUDA device, e.g. `cuda:0` |

### Tuning notes

- `planarity.residual_threshold` is the dominant quality/runtime lever: too low over-classifies cells as complex (slow), too high misses real detail.
- `complex_max_extent_m` and `complex_max_pts` are the scalability knobs — lower them if a unit runs out of GPU memory.
- The `*_voxel_factor` values trade detail against GPU time; flat surfaces reconstruct well at coarse resolution, so planar factors are larger than the complex factor.

## Run

1. Activate your Python environment (with GPU dependencies available).
2. Edit `configs/nksr_config.yaml` with your paths and parameters.
3. Run from the repository root:

```bash
python scripts/reconstruction/run_nksr_reconstruction.py
```

Useful flags:

```bash
--config FILE       Path to YAML config (default: configs/nksr_config.yaml)
--save-chunks       Keep the per-region chunk PLYs after merging
--save-boundaries   Also write chunk_boundaries.ply (region bbox wireframes, red=complex / blue=planar)
--save-voxel-grid   Also write voxel_grid.ply (fine-cell outlines)
--merge-only        Skip reconstruction; just stream-merge an existing chunks dir (no GPU)
```

## Pipeline steps

1. **Load** — reads xyz/colors from the LAS (or PLY), and normals from the PLY; recenters the cloud.
2. **Voxelize** — assigns points to fine cells (`fine_chunk_size`).
3. **Classify** — fits a plane per cell with SVD and labels it planar or complex by the residual.
4. **Region growing** — complex regions grow first, absorbing adjacent planar cells; remaining planar cells merge into coplanar units.
5. **Reconstruct** — each unit is reconstructed with NKSR at a resolution matched to its class; planar units use an overlap margin and are trimmed back to their core box.
6. **Merge** — per-unit meshes are stream-merged into one binary PLY (one chunk in memory at a time).

## Outputs

The merged mesh is written to `<output_dir>/<output_mesh>` (default `nksr_reconstruction.ply`). Per-region chunk PLYs are written under `<output_dir>/chunks/` during the run; they are removed after merging unless `--save-chunks` is given.
