# Screened Poisson Reconstruction

This document describes how to run the Screened Poisson surface reconstruction script and its dependencies. Poisson is a classical, global-implicit baseline.

## Overview

Screened Poisson reconstruction solves a Poisson equation for an indicator function from oriented points, producing a watertight implicit surface. It is the closest classical analogue to NKSR's implicit-field approach, which makes it a useful baseline. It depends on good, consistently-oriented normals — which the input PLY already provides.

Because the global solve hallucinates surface in unobserved regions, a **density trim** removes low-confidence vertices after reconstruction.

## Requirements

See the top-level `requirements.txt` for the dependency list. The reconstruction uses Open3D's built-in Poisson implementation, so no extra GPU/external packages are needed.

## Configuration

All parameters are set in `configs/poisson_config.yaml`. Key entries:

| Key | Description |
|---|---|
| `paths.pointcloud_ply` | Input PLY point cloud (must contain normals and colors) |
| `paths.output_dir` | Output directory |
| `paths.output_mesh` | Output mesh filename |
| `reconstruction.depth` | Octree depth — controls resolution (10 ≈ 0.1 m, 11 ≈ 0.05 m, 12 ≈ 0.025 m) |
| `reconstruction.width` | Target finest-cell width (0 = use `depth` instead) |
| `reconstruction.scale` | Ratio of the reconstruction cube to the samples' bounding box |
| `reconstruction.linear_fit` | Use linear interpolation of sample positions |
| `reconstruction.n_threads` | Threads for the solve and color transfer (-1 = all cores) |
| `trimming.density_percentile` | Remove vertices below this density percentile (5 = conservative, 15–20 = aggressive) |

### Tuning notes

- `depth` is the main quality/cost lever: higher depth gives finer detail but is slower and more memory-hungry.
- `density_percentile` controls how aggressively hallucinated exterior surface is removed — raise it if the mesh balloons beyond the observed geometry, lower it if real surface is being eaten away.

## Run

1. Activate your Python environment.
2. Edit `configs/poisson_config.yaml` with your paths and parameters.
3. Run from the repository root:

```bash
python scripts/reconstruction/run_poisson_reconstruction.py
```

## Pipeline steps

1. **Load** — reads the PLY point cloud (xyz, normals, colors).
2. **Poisson** — runs Open3D's screened Poisson reconstruction at the configured octree depth.
3. **Trim** — removes vertices whose density falls below the `density_percentile` threshold.
4. **Color transfer** — assigns each mesh vertex the color of its nearest input point (KD-tree lookup).
5. **Save** — writes the mesh to `output_dir/output_mesh`.

## Outputs

The mesh is written to `<output_dir>/<output_mesh>` (default `poisson_reconstruction_small_scene.ply`).
