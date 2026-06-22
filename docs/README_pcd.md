# pcdmeshing (Block Delaunay)

This document describes how to run the pcdmeshing surface reconstruction script and its dependencies. pcdmeshing is a Delaunay/visibility-based baseline.

## Overview

pcdmeshing reconstructs a triangle mesh by tetrahedralizing the points (Delaunay) and extracting a surface, optionally using visibility information to decide which tetrahedra are inside or outside. The scene is processed in spatial **blocks** so large clouds can be meshed in parallel and then stitched, with seam/discard margins to keep block boundaries consistent.

It represents a different reconstruction paradigm from the implicit methods (NKSR, Poisson) and the volumetric one (VDBFusion), which is the point of including it as a baseline.

## Requirements

See the top-level `requirements.txt` for the base dependency list. The `pcdmeshing` package itself is **not** on PyPI in a standard way — see the [main README](../README.md) "Dependencies" section and the [pcdmeshing project](https://github.com/cvg/pcdmeshing) for installation.

## Configuration

All parameters are set in `configs/pcd_config.yaml`. Key entries:

| Key | Description |
|---|---|
| `paths.pointcloud_ply` | Input PLY point cloud (xyz + normals + RGB) |
| `paths.output_dir` | Output directory |
| `paths.output_mesh` | Output mesh filename |
| `meshing.voxel_size` | Block size in metres (the scene is split into blocks of this size) |
| `meshing.margin_seam` | Overlap margin kept between adjacent blocks for seamless stitching (metres) |
| `meshing.margin_discard` | Outer margin discarded from each block (metres) |
| `meshing.num_parallel` | Number of blocks meshed in parallel |
| `meshing.max_edge_length` | Reject triangles with edges longer than this (metres) |
| `meshing.max_visibility` | Visibility threshold used when `use_visibility` is enabled |
| `meshing.use_visibility` | Use visibility information to filter the tetrahedralization |

### Tuning notes

- `voxel_size` (block size) trades memory against stitching overhead: smaller blocks use less memory per block but create more seams.
- `max_edge_length` is the main triangle-quality filter — lower it to remove long spurious triangles across gaps.
- `num_parallel` should be set to roughly the number of available CPU cores, bounded by memory.

## Run

1. Activate your Python environment (with `pcdmeshing` available).
2. Edit `configs/pcd_config.yaml` with your paths and parameters.
3. Run from the repository root:

```bash
python scripts/reconstruction/run_pcd_reconstruction.py
```

## Pipeline steps

1. **Load** — reads the PLY point cloud (xyz + normals + RGB).
2. **Block meshing** — `run_block_meshing` splits the cloud into blocks, reconstructs each (Delaunay, optionally visibility-filtered), and stitches them using the seam/discard margins.
3. **Save** — writes the mesh to `output_dir/output_mesh`.

## Outputs

The mesh is written to `<output_dir>/<output_mesh>` (default `pcd_reconstruction_small_scene.ply`).
