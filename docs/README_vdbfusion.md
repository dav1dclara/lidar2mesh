# VDBFusion (TSDF)

This document describes how to run the VDBFusion surface reconstruction script and its dependencies. VDBFusion is a classical, volumetric baseline.

## Overview

VDBFusion fuses the points into a **truncated signed-distance field (TSDF)** stored in a sparse VDB volume, then polygonizes it with Marching Cubes. It is a robust, well-established volumetric meshing approach and serves as the classical TSDF baseline against NKSR.

The script integrates the point cloud as a single static scan from a fixed sensor origin, then extracts a triangle mesh. It can optionally run a **parameter sweep** over voxel size, truncation, downsampling, and minimum weight to compare settings in one run.

## Requirements

See the top-level `requirements.txt` (`vdbfusion` is included). A working `vdbfusion` install with its native dependencies is required.

## Configuration

All parameters are set in `configs/vdbfusion_config.yaml`. Key entries:

| Key | Description |
|---|---|
| `input.las_path` | Input LAS point cloud |
| `fusion.voxel_size` | TSDF voxel size in metres (resolution) |
| `fusion.sdf_trunc` | Signed-distance truncation distance in metres |
| `fusion.space_carving` | Enable space carving during integration |
| `fusion.fill_holes` | Fill holes during mesh extraction |
| `fusion.min_weight` | Minimum TSDF weight for a voxel to contribute to the mesh |
| `preprocess.downsample_voxel_size` | Voxel downsampling of the input (`0.0` = no downsampling) |
| `static_mode.fixed_origin` | Sensor origin used for integration `[x, y, z]` |
| `output.output_dir` | Output directory |
| `output.map_name` | Base name for the output mesh/grid files |
| `sweep.enabled` | If true, run a parameter sweep instead of a single reconstruction |
| `sweep.output_subdir` | Subdirectory under `output_dir` for sweep results |
| `sweep.min_weight_values` / `downsample_voxel_sizes` / `sdf_trunc_values` / `voxel_size_values` | Lists of values to sweep over (cartesian product) |

### Tuning notes

- `voxel_size` controls resolution and cost: smaller is finer but slower and more memory-hungry.
- `sdf_trunc` is typically a small multiple of `voxel_size`; too small produces holes, too large oversmooths.
- `min_weight` trims weakly-observed voxels — raise it to remove noisy/hallucinated surface.
- Output filenames encode the parameters (`_ds…_vs…_st…_mw…`) so sweep results don't overwrite each other.

## Run

1. Activate your Python environment.
2. Point `input.las_path` in `configs/vdbfusion_config.yaml` at your LAS cloud and set the fusion parameters.
3. Run from the repository root:

```bash
python scripts/reconstruction/run_vdbfusion_reconstruction.py --config configs/vdbfusion_config.yaml
```

## Pipeline steps

1. **Load** — reads the LAS cloud, removes non-finite points, optionally voxel-downsamples.
2. **Integrate** — fuses the points into a TSDF VDB volume from the fixed origin.
3. **Extract** — polygonizes the TSDF into a triangle mesh (`fill_holes`, `min_weight`).
4. **Save** — writes the mesh and the VDB grid; when sweeping, iterates over all parameter combinations.

## Outputs

For each configuration the script writes a `.ply` mesh and a `.vdb` grid under `output_dir` (or `output_dir/<sweep.output_subdir>` when sweeping), named from `map_name` plus the encoded parameters.
