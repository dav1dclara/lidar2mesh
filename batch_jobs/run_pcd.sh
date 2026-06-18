#!/bin/bash
#SBATCH --job-name=pcd_meshing
#SBATCH --output=/work/scratch/oscipal/logs/pcd_%j.log
#SBATCH --error=/work/scratch/oscipal/logs/pcd_%j.log
#SBATCH --time=32:00:00

mkdir -p /work/scratch/oscipal/logs

cd /work/courses/3dv/team13 || exit 1

# activate your environment — adjust if you use venv or a different conda env name
source $(conda info --base)/etc/profile.d/conda.sh
conda activate 3dv

python scripts/pcd_meshing.py
