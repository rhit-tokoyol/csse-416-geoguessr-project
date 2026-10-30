#!/bin/bash
#SBATCH --job-name=geo-train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=shard:8
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CSSE416

cd /home/kennym1/CSSE416/
mkdir -p logs output

echo "node: $(hostname) | CPUs: $SLURM_CPUS_PER_TASK"
nvidia-smi

jupyter nbconvert --to notebook --execute main.ipynb \
    --output "executed_${SLURM_JOB_ID}.ipynb" \
    --output-dir output \
    --ExecutePreprocessor.timeout=-1