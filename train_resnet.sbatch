#!/bin/bash
#SBATCH --job-name=geo-train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=shard:16
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CSSE416

cd /home/parkerij/CSSE416/
mkdir -p logs output

echo "node: $(hostname) | CPUs: $SLURM_CPUS_PER_TASK"
nvidia-smi

jupyter nbconvert --to notebook --execute resnet_model.ipynb \
    --output "executed_${SLURM_JOB_ID}.ipynb" \
    --output-dir output \
    --ExecutePreprocessor.timeout=-1