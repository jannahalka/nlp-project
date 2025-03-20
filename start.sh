#!/bin/bash
#SBATCH --job-name=train.0
#SBATCH --output=train.0.out
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --gres=gpu
#SBATCH --mem=30G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=scavenge

module load Python/3.12.3-GCCcore-13.3.0

source .venv/bin/activate

python3 main.py
