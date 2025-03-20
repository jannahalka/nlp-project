#!/bin/bash
#SBATCH --job-name=train.0
#SBATCH --output=train.0.out
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --gres=gpu
#SBATCH --mem=30G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=scavenge

python3 main.py
