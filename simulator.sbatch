#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=120:00:00

source ~/venv/torch/bin/activate
python3 ./simulator.py
