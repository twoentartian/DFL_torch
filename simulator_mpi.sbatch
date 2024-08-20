#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=120:00:00

module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0
source ~/venv/torch/bin/activate
mpiexec -n 4 python3 ./simulator_mpi.py
