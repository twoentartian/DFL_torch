#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=4
#SBATCH --time=120:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=y.tian-3@tudelft.nl

module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0
source ~/venv/torch/bin/activate
mpiexec -n 32 -x OMP_NUM_THREADS=2 python3 ./simulator_mpi.py
