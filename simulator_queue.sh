#!/bin/sh

mpiexec -n 8 python3 ./simulator_mpi.py --config simulator_config.py
mpiexec -n 8 python3 ./simulator_mpi.py --config simulator_config_2.py
