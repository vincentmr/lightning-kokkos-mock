#!/bin/sh
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4
#SBATCH --ntasks 128
#SBATCH --time 0-01:00:00
#SBATCH --constraint gpu
#SBATCH --account m4139

cd $SLURM_SUBMIT_DIR
bash run_bench.sh