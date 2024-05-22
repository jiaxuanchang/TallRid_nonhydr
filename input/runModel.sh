#!/bin/bash
#SBATCH --account=def-jklymak
#SBATCH --mail-user=julia27317@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=0-2:55:00
#SBATCH --mem-per-cpu=3G

# sbatch -J TideFr3000N04000 runModel.sh


# top=$1  Passed as qsub  -v top=h60h27f20 runModel.sh
which mpiexec

PARENT=TallRid_nonhydr

top=${SLURM_JOB_NAME}
results=${SCRATCH}/${PARENT}/results/
outdir=$results$top

cd $outdir/input
pwd

ls -al ../build/mitgcmuv
printf "Starting: $outdir\n"
mpiexec -N 24 ../build/mitgcmuv > mit.out
