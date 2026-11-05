#!/usr/bin/env bash
# E6 candidate/control process groups run serially in the same allocation.
#SBATCH --job-name=cauchy-audit-e6
#SBATCH --partition=amd_256,amd_512,amd_1T,amd_2T
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=180G
#SBATCH --time=00:15:00
set -e -o pipefail
task_root=${1:?task root with runtime.sh, python and e6-frozen binaries}
source "$task_root/runtime.sh"
export OMPI_CC=gcc OMPI_CXX=g++ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DNDS_DISABLE_ASYNC_MPI=0
hostname
module list 2>&1
mpirun --version
mpicxx --showme
scontrol show job "$SLURM_JOB_ID"
lscpu
timeout --signal=TERM --kill-after=10s 850s \
    "$task_root/python" "$task_root/source/scripts/benchmark/plan53_jacobian.py" \
    --baseline "$task_root/e6-frozen/baseline" --candidate "$task_root/e6-frozen/candidate" \
    --mpiexec "$BSSCA_MPI_ROOT/bin/mpirun" --records "$task_root/results-e6-$SLURM_JOB_ID"
