#!/usr/bin/env bash
# Actual RHS and bounded nonlinear stages, serial comparison on one node.
#SBATCH --job-name=cauchy-audit-rhs-stage
#SBATCH --partition=amd_256,amd_512,amd_1T,amd_2T
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --mem=180G
#SBATCH --time=00:15:00
set -e -o pipefail
task_root=${1:?task root with runtime.sh and python}
revision=${2:?exact source commit}
source "$task_root/runtime.sh"
export OMPI_CC=gcc OMPI_CXX=g++ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DNDS_DISABLE_ASYNC_MPI=1
export DNDS_MECH_PATH="$task_root/source/external/cfd_externals/install/data"
hostname
module list 2>&1
mpirun --version
mpicxx --showme
scontrol show job "$SLURM_JOB_ID"
lscpu
timeout --signal=TERM --kill-after=10s 850s \
    "$task_root/python" "$task_root/source/scripts/benchmark/plan53_rhs_stage.py" \
    --build "$task_root/source/build-cpu" --mpiexec "$BSSCA_MPI_ROOT/bin/mpirun" \
    --records "$task_root/results-rhs-stage-$SLURM_JOB_ID" --expected-source "$revision"
