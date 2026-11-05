#!/usr/bin/env bash
# Actual repeated RHS calls, serial compared process groups, no node pin.
#SBATCH --job-name=cauchy-audit-e3-rhs
#SBATCH --partition=amd_256,amd_512,amd_1T,amd_2T
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=180G
#SBATCH --time=00:15:00
set -e -o pipefail
task_root=${1:?task root with runtime.sh and python}
revision=${2:?exact source commit}
profile=${3:-flow5}
source "$task_root/runtime.sh"
export OMPI_CC=gcc OMPI_CXX=g++ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DNDS_DISABLE_ASYNC_MPI=0
if [[ "$profile" == split-mixture ]]; then
    export DNDS_MECH_PATH="$task_root/source/external/cfd_externals/install/data"
fi
hostname
module list 2>&1
mpirun --version
mpicxx --showme
scontrol show job "$SLURM_JOB_ID"
lscpu
timeout --signal=TERM --kill-after=10s 850s \
    "$task_root/python" "$task_root/source/scripts/benchmark/plan53_copy_rhs.py" \
    --binary "$task_root/source/build-cpu/test/cpp/cauchy_benchmark_coefficient_rhs" \
    --mpiexec "$BSSCA_MPI_ROOT/bin/mpirun" --records "$task_root/results-e3-rhs-$SLURM_JOB_ID" \
    --expected-source "$revision" --profile "$profile"
