#!/usr/bin/env bash
# One matched allocation; compared process groups run serially, no node pin.
#SBATCH --job-name=cauchy-audit-e3
#SBATCH --partition=amd_256,amd_512
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=180G
#SBATCH --time=00:15:00
set -e -o pipefail
task_root=${1:?task root with source and runtime.sh}
revision=${2:?exact source commit}
source "$task_root/runtime.sh"
export OMPI_CC=gcc OMPI_CXX=g++ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DNDS_DISABLE_ASYNC_MPI=0
hostname
module list 2>&1
mpirun --version
mpicxx --showme
scontrol show job "$SLURM_JOB_ID"
lscpu
timeout --signal=TERM --kill-after=10s 850s \
    "$task_root/python" "$task_root/source/scripts/benchmark/plan53_coefficient.py" \
    --binary "$task_root/source/build-cpu/test/cpp/cfv_benchmark_coefficient_publication" \
    --mpiexec "$BSSCA_MPI_ROOT/bin/mpirun" \
    --records "$task_root/results-e3-$SLURM_JOB_ID/records" \
    --raw "$task_root/results-e3-$SLURM_JOB_ID/raw" --expected-source "$revision"
