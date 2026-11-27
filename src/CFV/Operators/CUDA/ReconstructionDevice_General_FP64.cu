#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "CFV/Operators/CUDA/ReconstructionDevice_General.cuh"

namespace DNDS::CFV::Detail
{
    namespace
    {
        ReconstructionSweepResources LaunchGeneral(const GeneralSweepArguments &a, bool limited, bool execute = true)
        {
            ValidateSweepArguments(a, limited);
            DNDS_check_throw_info(!a.plan.quantized &&
                                      a.plan.n > 0 && a.mean.owned.Width() > 0 &&
                                      a.output.Width() == a.mean.owned.Width() * a.plan.n,
                                  "general FP64 reconstruction requires FP64 arithmetic and matching field widths");
            const index bytes = a.plan.raw ? a.output.Width() * index(sizeof(real)) : 0;
            static thread_local int ordinal = -1;
            static thread_local size_t sharedLimit = 0;
            if (ordinal != a.context.device.ordinal)
            {
                cudaDeviceProp properties{};
                DNDS_check_throw_info(cudaGetDeviceProperties(&properties, a.context.device.ordinal) == cudaSuccess,
                                      "general FP64 reconstruction device query failed");
                sharedLimit = properties.sharedMemPerBlock;
                ordinal = a.context.device.ordinal;
            }
            DNDS_check_throw_info(bytes <= index(sharedLimit), "general FP64 reconstruction scratch exceeds shared memory");
            if (!a.selection.RowCount())
                return {};
            ReconstructionSweepResources resource{bytes, 0, 0, int(std::min<index>(a.selection.RowCount(), 256)), 128};
            if (!execute)
            {
                cudaFuncAttributes attrs{};
                DNDS_check_throw_info(cudaFuncGetAttributes(&attrs, GeneralJacobi) == cudaSuccess &&
                                          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resource.activeBlocksPerSM, GeneralJacobi, 128, size_t(bytes)) == cudaSuccess,
                                      "general CUDA VR resource query failed");
                resource.registersPerThread = attrs.numRegs;
                return resource;
            }
            GeneralJacobi<<<unsigned(std::min<index>(a.selection.RowCount(), 256)), 128, size_t(bytes), ::DNDS::Runtime::CUDAStream(a.context)>>>(
                RawReconstructionSweepView(a.plan), a.selection, a.vectors, a.mean, a.input, a.boundary,
                a.output, a.status, a.target, a.alpha, a.plan.raw, limited);
            return resource;
        }
    }
    ReconstructionSweepResources InspectGeneral(const GeneralSweepArguments &a, bool limited) { return LaunchGeneral(a, limited, false); }
    void RunGeneralFP64(const GeneralSweepArguments &a) { LaunchGeneral(a, false); }
    void RunGeneralLimited(const GeneralSweepArguments &a) { LaunchGeneral(a, true); }
}
