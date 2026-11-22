#include "CFV/Detail/ReconstructionSweepArguments.hpp"

namespace DNDS::CFV::Detail
{
    namespace V2
    {
#define CFV_V2_SHAPE(m, c) ReconstructionSweepResources M##m##C##c(const StagedSweepArguments &, bool, int);
#include "sweep_v2/catalog.inc"
#undef CFV_V2_SHAPE
    }
    ReconstructionSweepResources DispatchV2(const StagedSweepArguments &a, ReconstructionSweepArithmetic arithmetic)
    {
        const auto &p = a.plan;
        const int m = int(p.n), c = int(a.mean.owned.Width());
        const bool automatic = arithmetic == ReconstructionSweepArithmetic::V2Auto;
        const auto compact = [](auto v)
        { return !v.rows || (v.kind == R::RowLayoutKind::Table && v.stride == v.width && !v.offsets && !v.sizes); };
        const bool eligible = !p.quantized && !p.inverseTriangle && (!a.limited || m >= 5) &&
                              compact(a.mean.owned) && compact(a.mean.ghost) && compact(a.input.owned) && compact(a.input.ghost) &&
                              compact(a.output) && compact(a.target) && compact(a.alpha) && compact(a.boundary) &&
                              compact(p.matrixAInvB) && compact(p.rawMatrixB32) && compact(p.localInverse) && compact(p.localInverse32) &&
                              compact(p.vectorAInvB) && compact(p.boundaryInjection) && compact(p.rotation) &&
                              compact(p.neighbors) && compact(p.offsets) && compact(p.boundaryOffsets) && compact(p.faces) &&
                              compact(p.faceSide) && compact(p.rotationActive) && compact(a.vectors.starts) && compact(a.status);
        if (eligible && c >= 4 && c <= 7)
        {
            const bool mma = arithmetic == ReconstructionSweepArithmetic::V2MMA ||
                             (automatic && m >= 5 && (m != 5 || !p.packedRawB || c == 7) && FP64TensorCoreAvailable(a.context.device));
            DNDS_check_throw_info(!mma || (m >= 5 && FP64TensorCoreAvailable(a.context.device)), "V2 warp MMA requires M5, M9 or M19 and FP64 tensor cores");
            const int threads = a.options.threads ? a.options.threads : m == 5 && !p.raw && mma ? 64
                                                                                                : 128;
            switch (10 * m + c)
            {
#define CFV_V2_SHAPE(m, c) \
    case 10 * m + c:       \
        return V2::M##m##C##c(a, mma, threads);
#include "sweep_v2/catalog.inc"
#undef CFV_V2_SHAPE
            }
        }
        DNDS_check_throw_info(automatic && !p.quantized, "forced V2 sweep requires compact AoS C4..7, supported modes and a full inverse; LVR requires M5, M9 or M19");
        GeneralSweepArguments general{a.context, p, a.selection, a.vectors, a.mean, a.input, a.boundary, a.output, a.status, a.target, a.alpha};
        if (!a.execute)
            return InspectGeneral(general, a.limited);
        if (a.limited)
            RunGeneralLimited(general);
        else
            RunGeneralFP64(general);
        return {};
    }
}
