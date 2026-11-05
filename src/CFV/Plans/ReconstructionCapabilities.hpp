#pragma once

#include "DNDS/Core/Defines.hpp"
#include <string>
#include <vector>

namespace DNDS::CFV
{
    enum class ReconstructionOperation
    {
        Sweep,
        WBAP,
        LinearSystem,
        LocalPreconditioner,
        InverseDiagnostics
    };

    /// CPU preparation contract, independent of polynomial/basis-cache settings.
    /// Defaults preserve the complete standalone API. Sampling is always present.
    /// Copies passed to preparation are frozen in the resulting plan; requesting
    /// an omitted operation never allocates or rebuilds matrices on first use.
    struct ReconstructionCapabilities
    {
        bool sweeps = true;
        bool wbap = true;
        bool linearSystem = true;
        bool localPreconditioner = true;
        bool inverseDiagnostics = true;
        static ReconstructionCapabilities ForSweeps(bool withWBAP = true)
        {
            ReconstructionCapabilities result;
            result.wbap = withWBAP;
            result.linearSystem = result.localPreconditioner = false;
            return result;
        }
        bool Supports(ReconstructionOperation operation) const
        {
            switch (operation)
            {
            case ReconstructionOperation::Sweep:
                return sweeps;
            case ReconstructionOperation::WBAP:
                return wbap;
            case ReconstructionOperation::LinearSystem:
                return linearSystem;
            case ReconstructionOperation::LocalPreconditioner:
                return localPreconditioner;
            case ReconstructionOperation::InverseDiagnostics:
                return inverseDiagnostics;
            }
            return false;
        }
        int Mask() const
        {
            return int(sweeps) | (int(wbap) << 1) | (int(linearSystem) << 2) |
                   (int(localPreconditioner) << 3) | (int(inverseDiagnostics) << 4);
        }
    };

    struct ReconstructionTableStorage
    {
        std::string name;
        bool allocated = false;
        uint64_t payloadBytes = 0;
        uint64_t allocationBytes = 0;
    };
    /// Rank-local storage accounting for this plan, excluding shared metrics,
    /// mesh, user workspaces and allocator bookkeeping/reserved pools. Payloads
    /// and unique Buffer allocation extents are separate. Runtime scratch peak
    /// counts preparation's face systems; native Eigen/standard-library scratch
    /// and process RSS are not inferred from these numbers.
    struct ReconstructionStorageReport
    {
        std::vector<ReconstructionTableStorage> tables;
        uint64_t retainedPayloadBytes = 0;
        uint64_t retainedAllocationBytes = 0;
        uint64_t preparationRuntimeScratchBytes = 0;
        uint64_t preparationRuntimePeakBytes = 0;
    };
}
