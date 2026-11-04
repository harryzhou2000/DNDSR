#pragma once

#include "Cauchy/Physics/EquationSystem.hpp"

namespace DNDS::Cauchy
{
    /// Setup-only ownership result. Immutable equation metadata and mutable
    /// worker capacity stay separate; neither kernels nor EquationSystem hide
    /// a global/thread-ID chemistry context. Domain retains both owners.
    struct EquationSetup
    {
        EquationSystem equation;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        /// Load/clone on the calling controller, before numerical work. This
        /// performs no MPI; callers coordinate local failures before collectives.
        /// Paths follow Cantera lookup, with DNDS_MECH_PATH as an explicit
        /// fallback for a relative file that exists there. No backend fallback.
        static EquationSetup Resolve(const EquationSettings &settings, index workerNumber = 1);
    };
}
