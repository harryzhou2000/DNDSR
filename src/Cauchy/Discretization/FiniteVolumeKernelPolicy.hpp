#pragma once
#include "Cauchy/Riemann/RiemannFlux.hpp"
namespace DNDS::Cauchy::FiniteVolumeDetail
{
    struct FluxKernelPolicy
    {
        Riemann::Requirements required;
        Riemann::FluxSettings settings;
        bool rusanov = false;
        bool transverseSignal = false;
        bool pressureStencil = false;
        bool noRsOnWall = false;
        bool noGRPOnWall = false;
    };
}
