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
        bool isothermalWallInviscidMirror = false;
        bool pressureLLFBlend = false;
        real pressureLLFBlendStart = 1.05, pressureLLFBlendFull = 2;
        DNDS_DEVICE_CALLABLE bool NeedsPressureBounds(bool useStencil) const noexcept
        {
            return (useStencil && pressureStencil) || pressureLLFBlend;
        }
    };
}
