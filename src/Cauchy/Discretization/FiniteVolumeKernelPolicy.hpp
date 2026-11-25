#pragma once
#include "Cauchy/Riemann/RiemannFlux.hpp"
#include "Cauchy/Discretization/CellCenterViscosityView.hpp"
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
        real dissipationCoefficient = 1;
        bool desDissipation = false;
        DNDS_DEVICE_CALLABLE real DissipationAtFace(const CellCenterViscosityView &values, index cell, index other) const
        {
            return desDissipation ? dissipationCoefficient + (1 - dissipationCoefficient) * values.ShieldAtFace(cell, other)
                                  : dissipationCoefficient;
        }
        DNDS_DEVICE_CALLABLE bool NeedsPressureBounds(bool useStencil) const noexcept
        {
            return (useStencil && pressureStencil) || pressureLLFBlend;
        }
    };
}
