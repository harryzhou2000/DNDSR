#pragma once
#include "CFV/Math/LimitedVR.hpp"

namespace DNDS::CFV::Math
{
    /// Equation-independent cell seeds, preserving face-incidence multiplicity.
    /// Receivers next to a physical/interface boundary have no correction seed.
    template <class Stencil, class Input, class History, class Output>
    DNDS_DEVICE_CALLABLE void BuildLimitedVRSeedCell(Stencil stencil, Input scalar, Input gate, History previous,
                                                     Output output, LimitedVRKernelParameters settings, index iCell) noexcept
    {
        std::array<LimitedVRExtremumStencil, 2> extrema;
        for (auto &entry : extrema)
            entry.boundary = stencil.boundary(iCell, 0);
        real weak = output.Width() == 4 ? gate(iCell, 3) : 0;
        if ((settings.pressureCorrection && settings.alphaMax > 0) ||
            (settings.contactCorrection && settings.contactAlphaMax > 0))
            for (index entry = stencil.offsets(iCell, 0); entry < stencil.offsets(iCell + 1, 0); ++entry)
                if (const auto other = stencil.neighbors(entry, 0); other != UnInitIndex)
                {
                    extrema[0].Observe(scalar(iCell, 0), scalar(other, 0), gate(other, 0), settings.equalityTolerance);
                    extrema[1].Observe(scalar(iCell, 1), scalar(other, 1), gate(other, 2), settings.equalityTolerance);
                    if (settings.weakFirstMaximum)
                        weak = std::max(weak, gate(other, 3));
                }
        output(iCell, 0) = settings.pressureCorrection
                               ? extrema[0].Correction(scalar(iCell, 0), previous(iCell, 0), settings.alphaMax, settings.pressureSupport,
                                                       settings.extremumThreshold, settings.retention, settings)
                               : 0;
        output(iCell, 1) = settings.contactCorrection
                               ? extrema[1].Correction(scalar(iCell, 1), previous(iCell, 1), settings.contactAlphaMax, settings.contactSupport,
                                                       settings.contactExtremumThreshold, settings.contactExtremumRetention, settings)
                               : 0;
        output(iCell, 2) = 0;
        if (output.Width() == 4)
            output(iCell, 3) = weak;
    }
    /// Independent out-of-place maximum; no atomics or communication. Input
    /// ghost freshness belongs to the operator/caller's declared work region.
    template <class Stencil, class Input, class Output, class Base>
    DNDS_DEVICE_CALLABLE void LimitedVRMaximumCell(Stencil stencil, Input input, Output output, index iCell,
                                                   std::array<bool, 4> active, bool excludeBoundary, bool saturatedOnly,
                                                   std::array<real, 4> caps, Base base, unsigned mergeMask) noexcept
    {
        for (int column = 0; column < output.Width(); ++column)
            output(iCell, column) = input(iCell, column);
        const bool excludeCorrection = excludeBoundary && stencil.boundary(iCell, 0);
        for (index entry = stencil.offsets(iCell, 0); entry < stencil.offsets(iCell + 1, 0); ++entry)
            if (const auto other = stencil.neighbors(entry, 0); other != UnInitIndex)
                for (int column = 0; column < output.Width(); ++column)
                    if (active[column] && (!excludeCorrection || column == 3))
                    {
                        const real value = input(other, column);
                        const real candidate = !saturatedOnly ? value : value >= caps[column] * (1 - 1e-12) ? caps[column]
                                                                                                            : 0;
                        output(iCell, column) = std::max(output(iCell, column), candidate);
                    }
        for (int column = 0; column < 2; ++column)
            if (mergeMask & (1u << column))
                output(iCell, column) = std::max(output(iCell, column), base(iCell, column));
    }
}
