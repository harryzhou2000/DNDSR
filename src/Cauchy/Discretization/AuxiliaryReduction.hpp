#pragma once

#include "DNDS/Core/Defines.hpp"
#include <algorithm>
#include <limits>

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    /// Private typed-view arithmetic shared by flow5 and equation preparation.
    /// The caller retains geometry/output leases and prepares the face channels.
    /// Incidence traversal and addition order are deliberately unchanged: each
    /// cell sums spectral bounds, maximizes the wave jump and minimizes pressure.
    /// Rotation contributes volume*|omega| only to the spectral sum.
    template <class Data, class Metric, class Writes, class Views, class Scratch>
    void ReduceAuxiliaryCells(const Data &data, const std::vector<index> &cells,
                              Metric metric, real rotationBound, const Writes &writes,
                              Views values, Scratch scratch)
    {
        for (const auto iCell : cells)
        {
            real sum = 0, wave = 0, pressure = std::numeric_limits<real>::max();
            for (index entry = data.auxiliaryCellOffsets[iCell]; entry < data.auxiliaryCellOffsets[iCell + 1]; ++entry)
            {
                const auto iAux = data.auxiliaryCellFaces[entry];
                if (writes[4])
                    sum += data.auxiliaryFaceArea[iAux] * (writes[1] ? values[1](iAux, 0) + values[1](iAux, 1) : scratch(iAux, 0));
                if (writes[5])
                    wave = std::max(wave, writes[2] ? values[2](iAux, 0) : scratch(iAux, 1));
                if (writes[6])
                    pressure = std::min(pressure, writes[3] ? values[3](iAux, 0) : scratch(iAux, 2));
            }
            if (writes[4])
                values[4](iCell, 0) = sum + metric.GetCellVol(iCell) * rotationBound;
            if (writes[5])
                values[5](iCell, 0) = wave;
            if (writes[6])
                values[6](iCell, 0) = pressure;
        }
    }
}
