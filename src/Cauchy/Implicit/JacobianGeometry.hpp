#pragma once

#include "DNDS/Core/Defines.hpp"
#include <algorithm>

namespace DNDS::Cauchy::JacobianDetail
{
    /// Geometry-only setup traversal shared by the two evaluated snapshots.
    /// Preserve centered face-incidence order and repeated periodic-image contributions.
    /// Boundary/self contributions stay diagonal. Point construction belongs
    /// to the caller: flow5 retains a Roe sample ordinal; equation physics does
    /// not. No EOS, numerical cache, lease or MPI ownership belongs here.
    template <class Offsets, class Samples, class Columns, class Rows, class MakePoint>
    void BuildStencilRows(const Offsets &offsets, const Samples &samples, const Columns &columns,
                          Rows &rows, MakePoint &&makePoint)
    {
        rows.resize(columns.size());
        for (index iCell = 0; iCell < index(rows.size()); ++iCell)
        {
            auto &row = rows[iCell];
            const auto &neighbors = columns[iCell];
            row.resize(neighbors.size());
            for (index entry = offsets[iCell]; entry < offsets[iCell + 1]; ++entry)
            {
                const auto &sample = samples[entry];
                if (sample.other == UnInitIndex || sample.other == iCell)
                    continue;
                const index slot = sample.jacobianColumn;
                DNDS_check_throw_info(slot >= 0 && slot < index(neighbors.size()) && neighbors[slot] == sample.other,
                                      "evaluated Jacobian stencil has an invalid prepared column");
                row[slot].push_back(makePoint(sample, entry));
            }
        }
    }
}
