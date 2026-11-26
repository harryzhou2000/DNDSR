#pragma once

#include "DNDS/Core/CheckedSize.hpp"
#include "DNDS/Core/Defines.hpp"
#include "Geom/Quadratures/Quadrature.hpp"

#include <algorithm>
#include <vector>

namespace DNDS::Cauchy::Detail
{
    struct CellQuadratureLayout
    {
        std::vector<index> offsets;
        std::vector<int> preparedTypes;
        index maxPoints = 0;
    };

    /// Build checked exclusive per-cell quadrature point offsets while priming
    /// the shape-at-quadrature cache on the controller thread for every unique element type.
    template <class MeshView, class ExtraPointsFn>
    CellQuadratureLayout BuildCellQuadratureOffsets(index numCells, const MeshView &view, int order, ExtraPointsFn &&extraPoints)
    {
        CellQuadratureLayout layout;
        layout.offsets.assign(size_t(numCells) + 1, 0);
        for (index iCell = 0; iCell < numCells; ++iCell)
        {
            const auto element = view.GetCellElement(iCell);
            const Geom::Elem::Quadrature quadrature(element, order);
            const index volumePoints = quadrature.GetNumPoints();
            const index cellPoints = CheckedSize::Add<index>(volumePoints, index(extraPoints(iCell, element, quadrature)));
            layout.offsets[size_t(iCell) + 1] = CheckedSize::Add<index>(layout.offsets[size_t(iCell)], cellPoints);
            layout.maxPoints = std::max(layout.maxPoints, cellPoints);
            if (std::find(layout.preparedTypes.begin(), layout.preparedTypes.end(), int(element.type)) == layout.preparedTypes.end())
            {
                for (int iQ = 0; iQ < volumePoints; ++iQ)
                    (void)Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ);
                layout.preparedTypes.push_back(int(element.type));
            }
        }
        return layout;
    }

    template <class MeshView>
    CellQuadratureLayout BuildCellQuadratureOffsets(index numCells, const MeshView &view, int order)
    {
        return BuildCellQuadratureOffsets(numCells, view, order, [](index, const auto &, const auto &) -> index
                                          { return 0; });
    }
}
