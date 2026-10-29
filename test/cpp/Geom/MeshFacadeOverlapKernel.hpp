#pragma once

#include "Geom/Mesh/MeshView.hpp"
#include <cmath>

namespace DNDS::Geom::OverlapTest
{
    namespace R = DNDS::Runtime;

    struct OwnedSolution
    {
        R::ArrayView<const real> owned;
        DNDS_DEVICE_CALLABLE real operator[](index cell) const
        {
            DNDS_HD_assert(cell >= 0 && size_t(cell) < owned.RowCount());
            return owned(size_t(cell), 0);
        }
    };
    struct SplitSolution
    {
        R::ArrayView<const real> owned, ghost;
        DNDS_DEVICE_CALLABLE real operator[](index cell) const
        {
            return size_t(cell) < owned.RowCount() ? owned(size_t(cell), 0) : ghost(size_t(cell) - owned.RowCount(), 0);
        }
    };

    template <class Solution>
    DNDS_DEVICE_CALLABLE real NeighborSum(UnstructuredMeshView mesh, index cell, Solution solution)
    {
        real result = 0;
        auto faces = mesh.CellFaces(cell);
        for (size_t slot = 0; slot < faces.size(); ++slot)
        {
            const auto face = faces[slot];
            const auto other = mesh.CellFaceOther(cell, face, rowsize(slot));
            if (other == UnInitIndex)
                continue;
            MeshQueryCoords points;
            mesh.GetCoordsOnFace(face, points);
            DNDS_HD_assert(points.cols() == 2);
            real squared = 0;
            for (int d = 0; d < 3; ++d)
            {
                const auto delta = points(d, 1) - points(d, 0);
                squared += delta * delta;
            }
            result += ::sqrt(squared) * (solution[other] - solution[cell]);
        }
        return result;
    }

    inline void CoreHost(UnstructuredMeshView mesh, R::ArrayView<const index> rows,
                         R::ArrayView<const real> owned, R::ArrayView<real> output)
    {
        for (size_t i = 0; i < rows.RowCount(); ++i)
            output(i, 0) = NeighborSum(mesh, rows(i, 0), OwnedSolution{owned});
    }
    inline void RingHost(UnstructuredMeshView mesh, R::ArrayView<const index> rows,
                         R::ArrayView<const real> owned, R::ArrayView<const real> ghost, R::ArrayView<real> output)
    {
        for (size_t i = 0; i < rows.RowCount(); ++i)
            output(i, 0) = NeighborSum(mesh, rows(i, 0), SplitSolution{owned, ghost});
    }

#ifdef DNDS_USE_CUDA
    cudaError_t PreloadOverlapKernels();
    cudaError_t LaunchCore(UnstructuredMeshView mesh, R::ArrayView<const index> rows,
                           R::ArrayView<const real> owned, R::ArrayView<real> output, cudaStream_t stream);
    cudaError_t LaunchRing(UnstructuredMeshView mesh, R::ArrayView<const index> rows,
                           R::ArrayView<const real> owned, R::ArrayView<const real> ghost,
                           R::ArrayView<real> output, cudaStream_t stream);
#endif
}
