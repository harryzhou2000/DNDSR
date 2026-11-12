#include "Cauchy/Discretization/CPU/FiniteVolumeReconstructionKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSampleKernels.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
namespace DNDS::Cauchy::CPU
{
    index PrepareFaceCharacteristicMatrices(FiniteVolumeStencilView stencil, Geom::UnstructuredMeshView mesh,
                                            CFV::FiniteVolumeHostView metric, RHSFieldView u, index firstComponent,
                                            Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> l,
                                            Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> r,
                                            const EquationSystem &equation)
    {
        index fallbacks = 0;
        const auto loadFlow = [&](index iCell)
        {
            FlowState value;
            for (int j = 0; j < 5; ++j)
                value[j] = u(iCell, firstComponent + j);
            return value;
        };
        for (index iAux = 0; iAux < stencil.stencilFaces.RowCount(); ++iAux)
        {
            const auto iFace = stencil.stencilFaces(iAux, 0);
            const auto cells = mesh.FaceCells(iFace);
            if (cells[0] < 0 || cells[0] >= metric.cells.RowCount())
                continue;
            const auto left = loadFlow(cells[0]);
            auto right = left;
            if (cells[1] >= 0 && cells[1] < metric.cells.RowCount())
                right = FiniteVolumeDetail::RotateState(loadFlow(cells[1]), CFV::Detail::OtherFrameInCell(mesh, iFace, 0));
            const auto normal = metric.GetFaceNorm(metric.FaceQuadratureBegin(iFace));
            const auto matrices = equation.PrepareCharacteristics(left, right, {normal[0], normal[1], normal[2]});
            fallbacks += matrices.componentwiseFallback;
            for (index j = 0; j < 25; ++j)
                l(iFace, j) = matrices.forward[j], r(iFace, j) = matrices.inverse[j];
        }
        return fallbacks;
    }
}
