#include "CFV/CPU/ReconstructionBasisKernels.hpp"
#include "CFV/Math/PeriodicFrame.hpp"

namespace DNDS::CFV::CPU
{
    Math::ReconstructionBasisView EvaluateFaceBasis(
        const ReconstructionBasisEvaluationView &input,
        index face, int side, index point, int derivativeOrder,
        Math::SmallMatrix &scratch)
    {
        const Math::TaylorBasis polynomial(input.dimension, input.degree);
        const int rows = polynomial.DerivativeNumber(derivativeOrder);
        if (input.cached.data && rows <= input.cachedDerivatives)
            return {input.cached[2 * point + side].data(), rows,
                    input.modes, input.cachedDerivatives};
        const index cell = input.mesh.FaceCells(face)[side];
        const auto position = Detail::FacePointInCell(
            input.mesh, face, side, input.geometry.GetFacePoint(point));
        polynomial.Reconstruction(
            scratch, position, input.geometry.GetCellCent(cell),
            input.geometry.cells(cell, 7), input.moments[cell], derivativeOrder);
        return {scratch.data(), int(scratch.rows()), input.modes,
                int(scratch.rows())};
    }
}
