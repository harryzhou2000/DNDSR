#include "Cauchy/Implicit/CPU/LinearBlockKernels.hpp"
#include "Cauchy/Implicit/CPU/LinearBlockKernels_Impl.hpp"
namespace DNDS::Cauchy::CPU
{
    void FillLinearFactor(LinearDiagonalRows diagonal, OffDiagonalView blocks, Algebra::BlockMatrixView<real> factor,
                          Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::CSRRows> sourceBlocks,
                          index components, bool fullDiagonal)
    {
        const auto d = Algebra::MakeEigenRowMatrices(diagonal, components, fullDiagonal ? components : 1);
        blocks.WithBlocks([&](const auto &values)
                          { FillILUBlocks(d, values, factor, sourceBlocks, fullDiagonal); });
    }
}
