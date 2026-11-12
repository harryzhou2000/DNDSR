#include "Cauchy/Implicit/CPU/LinearProblemKernels.hpp"
#include "DNDS/Algebra/EigenView.hpp"
#include "Solver/BlockInverse.hpp"

namespace DNDS::Cauchy::CPU
{
    NumericalResult<index> InvertLinearDiagonal(const DiagonalInverseRequest &request) noexcept
    {
        DNDS_assert(request.components > 0 &&
                    request.diagonal.RowCount() == request.inverse.RowCount());
        index fallbacks = 0;
        if (!request.fullBlock)
        {
            for (index iCell = 0; iCell < request.diagonal.RowCount(); ++iCell)
                for (index j = 0; j < request.components; ++j)
                {
                    const real entry = request.diagonal(iCell, j);
                    if (!(std::isfinite(entry) && entry != 0))
                        return NumericalStatus{NumericalError::InvalidState, 1, iCell};
                    request.inverse(iCell, j) = 1 / entry;
                }
            return fallbacks;
        }
        const auto invert = [&](auto width) -> NumericalStatus
        {
            constexpr int Size = decltype(width)::value;
            const auto input = Algebra::MakeEigenRowMatrices<Size, Size>(
                request.diagonal, request.components, request.components);
            const auto output = Algebra::MakeEigenRowMatrices<Size, Size>(
                request.inverse, request.components, request.components);
            for (index iCell = 0; iCell < request.diagonal.RowCount(); ++iCell)
            {
                if (!input[iCell].allFinite())
                    return NumericalStatus{NumericalError::InvalidState, 2, iCell};
                auto inverseResult = Direct::TryInvertDiagonalBlock(input[iCell]);
                if (!inverseResult)
                    return inverseResult.status.At(iCell);
                const auto &result = inverseResult.value;
                if (!result.value.allFinite())
                    return NumericalStatus{NumericalError::InvalidState, 3, iCell};
                output[iCell] = result.value;
                fallbacks += result.usedSVD;
            }
            return {};
        };
        NumericalStatus status;
        if (request.components == 5)
            status = invert(std::integral_constant<int, 5>{});
        else if (request.components == 6)
            status = invert(std::integral_constant<int, 6>{});
        else if (request.components == 7)
            status = invert(std::integral_constant<int, 7>{});
        else
            status = invert(std::integral_constant<int, Eigen::Dynamic>{});
        if (!status)
            return status;
        return fallbacks;
    }
}
