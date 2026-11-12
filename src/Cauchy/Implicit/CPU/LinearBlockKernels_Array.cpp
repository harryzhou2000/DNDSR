#include "Cauchy/Implicit/CPU/LinearBlockKernels.hpp"
#include "Cauchy/Implicit/CPU/LinearBlockKernels_Impl.hpp"

namespace DNDS::Cauchy::CPU
{
    void ApplyLinearRows(const LinearBlockInput &request, LinearReadRows owned, LinearReadRows ghost, LinearWriteRows output)
    {
        const auto diagonal = Algebra::MakeEigenRowMatrices(request.diagonal, request.components, request.fullDiagonal ? request.components : 1);
        request.blocks.WithActions([&](const auto &blocks)
                                   {
            if (request.components == 5 && request.fullDiagonal)
                ApplyBlockRows<5>(Algebra::MakeEigenRowMatrices<5, 1>(owned, 5, 1),
                    Algebra::MakeEigenRowMatrices<5, 1>(ghost, 5, 1), Algebra::MakeEigenRowMatrices<5, 1>(output, 5, 1),
                    Algebra::MakeEigenRowMatrices<5, 5>(request.diagonal, 5, 5), blocks, request.pattern, request.rows, 5, true);
            else
                ApplyBlockRows(Algebra::MakeEigenRowMatrices(owned, request.components, 1), Algebra::MakeEigenRowMatrices(ghost, request.components, 1), Algebra::MakeEigenRowMatrices(output, request.components, 1), diagonal,
                blocks, request.pattern, request.rows, request.components, request.fullDiagonal); });
    }
    void JacobiLinearRows(const LinearBlockInput &request, LinearReadRows rhs, LinearReadRows owned, LinearReadRows ghost, LinearWriteRows output, bool zeroStart)
    {
        const auto inverse = Algebra::MakeEigenRowMatrices(request.diagonal, request.components, request.fullDiagonal ? request.components : 1);
        request.blocks.WithActions([&](const auto &blocks)
                                   {
            const auto invoke = [&](auto zero)
            {
                JacobiRows<decltype(zero)::value>(Algebra::MakeEigenRowMatrices(rhs, request.components, 1), Algebra::MakeEigenRowMatrices(owned, request.components, 1), Algebra::MakeEigenRowMatrices(ghost, request.components, 1), Algebra::MakeEigenRowMatrices(output, request.components, 1),
                    inverse, blocks, request.pattern, request.rows, request.components, request.fullDiagonal);
            };
            if (zeroStart) invoke(std::true_type{}); else invoke(std::false_type{}); });
    }
    void LowerSGSRows(const LinearBlockInput &request, LinearReadRows rhs, LinearReadRows old, LinearReadRows ghost, LinearWriteRows output, bool zeroStart)
    {
        const auto inverse = Algebra::MakeEigenRowMatrices(request.diagonal, request.components, request.fullDiagonal ? request.components : 1);
        request.blocks.WithActions([&](const auto &blocks)
                                   {
            const auto invoke = [&](auto zero)
            {
                constexpr bool Zero = decltype(zero)::value;
                if (request.components == 5)
                    ScanSGSRows<false, 5, Zero>(Algebra::MakeEigenRowMatrices(rhs, request.components, 1), Algebra::MakeEigenRowMatrices(old, request.components, 1), Algebra::MakeEigenRowMatrices(output, request.components, 1), Algebra::MakeEigenRowMatrices(output, request.components, 1), Algebra::MakeEigenRowMatrices(ghost, request.components, 1), inverse,
                        blocks, request.pattern, request.rows, request.components, request.fullDiagonal);
                else
                    ScanSGSRows<false, Eigen::Dynamic, Zero>(Algebra::MakeEigenRowMatrices(rhs, request.components, 1), Algebra::MakeEigenRowMatrices(old, request.components, 1), Algebra::MakeEigenRowMatrices(output, request.components, 1), Algebra::MakeEigenRowMatrices(output, request.components, 1), Algebra::MakeEigenRowMatrices(ghost, request.components, 1), inverse,
                        blocks, request.pattern, request.rows, request.components, request.fullDiagonal);
            };
            if (zeroStart) invoke(std::true_type{}); else invoke(std::false_type{}); });
    }
    void UpperSGSRows(const LinearBlockInput &request, LinearReadRows rhs, LinearReadRows old, LinearReadRows forward, LinearReadRows ghost, LinearWriteRows output, bool zeroStart)
    {
        const auto inverse = Algebra::MakeEigenRowMatrices(request.diagonal, request.components, request.fullDiagonal ? request.components : 1);
        request.blocks.WithActions([&](const auto &blocks)
                                   {
            const auto invoke = [&](auto zero)
            {
                constexpr bool Zero = decltype(zero)::value;
                if (request.components == 5)
                    ScanSGSRows<true, 5, Zero>(Algebra::MakeEigenRowMatrices(rhs, request.components, 1), Algebra::MakeEigenRowMatrices(old, request.components, 1), Algebra::MakeEigenRowMatrices(forward, request.components, 1), Algebra::MakeEigenRowMatrices(output, request.components, 1), Algebra::MakeEigenRowMatrices(ghost, request.components, 1), inverse,
                        blocks, request.pattern, request.rows, request.components, request.fullDiagonal);
                else
                    ScanSGSRows<true, Eigen::Dynamic, Zero>(Algebra::MakeEigenRowMatrices(rhs, request.components, 1), Algebra::MakeEigenRowMatrices(old, request.components, 1), Algebra::MakeEigenRowMatrices(forward, request.components, 1), Algebra::MakeEigenRowMatrices(output, request.components, 1), Algebra::MakeEigenRowMatrices(ghost, request.components, 1), inverse,
                        blocks, request.pattern, request.rows, request.components, request.fullDiagonal);
            };
            if (zeroStart) invoke(std::true_type{}); else invoke(std::false_type{}); });
    }
}
