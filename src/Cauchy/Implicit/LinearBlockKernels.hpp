#pragma once

#include "DNDS/Algebra/EigenView.hpp"

namespace DNDS::Cauchy::Detail
{
    /// One ordered SGS scan. Input remote values are frozen by the enclosing
    /// phase scope. Lower local predecessors read forward; upper successors
    /// read output. No ownership/communication or per-row allocation here.
    template <bool Upper, int Size, bool ZeroInput = false, class RHS, class Old, class Forward, class Output, class Ghost, class Inverse, class Blocks, class Pattern>
    void ScanSGSRows(RHS rhs, Old old, Forward forward, Output output, Ghost ghost, Inverse inverse, const Blocks &blocks,
                     const Pattern &pattern, const std::vector<index> &rows, index components, bool fullDiagonal)
    {
        constexpr bool scratch = std::is_same_v<Output, Algebra::DofComponentView<real>>;
        if constexpr (scratch && std::is_same_v<RHS, Algebra::DofComponentView<const real>> &&
                      std::is_same_v<Old, Algebra::DofComponentView<const real>> &&
                      std::is_same_v<Ghost, Algebra::DofComponentView<const real>>)
        {
            // Resolve the common coupled AoS case once, under the enclosing
            // logical-state leases. Split, permuted and column-major fields
            // keep the generic row-gather kernel below. The SGS scan order and
            // its frozen remote inputs are unchanged.
            const auto source = Algebra::TryOwnedTable(rhs), previous = Algebra::TryOwnedTable(old);
            const auto temporary = Algebra::TryOwnedTable(forward);
            const auto destination = Algebra::TryOwnedTable(output);
            const auto remote = ghost.RowCount() ? Algebra::TryGhostTable(ghost)
                                                 : std::optional<Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>>{
                                                       Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>{nullptr, nullptr, 0, components, components}};
            if (source && previous && temporary && destination && remote)
            {
                const auto scan = [&](auto width)
                {
                    constexpr int N = decltype(width)::value;
                    ScanSGSRows<Upper, N, ZeroInput>(Algebra::MakeEigenRowMatrices<N, 1>(*source, components, 1),
                                                     Algebra::MakeEigenRowMatrices<N, 1>(*previous, components, 1),
                                                     Algebra::MakeEigenRowMatrices<N, 1>(*temporary, components, 1),
                                                     Algebra::MakeEigenRowMatrices<N, 1>(*destination, components, 1),
                                                     Algebra::MakeEigenRowMatrices<N, 1>(*remote, components, 1),
                                                     inverse, blocks, pattern, rows, components, fullDiagonal);
                };
                if (components == 5)
                    scan(std::integral_constant<int, 5>{});
                else if (components == 6)
                    scan(std::integral_constant<int, 6>{});
                else if (components == 7)
                    scan(std::integral_constant<int, 7>{});
                else
                    scan(std::integral_constant<int, Eigen::Dynamic>{});
                return;
            }
        }
        Eigen::VectorXd otherScratch(scratch ? components : 0), outputScratch(scratch ? components : 0);
        const auto readRow = [&](auto input, index iRow)
        {
            if constexpr (scratch)
                return Algebra::PackEigenRow(input, iRow, otherScratch);
            else
                return input[iRow]; // Explicit dense-view specialization only.
        };
        Eigen::Matrix<real, Size, 1> value(components);
        Eigen::Matrix<real, Blocks::needs_scratch ? Size : 0, 1> product(Blocks::needs_scratch ? components : 0);
        const index ownedRows = index(pattern.partition->owned);
        for (index scan = 0; scan < index(rows.size()); ++scan)
        {
            const index iCell = rows[Upper ? index(rows.size()) - 1 - scan : scan];
            value = readRow(rhs, iCell);
            const auto &columns = pattern.columns[iCell];
            for (index j = 0; j < index(columns.size()); ++j)
            {
                const auto column = columns[j];
                if constexpr (ZeroInput)
                {
                    // Dedicated kernel: old data is absent, not stored zeros.
                    // Upper remote values are analytic zeros in local SGS;
                    // a middle-exchange upper scan uses the ordinary kernel.
                    if (column >= ownedRows)
                        continue;
                    if constexpr (!Upper)
                        if (pattern.position[column] >= pattern.position[iCell])
                            continue;
                }
                const auto subtract = [&](const auto &input)
                {
                    blocks.template Accumulate<Size, true>(iCell, j, input, value, product);
                };
                if (column >= ownedRows)
                    subtract(readRow(ghost, column - ownedRows));
                else if (pattern.position[column] < pattern.position[iCell])
                    subtract(readRow(forward, column));
                else if constexpr (Upper)
                    subtract(readRow(output, column));
                else if constexpr (!ZeroInput)
                    subtract(readRow(old, column));
            }
            auto result = [&]
            {
                if constexpr (scratch)
                    return Eigen::Ref<Eigen::VectorXd>(outputScratch);
                else
                    return output[iCell];
            }();
            if (fullDiagonal)
            {
                if constexpr (Size != Eigen::Dynamic)
                    result.noalias() = inverse[iCell].template topLeftCorner<Size, Size>() * value;
                else
                    result.noalias() = inverse[iCell] * value;
            }
            else
                result.array() = inverse[iCell].array() * value.array();
            if constexpr (scratch)
                Algebra::UnpackEigenRow(output, iCell, result);
        }
    }

    /// Matrix action shared by physical-field maps and logical component views.
    /// All ownership, matrix/field validation and row-dependency checks belong
    /// to the enclosing access scope. The field path keeps direct Eigen maps;
    /// only separate logical fields need one-row scratch, allocated before loops.
    template <int Size = Eigen::Dynamic, class Owned, class Ghost, class Output, class Diagonal, class Blocks, class Pattern>
    void ApplyBlockRows(Owned owned, Ghost ghost, Output output, Diagonal diagonal, const Blocks &blocks,
                        const Pattern &pattern, const std::vector<index> &rows, index components, bool fullDiagonal)
    {
        constexpr bool scratch = std::is_same_v<Output, Algebra::DofComponentView<real>>;
        if constexpr (scratch && std::is_same_v<Owned, Algebra::DofComponentView<const real>> &&
                      std::is_same_v<Ghost, Algebra::DofComponentView<const real>>)
        {
            const auto source = Algebra::TryOwnedTable(owned);
            const auto destination = Algebra::TryOwnedTable(output);
            auto remote = ghost.RowCount() ? Algebra::TryGhostTable(ghost)
                                           : std::optional<Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>>{
                                                 Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>{nullptr, nullptr, 0, components, components}};
            if (source && destination && remote)
            {
                const auto apply = [&](auto width)
                {
                    constexpr int N = decltype(width)::value;
                    ApplyBlockRows<N>(Algebra::MakeEigenRowMatrices<N, 1>(*source, components, 1),
                                      Algebra::MakeEigenRowMatrices<N, 1>(*remote, components, 1),
                                      Algebra::MakeEigenRowMatrices<N, 1>(*destination, components, 1),
                                      diagonal, blocks, pattern, rows, components, fullDiagonal);
                };
                if (components == 5)
                    apply(std::integral_constant<int, 5>{});
                else if (components == 6)
                    apply(std::integral_constant<int, 6>{});
                else if (components == 7)
                    apply(std::integral_constant<int, 7>{});
                else
                    apply(std::integral_constant<int, Eigen::Dynamic>{});
                return;
            }
        }
        Eigen::VectorXd ownScratch(scratch ? components : 0), otherScratch(scratch ? components : 0), outputScratch(scratch ? components : 0);
        const auto readRow = [&](auto input, index iRow, auto &storage)
        {
            if constexpr (scratch)
                return Algebra::PackEigenRow(input, iRow, storage);
            else
                return input[iRow];
        };
        Eigen::Matrix<real, Blocks::needs_scratch ? Size : 0, 1> product(Blocks::needs_scratch ? components : 0);
        const index ownedRows = index(pattern.partition->owned);
        for (const auto iCell : rows)
        {
            auto result = [&]
            {
                if constexpr (scratch)
                    return Eigen::Ref<Eigen::VectorXd>(outputScratch);
                else
                    return output[iCell];
            }();
            const auto value = readRow(owned, iCell, ownScratch);
            if (fullDiagonal)
            {
                if constexpr (Size != Eigen::Dynamic)
                    result.noalias() = diagonal[iCell].template topLeftCorner<Size, Size>() * value;
                else
                    result.noalias() = diagonal[iCell] * value;
            }
            else
            {
                if constexpr (Size != Eigen::Dynamic)
                    result.array() = diagonal[iCell].col(0).template head<Size>().array() * value.array();
                else
                    result.array() = diagonal[iCell].col(0).array() * value.array();
            }
            for (index j = 0; j < index(pattern.columns[iCell].size()); ++j)
            {
                const index column = pattern.columns[iCell][j];
                if (column < ownedRows)
                    blocks.template Accumulate<Size, false>(iCell, j, readRow(owned, column, otherScratch), result, product);
                else
                    blocks.template Accumulate<Size, false>(iCell, j, readRow(ghost, column - ownedRows, otherScratch), result, product);
            }
            if constexpr (scratch)
                Algebra::UnpackEigenRow(output, iCell, result);
        }
    }
}
