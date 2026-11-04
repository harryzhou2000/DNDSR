#pragma once

#include "DNDS/Algebra/EigenView.hpp"

namespace DNDS::Cauchy::Detail
{
    /// One ordered SGS scan. Input remote values are frozen by the enclosing
    /// phase scope. Lower local predecessors read forward; upper successors
    /// read output. No ownership/communication or per-row allocation here.
    template <bool Upper, int Size, class RHS, class Old, class Forward, class Output, class Ghost, class Inverse, class Blocks, class Pattern>
    void ScanSGSRows(RHS rhs, Old old, Forward forward, Output output, Ghost ghost, Inverse inverse, const Blocks &blocks,
                     const Pattern &pattern, const std::vector<index> &rows, index components, bool fullDiagonal)
    {
        constexpr bool scratch = std::is_same_v<Output, Algebra::DofComponentView<real>>;
        Eigen::VectorXd otherScratch(scratch ? components : 0), outputScratch(scratch ? components : 0);
        Eigen::Matrix<real, Size, 1> value(components);
        const index ownedRows = index(pattern.partition->owned);
        for (index scan = 0; scan < index(rows.size()); ++scan)
        {
            const index iCell = rows[Upper ? index(rows.size()) - 1 - scan : scan];
            value = Algebra::ReadEigenRow(rhs, iCell, otherScratch);
            const auto &columns = pattern.columns[iCell];
            for (index j = 0; j < index(columns.size()); ++j)
            {
                const auto column = columns[j];
                if (column >= ownedRows)
                    value.noalias() -= blocks(iCell, j) * Algebra::ReadEigenRow(ghost, column - ownedRows, otherScratch);
                else if (pattern.position[column] < pattern.position[iCell])
                    value.noalias() -= blocks(iCell, j) * Algebra::ReadEigenRow(forward, column, otherScratch);
                else if constexpr (Upper)
                    value.noalias() -= blocks(iCell, j) * Algebra::ReadEigenRow(output, column, otherScratch);
                else
                    value.noalias() -= blocks(iCell, j) * Algebra::ReadEigenRow(old, column, otherScratch);
            }
            auto result = Algebra::WriteEigenRow(output, iCell, outputScratch);
            if (fullDiagonal)
                result.noalias() = inverse[iCell] * value;
            else
                result.array() = inverse[iCell].array() * value.array();
            Algebra::PublishEigenRow(output, iCell, result);
        }
    }

    /// Matrix action shared by physical-field maps and logical component views.
    /// All ownership, matrix/field validation and row-dependency checks belong
    /// to the enclosing access scope. The field path keeps direct Eigen maps;
    /// only separate logical fields need one-row scratch, allocated before loops.
    template <class Owned, class Ghost, class Output, class Diagonal, class Blocks, class Pattern>
    void ApplyBlockRows(Owned owned, Ghost ghost, Output output, Diagonal diagonal, const Blocks &blocks,
                        const Pattern &pattern, const std::vector<index> &rows, index components, bool fullDiagonal)
    {
        constexpr bool scratch = std::is_same_v<Output, Algebra::DofComponentView<real>>;
        Eigen::VectorXd ownScratch(scratch ? components : 0), otherScratch(scratch ? components : 0), outputScratch(scratch ? components : 0);
        const index ownedRows = index(pattern.partition->owned);
        for (const auto iCell : rows)
        {
            auto result = Algebra::WriteEigenRow(output, iCell, outputScratch);
            const auto value = Algebra::ReadEigenRow(owned, iCell, ownScratch);
            if (fullDiagonal)
                result.noalias() = diagonal[iCell] * value;
            else
                result.array() = diagonal[iCell].array() * value.array();
            for (index j = 0; j < index(pattern.columns[iCell].size()); ++j)
            {
                const index column = pattern.columns[iCell][j];
                if (column < ownedRows)
                    result.noalias() += blocks(iCell, j) * Algebra::ReadEigenRow(owned, column, otherScratch);
                else
                    result.noalias() += blocks(iCell, j) * Algebra::ReadEigenRow(ghost, column - ownedRows, otherScratch);
            }
            Algebra::PublishEigenRow(output, iCell, result);
        }
    }
}
