#pragma once

#include "DNDS/Algebra/EigenView.hpp"

namespace DNDS::Cauchy::Detail
{
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
