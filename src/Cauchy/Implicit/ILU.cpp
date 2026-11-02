#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Solver/NativeBlockLU.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct LinearProblemContext::ILUState
    {
        std::shared_ptr<Direct::SerialSymLUStructure> symbolic;
        std::unique_ptr<Direct::NativeBlockLU> factors;
        std::vector<index> forwardCore, backwardCore;
        uint64_t diagonalVersion = 0, offDiagonalVersion = 0;
        bool current = false;
    };
    void LinearProblemContext::PrepareILUStructure(int fill, bool coreFirst, const std::vector<index> &localParts)
    {
        ValidateMatrix();
        DNDS_check_throw_info(fill >= -1 && fill <= 5 && _space.Distributed(), "ILU structure needs a domain communicator and fill -1..5");
        auto prepared = std::make_shared<ILUState>();
        const index rows = index(_pattern->partition->owned);
        DNDS_check_throw_info(localParts.empty() || (localParts.size() >= 2 && localParts.front() == 0 && localParts.back() == rows &&
                                                     std::is_sorted(localParts.begin(), localParts.end())),
                              "invalid ILU local part boundaries");
        const auto parts = localParts.empty() ? std::vector<index>{0, rows} : localParts;
        std::vector<std::vector<index>> adjacency(rows);
        // Generic symbolic construction preserves supplied cross-part edges;
        // its independent-part numeric scans do not remove them. Build the
        // induced block-Jacobi graphs explicitly, before ordering or fill.
        for (index part = 0; part + 1 < index(parts.size()); ++part)
            for (index iCell = parts[part]; iCell < parts[part + 1]; ++iCell)
                for (auto column : _pattern->columns[iCell])
                    if (column >= parts[part] && column < parts[part + 1])
                    {
                        adjacency[iCell].push_back(column);
                        adjacency[column].push_back(iCell);
                    }
        for (auto &row : adjacency)
        {
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }
        prepared->symbolic = std::make_shared<Direct::SerialSymLUStructure>(MPIInfo(_space.Communicator()), rows);
        if (coreFirst)
            prepared->symbolic->SetCoreFirstOrdering(_pattern->core, parts);
        prepared->symbolic->ObtainSymmetricSymbolicFactorization(adjacency, parts, fill, false);
        if (fill >= 0)
        {
            prepared->forwardCore = prepared->symbolic->ReadySolveRows(_pattern->core, true);
            prepared->backwardCore = prepared->symbolic->ReadySolveRows(prepared->forwardCore, false);
        }
        prepared->factors = std::make_unique<Direct::NativeBlockLU>(prepared->symbolic, _components);
        _ilu = std::move(prepared);
    }
    void LinearProblemContext::FactorizeILU()
    {
        ValidateMatrix();
        DNDS_check_throw_info(_ilu, "ILU structure is not prepared");
        _ilu->current = false;
        const auto diagonal = _diagonal.Read();
        const auto offDiagonal = _offDiagonal.Read();
        const auto d = Algebra::MakeEigenRowMatrices(diagonal.View<R::TableRows>(), _components,
                                                     _diagonalStorage == DiagonalStorage::Block ? _components : 1);
        const auto blocks = offDiagonal.View();
        {
            auto write = _ilu->factors->Matrix().Write();
            const auto matrix = write.View();
            for (index iCell = 0; iCell < index(_pattern->partition->owned); ++iCell)
            {
                if (_diagonalStorage == DiagonalStorage::Block)
                    matrix(iCell, 0) = d[iCell];
                else
                    matrix(iCell, 0) = d[iCell].col(0).asDiagonal();
                const auto copy = [&](const std::vector<index> &columns, index first)
                {
                    for (index j = 0; j < index(columns.size()); ++j)
                    {
                        auto block = matrix(iCell, first + j);
                        block.setZero();
                        const auto &source = _pattern->columns[iCell];
                        const auto found = std::find(source.begin(), source.end(), columns[j]);
                        if (found != source.end())
                            block = blocks(iCell, found - source.begin());
                    }
                };
                copy(_ilu->symbolic->lowerTriStructure[iCell], 1);
                copy(_ilu->symbolic->upperTriStructure[iCell], 1 + index(_ilu->symbolic->lowerTriStructure[iCell].size()));
            }
        }
        _ilu->factors->Factorize();
        _ilu->diagonalVersion = _diagonal.Version();
        _ilu->offDiagonalVersion = _offDiagonal.Values().Version();
        _ilu->current = true;
    }
    bool LinearProblemContext::ILUFactorsCurrent() const
    {
        try
        {
            ValidateMatrix();
            const auto diagonal = _diagonal.Read();
            const auto offDiagonal = _offDiagonal.Read();
            return _ilu && _ilu->current && _ilu->diagonalVersion == _diagonal.Version() &&
                   _ilu->offDiagonalVersion == _offDiagonal.Values().Version() && _ilu->factors->FactorsCurrent();
        }
        catch (const std::exception &)
        {
            return false;
        }
    }
    const std::vector<index> &LinearProblemContext::ILUForwardCore() const
    {
        DNDS_check_throw_info(_ilu, "ILU structure is not prepared");
        return _ilu->forwardCore;
    }
    const std::vector<index> &LinearProblemContext::ILUBackwardCore() const
    {
        DNDS_check_throw_info(_ilu, "ILU structure is not prepared");
        return _ilu->backwardCore;
    }
    void LinearProblemContext::ApplyILU(const Field &rhs, Field &forward, Field &output) const
    {
        ValidateMatrix();
        ValidateOutputs(rhs, forward);
        ValidateOutputs(rhs, output);
        ValidateOutputs(forward, output);
        const auto diagonal = _diagonal.Read();
        const auto offDiagonal = _offDiagonal.Read();
        DNDS_check_throw_info(_ilu && _ilu->current && _ilu->diagonalVersion == _diagonal.Version() &&
                                  _ilu->offDiagonalVersion == _offDiagonal.Values().Version(),
                              "ILU factors are missing or stale against the operator");
        _ilu->factors->Solve(rhs.Owned(), forward.Owned(), output.Owned());
    }
}
