#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Cauchy/Implicit/LinearBlockKernels.hpp"
#include "Solver/BlockInverse.hpp"
#include <limits>
#include <optional>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        index DiagonalWidth(index components, DiagonalStorage mode)
        {
            DNDS_check_throw_info(components > 0 && components <= std::numeric_limits<rowsize>::max() / components,
                                  "invalid linear block dimension");
            DNDS_check_throw_info(mode == DiagonalStorage::Entries || mode == DiagonalStorage::Block, "invalid linear diagonal mode");
            return mode == DiagonalStorage::Block ? components * components : components;
        }
        std::vector<index> Counts(const std::vector<std::vector<index>> &columns)
        {
            std::vector<index> result;
            result.reserve(columns.size());
            for (const auto &row : columns)
                result.push_back(index(row.size()));
            return result;
        }

    }
    std::shared_ptr<const LinearProblemContext::Pattern> LinearProblemContext::BuildPattern(
        const Algebra::DofSpace &space, std::vector<std::vector<index>> columns, std::vector<index> core)
    {
        DNDS_check_throw_info(space.OwnedRows() <= size_t(INT64_MAX) && space.GhostRows() <= size_t(INT64_MAX) - space.OwnedRows(),
                              "linear row space exceeds signed index range");
        auto data = std::make_shared<Pattern>(space.Pin(), std::move(columns));
        const index rows = index(data->partition->owned), allRows = rows + index(data->partition->ghost);
        DNDS_check_throw_info(index(data->columns.size()) == rows, "linear stencil row count differs from its DOF space");
        std::vector<unsigned char> isCore(rows, 0);
        for (auto iCell : core)
        {
            DNDS_check_throw_info(iCell >= 0 && iCell < rows && !isCore[iCell], "invalid or duplicate linear core row");
            isCore[iCell] = 1;
        }
        for (index iCell = 0; iCell < rows; ++iCell)
        {
            auto sorted = data->columns[iCell];
            std::sort(sorted.begin(), sorted.end());
            DNDS_check_throw_info(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "duplicate linear stencil column");
            for (auto column : sorted)
            {
                DNDS_check_throw_info(column >= 0 && column < allRows && column != iCell && (!isCore[iCell] || column < rows),
                                      "invalid linear stencil column or core ghost dependency");
                data->hasGhostColumns |= column >= rows;
            }
            if (!isCore[iCell])
                data->dependent.push_back(iCell);
        }
        data->core = std::move(core);
        data->order = data->core;
        data->order.insert(data->order.end(), data->dependent.begin(), data->dependent.end());
        data->position.resize(rows);
        for (index i = 0; i < rows; ++i)
            data->position[data->order[i]] = i;
        return data;
    }
    LinearProblemContext::LinearProblemContext(Algebra::DofSpace space, index components,
                                               std::vector<std::vector<index>> columns,
                                               std::vector<index> core, DiagonalStorage diagonal, OffDiagonalStorage offDiagonal)
        : _space(std::move(space)), _pattern(BuildPattern(_space, std::move(columns), std::move(core))), _components(components), _diagonalStorage(diagonal),
          _diagonal(R::RowLayout::Dense(_pattern->partition->owned, DiagonalWidth(components, diagonal))),
          _inverse(_diagonal.Layout()), _offDiagonalStorage(offDiagonal)
    {
        DNDS_check_throw_info(offDiagonal == OffDiagonalStorage::Assembled || offDiagonal == OffDiagonalStorage::SemiMatrixFree,
                              "invalid off-diagonal storage mode");
        if (offDiagonal == OffDiagonalStorage::Assembled)
            _offDiagonal.emplace(Counts(_pattern->columns), components);
        _diagonalStoragePin = _diagonal.SharedStorage();
        _diagonalShape = _diagonal.ShapeVersion();
        if (_offDiagonal)
            _offDiagonalStoragePin = _offDiagonal->Values().SharedStorage();
    }
    void LinearProblemContext::ValidateMatrix() const
    {
        Layout();
        DNDS_check_throw_info(_diagonal.SharedStorage() == _diagonalStoragePin && _diagonal.ShapeVersion() == _diagonalShape &&
                                  (!_offDiagonal || _offDiagonal->Values().SharedStorage() == _offDiagonalStoragePin),
                              "linear matrix allocation or shape changed");
    }
    void LinearProblemContext::ValidateField(const Field &field) const
    {
        Layout();
        field.Validate();
        DNDS_check_throw_info(field.Space().Same(_space) && index(field.Width()) == _components && field.Device() == R::Device::Host(),
                              "linear field space, component count or backend differs");
    }
    void LinearProblemContext::ValidateOutputs(const Field &input, const Field &output) const
    {
        ValidateField(input);
        ValidateField(output);
        for (const auto *source : {&input.Owned(), &input.Ghost(), &_diagonal, &_inverse})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(output.Owned().Storage(), source->Storage()), "linear output aliases input or matrix storage");
        if (_offDiagonal)
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(output.Owned().Storage(), _offDiagonal->Values().Storage()), "linear output aliases matrix storage");
    }
    Algebra::BlockMatrixStorage &LinearProblemContext::OffDiagonal()
    {
        DNDS_check_throw_info(_offDiagonal, "semi-matrix-free context has no assembled off-diagonal storage");
        return *_offDiagonal;
    }
    const Algebra::BlockMatrixStorage &LinearProblemContext::OffDiagonal() const
    {
        DNDS_check_throw_info(_offDiagonal, "semi-matrix-free context has no assembled off-diagonal storage");
        return *_offDiagonal;
    }
    void LinearProblemContext::PublishOffDiagonalEvaluator(std::shared_ptr<const OffDiagonalEvaluator> evaluator)
    {
        ValidateMatrix();
        DNDS_check_throw_info(!_offDiagonal && evaluator && evaluator->Components() == _components &&
                                  evaluator->RowSpaceIdentity() == _pattern->partition && evaluator->Columns() == _pattern->columns &&
                                  _offDiagonalGeneration < UINT64_MAX,
                              "invalid evaluated off-diagonal publication");
        auto publication = _diagonal.Write();
        _offDiagonalEvaluator = std::move(evaluator);
        ++_offDiagonalGeneration;
        publication.CancelUnmodified();
    }
    uint64_t LinearProblemContext::OffDiagonalGeneration() const
    {
        return _offDiagonal ? _offDiagonal->Values().Version() : _offDiagonalGeneration;
    }
    LinearProblemContext::OffDiagonalReadAccess LinearProblemContext::ReadOffDiagonal() const
    {
        ValidateMatrix();
        if (_offDiagonal)
            return {_offDiagonal->Read(), _components};
        auto publication = _diagonal.Read();
        DNDS_check_throw_info(_offDiagonalEvaluator, "semi-matrix-free off-diagonal data have not been prepared");
        return {std::move(publication), _offDiagonalEvaluator, _components};
    }
    void LinearProblemContext::InvertDiagonal()
    {
        _inverseCurrent = false;
        _inverseFallbacks = 0;
        ValidateMatrix();
        DNDS_check_throw_info(_diagonal.GetDevice() == R::Device::Host() && _diagonal.Layout() == _inverse.Layout(), "linear diagonal shape/backend changed");
        const auto read = _diagonal.Read();
        std::optional<R::Array<real>::WriteLease> write(_inverse.Write());
        const auto input = read.View<R::TableRows>();
        const auto output = write->View<R::TableRows>();
        const index width = DiagonalWidth(_components, _diagonalStorage);
        DNDS_check_throw_info(input.RowCount() == index(_pattern->partition->owned) && input.Width() == width, "linear diagonal shape changed");
        if (_diagonalStorage == DiagonalStorage::Entries)
            for (index iCell = 0; iCell < input.RowCount(); ++iCell)
                for (index j = 0; j < _components; ++j)
                {
                    DNDS_check_throw_info(std::isfinite(input(iCell, j)) && input(iCell, j) != 0, "nonfinite or zero linear diagonal entry");
                    output(iCell, j) = 1 / input(iCell, j);
                }
        else
        {
            const auto matrices = Algebra::MakeEigenRowMatrices(input, _components, _components);
            const auto inverses = Algebra::MakeEigenRowMatrices(output, _components, _components);
            for (index iCell = 0; iCell < input.RowCount(); ++iCell)
            {
                auto result = Direct::InvertDiagonalBlock(matrices[iCell]);
                inverses[iCell] = result.value;
                _inverseFallbacks += result.usedSVD;
            }
        }
        _diagonalVersion = _diagonal.Version();
        // Publish before recording the inverse generation.
        write.reset();
        _inverseVersion = _inverse.Version();
        _inverseCurrent = true;
    }
    void LinearProblemContext::Apply(const Field &input, Field &output, LinearRegion region) const
    {
        ValidateOutputs(input, output);
        ValidateMatrix();
        DNDS_check_throw_info(region == LinearRegion::All || region == LinearRegion::Core || region == LinearRegion::HaloDependent, "invalid linear operator region");
        const auto owned = input.Owned().Read();
        std::optional<R::Array<real>::ReadLease> ghost;
        if (region != LinearRegion::Core && _pattern->hasGhostColumns)
            ghost.emplace(input.Ghost().Read());
        const auto diagonal = _diagonal.Read();
        const auto offDiagonal = ReadOffDiagonal();
        auto destination = output.Owned().Write();
        const auto x = Algebra::MakeEigenRowMatrices(owned.View<R::UniformTableRows>(), _components, 1);
        const auto y = Algebra::MakeEigenRowMatrices(destination.View<R::UniformTableRows>(), _components, 1);
        const auto d = diagonal.View<R::TableRows>();
        const auto matrix = Algebra::MakeEigenRowMatrices(d, _components, _diagonalStorage == DiagonalStorage::Block ? _components : 1);
        const auto remote = ghost ? ghost->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{};
        const auto xGhost = Algebra::MakeEigenRowMatrices(remote, _components, 1);
        offDiagonal.WithBlocks([&](const auto &blocks)
                               {
            const auto &rows = region == LinearRegion::All ? _pattern->order : region == LinearRegion::Core ? _pattern->core : _pattern->dependent;
            if (_components == 5 && _diagonalStorage == DiagonalStorage::Block)
                Detail::ApplyBlockRows<5>(Algebra::MakeEigenRowMatrices<5, 1>(owned.View<R::UniformTableRows>(), 5, 1),
                                          Algebra::MakeEigenRowMatrices<5, 1>(remote, 5, 1),
                                          Algebra::MakeEigenRowMatrices<5, 1>(destination.View<R::UniformTableRows>(), 5, 1),
                                          Algebra::MakeEigenRowMatrices<5, 5>(d, 5, 5), blocks, *_pattern, rows, 5, true);
            else
                Detail::ApplyBlockRows(x, xGhost, y, matrix, blocks, *_pattern, rows, _components, _diagonalStorage == DiagonalStorage::Block); });
    }
    void LinearProblemContext::ApplySGS(const Field &rhs, const Field &input, Field &forward, Field &output) const
    {
        auto iteration = BeginSGS(rhs, input, forward, output);
        iteration.Lower();
        iteration.Upper();
        iteration.Finish();
    }
}
