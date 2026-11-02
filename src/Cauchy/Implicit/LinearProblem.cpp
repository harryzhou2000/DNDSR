#include "Cauchy/Implicit/LinearProblem.hpp"
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
        template <class Kernel>
        void DispatchVector(index components, Kernel &&kernel)
        {
            // The common flow5 temporary stays on the stack. The general
            // profile allocates its local vector once per scan, not per cell.
            if (components == 5)
                kernel(std::integral_constant<int, 5>{});
            else
                kernel(std::integral_constant<int, Eigen::Dynamic>{});
        }
    }
    LinearProblemContext::LinearProblemContext(Algebra::DofSpace space, index components,
                                               std::vector<std::vector<index>> columns,
                                               std::vector<index> core, DiagonalStorage diagonal)
        : _space(std::move(space)), _components(components), _diagonalStorage(diagonal), _columns(std::move(columns)),
          _diagonal(R::RowLayout::Dense(_space.OwnedRows(), DiagonalWidth(components, diagonal))),
          _inverse(_diagonal.Layout()), _offDiagonal(Counts(_columns), components)
    {
        const index rows = index(_space.OwnedRows()), allRows = rows + index(_space.GhostRows());
        DNDS_check_throw_info(index(_columns.size()) == rows, "linear stencil row count differs from its DOF space");
        std::vector<unsigned char> isCore(rows, 0);
        for (auto iCell : core)
        {
            DNDS_check_throw_info(iCell >= 0 && iCell < rows && !isCore[iCell], "invalid or duplicate linear core row");
            isCore[iCell] = 1;
        }
        for (index iCell = 0; iCell < rows; ++iCell)
        {
            auto sorted = _columns[iCell];
            std::sort(sorted.begin(), sorted.end());
            DNDS_check_throw_info(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "duplicate linear stencil column");
            for (auto column : sorted)
                DNDS_check_throw_info(column >= 0 && column < allRows && column != iCell && (!isCore[iCell] || column < rows),
                                      "invalid linear stencil column or core ghost dependency");
            if (!isCore[iCell])
                _dependent.push_back(iCell);
        }
        _core = std::move(core);
        _order = _core;
        _order.insert(_order.end(), _dependent.begin(), _dependent.end());
        _position.resize(rows);
        for (index i = 0; i < rows; ++i)
            _position[_order[i]] = i;
        _diagonalIdentity = _diagonal.Identity();
        _diagonalShape = _diagonal.ShapeVersion();
        _offDiagonalIdentity = _offDiagonal.Values().Identity();
    }
    void LinearProblemContext::ValidateMatrix() const
    {
        DNDS_check_throw_info(_diagonal.Identity() == _diagonalIdentity && _diagonal.ShapeVersion() == _diagonalShape &&
                                  _offDiagonal.Values().Identity() == _offDiagonalIdentity,
                              "linear matrix allocation or shape changed");
    }
    void LinearProblemContext::ValidateField(const Field &field) const
    {
        field.Validate();
        DNDS_check_throw_info(field.Space().Same(_space) && index(field.Width()) == _components && field.Device() == R::Device::Host(),
                              "linear field space, component count or backend differs");
    }
    void LinearProblemContext::ValidateOutputs(const Field &input, const Field &output) const
    {
        ValidateField(input);
        ValidateField(output);
        for (const auto *source : {&input.Owned(), &input.Ghost(), &_diagonal, &_inverse, &_offDiagonal.Values()})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(output.Owned().Storage(), source->Storage()), "linear output aliases input or matrix storage");
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
        DNDS_check_throw_info(input.RowCount() == index(_space.OwnedRows()) && input.Width() == width, "linear diagonal shape changed");
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
        if (region != LinearRegion::Core)
            ghost.emplace(input.Ghost().Read());
        const auto diagonal = _diagonal.Read();
        const auto offDiagonal = _offDiagonal.Read();
        auto destination = output.Owned().Write();
        const auto x = Algebra::MakeEigenRowMatrices(owned.View<R::UniformTableRows>(), _components, 1);
        const auto y = Algebra::MakeEigenRowMatrices(destination.View<R::UniformTableRows>(), _components, 1);
        const auto d = diagonal.View<R::TableRows>();
        const auto blocks = offDiagonal.View();
        const auto matrix = Algebra::MakeEigenRowMatrices(d, _components, _diagonalStorage == DiagonalStorage::Block ? _components : 1);
        const auto remote = ghost ? ghost->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{};
        const auto xGhost = Algebra::MakeEigenRowMatrices(remote, _components, 1);
        const auto compute = [&](index iCell)
        {
            if (_diagonalStorage == DiagonalStorage::Block)
                y[iCell].noalias() = matrix[iCell] * x[iCell];
            else
                y[iCell].array() = matrix[iCell].array() * x[iCell].array();
            for (index j = 0; j < index(_columns[iCell].size()); ++j)
            {
                const auto column = _columns[iCell][j];
                if (column < index(_space.OwnedRows()))
                    y[iCell].noalias() += blocks(iCell, j) * x[column];
                else
                    y[iCell].noalias() += blocks(iCell, j) * xGhost[column - index(_space.OwnedRows())];
            }
        };
        if (region == LinearRegion::All)
            for (index iCell = 0; iCell < index(_space.OwnedRows()); ++iCell)
                compute(iCell);
        else
            for (auto iCell : region == LinearRegion::Core ? _core : _dependent)
                compute(iCell);
    }
    void LinearProblemContext::ApplySGS(const Field &rhs, const Field &input, Field &forward, Field &output) const
    {
        ValidateOutputs(input, forward);
        ValidateOutputs(input, output);
        ValidateOutputs(rhs, forward);
        ValidateOutputs(rhs, output);
        ValidateOutputs(forward, output);
        ValidateMatrix();
        const auto diagonal = _diagonal.Read(), inverse = _inverse.Read();
        DNDS_check_throw_info(_inverseCurrent && _diagonal.Version() == _diagonalVersion && _inverse.Version() == _inverseVersion,
                              "linear diagonal inverse is missing or stale");
        const auto offDiagonal = _offDiagonal.Read();
        const auto bRead = rhs.Owned().Read(), owned = input.Owned().Read(), ghost = input.Ghost().Read();
        auto middle = forward.Owned().Write();
        std::optional<R::Array<real>::WriteLease> destination;
        try
        {
            destination.emplace(output.Owned().Write());
        }
        catch (...)
        {
            middle.CancelUnmodified();
            throw;
        }
        const auto b = Algebra::MakeEigenRowMatrices(bRead.View<R::UniformTableRows>(), _components, 1);
        const auto old = Algebra::MakeEigenRowMatrices(owned.View<R::UniformTableRows>(), _components, 1);
        const auto remote = Algebra::MakeEigenRowMatrices(ghost.View<R::UniformTableRows>(), _components, 1);
        const auto lower = Algebra::MakeEigenRowMatrices(middle.View<R::UniformTableRows>(), _components, 1);
        const auto result = Algebra::MakeEigenRowMatrices(destination->View<R::UniformTableRows>(), _components, 1);
        const auto inv = Algebra::MakeEigenRowMatrices(inverse.View<R::TableRows>(), _components,
                                                       _diagonalStorage == DiagonalStorage::Block ? _components : 1);
        const auto blocks = offDiagonal.View();
        DispatchVector(_components, [&](auto size)
                       {
            Eigen::Matrix<real, decltype(size)::value, 1> value(_components);
            for (auto iCell : _order)
            {
                value = b[iCell];
                for (index j = 0; j < index(_columns[iCell].size()); ++j)
                {
                    const index column = _columns[iCell][j];
                    if (column >= index(_space.OwnedRows())) value.noalias() -= blocks(iCell, j) * remote[column - index(_space.OwnedRows())];
                    else if (_position[column] < _position[iCell]) value.noalias() -= blocks(iCell, j) * lower[column];
                    else value.noalias() -= blocks(iCell, j) * old[column];
                }
                if (_diagonalStorage == DiagonalStorage::Block) lower[iCell].noalias() = inv[iCell] * value;
                else lower[iCell].array() = inv[iCell].array() * value.array();
            }
            for (index pos = index(_order.size()); pos > 0; --pos)
            {
                const auto iCell = _order[pos - 1];
                value = b[iCell];
                for (index j = 0; j < index(_columns[iCell].size()); ++j)
                {
                    const index column = _columns[iCell][j];
                    if (column >= index(_space.OwnedRows())) value.noalias() -= blocks(iCell, j) * remote[column - index(_space.OwnedRows())];
                    else if (_position[column] < _position[iCell]) value.noalias() -= blocks(iCell, j) * lower[column];
                    else value.noalias() -= blocks(iCell, j) * result[column];
                }
                if (_diagonalStorage == DiagonalStorage::Block) result[iCell].noalias() = inv[iCell] * value;
                else result[iCell].array() = inv[iCell].array() * value.array();
            } });
    }
}
