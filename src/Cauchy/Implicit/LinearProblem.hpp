#pragma once

#include "DNDS/Algebra/BlockMatrixStorage.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include <memory>
#include <vector>

namespace DNDS::Cauchy
{
    enum class LinearRegion
    {
        All,
        Core,
        HaloDependent
    };
    enum class DiagonalStorage
    {
        Entries,
        Block
    };

    /// Frozen rank-owned block rows, including off-rank operator columns. The
    /// row space and stencil are immutable; numerical values are filled through
    /// native Array leases. No EOS, mesh mutation, MPI, nonlinear or time policy
    /// is hidden here. The caller serializes numeric preparation and execution.
    class LinearProblemContext
    {
        using Field = Algebra::DofField<real>;
        Algebra::DofSpace _space;
        index _components;
        DiagonalStorage _diagonalStorage;
        std::vector<std::vector<index>> _columns;
        std::vector<index> _order, _position, _core, _dependent;
        Runtime::Array<real> _diagonal, _inverse;
        Algebra::BlockMatrixStorage _offDiagonal;
        uint64_t _diagonalVersion = 0, _inverseVersion = 0;
        uint64_t _diagonalIdentity = 0, _diagonalShape = 0, _offDiagonalIdentity = 0;
        bool _inverseCurrent = false;
        index _inverseFallbacks = 0;
        void ValidateMatrix() const;
        void ValidateField(const Field &field) const;
        void ValidateOutputs(const Field &input, const Field &output) const;

    public:
        /// Off-diagonal row columns use the owned+ghost local indexing of the
        /// supplied DOF space. Rows must have no duplicates or self column.
        /// Core rows may be conservative, but must not reference a ghost input.
        /// Logical scans always use [core][noncore], preserving each input list
        /// order; physical arrays and MPI mappings are not reordered.
        LinearProblemContext(Algebra::DofSpace space, index components,
                             std::vector<std::vector<index>> columns,
                             std::vector<index> core, DiagonalStorage diagonal);
        LinearProblemContext(const LinearProblemContext &) = delete;
        LinearProblemContext &operator=(const LinearProblemContext &) = delete;
        LinearProblemContext(LinearProblemContext &&) noexcept = default;
        LinearProblemContext &operator=(LinearProblemContext &&) noexcept = default;
        const Algebra::DofSpace &Space() const { return _space; }
        index Components() const { return _components; }
        DiagonalStorage DiagonalMode() const { return _diagonalStorage; }
        const std::vector<std::vector<index>> &Columns() const { return _columns; }
        const std::vector<index> &Order() const { return _order; }
        const std::vector<index> &Core() const { return _core; }
        const std::vector<index> &HaloDependent() const { return _dependent; }
        Runtime::Array<real> &Diagonal() { return _diagonal; }
        const Runtime::Array<real> &Diagonal() const { return _diagonal; }
        Algebra::BlockMatrixStorage &OffDiagonal() { return _offDiagonal; }
        const Algebra::BlockMatrixStorage &OffDiagonal() const { return _offDiagonal; }
        /// Explicit preparation of diagonal inverses. Block mode follows the
        /// active Euler guarded LU/SVD inversion policy. Later diagonal writes
        /// invalidate the inverses; active writers are rejected by read leases.
        void InvertDiagonal();
        index LastInverseFallbacks() const { return _inverseFallbacks; }
        /// Stateless matrix action. Core acquires no input ghost lease. Every
        /// output allocation must be distinct from inputs and numeric storage.
        void Apply(const Field &input, Field &output, LinearRegion region = LinearRegion::All) const;
        /// Two-buffer local symmetric GS application with distinct forward
        /// scratch. Frozen remote inputs are used throughout both scans; there
        /// is no MPI between lower and upper. This unsplit convenience owns all
        /// leases for the call; staged/communication orchestration is separate.
        void ApplySGS(const Field &rhs, const Field &input, Field &forward, Field &output) const;
    };
}
