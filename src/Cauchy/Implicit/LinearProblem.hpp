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
    /// Integrated residual convention: A=M/dTau+physicalMass*M-residualScale*J_R.
    /// Roe dissipation requires a full diagonal block; scalar dissipation keeps
    /// the active Euler diagonal spectral approximation, including in block mode.
    struct JacobianPreparation
    {
        real physicalMass = 0, residualScale = 1;
        bool roeDissipation = false;
    };

    /// Frozen rank-owned block rows, including off-rank operator columns. The
    /// row space and stencil are immutable; numerical values are filled through
    /// native Array leases. No EOS, mesh mutation, MPI, nonlinear or time policy
    /// is hidden here. The caller serializes numeric preparation and execution.
    class LinearProblemContext
    {
        using Field = Algebra::DofField<real>;
        struct Pattern
        {
            std::shared_ptr<const Algebra::DofSpace::Partition> partition;
            std::vector<std::vector<index>> columns;
            std::vector<index> order, position, core, dependent;
            bool hasGhostColumns = false;
            Pattern(std::shared_ptr<const Algebra::DofSpace::Partition> rowsIdentity, std::vector<std::vector<index>> rows)
                : partition(std::move(rowsIdentity)), columns(std::move(rows)) {}
        };
        Algebra::DofSpace _space;
        std::shared_ptr<const Pattern> _pattern;
        index _components;
        DiagonalStorage _diagonalStorage;
        Runtime::Array<real> _diagonal, _inverse;
        Algebra::BlockMatrixStorage _offDiagonal;
        uint64_t _diagonalVersion = 0, _inverseVersion = 0;
        uint64_t _diagonalIdentity = 0, _diagonalShape = 0, _offDiagonalIdentity = 0;
        bool _inverseCurrent = false;
        index _inverseFallbacks = 0;
        struct ILUState;
        std::shared_ptr<ILUState> _ilu;
        static std::shared_ptr<const Pattern> BuildPattern(const Algebra::DofSpace &space, std::vector<std::vector<index>> columns, std::vector<index> core);
        const Pattern &Layout() const
        {
            DNDS_check_throw_info(_pattern, "linear context was moved");
            return *_pattern;
        }
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
        const Algebra::DofSpace &Space() const
        {
            Layout();
            return _space;
        }
        index Components() const { return _components; }
        DiagonalStorage DiagonalMode() const { return _diagonalStorage; }
        const std::vector<std::vector<index>> &Columns() const { return Layout().columns; }
        const std::vector<index> &Order() const { return Layout().order; }
        const std::vector<index> &Core() const { return Layout().core; }
        const std::vector<index> &HaloDependent() const { return Layout().dependent; }
        void Validate() const { ValidateMatrix(); }
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
        /// Move-only access scope for one staged symmetric-GS application.
        /// Lower runs Core then HaloDependent; Upper runs HaloDependent then
        /// Core. All is available for either complete scan. Finish publishes
        /// the result only after both scans complete. Abandoning a started
        /// iteration invalidates unfinished outputs instead of publishing them.
        /// The scope retains pattern, allocations and numerical input leases;
        /// it remains valid even if the original context/field handles expire.
        class SGSIteration
        {
            struct State;
            std::unique_ptr<State> _state;
            explicit SGSIteration(std::unique_ptr<State> state);
            friend class LinearProblemContext;

        public:
            SGSIteration(const SGSIteration &) = delete;
            SGSIteration &operator=(const SGSIteration &) = delete;
            SGSIteration(SGSIteration &&) noexcept;
            SGSIteration &operator=(SGSIteration &&) noexcept;
            ~SGSIteration();
            void Lower(LinearRegion region = LinearRegion::All);
            void Upper(LinearRegion region = LinearRegion::All);
            void Finish();
        };
        /// Core lower work may start while input ghosts are being written.
        /// Complete the input halo before lower dependent work. By default
        /// remote old values remain frozen across both scans. If the explicit
        /// middleExchange option is true, caller exchanges completed forward
        /// values before Upper; that join is exposed. No MPI is posted here.
        /// Callers serialize phase calls and do not replace field handles or
        /// routing during the iteration. Synchronous CPU entry points only.
        SGSIteration BeginSGS(const Field &rhs, const Field &input, Field &forward, Field &output, bool middleExchange = false) const;
        /// Two-buffer local symmetric GS application with distinct forward
        /// scratch. Frozen remote inputs are used throughout both scans; there
        /// is no MPI between lower and upper. This unsplit convenience owns all
        /// leases for the call; staged/communication orchestration is separate.
        void ApplySGS(const Field &rhs, const Field &input, Field &forward, Field &output) const;
        /// Rank-local incomplete/full factor structure. Off-rank columns stay
        /// in Apply but are excluded from local factors. Core-first ordering
        /// is optional and precedes symbolic fill; local parts are independent.
        void PrepareILUStructure(int fill = 0, bool coreFirst = true, const std::vector<index> &localParts = {});
        /// Copy current native coefficients into the prepared factor template,
        /// then factor numerically. No inverse/SGS state or MPI halo is changed.
        void FactorizeILU();
        bool ILUFactorsCurrent() const;
        /// Dependency-safe ready rows from the conservative input core through
        /// both symbolic scans. Complete LU provides no bounded-core promise.
        const std::vector<index> &ILUForwardCore() const;
        const std::vector<index> &ILUBackwardCore() const;
        /// One full local factor solve, no communication between scans. Parent
        /// matrix generations are checked while reads are pinned. RHS, forward
        /// scratch and output are separate native fields on the same row space.
        void ApplyILU(const Field &rhs, Field &forward, Field &output) const;
    };
}
