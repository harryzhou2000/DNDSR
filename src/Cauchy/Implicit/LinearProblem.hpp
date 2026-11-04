#pragma once

#include "DNDS/Algebra/BlockMatrixStorage.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Algebra/ComponentSelection.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace DNDS::Cauchy
{
    class EquationDofs;
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
    enum class OffDiagonalStorage
    {
        Assembled,
        SemiMatrixFree
    };
    /// Immutable physical block producer. Columns and block ordinals belong to
    /// LinearProblemContext. Implementations own frozen numerical inputs; they
    /// must not borrow live means, perform communication or mutate hidden caches.
    /// Blocks use column-major component order, including periodic transforms.
    class OffDiagonalEvaluator
    {
    public:
        virtual ~OffDiagonalEvaluator() = default;
        virtual index Components() const = 0;
        virtual std::shared_ptr<const Algebra::DofSpace::Partition> RowSpaceIdentity() const = 0;
        virtual const std::vector<std::vector<index>> &Columns() const = 0;
        virtual void EvaluateBlock(index iRow, index iBlock, Eigen::Ref<Eigen::MatrixXd> output) const = 0;
    };
    /// Integrated residual convention: A=M/dTau+physicalMass*M-residualScale*J_R.
    /// Roe dissipation requires a full diagonal block; scalar dissipation keeps
    /// the active Euler diagonal spectral approximation, including in block mode.
    struct JacobianPreparation
    {
        real physicalMass = 0, residualScale = 1;
        bool roeDissipation = false;
        /// Empty means full flow5. A subset embeds its principal matrix in
        /// flow5 storage: unselected rows/columns are zero, diagonal is one.
        /// The matching RHS must be zero in those unselected equations.
        std::vector<index> updateComponents;
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
        OffDiagonalStorage _offDiagonalStorage;
        std::optional<Algebra::BlockMatrixStorage> _offDiagonal;
        std::shared_ptr<const OffDiagonalEvaluator> _offDiagonalEvaluator;
        uint64_t _offDiagonalGeneration = 0;
        uint64_t _diagonalVersion = 0, _inverseVersion = 0;
        Runtime::ArrayStorage::Shared _diagonalStoragePin, _offDiagonalStoragePin;
        uint64_t _diagonalShape = 0;
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
        void ValidateField(const EquationDofs &field) const;
        void ValidateOutputs(const EquationDofs &input, const EquationDofs &output) const;

    public:
        /// Off-diagonal row columns use the owned+ghost local indexing of the
        /// supplied DOF space. Rows must have no duplicates or self column.
        /// Core rows may be conservative, but must not reference a ghost input.
        /// Logical scans always use [core][noncore], preserving each input list
        /// order; physical arrays and MPI mappings are not reordered.
        LinearProblemContext(Algebra::DofSpace space, index components,
                             std::vector<std::vector<index>> columns,
                             std::vector<index> core, DiagonalStorage diagonal,
                             OffDiagonalStorage offDiagonal = OffDiagonalStorage::Assembled);
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
        OffDiagonalStorage OffDiagonalMode() const { return _offDiagonalStorage; }
        bool HasAssembledOffDiagonal() const { return bool(_offDiagonal); }
        /// Explicit storage access is unavailable in semi-matrix-free mode.
        Algebra::BlockMatrixStorage &OffDiagonal();
        const Algebra::BlockMatrixStorage &OffDiagonal() const;
        /// Numeric preparation only; serialize with all context access. The
        /// returned immutable producer may supply reusable geometry to the next
        /// preparation. Numerical consumers use ReadOffDiagonal instead.
        const std::shared_ptr<const OffDiagonalEvaluator> &PreparedOffDiagonalEvaluator() const { return _offDiagonalEvaluator; }
        /// Publish a fully frozen producer. A short exclusive diagonal lease
        /// guards this publication against active operators/SGS/factor reads.
        /// No diagonal value or generation is changed by this publication.
        void PublishOffDiagonalEvaluator(std::shared_ptr<const OffDiagonalEvaluator> evaluator);
        uint64_t OffDiagonalGeneration() const;
        /// Retains numerical storage or the frozen evaluator. Evaluated reads
        /// also pin the diagonal as publication guard. Retain this scope through
        /// all block use; no source means/communicators are retained by it.
        class OffDiagonalReadAccess
        {
            std::optional<Algebra::BlockMatrixStorage::ReadAccess> _assembled;
            std::optional<Runtime::Array<real>::ReadLease> _publication;
            std::shared_ptr<const OffDiagonalEvaluator> _evaluator;
            index _components;
            friend class LinearProblemContext;
            OffDiagonalReadAccess(Algebra::BlockMatrixStorage::ReadAccess assembled, index components)
                : _assembled(std::move(assembled)), _components(components) {}
            OffDiagonalReadAccess(Runtime::Array<real>::ReadLease publication, std::shared_ptr<const OffDiagonalEvaluator> evaluator, index components)
                : _publication(std::move(publication)), _evaluator(std::move(evaluator)), _components(components) {}
            template <int Size>
            struct EvaluatedBlocks
            {
                const OffDiagonalEvaluator *evaluator;
                index components;
                Eigen::Matrix<real, Size, Size> operator()(index iRow, index iBlock) const
                {
                    if constexpr (Runtime::AccessBoundsChecks)
                    {
                        const auto &columns = evaluator->Columns();
                        DNDS_assert(iRow >= 0 && iRow < index(columns.size()) && iBlock >= 0 && iBlock < index(columns[iRow].size()));
                    }
                    Eigen::Matrix<real, Size, Size> result(components, components);
                    evaluator->EvaluateBlock(iRow, iBlock, result);
                    return result;
                }
            };

        public:
            /// Dispatch once outside loops. The callback receives a callable
            /// (row,block): assembled mode returns the ordinary stable map;
            /// evaluated mode returns an owning block value, with flow5 on the
            /// stack. Querying another block never overwrites a previous result.
            /// Other component counts use the generic dynamic dense matrix.
            template <class Callback>
            void WithBlocks(Callback &&callback) const
            {
                DNDS_check_throw_info(_assembled || _evaluator, "off-diagonal read scope is moved");
                if (_assembled)
                    callback(_assembled->View());
                else if (_components == 5)
                    callback(EvaluatedBlocks<5>{_evaluator.get(), _components});
                else
                    callback(EvaluatedBlocks<Eigen::Dynamic>{_evaluator.get(), _components});
            }
        };
        OffDiagonalReadAccess ReadOffDiagonal() const;
        /// Explicit preparation of diagonal inverses. Block mode follows the
        /// active Euler guarded LU/SVD inversion policy. Later diagonal writes
        /// invalidate the inverses; active writers are rejected by read leases.
        void InvertDiagonal();
        index LastInverseFallbacks() const { return _inverseFallbacks; }
        /// Stateless matrix action. Core acquires no input ghost lease. Every
        /// output allocation must be distinct from inputs and numeric storage.
        void Apply(const Field &input, Field &output, LinearRegion region = LinearRegion::All) const;
        /// Coupled matrix action on logical blocks without whole-state packing.
        /// Input/output may group components differently; semantics, row space
        /// and device must agree. Core reads no ghost part. Caller retains the
        /// matrix and serializes preparation; return is synchronous on CPU.
        void Apply(const EquationDofs &input, EquationDofs &output, LinearRegion region = LinearRegion::All) const;
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
        /// Same scan/order/communication contract for logical equation fields.
        /// Owns numeric aliases and leases, not an MPI context. The controller
        /// retains communication owners and completes each declared halo phase.
        class EquationSGSIteration
        {
            struct State;
            std::unique_ptr<State> _state;
            explicit EquationSGSIteration(std::unique_ptr<State> state);
            friend class LinearProblemContext;

        public:
            EquationSGSIteration(const EquationSGSIteration &) = delete;
            EquationSGSIteration &operator=(const EquationSGSIteration &) = delete;
            EquationSGSIteration(EquationSGSIteration &&) noexcept;
            EquationSGSIteration &operator=(EquationSGSIteration &&) noexcept;
            ~EquationSGSIteration();
            void Lower(LinearRegion region = LinearRegion::All);
            void Upper(LinearRegion region = LinearRegion::All);
            void Finish();
        };
        EquationSGSIteration BeginSGS(const EquationDofs &rhs, const EquationDofs &input, EquationDofs &forward, EquationDofs &output, bool middleExchange = false) const;
        /// Two-buffer local symmetric GS application with distinct forward
        /// scratch. Frozen remote inputs are used throughout both scans; there
        /// is no MPI between lower and upper. This unsplit convenience owns all
        /// leases for the call; staged/communication orchestration is separate.
        void ApplySGS(const Field &rhs, const Field &input, Field &forward, Field &output) const;
        void ApplySGS(const EquationDofs &rhs, const EquationDofs &input, EquationDofs &forward, EquationDofs &output) const;
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
        /// Same local factors/order on logical blocks. Only one-row dense
        /// scratch is gathered at a time; no distributed state repacking.
        void ApplyILU(const EquationDofs &rhs, EquationDofs &forward, EquationDofs &output) const;
    };
}
