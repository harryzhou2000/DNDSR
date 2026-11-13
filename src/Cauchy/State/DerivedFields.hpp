#pragma once
#include "DNDS/Core/NumericalStatus.hpp"

#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Physics/FlowState.hpp"
#include "Physics/Chemistry/ChemicalWorker.hpp"
#include <initializer_list>
#include <optional>

namespace DNDS::Runtime
{
    class EagerEngine;
    class ExecutionStream;
}

namespace DNDS::Cauchy
{
    class EquationSystem;
    class PointEvaluator;
    enum class DerivedQuantity
    {
        Temperature,
        Pressure,
        SoundSpeed,
        MolecularViscosity,
        ThermalConductivity,
        HeatCapacityP,
        HeatCapacityV,
        GasConstant,
        DiffusionBound,
        TotalEnthalpy,                    ///< (rhoE+p)/rho, retaining the equation's energy origin.
        PressureInternalEnergyDerivative, ///< dp/d(rho*e) at fixed rho,Y; ideal mixture R_mix/cv.
        Count
    };
    constexpr int DerivedQuantityNumber = int(DerivedQuantity::Count);
    using DerivedPointValues = std::array<real, DerivedQuantityNumber>;

    /// Internal physics/scheme selection; these are properties, not equations.
    struct DerivedRequest
    {
        std::array<bool, DerivedQuantityNumber> enabled{};
        DerivedRequest() = default;
        DerivedRequest(std::initializer_list<DerivedQuantity> quantities);
        bool Has(DerivedQuantity quantity) const;
        int ComponentNumber() const;
    };

    /// Optional accompanying point properties. Copies share one allocation and
    /// its publication metadata; they never add components to EquationDofs.
    /// No halo plan is created. One instance covers owned OR ghost rows.
    ///
    /// Serialize controller publication/refresh calls. Current() is a snapshot
    /// query, not access permission; ReadCurrent retains source and output read
    /// leases. Refresh has immediate and queued CPU entries; the latter uses
    /// declared Array operands and publishes after same-stream retirement.
    /// Point providers must perform no MPI or stencil work; prepared local
    /// inputs and an exclusive worker context are required by mixture providers.
    class DerivedFields
    {
        struct Data;
        std::shared_ptr<Data> _data;
        friend bool RefreshDerivedFields(const EquationDofs &, const EquationSystem &, DerivedFields &, bool);
        friend bool RefreshDerivedFields(const EquationDofs &, const EquationSystem &, DerivedFields &,
                                         Physics::Chemistry::ChemicalWorkerPool::Lease, bool);
        friend void RefreshDerivedFields(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                         const EquationDofs &, const EquationSystem &, DerivedFields &,
                                         std::optional<Physics::Chemistry::ChemicalWorkerPool>, bool);

    public:
        DerivedFields(Algebra::DofSpace space, DerivedRequest request, bool ghost = false);
        bool Enabled() const;
        bool GhostPart() const;
        index RowNumber() const;
        const DerivedRequest &Request() const;
        /// Immutable selected-column metadata; grants no payload access or
        /// freshness. Queued consumers still declare Values() as an operand
        /// and validate the publication against their source state.
        int Column(DerivedQuantity quantity) const;
        /// Debug/export owner. Writes through an exported shallow handle
        /// invalidate publication by generation; prefer checked read scopes.
        const Runtime::Array<real> &Values() const;
        /// Validate/reserve a future refresh destination without changing its
        /// generation. Use before posting surrounding communications.
        void PreflightRefresh() const;
        bool Current(const EquationDofs &state, const EquationSystem &equation) const;

        class ReadScope
        {
            friend class DerivedFields;
            std::shared_ptr<const Data> _data;
            Runtime::Array<real>::ReadLease _read;
            std::optional<EquationDofs::ReadScope> _source;
            bool _guessOnly;
            ReadScope(std::shared_ptr<const Data> data, bool guessOnly);

        public:
            ReadScope(ReadScope &&) noexcept = default;
            ReadScope(const ReadScope &) = delete;
            EquationBlockArrayView<const real> Field(DerivedQuantity quantity) const;
            Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> View() const;
            int Column(DerivedQuantity quantity) const;
            /// Query the selected projection without demanding an absent column.
            bool Has(DerivedQuantity quantity) const;
        };
        ReadScope ReadCurrent(const EquationDofs &state, const EquationSystem &equation) const;
        /// Stale solution values are allowed only as temperature guesses.
        /// Row space, model/units and prior successful publication must match.
        /// The EOS consumer must still validate its guess and solve its state.
        ReadScope ReadTemperatureGuess(const EquationDofs &state, const EquationSystem &equation) const;
    };

    /// Pure local perfect-gas profile. The caller supplies an admissible state;
    /// coarse backend/schema validation belongs outside the point loop.
    DerivedPointValues EvaluateDerivedPoint(const EquationSystem &equation, const FlowState &state);
    /// Same point core for a reserved mixture/perfect-gas worker. Species are
    /// explicit logical inputs, not inferred from the tail of a physical Array.
    /// Nonthrowing point-property projection; borrowed worker data is consumed here.
    NumericalResult<DerivedPointValues> TryEvaluateDerivedPoint(PointEvaluator &evaluator, const FlowState &state,
                                                                Physics::Chemistry::ConstSpeciesBufferView rhoY = {}, real temperatureGuess = 0) noexcept;
    DerivedPointValues EvaluateDerivedPoint(PointEvaluator &evaluator, const FlowState &state,
                                            Physics::Chemistry::ConstSpeciesBufferView rhoY = {}, real temperatureGuess = 0);

    /// Refresh every enabled property at every row of the selected part. State
    /// is immutable; no MPI, stencil, reduction, clock or hidden getter update.
    /// Publish only after success. False means disabled or already current when
    /// onlyIfStale=true. Missing/stale ghost input is a caller readiness error.
    bool RefreshDerivedFields(const EquationDofs &state, const EquationSystem &equation,
                              DerivedFields &output, bool onlyIfStale = false);
    /// Explicit mixture context; consumed for this synchronous loop. Source is
    /// immutable. Compatible prior temperatures seed the new UV solve in place
    /// under the output writer, before each row is overwritten. On failure the
    /// bundle remains unpublished. Retain/move the lease with CPU worker work.
    bool RefreshDerivedFields(const EquationDofs &state, const EquationSystem &equation,
                              DerivedFields &output, Physics::Chemistry::ChemicalWorkerPool::Lease worker,
                              bool onlyIfStale = false);
    /// Queued CPU counterpart using the same point kernel. All source/value
    /// arrays are scheduled operands; the worker evaluator is acquired only
    /// during execution. Publication follows write retirement on the same
    /// stream, so later queued consumers can validate it without a host drain.
    /// The caller orders prior source producers and subsequent cache consumers.
    /// Failed allocations must be replaced, as for immediate refresh. ReadWrite
    /// declares the optional old-temperature guess; invalid data is never used
    /// as a guess. A queued cache hit may still advance its write generation.
    void RefreshDerivedFields(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                              const EquationDofs &, const EquationSystem &, DerivedFields &,
                              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = {},
                              bool onlyIfStale = false);
}
