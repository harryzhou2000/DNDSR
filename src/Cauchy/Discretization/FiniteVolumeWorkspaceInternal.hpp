#pragma once

#include "Cauchy/Discretization/FiniteVolumeReconstructionStateInternal.hpp"
#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct FiniteVolume::Workspace
    {
        R::SessionAccess session;
        struct Field
        {
            std::optional<CFV::ReconstructionWorkspace> reconstruction;
            std::optional<CFV::ReconstructionBoundaryValues> reconstructionBoundary;
            Runtime::Array<real> boundaryValues;
            std::optional<CFV::FVField> gradient;
            std::optional<R::Array<real>> gradientBoundary;
            std::vector<index> vectorStarts;
        };
        std::shared_ptr<const Data> owner;
        enum ProfilePhase
        {
            BoundaryPhase,
            GradientPhase,
            VRPhase,
            PPPhase,
            RHSPhase,
            SamplingPhase,
            SourcePhase,
            TotalPhase,
            ProfilePhases
        };
        bool profiling = std::getenv("DNDS_CAUCHY_PROFILE_EQUATION_FV") != nullptr;
        bool profileAllRanks = std::getenv("DNDS_CAUCHY_PROFILE_ALL_RANKS") != nullptr;
        std::array<real, ProfilePhases> profileSeconds{};
        std::array<index, ProfilePhases> profileCalls{};
        struct PhaseTime
        {
            using Clock = std::chrono::steady_clock;
            Workspace *work;
            ProfilePhase phase;
            Clock::time_point start{};
            PhaseTime(Workspace &input, ProfilePhase section) : work(input.profiling ? &input : nullptr), phase(section)
            {
                if (work)
                    start = Clock::now();
            }
            ~PhaseTime()
            {
                if (!work)
                    return;
                work->profileSeconds[phase] += std::chrono::duration<real>(Clock::now() - start).count();
                const int rank = work->owner->mesh->getMPI().rank;
                if (++work->profileCalls[phase] % 32 == 0 && phase == TotalPhase && (!rank || work->profileAllRanks))
                {
                    const auto record = Json{{"rank", rank}, {"owned_cells", work->owner->metrics.NumCell()}, {"meanOnly", work->meanOnly}, {"reconstruction", work->owner->settings.numerics.reconstruction}, {"firstOrderConvection", work->owner->settings.numerics.firstOrderConvection}, {"phases", {"boundary", "gradient", "vr", "pp", "rhs", "sampling", "source", "evaluate"}}, {"seconds", work->profileSeconds}, {"calls", work->profileCalls}}.dump();
                    std::fprintf(stdout, "CAUCHY_EQUATION_FV_PROFILE %s\n", record.c_str());
                }
            }
        };
        const bool meanOnly;
        index componentNumber = 0, physicalFieldNumber = 0;
        std::vector<Field> fields;
        EquationDofs boundary;
        /// Operation-scoped retained binding. It is installed by the
        /// orchestration call and cleared on every exit; the workspace never
        /// becomes the durable coefficient owner.
        std::optional<EquationDofs> activeCoefficients, activeFluxCoefficients;
        // Accepted direct-FV physical gradients, separate from source sampling.
        // WBAP/PP consume their explicit polynomial projection; the direct RHS
        // reads only these physical gradients and never evaluates a Taylor basis.
        std::vector<CFV::FVField> directGradients, directViscousGradients;
        /// Unfiltered explicit P1 keeps the GradientPlan publication directly;
        /// no gradient-to-modal-to-gradient round trip or coefficient halo.
        bool nativeDirectGradient = false;
        std::vector<R::TransferBinding> directGradientHalo, directGradientInputHalo,
            directViscousGradientHalo;
        /// Legacy-compatible primitive-gradient viscosity is prepared inside
        /// this workspace but consumed by the same fixed-flow RHS leaf.
        std::optional<CFV::FVField> primitive, primitiveGradient;
        std::optional<R::Array<real>> primitiveBoundary;
        std::optional<R::TransferBinding> primitiveGradientHalo;
        struct DirectGradientPublication
        {
            struct BoundaryField
            {
                std::weak_ptr<const R::ArrayStorage> storage;
                uint64_t shape = 0, version = 0;
            };
            // Weak identities reject rebinding/ABA without retaining old fields.
            // The workspace/operation owns the arrays and packet; this stamp is
            // validation metadata, never an asynchronous lifetime mechanism.
            std::weak_ptr<const R::ArrayStorage> mean, input, output;
            uint64_t meanShape = 0, meanVersion = 0, inputShape = 0,
                     inputVersion = 0, outputShape = 0, outputVersion = 0;
            R::VersionStamp geometry;
            std::vector<BoundaryField> boundary;
            real time = std::numeric_limits<real>::quiet_NaN();
            std::string reconstruction, gradient, limiter;

            void Publish(const Data &plan, const R::Array<real> &meanValues,
                         const R::Array<real> &inputValues,
                         const R::Array<real> &outputValues,
                         const std::optional<BoundaryInputs> &incoming,
                         real stageTime)
            {
                const auto meanRead = meanValues.Read();
                const auto inputRead = inputValues.Read();
                const auto outputRead = outputValues.Read();
                mean = meanValues.SharedStorage();
                input = inputValues.SharedStorage();
                output = outputValues.SharedStorage();
                meanShape = meanValues.ShapeVersion();
                meanVersion = meanValues.Version();
                inputShape = inputValues.ShapeVersion();
                inputVersion = inputValues.Version();
                outputShape = outputValues.ShapeVersion();
                outputVersion = outputValues.Version();
                geometry = plan.boundaryLayout->stamp;
                boundary.clear();
                if (plan.hasInterfaces)
                {
                    DNDS_check_throw_info(
                        incoming && incoming->Time() == stageTime,
                        "direct gradient publication needs the current interface packet");
                    const auto &values = incoming->QuadratureValues();
                    boundary.reserve(size_t(values.PhysicalFieldNumber()));
                    for (index iField = 0;
                         iField < values.PhysicalFieldNumber(); ++iField)
                    {
                        const auto &array = values.PhysicalField(iField).Owned();
                        const auto read = array.Read();
                        boundary.push_back({array.SharedStorage(),
                                            array.ShapeVersion(),
                                            array.Version()});
                    }
                }
                time = stageTime;
                reconstruction = plan.settings.numerics.reconstruction;
                gradient = plan.settings.numerics.gradient;
                limiter = plan.settings.numerics.limiter;
            }

            bool Current(const Data &plan, const R::Array<real> &meanValues,
                         const R::Array<real> &inputValues,
                         const R::Array<real> &outputValues,
                         const std::optional<BoundaryInputs> &incoming,
                         real stageTime) const
            {
                // Acquire before reading generations: checking first would
                // permit a writer to publish a new generation in the gap.
                const auto meanRead = meanValues.Read();
                const auto inputRead = inputValues.Read();
                const auto outputRead = outputValues.Read();
                if (time != stageTime || !geometry.Valid() ||
                    reconstruction != plan.settings.numerics.reconstruction ||
                    gradient != plan.settings.numerics.gradient ||
                    limiter != plan.settings.numerics.limiter ||
                    mean.lock() != meanValues.SharedStorage() ||
                    input.lock() != inputValues.SharedStorage() ||
                    output.lock() != outputValues.SharedStorage() ||
                    meanShape != meanValues.ShapeVersion() ||
                    meanVersion != meanValues.Version() ||
                    inputShape != inputValues.ShapeVersion() ||
                    inputVersion != inputValues.Version() ||
                    outputShape != outputValues.ShapeVersion() ||
                    outputVersion != outputValues.Version())
                    return false;
                if (!plan.hasInterfaces)
                    return boundary.empty();
                if (!incoming || incoming->Time() != stageTime ||
                    boundary.size() !=
                        size_t(incoming->QuadratureValues()
                                   .PhysicalFieldNumber()))
                    return false;
                const auto &values = incoming->QuadratureValues();
                for (index iField = 0;
                     iField < values.PhysicalFieldNumber(); ++iField)
                {
                    const auto &array = values.PhysicalField(iField).Owned();
                    const auto read = array.Read();
                    const auto &saved = boundary[size_t(iField)];
                    if (saved.storage.lock() != array.SharedStorage() ||
                        saved.shape != array.ShapeVersion() ||
                        saved.version != array.Version())
                        return false;
                }
                return true;
            }
        };
        std::shared_ptr<PositivityWorkspace> positivity;
        std::unique_ptr<LimitedVRWorkspace> limitedVR;
        EquationThermalPolicy thermal;
        /// Setup-resolved fixed-flow leaf. It owns immutable kernel reads; the
        /// surrounding Workspace still owns the only reconstruction/RHS
        /// schedule and all operand leases.
        std::optional<RHSOperator> fixedFlowRHS;
        /// One synchronous fixed-flow invocation's leases and plain views.
        /// Capacity is resolved by the prepared operator contract; no heap
        /// allocation occurs while packaging a core/dependent RHS call.
        struct RHSAccessScope
        {
            struct FieldRead
            {
                std::optional<R::Array<real>::ReadLease> owned, ghost;
            };
            static constexpr size_t FieldCapacity = 8;
            bool ghosts = false;
            size_t fieldCount = 0;
            std::array<FieldRead, FieldCapacity> fields;
            std::optional<R::Array<real>::ReadLease> thermodynamics;
            std::optional<DerivedFields::ReadScope> derivedOwned, derivedGhost;
            std::optional<R::Array<real>::WriteLease> result, lowOrder, face;
            RHSReadViews input;
            RHSWriteViews output;

            explicit RHSAccessScope(bool includeGhosts) : ghosts(includeGhosts) {}
            RHSFieldView ReadField(const CFV::FVField &field)
            {
                DNDS_check_throw_info(fieldCount < fields.size(),
                                      "prepared RHS access capacity exceeded");
                auto &slot = fields[fieldCount++];
                slot.owned.emplace(field.Owned().Read());
                if (ghosts)
                    slot.ghost.emplace(field.Ghost().Read());
                return {slot.owned->View<R::UniformTableRows>(),
                        slot.ghost ? slot.ghost->View<R::UniformTableRows>() : RHSScalarView{}};
            }
            RHSFlowFieldView ReadFlowField(const CFV::FVField &field)
            {
                DNDS_check_throw_info(fieldCount < fields.size(),
                                      "prepared RHS access capacity exceeded");
                auto &slot = fields[fieldCount++];
                slot.owned.emplace(field.Owned().Read());
                if (ghosts)
                    slot.ghost.emplace(field.Ghost().Read());
                return {slot.owned->View<RHSFlowRows>(),
                        slot.ghost ? slot.ghost->View<RHSFlowRows>() : decltype(slot.owned->View<RHSFlowRows>()){}};
            }
        };
        RHSLocalDiagnostics diagnostics;
        std::array<index, 2> faceFallbacks{};
        std::optional<CFV::FaceCharacteristicTransform> flowCharacteristic;
        std::optional<CFV::GradientFaceCharacteristicTransform> gradientFlowCharacteristic;
        index flowCharacteristicField = -1;
        index characteristicFallbacks = 0;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        /// Published receiver-frame interface traces. The workspace retains
        /// their allocation/publication identity; each evaluation pins the
        /// numeric rows through its final dependent consumer.
        std::optional<BoundaryInputs> incoming;
        /// Operator-owned packed auxiliary channels needed by flux profiles
        /// whose RHS cannot be evaluated from one face pair alone. Ordinary
        /// fluxes avoid this allocation and their existing fused path.
        std::optional<AuxiliaryValues> rhsAuxiliary;
        R::MetadataBuffer<index> logicalFirst, vectorStarts;
        R::MetadataBuffer<index> logicalField, logicalComponent;
        index flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        R::Array<index> sourceOffsets{R::RowLayout::Dense(0, 1)};
        R::Array<real> sourceWeights;
        std::optional<CFV::PolynomialSamplingPlan> sourceSampling;
        std::vector<R::Array<real>> sourceValues;
        bool sourceValuesAreMeans = false;
        std::vector<std::array<R::Array<real>, 3>> sourceGradients;
        // The default approximate source Jacobian uses cell means plus
        // gradients at one cell center. The opt-in rhsQuadrature mode reuses
        // the separately published RHS source samples.
        std::optional<CFV::PolynomialSamplingPlan> sourceJacobianSampling;
        std::vector<std::array<R::Array<real>, 3>> sourceJacobianGradients;
        R::Array<real> boundaryMomentumFlux;
        /// Integrated residual contribution in the canonical owned-parent
        /// frame, one row per local stencil face. Refilled on every RHS call.
        R::Array<real> faceFlux;
        /// Optional first-order conservative flux path used by explicit flow5
        /// positivity. It is an output of the same equation RHS operator, not
        /// a second finite-volume frontend or a second reconstruction pass.
        std::optional<EquationDofs> lowOrderResidual;
        std::optional<R::Array<real>> lowOrderFaceFlux;
        bool boundaryMomentumCurrent = false;
        std::optional<MeanGradients> meanGradients;
        bool chemical;
        int iterations = 0;
        real relativeUpdate = 0;
        index directPPLimited = 0;
        real directPPMinimum = 1;
        /// Reused direct-P1 CPU scratch. Workspace session access serializes
        /// these buffers; no allocation occurs in the cell/point operator.
        R::Array<real> directLimiterScratch, directPPSamples;
        struct WeakPublication
        {
            std::vector<std::weak_ptr<const R::ArrayStorage>> storage;
            std::vector<uint64_t> shapes, versions;
            explicit WeakPublication(const EquationDofs &values)
            {
                storage.reserve(size_t(values.PhysicalFieldNumber()));
                shapes.reserve(storage.capacity());
                versions.reserve(storage.capacity());
                for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                {
                    const auto &array = values.PhysicalField(iField).Owned();
                    storage.push_back(array.SharedStorage());
                    shapes.push_back(array.ShapeVersion());
                    versions.push_back(array.Version());
                }
            }
            bool Current(const EquationDofs &values) const
            {
                if (storage.size() != size_t(values.PhysicalFieldNumber()))
                    return false;
                for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                {
                    const auto &array = values.PhysicalField(iField).Owned();
                    if (storage[size_t(iField)].lock() != array.SharedStorage() ||
                        shapes[size_t(iField)] != array.ShapeVersion() ||
                        versions[size_t(iField)] != array.Version())
                        return false;
                }
                return true;
            }
        };
        std::optional<EquationDofOperations::Binding> sourceState;
        std::optional<WeakPublication> sourceCoefficients;
        real sourceTime = 0;
        std::optional<EquationDofOperations::Binding> rhsState;
        std::optional<WeakPublication> rhsCoefficients;
        real rhsTime = 0;
        // Keep auxiliary cache metadata after the existing numerical fields.
        std::vector<std::array<DirectGradientPublication, 2>> directGradientPublications;

        class ActiveReconstruction
        {
            Workspace *_work = nullptr;

        public:
            ActiveReconstruction(Workspace &work, const EquationDofs &coefficients,
                                 const EquationDofs &fluxCoefficients) : _work(&work)
            {
                DNDS_check_throw_info(!work.activeCoefficients && !work.activeFluxCoefficients,
                                      "workspace already has an active reconstruction binding");
                work.activeCoefficients.emplace(coefficients);
                work.activeFluxCoefficients.emplace(fluxCoefficients);
            }
            ActiveReconstruction(const ActiveReconstruction &) = delete;
            ActiveReconstruction &operator=(const ActiveReconstruction &) = delete;
            ActiveReconstruction(ActiveReconstruction &&other) noexcept : _work(other._work) { other._work = nullptr; }
            ~ActiveReconstruction()
            {
                if (_work)
                {
                    _work->activeCoefficients.reset();
                    _work->activeFluxCoefficients.reset();
                }
            }
        };
        ActiveReconstruction BindReconstruction(const EquationDofs &coefficients,
                                                const EquationDofs &fluxCoefficients)
        {
            return ActiveReconstruction(*this, coefficients, fluxCoefficients);
        }
        const EquationDofs &Coefficients() const
        {
            DNDS_check_throw_info(activeCoefficients, "workspace has no active reconstruction binding");
            return *activeCoefficients;
        }
        const EquationDofs &FluxCoefficients() const
        {
            DNDS_check_throw_info(activeFluxCoefficients, "workspace has no active finalized reconstruction binding");
            return *activeFluxCoefficients;
        }
        EquationDofs &FluxCoefficients()
        {
            DNDS_check_throw_info(activeFluxCoefficients, "workspace has no active finalized reconstruction binding");
            return *activeFluxCoefficients;
        }

        static std::vector<Field> BuildFields(const Data &data, const EquationDofs &state, bool meanOnly);
        void BuildSourceSampling(const EquationDofs &state);
        static EquationDofs BindBoundary(const EquationDofs &state, const std::vector<Field> &fields, index rows);
        Workspace(std::shared_ptr<const Data> data, EquationDofs &state,
                  std::optional<Physics::Chemistry::ChemicalWorkerPool> pool, bool meanTrial);
        void BindHalos(EquationDofs &state);
        void RefreshDirectGradients(const EquationDofs &state, bool ghost, real time);
        void PrepareDirectGradientLimiter(const EquationDofs &state, bool barth);
        void ApplyDirectGradientPositivity(const EquationDofs &state);
        void PreparePrimitiveGradient(const EquationDofs &state, CFV::ReconstructionRegion region);
        void FillBoundary(const EquationDofs &state, bool polynomial, real time,
                          EquationPointEvaluator &evaluator, RHSScalarView incoming = {});
        void ApplyFixedFlowRHS(const EquationDofs &state, EquationDofs &result,
                               CFV::ReconstructionRegion region, real time,
                               OptionalRef<const AuxiliaryValues> auxiliary,
                               OptionalRef<R::Array<real>> faceOutput,
                               RHSScalarView incoming);
        void ApplyRHS(const EquationDofs &state, EquationDofs &result,
                      CFV::ReconstructionRegion region, EquationPointEvaluator &evaluator,
                      real time, OptionalRef<const AuxiliaryValues> auxiliary = {},
                      OptionalRef<R::Array<real>> faceOutput = {},
                      RHSScalarView incoming = {});
        void ApplyEquationRHS(const EquationDofs &state, CFV::ReconstructionRegion region,
                              EquationPointEvaluator &evaluator, real time,
                              OptionalRef<const AuxiliaryValues> auxiliary,
                              const EquationDofs::ReadScope &mean,
                              EquationLayoutView<const real> owned, EquationLayoutView<const real> remote,
                              EquationLayoutView<real> destination, EquationLayoutView<real> lowDestination, bool produceLow,
                              OptionalRef<R::Array<real>> faceOutput, RHSScalarView incoming);
        void SampleSourceState(const EquationDofs &state, real time);
        bool AddRHSQuadratureSourceJacobian(
            const EquationDofs &state, const R::Array<real>::WriteLease &diagonal,
            DiagonalStorage storage, real scale, EquationPointEvaluator &evaluator,
            OptionalRef<const EquationFVDetail::MeanLinearizationOutput> linearization) const;
        bool AddMeanSourceJacobian(
            const EquationDofs &state, const R::Array<real>::WriteLease &diagonal,
            DiagonalStorage storage, real scale, EquationPointEvaluator &evaluator,
            OptionalRef<const EquationFVDetail::MeanLinearizationOutput> linearization = {}) const;
        void PublishSourceState(const EquationDofs &state, real time);
        bool SourceSamplesCurrent(const EquationDofs &state, real time) const;
        void PrepareVolumeSource(const EquationDofs &state,
                                 OptionalRef<EquationDofs> result,
                                 EquationPointEvaluator &evaluator, real time = 0,
                                 OptionalRef<EquationDofs> lowOrder = {});
    };

    struct FiniteVolume::MeanWorkspace
    {
        /// Zero-polynomial probe storage is temporary by definition; it is not
        /// the solver's warm reconstruction publication.
        std::shared_ptr<ReconstructionState> reconstruction;
        std::shared_ptr<Workspace> work;
    };

}
