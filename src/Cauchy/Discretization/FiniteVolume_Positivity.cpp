#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePositivityKernels.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "CFV/Math/TaylorBasis.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        /// Owns only read leases and logical column metadata. Values remain in
        /// per-physical-field sample arrays; no full-state sample packing.
        struct SampleRead
        {
            std::vector<R::Array<real>::ReadLease> leases;
            std::optional<Algebra::DofComponentLayout<const real>> layout;
            SampleRead(const EquationDofs &prototype, const std::vector<R::Array<real>> &arrays, index rows)
            {
                leases.reserve(arrays.size());
                for (const auto &array : arrays)
                    leases.emplace_back(array.Read());
                std::vector<Algebra::DofComponentColumn<const real>> columns;
                columns.reserve(size_t(prototype.ComponentNumber()));
                for (index iBlock = 0; iBlock < prototype.BlockNumber(); ++iBlock)
                {
                    const auto location = prototype.LocateBlock(iBlock);
                    const auto parent = leases[location.physicalField].View<R::UniformTableRows>();
                    for (index j = 0; j < location.componentNumber; ++j)
                        columns.push_back({parent, {}, location.firstComponent + j});
                }
                layout.emplace(std::move(columns), rows, 0);
            }
            auto View() const { return layout->View(); }
        };
    }
    struct FiniteVolume::PositivityWorkspace::State
    {
        std::shared_ptr<const Data> owner;
        EquationDofs prototype;
        ThermalPolicy policy;
        RANSReconstructionView rans;
        index ransFirst = -1;
        std::vector<R::Array<real>> higherValues, baseValues;
        std::optional<EquationDofs> reference;
        R::Array<index> degree;
        R::Array<real> beta, recoveryScale;
        R::Array<uint8_t> recover;
        index limited = 0;
        real minimum = 1;
        State(std::shared_ptr<const Data> data, const EquationDofs &input, const EquationDofs &coefficients,
              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
            : owner(std::move(data)), prototype(input.NumericAlias()),
              policy(owner->equation, owner->settings.numerics.densityFloor, owner->settings.numerics.temperatureFloor,
                     owner->settings.numerics.sensibleEnergyFloor, owner->settings.numerics.pressureFloor,
                     workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}),
              degree(R::RowLayout::Dense(owner->metrics.NumCell(), 1)),
              beta(degree.Layout()), recoveryScale(degree.Layout()), recover(degree.Layout())
        {
            DNDS_check_throw_info(owner->positivitySamples, "equation PP sampling geometry is unavailable");
            if (owner->equation.HasRANS())
            {
                rans = ResolveRANSReconstructionPolicy(owner->equation);
                index first = 0;
                for (const auto &block : owner->equation.Schema().blocks)
                {
                    if (block.name == "rans")
                        ransFirst = first;
                    first += index(block.components.size());
                }
                DNDS_assert(ransFirst >= 0);
            }
            for (index iField = 0; iField < input.PhysicalFieldNumber(); ++iField)
                higherValues.emplace_back(FiniteVolumeDetail::FieldLayout(input.PhysicalField(iField).Owned().Layout(), owner->positivitySamples->Size(), index(input.PhysicalField(iField).Width())));
            if (!owner->settings.numerics.positivityCompressToMean)
            {
                reference.emplace(EquationDofOperations::NewVector(coefficients));
                for (const auto &array : higherValues)
                    baseValues.emplace_back(array.Layout());
            }
        }
        void Sample(const EquationDofs &input, const EquationDofs &coefficients, std::vector<R::Array<real>> &values) const
        {
            for (index iField = 0; iField < input.PhysicalFieldNumber(); ++iField)
                owner->positivitySamples->EvaluateValues(input.PhysicalField(iField), coefficients.PhysicalField(iField), values[iField]);
        }
    };
    FiniteVolume::PositivityWorkspace::PositivityWorkspace(std::shared_ptr<const Data> owner,
                                                           const EquationDofs &prototype, const EquationDofs &coefficients, std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
        : state(std::make_shared<State>(std::move(owner), prototype, coefficients, std::move(workers))) {}
    std::pair<index, real> FiniteVolume::PositivityWorkspace::LocalStatistics() const { return {state->limited, state->minimum}; }

    void FiniteVolume::PositivityWorkspace::Apply(const EquationDofs &input, const EquationDofs &unlimited, EquationDofs &flux, bool refresh)
    {
        auto &data = *state;
        const auto &n = data.owner->settings.numerics;
        DNDS_check_throw_info(data.owner->reconstruction, "polynomial positivity requires variational reconstruction");
        if (refresh)
            EquationDofOperations::Copy(flux, unlimited);
        const auto meanScope = input.ReadOwned();
        const auto meanLayout = meanScope.ComponentLayout();
        const auto offsetRead = data.owner->stencil.positivityOffsets.Read();
        CPU::PolynomialPositivityRequest request;
        request.means = meanLayout.View();
        request.offsets = offsetRead.View<R::FixedRows<1>>();
        request.bounds = data.policy.View();
        request.rans = data.rans;
        request.ransFirst = data.ransFirst;
        request.modes = data.owner->reconstruction->NRecDOF();
        request.maxOrder = n.cfv.maxOrder;
        request.betaPower = n.positivityBetaPower;
        request.betaCutoff = n.positivityBetaCutoff;
        request.clippedPolicy = n.positivityClippedReferencePolicy == "backtracking"
                                    ? ReconstructionClippedReferencePolicy::Backtracking
                                    : ReconstructionClippedReferencePolicy::FallbackToBase;
        if (n.positivityReconstruction == "Segment")
        {
            DNDS_check_throw_info(!data.owner->equation.HasMixture() && !data.owner->equation.HasRANS() &&
                                      data.prototype.ComponentNumber() == 5,
                                  "Segment reconstruction PP is a flow5 compatibility policy");
            request.segmentBounds = data.owner->FixedFlowThermalBounds();
        }
        const index points = data.owner->positivitySamples->Size();
        RequireNumericalSuccess(CPU::ValidateReconstructionMeans(request), "ValidateReconstructionMeans");
        data.Sample(input, flux, data.higherValues);
        bool allGood;
        {
            const SampleRead read(data.prototype, data.higherValues, points);
            auto write = data.degree.Write();
            allGood = RequireNumericalSuccess(CPU::MarkAdmissibleDegrees(request, read.View(), write.View<R::FixedRows<1>>()), "MarkAdmissibleDegrees");
        }
        if (allGood && !request.segmentBounds)
        {
            data.limited = 0;
            data.minimum = 1;
            return;
        }
        if (data.reference)
        {
            EquationDofOperations::Copy(*data.reference, flux);
            for (int order = n.cfv.maxOrder - 1; order >= 0; --order)
            {
                const int keep = order ? CFV::Math::PolynomialSize(data.owner->metrics.Dimension(), order) - 1 : 0;
                {
                    const auto status = data.degree.Read();
                    auto write = data.reference->WriteOwned();
                    const auto layout = write.ComponentLayout();
                    CPU::TruncateReferenceModes(request, status.View<R::FixedRows<1>>(), layout.View(), keep);
                }
                data.Sample(input, *data.reference, data.baseValues);
                const SampleRead read(data.prototype, data.baseValues, points);
                const auto referenceScope = data.reference->ReadOwned();
                const auto referenceLayout = referenceScope.ComponentLayout();
                auto degreeWrite = data.degree.Write();
                auto write = flux.WriteOwned();
                const auto layout = write.ComponentLayout();
                CPU::SelectReferenceDegree(request, read.View(), referenceLayout.View(),
                                           degreeWrite.View<R::FixedRows<1>>(), layout.View(), order);
            }
            data.Sample(input, flux, data.higherValues);
        }
        auto betaWrite = data.beta.Write();
        auto recoverWrite = data.recover.Write();
        auto scaleWrite = data.recoveryScale.Write();
        const auto beta = betaWrite.View<R::FixedRows<1>>();
        const auto recover = recoverWrite.View<R::FixedRows<1>>();
        const auto recoveryScale = scaleWrite.View<R::FixedRows<1>>();
        {
            const SampleRead read(data.prototype, data.higherValues, points);
            std::optional<SampleRead> baseRead;
            std::optional<EquationDofs::ReadScope> referenceScope;
            std::optional<Algebra::DofComponentLayout<const real>> referenceLayout;
            if (data.reference)
            {
                baseRead.emplace(data.prototype, data.baseValues, points);
                referenceScope.emplace(data.reference->ReadOwned());
                referenceLayout.emplace(referenceScope->ComponentLayout());
            }
            auto write = flux.WriteOwned();
            const auto layout = write.ComponentLayout();
            const auto degreeRead = data.degree.Read();
            RequireNumericalSuccess(CPU::CompressPolynomial(request, read.View(),
                                                            baseRead ? std::make_optional(baseRead->View()) : std::nullopt,
                                                            referenceLayout ? std::make_optional(referenceLayout->View()) : std::nullopt,
                                                            degreeRead.View<R::FixedRows<1>>(), layout.View(), beta),
                                    "CompressPolynomial");
        }
        // Sampling remains an independently owned CFV operator. Numerical
        // acceptance and coefficient updates are separate pure CPU leaves.
        for (int iteration = 0; iteration < 1000; ++iteration)
        {
            data.Sample(input, flux, data.higherValues);
            bool any;
            {
                const SampleRead read(data.prototype, data.higherValues, points);
                any = RequireNumericalSuccess(CPU::MarkPolynomialRecovery(request, read.View(), beta, recover, recoveryScale), "MarkPolynomialRecovery");
            }
            if (!any)
                break;
            std::optional<EquationDofs::ReadScope> referenceScope;
            std::optional<Algebra::DofComponentLayout<const real>> referenceLayout;
            if (data.reference)
            {
                referenceScope.emplace(data.reference->ReadOwned());
                referenceLayout.emplace(referenceScope->ComponentLayout());
            }
            auto write = flux.WriteOwned();
            const auto layout = write.ComponentLayout();
            CPU::RecoverPolynomial(request, referenceLayout ? std::make_optional(referenceLayout->View()) : std::nullopt,
                                   layout.View(), recover.ConstView(), recoveryScale.ConstView());
        }
        data.Sample(input, flux, data.higherValues);
        {
            const SampleRead read(data.prototype, data.higherValues, points);
            RequireNumericalSuccess(CPU::VerifyReconstructedPoints(request, read.View()), "VerifyReconstructedPoints");
        }
        const auto statistics = CPU::PolynomialPositivityStatistics(beta.ConstView());
        data.limited = statistics.first;
        data.minimum = statistics.second;
    }
}
