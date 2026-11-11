#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
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
        EquationThermalPolicy policy;
        RANSReconstructionView rans;
        index ransFirst = -1;
        std::vector<R::Array<real>> higherValues, baseValues;
        std::optional<EquationDofs> reference;
        R::Array<index> degree;
        index limited = 0;
        real minimum = 1;
        State(std::shared_ptr<const Data> data, const EquationDofs &input, const EquationDofs &coefficients,
              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
            : owner(std::move(data)), prototype(input.NumericAlias()),
              policy(owner->equation, owner->settings.numerics.densityFloor, owner->settings.numerics.temperatureFloor,
                     owner->settings.numerics.sensibleEnergyFloor, owner->settings.numerics.pressureFloor,
                     workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}),
              degree(R::RowLayout::Dense(owner->metrics.NumCell(), 1))
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
        template <class Values>
        bool ReferenceGood(const Values &values, index q, const EquationThermalView &bounds) const
        {
            return bounds.Admissible(EquationPositivityDetail::PointRow<Values>{values, q}, false);
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
        const auto bounds = data.policy.View();
        const index cells = data.owner->metrics.NumCell(), points = data.owner->positivitySamples->Size();
        const int modes = data.owner->reconstruction.NRecDOF();
        if (refresh)
            EquationDofOperations::Copy(flux, unlimited);
        const auto meanScope = input.ReadOwned();
        const auto meanLayout = meanScope.ComponentLayout();
        const auto means = meanLayout.View();
        for (index iCell = 0; iCell < cells; ++iCell)
            DNDS_check_throw_info(bounds.Admissible(EquationPositivityDetail::PointRow<decltype(means)>{means, iCell}), "equation reconstruction PP requires admissible means");
        data.Sample(input, flux, data.higherValues);
        bool allGood = true;
        {
            const SampleRead read(data.prototype, data.higherValues, points);
            const auto values = read.View();
            for (index q = 0; q < points; ++q)
                for (index j = 0; j < values.ComponentNumber(); ++j)
                    DNDS_check_throw_info(std::isfinite(values(q, j)), "nonfinite equation reconstruction PP input");
            auto write = data.degree.Write();
            const auto degrees = write.View<R::FixedRows<1>>();
            for (index iCell = 0; iCell < cells; ++iCell)
            {
                bool good = true;
                for (index q = data.owner->positivityOffsets[iCell]; q < data.owner->positivityOffsets[iCell + 1]; ++q)
                    good = good && data.ReferenceGood(values, q, bounds);
                degrees(iCell, 0) = good ? n.cfv.maxOrder : -1;
                allGood = allGood && good;
                if (allGood && data.rans.components)
                {
                    struct RepeatedMean
                    {
                        decltype(means) values;
                        index cell;
                        real operator()(index, index component) const { return values(cell, component); }
                    } base{means, iCell};
                    // Flow admissibility intentionally excludes RANS. The
                    // fast exit must also prove its independent scalar repair
                    // unnecessary; SA's compile-time no-compression policy is
                    // already implemented by Compute returning one.
                    allGood = data.rans.Compute(base, values, data.ransFirst,
                                                data.owner->positivityOffsets[iCell], data.owner->positivityOffsets[iCell + 1]) == 1;
                }
            }
        }
        // Same full-polynomial early exit as the per-cell Euler policy. The
        // actual coefficients generated these samples and remain unchanged;
        // no write scope, repeat sampling or publication is needed. This is
        // local only: the enclosing operator owns collective ordering.
        if (allGood && n.positivityReconstruction != "Segment")
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
                    const auto degrees = status.View<R::FixedRows<1>>();
                    auto write = data.reference->WriteOwned();
                    const auto layout = write.ComponentLayout();
                    const auto coefficients = layout.View();
                    for (index iCell = 0; iCell < cells; ++iCell)
                        if (degrees(iCell, 0) < 0)
                            for (index j = 0; j < input.ComponentNumber(); ++j)
                                if (bounds.CompressesReconstructionComponent(j))
                                    for (int mode = keep; mode < modes; ++mode)
                                        coefficients(iCell, j * modes + mode) = 0;
                }
                data.Sample(input, *data.reference, data.baseValues);
                const SampleRead read(data.prototype, data.baseValues, points);
                const auto values = read.View();
                const auto referenceScope = data.reference->ReadOwned();
                const auto referenceLayout = referenceScope.ComponentLayout();
                const auto reference = referenceLayout.View();
                auto degreeWrite = data.degree.Write();
                const auto degrees = degreeWrite.View<R::FixedRows<1>>();
                auto write = flux.WriteOwned();
                const auto layout = write.ComponentLayout();
                const auto coefficients = layout.View();
                for (index iCell = 0; iCell < cells; ++iCell)
                    if (degrees(iCell, 0) < 0)
                    {
                        bool good = true;
                        for (index q = data.owner->positivityOffsets[iCell]; q < data.owner->positivityOffsets[iCell + 1]; ++q)
                            good = good && data.ReferenceGood(values, q, bounds);
                        if (good)
                            degrees(iCell, 0) = order;
                        else
                            for (index component = 0; component < input.ComponentNumber(); ++component)
                                if (bounds.CompressesReconstructionComponent(component))
                                    for (int mode = 0; mode < modes; ++mode)
                                        coefficients(iCell, component * modes + mode) = reference(iCell, component * modes + mode);
                    }
            }
            data.Sample(input, flux, data.higherValues);
        }
        data.limited = 0;
        data.minimum = 1;
        {
            const SampleRead read(data.prototype, data.higherValues, points);
            const auto higher = read.View();
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
            const auto coefficients = layout.View();
            const auto degreeRead = data.degree.Read();
            const auto degrees = degreeRead.View<R::FixedRows<1>>();
            for (index iCell = 0; iCell < cells; ++iCell)
            {
                const EquationPositivityDetail::PointRow<decltype(means)> mean{means, iCell};
                struct MeanValues
                {
                    decltype(mean) value;
                    real operator()(index, index j) const { return value[j]; }
                } base{mean};
                const index begin = data.owner->positivityOffsets[iCell], end = data.owner->positivityOffsets[iCell + 1];
                // RANS was excluded from every flow base test/write above.
                // Correct its original polynomial about the mean even when
                // flow is already admissible and will take the early exit.
                const real scale = data.rans.Compute(base, higher, data.ransFirst, begin, end);
                DNDS_check_throw_info(std::isfinite(scale) && scale >= 0 && scale <= 1, "invalid RANS reconstruction compression");
                if (scale < 1)
                    for (int component = 0; component < data.rans.components; ++component)
                        for (int mode = 0; mode < modes; ++mode)
                            coefficients(iCell, (data.ransFirst + component) * modes + mode) *= scale;
                if (degrees(iCell, 0) == n.cfv.maxOrder)
                    continue;
                real beta = 1;
                if (n.positivityReconstruction == "Segment")
                {
                    DNDS_check_throw_info(!data.owner->equation.HasMixture() && !data.owner->equation.HasRANS() &&
                                              data.prototype.ComponentNumber() == 5,
                                          "Segment reconstruction PP is a flow5 compatibility policy");
                    const auto flowMean = bounds.Flow(mean);
                    for (index q = begin; q < end; ++q)
                        beta = std::min(beta, AdmissibleFraction(flowMean, bounds.Flow(EquationPositivityDetail::PointRow<decltype(higher)>{higher, q}),
                                                                 data.owner->PerfectGasBounds()));
                    if (beta < n.positivityBetaCutoff)
                        beta = 0;
                }
                else
                    beta = baseRead ? bounds.ReconstructionFraction(mean, baseRead->View(), higher, begin, end, n.positivityBetaPower, n.positivityBetaCutoff)
                                    : bounds.ReconstructionFraction(mean, base, higher, begin, end, n.positivityBetaPower, n.positivityBetaCutoff);
                DNDS_check_throw_info(std::isfinite(beta) && beta >= 0 && beta <= 1, "invalid equation reconstruction PP beta");
                if (beta < 1)
                {
                    ++data.limited;
                    data.minimum = std::min(data.minimum, beta);
                    for (index component = 0; component < input.ComponentNumber(); ++component)
                        if (bounds.CompressesReconstructionComponent(component))
                            for (int mode = 0; mode < modes; ++mode)
                            {
                                const index j = component * modes + mode;
                                const real reference = referenceLayout ? referenceLayout->View()(iCell, j) : 0;
                                coefficients(iCell, j) = reference + beta * (coefficients(iCell, j) - reference);
                            }
                }
            }
        }
        // Validate the actual published coefficients, including roundoff and
        // physical-layout indexing, rather than only the algebraic point blend.
        data.Sample(input, flux, data.higherValues);
        const SampleRead read(data.prototype, data.higherValues, points);
        const auto values = read.View();
        for (index q = 0; q < points; ++q)
            DNDS_check_throw_info(bounds.Admissible(EquationPositivityDetail::PointRow<decltype(values)>{values, q}, false), "published equation reconstruction violates its thermal bounds");
    }
}
