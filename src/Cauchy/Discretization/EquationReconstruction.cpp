#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
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
    struct FiniteVolume::EquationPositivityWorkspace::State
    {
        std::shared_ptr<const Data> owner;
        EquationDofs prototype;
        EquationThermalPolicy policy;
        std::vector<R::Array<real>> higherValues, baseValues;
        std::optional<EquationDofs> reference;
        std::optional<R::Array<index>> degree;
        index limited = 0;
        real minimum = 1;
        State(std::shared_ptr<const Data> data, const EquationDofs &input, const EquationDofs &coefficients,
              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
            : owner(std::move(data)), prototype(input.NumericAlias()),
              policy(owner->equation, owner->settings.numerics.densityFloor, owner->settings.numerics.temperatureFloor,
                     owner->settings.numerics.sensibleEnergyFloor, owner->settings.numerics.pressureFloor,
                     workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{})
        {
            DNDS_check_throw_info(owner->positivitySamples, "equation PP sampling geometry is unavailable");
            for (index iField = 0; iField < input.PhysicalFieldNumber(); ++iField)
                higherValues.emplace_back(FiniteVolumeDetail::FieldLayout(input.PhysicalField(iField).Owned().Layout(), owner->positivitySamples->Size(), index(input.PhysicalField(iField).Width())));
            if (!owner->settings.numerics.positivityCompressToMean)
            {
                reference.emplace(EquationDofOperations::NewVector(coefficients));
                degree.emplace(R::RowLayout::Dense(owner->metrics.NumCell(), 1));
                for (const auto &array : higherValues)
                    baseValues.emplace_back(array.Layout());
            }
        }
        void Sample(const EquationDofs &input, const EquationDofs &coefficients, std::vector<R::Array<real>> &values) const
        {
            for (index iField = 0; iField < input.PhysicalFieldNumber(); ++iField)
                owner->positivitySamples->Evaluate(input.PhysicalField(iField), coefficients.PhysicalField(iField), values[iField]);
        }
    };
    FiniteVolume::EquationPositivityWorkspace::EquationPositivityWorkspace(std::shared_ptr<const Data> owner,
                                                                           const EquationDofs &prototype, const EquationDofs &coefficients, std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
        : state(std::make_shared<State>(std::move(owner), prototype, coefficients, std::move(workers))) {}
    std::pair<index, real> FiniteVolume::EquationPositivityWorkspace::LocalStatistics() const { return {state->limited, state->minimum}; }

    void FiniteVolume::EquationPositivityWorkspace::Apply(const EquationDofs &input, const EquationDofs &unlimited, EquationDofs &flux, bool refresh)
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
        {
            const SampleRead read(data.prototype, data.higherValues, points);
            const auto values = read.View();
            for (index q = 0; q < points; ++q)
                for (index j = 0; j < values.ComponentNumber(); ++j)
                    DNDS_check_throw_info(std::isfinite(values(q, j)), "nonfinite equation reconstruction PP input");
        }
        if (data.reference)
        {
            EquationDofOperations::Copy(*data.reference, flux);
            {
                const SampleRead read(data.prototype, data.higherValues, points);
                const auto values = read.View();
                auto write = data.degree->Write();
                const auto degrees = write.View<R::FixedRows<1>>();
                for (index iCell = 0; iCell < cells; ++iCell)
                {
                    bool good = true;
                    for (index q = data.owner->positivityOffsets[iCell]; q < data.owner->positivityOffsets[iCell + 1]; ++q)
                        good = good && bounds.Admissible(EquationPositivityDetail::PointRow<decltype(values)>{values, q});
                    degrees(iCell, 0) = good ? n.cfv.maxOrder : -1;
                }
            }
            for (int order = n.cfv.maxOrder - 1; order >= 0; --order)
            {
                const int keep = order ? CFV::Math::PolynomialSize(data.owner->metrics.Dimension(), order) - 1 : 0;
                {
                    const auto status = data.degree->Read();
                    const auto degrees = status.View<R::FixedRows<1>>();
                    auto write = data.reference->WriteOwned();
                    const auto layout = write.ComponentLayout();
                    const auto coefficients = layout.View();
                    for (index iCell = 0; iCell < cells; ++iCell)
                        if (degrees(iCell, 0) < 0)
                            for (index j = 0; j < input.ComponentNumber(); ++j)
                                for (int mode = keep; mode < modes; ++mode)
                                    coefficients(iCell, j * modes + mode) = 0;
                }
                data.Sample(input, *data.reference, data.baseValues);
                const SampleRead read(data.prototype, data.baseValues, points);
                const auto values = read.View();
                const auto referenceScope = data.reference->ReadOwned();
                const auto referenceLayout = referenceScope.ComponentLayout();
                const auto reference = referenceLayout.View();
                auto degreeWrite = data.degree->Write();
                const auto degrees = degreeWrite.View<R::FixedRows<1>>();
                auto write = flux.WriteOwned();
                const auto layout = write.ComponentLayout();
                const auto coefficients = layout.View();
                for (index iCell = 0; iCell < cells; ++iCell)
                    if (degrees(iCell, 0) < 0)
                    {
                        bool good = true;
                        for (index q = data.owner->positivityOffsets[iCell]; q < data.owner->positivityOffsets[iCell + 1]; ++q)
                            good = good && bounds.Admissible(EquationPositivityDetail::PointRow<decltype(values)>{values, q});
                        if (good)
                            degrees(iCell, 0) = order;
                        else
                            for (index j = 0; j < coefficients.ComponentNumber(); ++j)
                                coefficients(iCell, j) = reference(iCell, j);
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
            for (index iCell = 0; iCell < cells; ++iCell)
            {
                const EquationPositivityDetail::PointRow<decltype(means)> mean{means, iCell};
                struct MeanValues
                {
                    decltype(mean) value;
                    real operator()(index, index j) const { return value[j]; }
                } base{mean};
                const index begin = data.owner->positivityOffsets[iCell], end = data.owner->positivityOffsets[iCell + 1];
                const auto beta = baseRead ? bounds.ReconstructionFraction(mean, baseRead->View(), higher, begin, end, n.positivityBetaPower, n.positivityBetaCutoff)
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
