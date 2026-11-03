#include "CFV/Plans/LimitedVRPlan.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    void LimitedVRPlan::ValidateRegion(ReconstructionRegion region) const
    {
        _metrics.Validate();
        _partition.Validate();
        DNDS_check_throw_info(region == ReconstructionRegion::All || region == ReconstructionRegion::Core || region == ReconstructionRegion::HaloDependent,
                              "invalid LVR work region");
    }
    void LimitedVRPlan::BuildExtremumSeeds(const FVField &meanScalars, const FVField &baseGates, const R::Array<real> &previous,
                                           FVField &output, const LimitedVRSettings &settings, ReconstructionRegion region) const
    {
        ValidateRegion(region);
        settings.Validate();
        _metrics.ValidateField(meanScalars, 2);
        _metrics.ValidateField(baseGates, 3);
        _metrics.ValidateField(output, 3);
        DNDS_check_throw_info(previous.GetDevice() == R::Device::Host() && !previous.Layout().IsVariable() && previous.RowCount() == _metrics.NumCell() && previous.Layout().Width() == 2,
                              "LVR previous-gate shape or placement differs");
        for (const auto *source : {&meanScalars.Owned(), &meanScalars.Ghost(), &baseGates.Owned(), &baseGates.Ghost(), &previous})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(source->Storage(), output.Owned().Storage()), "LVR seed output aliases input");
        auto scalarRead = meanScalars.Owned().Read(), gateRead = baseGates.Owned().Read(), history = previous.Read();
        std::optional<R::Array<real>::ReadLease> scalarGhost, gateGhost;
        if (region != ReconstructionRegion::Core)
            scalarGhost.emplace(meanScalars.Ghost().Read()), gateGhost.emplace(baseGates.Ghost().Read());
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> scalar{scalarRead.View<R::UniformTableRows>(), scalarGhost ? scalarGhost->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{}};
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> gate{gateRead.View<R::UniformTableRows>(), gateGhost ? gateGhost->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{}};
        const auto prior = history.View<R::UniformTableRows>();
        auto offsets = _offsets.Read(), neighbors = _neighbors.Read(), boundary = _boundary.Read();
        const auto o = offsets.View<R::FixedRows<1>>(), n = neighbors.View<R::FixedRows<1>>(), b = boundary.View<R::FixedRows<1>>();
        auto write = output.Owned().Write();
        const auto result = write.View<R::UniformTableRows>();
        ForCells(region, [&](index iCell)
                 {
            std::array<Math::LimitedVRExtremumStencil, 2> stencil;
            for (auto &entry : stencil) entry.boundary = b(iCell, 0);
            for (index entry = o(iCell, 0); entry < o(iCell + 1, 0); ++entry)
                if (const auto other = n(entry, 0); other != UnInitIndex)
                {
                    stencil[0].Observe(scalar(iCell, 0), scalar(other, 0), gate(other, 0), settings.equalityTolerance);
                    stencil[1].Observe(scalar(iCell, 1), scalar(other, 1), gate(other, 2), settings.equalityTolerance);
                }
            result(iCell, 0) = stencil[0].Correction(scalar(iCell, 0), prior(iCell, 0), settings.alphaMax, settings.pressureSupport, settings);
            result(iCell, 1) = settings.variant == "v3" ? stencil[1].Correction(scalar(iCell, 1), prior(iCell, 1), settings.contactAlphaMax, settings.contactSupport, settings) : 0;
            result(iCell, 2) = 0; });
    }
    void LimitedVRPlan::NeighborMaximum(const FVField &input, FVField &output, std::array<bool, 3> active,
                                        bool excludeBoundary, bool saturatedOnly, std::array<real, 3> caps, ReconstructionRegion region) const
    {
        ValidateRegion(region);
        _metrics.ValidateField(input, 3);
        _metrics.ValidateField(output, 3);
        for (auto cap : caps)
            DNDS_check_throw_info(std::isfinite(cap) && cap >= 0 && cap <= 1, "invalid LVR saturation cap");
        for (const auto *source : {&input.Owned(), &input.Ghost()})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(source->Storage(), output.Owned().Storage()), "LVR maximum output aliases input");
        auto owned = input.Owned().Read();
        std::optional<R::Array<real>::ReadLease> ghost;
        if (region != ReconstructionRegion::Core)
            ghost.emplace(input.Ghost().Read());
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> source{owned.View<R::UniformTableRows>(), ghost ? ghost->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{}};
        auto offsets = _offsets.Read(), neighbors = _neighbors.Read(), boundary = _boundary.Read();
        const auto o = offsets.View<R::FixedRows<1>>(), n = neighbors.View<R::FixedRows<1>>(), b = boundary.View<R::FixedRows<1>>();
        auto write = output.Owned().Write();
        const auto result = write.View<R::UniformTableRows>();
        ForCells(region, [&](index iCell)
                 {
            for (int column = 0; column < 3; ++column) result(iCell, column) = source(iCell, column);
            if (excludeBoundary && b(iCell, 0)) return;
            for (index entry = o(iCell, 0); entry < o(iCell + 1, 0); ++entry)
                if (const auto other = n(entry, 0); other != UnInitIndex)
                    for (int column = 0; column < 3; ++column)
                        if (active[column])
                        {
                            const real value = source(other, column);
                            const real candidate = !saturatedOnly ? value : value >= caps[column] * (1 - 1e-12) ? caps[column] : 0;
                            result(iCell, column) = std::max(result(iCell, column), candidate);
                        } });
    }
}
