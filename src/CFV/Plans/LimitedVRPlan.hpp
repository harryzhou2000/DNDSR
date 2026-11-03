#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Math/LimitedVR.hpp"

namespace DNDS::CFV
{
    /// Immutable face-incidence stencil for scalar LVR gates. No EOS, target
    /// construction, communication or retained iteration history is hidden here.
    /// Copies share immutable metadata buffers and the metric/mesh publication.
    class LimitedVRPlan
    {
        FiniteVolumePlan _metrics;
        Geom::EntityWorkPartition _partition;
        Runtime::Array<index> _offsets, _neighbors, _boundary;
        explicit LimitedVRPlan(FiniteVolumePlan metrics) : _metrics(std::move(metrics)), _partition(_metrics.CellWorkPartition()) {}
        template <class Function>
        void ForCells(ReconstructionRegion region, Function &&function) const
        {
            if (region == ReconstructionRegion::All)
                for (index iCell = 0; iCell < _metrics.NumCell(); ++iCell)
                    function(iCell);
            else
                for (auto iCell : region == ReconstructionRegion::Core ? _partition.Core() : _partition.HaloDependent())
                    function(iCell);
        }
        void ValidateRegion(ReconstructionRegion region) const;

    public:
        static LimitedVRPlan Prepare(FiniteVolumePlan metrics);
        /// Mean scalars are [p,rho]; base gates [shock,contact,mean-contact
        /// support]. Previous is an owned table of [shock,contact] from the last
        /// RHS input. Output is [pressure seed,density seed,0]. Only v3 adds the
        /// density seed. Merge base gates after expanding each seed's own depth.
        /// No MPI: caller completes base/input halos before dependent/All work.
        void BuildExtremumSeeds(const FVField &meanScalars, const FVField &baseGates,
                                const Runtime::Array<real> &previous, FVField &output,
                                const LimitedVRSettings &settings, ReconstructionRegion region = ReconstructionRegion::All) const;
        /// One synchronous, out-of-place interior-face maximum round on three
        /// scalar gate columns. Inactive columns copy unchanged. Correction
        /// seeds exclude boundary-adjacent receivers; extra shock layers do not.
        /// Saturation filtering uses caps[column]*(1-1e-12), preserving each
        /// local unsaturated value. Input halos must be current for dependent/
        /// All; Core reads owned only. Caller finishes both phases before copy
        /// or the next exchange. Distinct output; no allocation in cell loops.
        void NeighborMaximum(const FVField &input, FVField &output, std::array<bool, 3> active,
                             bool excludeBoundary, bool saturatedOnly = false, std::array<real, 3> caps = {1, 1, 1},
                             ReconstructionRegion region = ReconstructionRegion::All) const;
    };
}
