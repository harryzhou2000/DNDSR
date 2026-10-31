#pragma once

#include "FiniteVolumePlan.hpp"

namespace DNDS::CFV
{
    /// Immutable point-sampling cache: copies local cell IDs and computes all
    /// Taylor basis derivatives once. Points are snapshotted, not borrowed.
    /// Copies share these Runtime buffers; geometry/order changes invalidate
    /// the retained reconstruction plan. A moved interface needs a new cache.
    /// Outputs remain in each donor's reference/component frame.
    class PolynomialSamplingPlan
    {
        VariationalReconstructionPlan _reconstruction;
        Runtime::Array<index> _cells;
        Runtime::Array<real> _basis;
        bool _needsGhost = false;

    public:
        PolynomialSamplingPlan(VariationalReconstructionPlan reconstruction,
                               const Runtime::Array<index> &cells, const Runtime::Array<real> &points);
        index Size() const { return _cells.RowCount(); }
        void Validate() const { _reconstruction.Validate(); }
        /// derivative is the central Taylor derivative-mode ordinal. Zero
        /// samples value; higher ordinals sample derivatives without the mean.
        /// Only caches containing ghost cells acquire input ghost leases.
        void Evaluate(const FVField &mean, const FVField &coefficients, Runtime::Array<real> &output, int derivative = 0) const;
    };
}
