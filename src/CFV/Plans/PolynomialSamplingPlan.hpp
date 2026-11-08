#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include <array>

namespace DNDS::CFV
{
    /// Caller-owned point values and physical gradients. Null entries omit an
    /// output; gradient slots refer to mesh-reference Cartesian directions.
    struct PolynomialSampleOutputs
    {
        Runtime::Array<real> *values = nullptr;
        std::array<Runtime::Array<real> *, 3> gradients{};
    };

    /// Immutable point-sampling cache: copies local cell IDs and computes all
    /// Taylor basis values/derivatives under the reconstruction cache policy. Points are
    /// snapshotted, not borrowed.
    /// Copies share these Runtime buffers; geometry/order changes invalidate
    /// the retained reconstruction plan. A moved interface needs a new cache.
    /// Outputs remain in each donor's reference/component frame.
    class PolynomialSamplingPlan
    {
        VariationalReconstructionPlan _reconstruction;
        Runtime::Array<index> _cells;
        Runtime::Array<real> _points;
        Runtime::Array<real> _basis;
        int _cachedDerivatives = 0;
        bool _needsGhost = false;
        bool _cacheDerivatives = true;

    public:
        /// cacheDerivatives=false permits value queries only (e.g. PP).
        /// Otherwise all derivatives may be queried; orders outside the selected
        /// cache coverage are evaluated on demand. Global cache-off stores none.
        PolynomialSamplingPlan(VariationalReconstructionPlan reconstruction,
                               const Runtime::Array<index> &cells, const Runtime::Array<real> &points, bool cacheDerivatives = true);
        index Size() const { return _cells.RowCount(); }
        int CachedDerivativeNumber() const { return _cachedDerivatives; }
        void Validate() const { _reconstruction.Validate(); }
        /// derivative is the central Taylor derivative-mode ordinal. Zero
        /// samples value; higher ordinals sample derivatives without the mean.
        /// Only caches containing ghost cells acquire input ghost leases.
        void Evaluate(const FVField &mean, const FVField &coefficients, Runtime::Array<real> &output, int derivative = 0) const;
        /// Value-only counterpart with the same shape, placement, geometry and
        /// lease contract. Common cached owned AoS queries use bounded CPU
        /// kernels; all other queries call Evaluate(..., 0). Keeping this
        /// separate leaves derivative evaluation independent of specialization.
        void EvaluateValues(const FVField &mean, const FVField &coefficients, Runtime::Array<real> &output) const;
        /// Synchronous CPU sampling of values and/or first derivatives. All
        /// supplied outputs must have Size() rows and mean.Width() components,
        /// with no input/output or output/output overlap. At least one output
        /// is required. No output allocation, halo exchange or hidden layout conversion
        /// occurs. Only ghost queries acquire input ghost leases. On failure,
        /// discard the output bundle; publication belongs to the caller.
        void EvaluateValueGradient(const FVField &mean, const FVField &coefficients, const PolynomialSampleOutputs &outputs) const;
    };
}
