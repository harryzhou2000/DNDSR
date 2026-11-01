#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::CFV
{
    enum class GradientMethod
    {
        GreenGauss,
        LeastSquares,
        GGMP
    };

    /// Independent cell-centered FV gradient preparation. BC inputs are physical
    /// face-quadrature values, not reflected ghost states. LS minimizes residuals
    /// with inverse-distance-squared weights using the design matrix SVD (not
    /// normal equations). GGMP eliminates face values from the coupled corrected
    /// Green-Gauss equations. No legacy VR owner or boundary callback is used.
    class GradientPlan
    {
        struct Data
        {
            FiniteVolumePlan metrics;
            Runtime::Array<index> offsets, neighbors, boundaryIds;
            Runtime::Array<real> weights, rotations;
            std::vector<BoundaryQuadraturePoint> boundary;
            Data(FiniteVolumePlan plan) : metrics(std::move(plan)) {}
        };
        std::shared_ptr<const Data> _data;
        explicit GradientPlan(std::shared_ptr<const Data> data) : _data(std::move(data)) {}

    public:
        static GradientPlan Prepare(Geom::UnstructuredMesh &mesh, FiniteVolumePlan metrics,
                                    GradientMethod method = GradientMethod::LeastSquares, real relativeCutoff = 1e-13);
        index NumBoundaryQuadrature() const;
        BoundaryQuadraturePoint GetBoundaryQuadrature(index iPoint) const;
        Runtime::Array<real> BuildBoundaryValues(index components) const;
        /// Convert owned gradients to linear Taylor coefficients and zero all
        /// higher modes. Zero-mean basis moments preserve cell averages. This
        /// is an explicit overwrite, not a limiter or second reconstruction.
        void ConvertToCoefficients(const FVField &gradient, const VariationalReconstructionPlan &reconstruction,
                                   FVField &coefficients) const;
        /// Owned output rows hold [component*dimension+direction]. Release input
        /// writers and finish input halo/BC producers first. Output must be
        /// distinct from inputs; ordinary Runtime leases reject alias conflicts.
        void Evaluate(const FVField &mean, const Runtime::Array<real> &boundaryValues, FVField &gradient,
                      const std::vector<index> &periodicVectorStarts = {}) const;
    };
}
