#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::CFV
{
    namespace Detail
    {
        struct GradientIdentity
        {
        };
    }
    enum class GradientMethod
    {
        GreenGauss,
        LeastSquares,
        GGMP
    };

    /// Optional equation-variable transform for one-level physical-gradient
    /// WBAP. Spatial gradient-frame rotation remains a separate mesh operation.
    /// `count == 0` is a valid disabled descriptor and owns no matrix buffers.
    struct GradientFaceCharacteristicTransform
    {
        index first = 0, count = 0;
        Runtime::Array<real> forward, inverse;
        std::shared_ptr<const Detail::GradientIdentity> plan;
        uint64_t forwardVersion = 0, inverseVersion = 0;
        uint64_t forwardIdentity = 0, inverseIdentity = 0;
    };
    using GradientFaceCharacteristicTransformRef =
        std::optional<std::reference_wrapper<const GradientFaceCharacteristicTransform>>;

    /// Independent cell-centered FV gradient preparation. BC inputs are physical
    /// face-quadrature values, not reflected ghost states. LS minimizes residuals
    /// with inverse-distance-squared weights using the design matrix SVD (not
    /// normal equations). GGMP eliminates face values from the coupled corrected
    /// Green-Gauss equations. No legacy VR owner or boundary callback is used.
    class GradientPlan
    {
        struct Data
        {
            std::shared_ptr<const Detail::GradientIdentity> identity =
                std::make_shared<const Detail::GradientIdentity>();
            FiniteVolumePlan metrics;
            Geom::EntityWorkPartition partition;
            Runtime::Array<index> offsets, neighbors, faces, boundaryIds;
            Runtime::Array<uint8_t> rotationActive;
            Runtime::Array<real> weights, rotations;
            bool hasRotatedFrames = false;
            std::vector<BoundaryQuadraturePoint> boundary;
            Data(FiniteVolumePlan plan) : metrics(std::move(plan)), partition(metrics.CellWorkPartition()) {}
        };
        std::shared_ptr<const Data> _data;
        explicit GradientPlan(std::shared_ptr<const Data> data) : _data(std::move(data)) {}

    public:
        static GradientPlan Prepare(Geom::UnstructuredMesh &mesh, FiniteVolumePlan metrics,
                                    GradientMethod method = GradientMethod::LeastSquares, real relativeCutoff = 1e-13);
        index NumBoundaryQuadrature() const;
        BoundaryQuadraturePoint GetBoundaryQuadrature(index iPoint) const;
        Runtime::Array<real> BuildBoundaryValues(index components) const;
        GradientFaceCharacteristicTransform BuildFaceTransform(index first, index count) const;
        void FinalizeFaceTransform(GradientFaceCharacteristicTransform &transform) const;
        /// Convert owned gradients to linear Taylor coefficients and zero all
        /// higher modes. Zero-mean basis moments preserve cell averages. This
        /// is an explicit overwrite, not a limiter or second reconstruction.
        void ConvertToCoefficients(const FVField &gradient, const VariationalReconstructionPlan &reconstruction,
                                   FVField &coefficients) const;
        /// Explicit compatibility projection of linear modes into physical
        /// gradients, without basis evaluation. Higher modes are ignored: the
        /// caller must supply a linear polynomial, e.g. the accepted direct-FV
        /// limiter result. ghost=false overwrites owned rows; ghost=true only
        /// ghost rows, after the coefficient exchange has completed. No MPI.
        void ExtractLinearGradients(const FVField &coefficients, const VariationalReconstructionPlan &reconstruction,
                                    FVField &gradient, bool ghost = false) const;
        /// Owned output rows hold [component*dimension+direction]. Core reads
        /// owned means only, so an input ghost write may be active. Complete
        /// the input halo before HaloDependent/All. BC values must be ready.
        /// Partial calls overwrite only their selected output rows; the caller
        /// completes both phases before consuming a full gradient. Calls sharing
        /// an output are serialized. Output must be distinct from all inputs.
        void Evaluate(const FVField &mean, const Runtime::Array<real> &boundaryValues, FVField &gradient,
                      const std::vector<index> &periodicVectorStarts = {}, ReconstructionRegion region = ReconstructionRegion::All) const;
        /// Apply one physical-gradient 3WBAP level. Neighbor gradients are first
        /// mapped into the center cell's periodic spatial/component frame. A
        /// nonempty characteristic range is transformed per spatial direction;
        /// all remaining components are scalar-limited. Output owned rows are
        /// replaced and remain halo-stale. No polynomial modes or secondary
        /// reconstruction matrices are constructed.
        void LimitWBAP3(const FVField &input, FVField &output,
                        const std::vector<index> &periodicVectorStarts = {},
                        GradientFaceCharacteristicTransformRef transform = std::nullopt,
                        ReconstructionRegion region = ReconstructionRegion::All,
                        LimiterIndicatorRef indicator = std::nullopt,
                        real threshold = 0, real centerWeight = 10) const;
    };
}
