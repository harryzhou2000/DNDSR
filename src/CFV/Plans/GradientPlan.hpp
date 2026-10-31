#pragma once

#include "FiniteVolumePlan.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Access/EntityWorkPartition.hpp"
#include "CFV/Math/PolynomialReconstruction.hpp"

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
            Runtime::Array<real> weights;
            std::vector<BoundaryQuadraturePoint> boundary;
            Data(FiniteVolumePlan plan) : metrics(std::move(plan)) {}
        };
        std::shared_ptr<const Data> _data;
        explicit GradientPlan(std::shared_ptr<const Data> data) : _data(std::move(data)) {}

    public:
        static GradientPlan Prepare(Geom::UnstructuredMesh &mesh, FiniteVolumePlan metrics,
                                    GradientMethod method = GradientMethod::LeastSquares, real relativeCutoff = 1e-13)
        {
            namespace R = Runtime;
            using namespace Geom;
            DNDS_check_throw_info(method == GradientMethod::GreenGauss || method == GradientMethod::LeastSquares || method == GradientMethod::GGMP,
                                  "unknown FV gradient method");
            DNDS_check_throw_info(std::isfinite(relativeCutoff) && relativeCutoff >= 0, "invalid gradient cutoff");
            auto metricRead = metrics.AcquireRead();
            auto meshRead = mesh.AcquireRead(MeshReadRequirements::CellFlux());
            DNDS_check_throw_info(mesh.Entities(EntityKind::Cell).SameOrder(metrics.Cells()), "gradient metrics and mesh orders differ");
            const auto view = meshRead.HostView();
            const auto geometry = metricRead.HostView();
            const int dim = metrics.Dimension();
            // Component/frame rotation policy is explicit future input; scalar
            // geometric periodic handling must not silently rotate vector data.
            DNDS_check_throw_info(!view.IsPeriodic(), "FV gradient periodic component transforms require explicit frame data");
            auto data = std::make_shared<Data>(metrics);
            index entries = 0, boundaryCount = 0;
            for (index iCell = 0; iCell < metrics.NumCell(); ++iCell)
                for (index slot = 0; slot < view.CellFaces(iCell).size(); ++slot)
                {
                    const index face = view.CellFaces(iCell)[slot];
                    const index count = geometry.FaceQuadratureEnd(face) - geometry.FaceQuadratureBegin(face);
                    entries += count;
                    if (view.CellFaceOther(iCell, face, rowsize(slot)) == UnInitIndex)
                        boundaryCount += count;
                }
            data->offsets = R::Array<index>(R::RowLayout::Dense(metrics.NumCell() + 1, 1));
            data->neighbors = R::Array<index>(R::RowLayout::Dense(entries, 1));
            data->boundaryIds = R::Array<index>(R::RowLayout::Dense(entries, 1));
            data->weights = R::Array<real>(R::RowLayout::Dense(entries, 3));
            data->boundary.resize(size_t(boundaryCount));
            auto offsetWrite = data->offsets.Write(), neighborWrite = data->neighbors.Write(), boundaryWrite = data->boundaryIds.Write();
            auto weightWrite = data->weights.Write();
            const auto offsets = offsetWrite.View<R::FixedRows<1>>(), neighbors = neighborWrite.View<R::FixedRows<1>>(), boundaries = boundaryWrite.View<R::FixedRows<1>>();
            const auto weights = weightWrite.View<R::FixedRows<3>>();
            index next = 0, nextBoundary = 0;
            for (index iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                const auto center = geometry.GetCellBary(iCell);
                offsets(iCell, 0) = next;
                index count = 0;
                for (auto face : view.CellFaces(iCell))
                    count += geometry.FaceQuadratureEnd(face) - geometry.FaceQuadratureBegin(face);
                Eigen::MatrixXd design(count, dim), initial(dim, count);
                Eigen::VectorXd lengths(count);
                Eigen::MatrixXd correction = Eigen::MatrixXd::Identity(dim, dim);
                index local = 0;
                for (index slot = 0; slot < view.CellFaces(iCell).size(); ++slot)
                {
                    const index face = view.CellFaces(iCell)[slot];
                    const index other = view.CellFaceOther(iCell, face, rowsize(slot));
                    const bool back = view.CellIsFaceBack(iCell, face, rowsize(slot));
                    for (index q = geometry.FaceQuadratureBegin(face); q < geometry.FaceQuadratureEnd(face); ++q, ++next, ++local)
                    {
                        const auto point = geometry.GetFacePoint(q);
                        const tPoint areaNormal = geometry.GetFaceNorm(q) * ((back ? 1 : -1) * geometry.GetFaceWeight(q) / geometry.GetCellVol(iCell));
                        const tPoint displacement = (other == UnInitIndex ? point : geometry.GetCellBary(other)) - center;
                        const real length = displacement.head(dim).norm();
                        DNDS_check_throw_info(length > 0 && std::isfinite(length), "degenerate FV gradient sample displacement");
                        // Boundary quadrature is multiple physical samples. For
                        // neighbor means repeat the same equation with quadrature
                        // weights so face rule refinement cannot reweight a face.
                        const real quadratureScale = std::sqrt(geometry.GetFaceWeight(q));
                        design.row(local) = displacement.head(dim).transpose() / length * quadratureScale;
                        lengths[local] = quadratureScale / length;
                        neighbors(next, 0) = other;
                        boundaries(next, 0) = UnInitIndex;
                        if (other == UnInitIndex)
                        {
                            boundaries(next, 0) = nextBoundary;
                            data->boundary[nextBoundary++] = {iCell, face, int(q - geometry.FaceQuadratureBegin(face)), point, areaNormal.normalized()};
                        }
                        real fraction = 1;
                        if (other != UnInitIndex)
                        {
                            const real ownDistance = (point - center).norm(), otherDistance = (geometry.GetCellBary(other) - point).norm();
                            fraction = method == GradientMethod::GGMP ? 0.5 : ownDistance / (ownDistance + otherDistance);
                            if (method == GradientMethod::GGMP)
                                correction.noalias() -= areaNormal.head(dim) * (point - (center + geometry.GetCellBary(other)) * 0.5).head(dim).transpose();
                        }
                        initial.col(local) = areaNormal.head(dim) * fraction;
                    }
                }
                Eigen::MatrixXd inverse;
                if (method == GradientMethod::LeastSquares)
                {
                    HardEigen::EigenSVDInverse(design, inverse, relativeCutoff);
                    for (index i = 0; i < count; ++i)
                        initial.col(i) = inverse.col(i) * lengths[i];
                }
                else if (method == GradientMethod::GGMP)
                {
                    HardEigen::EigenSVDInverse(correction, inverse, relativeCutoff);
                    initial = (inverse * initial).eval();
                }
                for (index i = 0; i < count; ++i)
                    for (int d = 0; d < 3; ++d)
                        weights(next - count + i, d) = d < dim ? initial(d, i) : 0;
            }
            offsets(metrics.NumCell(), 0) = next;
            return GradientPlan(std::move(data));
        }
        index NumBoundaryQuadrature() const { return index(_data->boundary.size()); }
        BoundaryQuadraturePoint GetBoundaryQuadrature(index iPoint) const
        {
            _data->metrics.Validate();
            DNDS_check_throw_info(iPoint >= 0 && iPoint < NumBoundaryQuadrature(), "gradient boundary point out of range");
            return _data->boundary[iPoint];
        }
        Runtime::Array<real> BuildBoundaryValues(index components) const
        {
            DNDS_check_throw_info(components > 0, "gradient boundary components must be positive");
            return Runtime::Array<real>(Runtime::RowLayout::Dense(NumBoundaryQuadrature(), components));
        }
        /// Convert owned gradients to linear Taylor coefficients and zero all
        /// higher modes. Zero-mean basis moments preserve cell averages. This
        /// is an explicit overwrite, not a limiter or second reconstruction.
        void ConvertToCoefficients(const FVField &gradient, const VariationalReconstructionPlan &reconstruction,
                                   FVField &coefficients) const
        {
            const index dim = _data->metrics.Dimension();
            DNDS_check_throw_info(gradient.Width() % dim == 0 && gradient.Width() > 0 &&
                                      reconstruction.Metrics().Cells().SameOrder(_data->metrics.Cells()),
                                  "gradient conversion has wrong component count or mesh order");
            reconstruction.Validate();
            const index components = gradient.Width() / dim, n = reconstruction.NRecDOF();
            _data->metrics.ValidateField(gradient, dim * components);
            _data->metrics.ValidateField(coefficients, n * components);
            auto geometry = _data->metrics.AcquireRead();
            auto input = gradient.Owned().Read();
            auto output = coefficients.Owned().Write();
            const auto values = input.View<Runtime::TableRows>();
            const auto result = output.View<Runtime::TableRows>();
            const auto cells = geometry.HostView().cells;
            for (index iCell = 0; iCell < _data->metrics.NumCell(); ++iCell)
            {
                std::fill(result[iCell].begin(), result[iCell].end(), 0);
                for (index component = 0; component < components; ++component)
                    for (index d = 0; d < dim; ++d)
                        result(iCell, component * n + d) = values(iCell, component * dim + d) * cells(iCell, 7);
            }
        }
        /// Owned output rows hold [component*dimension+direction]. Release input
        /// writers and finish input halo/BC producers first. Output must be
        /// distinct from inputs; ordinary Runtime leases reject alias conflicts.
        void Evaluate(const FVField &mean, const Runtime::Array<real> &boundaryValues, FVField &gradient) const
        {
            namespace R = Runtime;
            auto geometry = _data->metrics.AcquireRead();
            const index components = mean.Width(), dim = _data->metrics.Dimension();
            _data->metrics.ValidateField(mean, components);
            _data->metrics.ValidateField(gradient, dim * components);
            DNDS_check_throw_info(boundaryValues.GetDevice() == R::Device::Host() && boundaryValues.Layout().Kind() == R::RowLayoutKind::Table &&
                                      boundaryValues.RowCount() == NumBoundaryQuadrature() && boundaryValues.Layout().Width() == components,
                                  "gradient boundary value shape or placement differs");
            auto owned = mean.Owned().Read(), ghost = mean.Ghost().Read(), boundary = boundaryValues.Read();
            R::PairReadView<real, false, R::TableRows> u{owned.View<R::TableRows>(), ghost.View<R::TableRows>()};
            auto outWrite = gradient.Owned().Write();
            auto offsets = _data->offsets.Read(), neighbors = _data->neighbors.Read(), boundaryIds = _data->boundaryIds.Read();
            auto weights = _data->weights.Read();
            const auto offset = offsets.View<R::FixedRows<1>>(), neighbor = neighbors.View<R::FixedRows<1>>(), bnd = boundaryIds.View<R::FixedRows<1>>();
            const auto w = weights.View<R::FixedRows<3>>();
            const auto bv = boundary.View<R::TableRows>();
            const auto output = outWrite.View<R::TableRows>();
            for (index iCell = 0; iCell < _data->metrics.NumCell(); ++iCell)
            {
                std::fill(output[iCell].begin(), output[iCell].end(), 0);
                for (index entry = offset(iCell, 0); entry < offset(iCell + 1, 0); ++entry)
                    for (index component = 0; component < components; ++component)
                    {
                        const real difference = (neighbor(entry, 0) == UnInitIndex ? bv(bnd(entry, 0), component) : u(neighbor(entry, 0), component)) - u(iCell, component);
                        for (index d = 0; d < dim; ++d)
                            output(iCell, component * dim + d) += w(entry, d) * difference;
                    }
            }
        }
    };
}
