#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Math/PolynomialReconstruction.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "CFV/Math/PeriodicFrame.hpp"

namespace DNDS::CFV
{
    GradientPlan GradientPlan::Prepare(Geom::UnstructuredMesh &mesh, FiniteVolumePlan metrics,
                                       GradientMethod method, real relativeCutoff)
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
        data->rotations = R::Array<real>(R::RowLayout::Dense(entries, 9));
        data->boundary.resize(size_t(boundaryCount));
        auto offsetWrite = data->offsets.Write(), neighborWrite = data->neighbors.Write(), boundaryWrite = data->boundaryIds.Write();
        auto weightWrite = data->weights.Write();
        auto rotationWrite = data->rotations.Write();
        const auto rotations = rotationWrite.View<R::FixedRows<9>>();
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
            if (metrics.Axisymmetric())
                for (int d = 0; d < dim; ++d)
                    correction(1, d) += geometry.radialMoment(iCell, d);
            index local = 0;
            for (index slot = 0; slot < view.CellFaces(iCell).size(); ++slot)
            {
                const index face = view.CellFaces(iCell)[slot];
                const index other = view.CellFaceOther(iCell, face, rowsize(slot));
                DNDS_check_throw_info(other == UnInitIndex || (other >= 0 && other < metrics.NumCellProc()),
                                      "gradient stencil requires a complete face-neighbor ghost mapping");
                const bool back = view.CellIsFaceBack(iCell, face, rowsize(slot));
                const int side = back ? 0 : 1;
                const tGPoint rotation = Detail::OtherFrameInCell(view, face, side);
                for (index q = geometry.FaceQuadratureBegin(face); q < geometry.FaceQuadratureEnd(face); ++q, ++next, ++local)
                {
                    const tPoint point = Detail::FacePointInCell(view, face, side, geometry.GetFacePoint(q));
                    const tPoint normal = Detail::OtherFrameInCell(view, face, 1) * geometry.GetFaceNorm(q);
                    // Face quadrature normals are in parent's0 frame.
                    const tPoint ownNormal = (side == 0 ? geometry.GetFaceNorm(q) : normal) * (back ? 1 : -1);
                    const tPoint areaNormal = ownNormal * (geometry.GetFaceWeight(q) / geometry.GetCellVol(iCell));
                    const tPoint otherCenter = other == UnInitIndex ? point : Detail::OtherPointInCell(view, face, side, geometry.GetCellBary(other));
                    const tPoint displacement = otherCenter - center;
                    const real length = displacement.head(dim).norm();
                    DNDS_check_throw_info(length > 0 && std::isfinite(length), "degenerate FV gradient sample displacement");
                    // Boundary quadrature is multiple physical samples. For
                    // neighbor means repeat the same equation with quadrature
                    // weights so face rule refinement cannot reweight a face.
                    const real quadratureScale = std::sqrt(geometry.GetFaceWeight(q));
                    design.row(local) = displacement.head(dim).transpose() / length * quadratureScale;
                    lengths[local] = quadratureScale / length;
                    neighbors(next, 0) = other;
                    Eigen::Map<tGPoint>(rotations[next].data()) = rotation;
                    boundaries(next, 0) = UnInitIndex;
                    if (other == UnInitIndex)
                    {
                        boundaries(next, 0) = nextBoundary;
                        data->boundary[nextBoundary++] = {iCell, face, int(q - geometry.FaceQuadratureBegin(face)), point, ownNormal};
                    }
                    real fraction = 1;
                    if (other != UnInitIndex)
                    {
                        const real ownDistance = std::abs((point - center).dot(ownNormal)), otherDistance = std::abs((otherCenter - point).dot(ownNormal));
                        DNDS_check_throw_info(ownDistance + otherDistance > 0, "gradient face centers have zero normal separation");
                        fraction = method == GradientMethod::GGMP ? 0.5 : ownDistance / (ownDistance + otherDistance);
                        if (method == GradientMethod::GGMP)
                            correction.noalias() -= areaNormal.head(dim) * (point - (center + otherCenter) * 0.5).head(dim).transpose();
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
            else if (method == GradientMethod::GGMP || metrics.Axisymmetric())
            {
                HardEigen::EigenSVDInverse(correction, inverse, relativeCutoff);
                initial = (inverse * initial).eval();
            }
            for (index i = 0; i < count; ++i)
                for (int d = 0; d < 3; ++d)
                    weights(next - count + i, d) = d < dim ? initial(d, i) : 0;
        }
        offsets(metrics.NumCell(), 0) = next;
        // Prove owned-only access once, using the actual prepared entries.
        // The hot core kernel needs neither ghost views nor row-mask checks.
        for (auto iCell : data->partition.Core())
            for (index entry = offsets(iCell, 0); entry < offsets(iCell + 1, 0); ++entry)
                DNDS_check_throw_info(neighbors(entry, 0) == UnInitIndex || neighbors(entry, 0) < metrics.NumCell(),
                                      "gradient core stencil references a ghost input");
        return GradientPlan(std::move(data));
    }
}
