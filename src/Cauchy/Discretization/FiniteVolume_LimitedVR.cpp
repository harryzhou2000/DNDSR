#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;
    FiniteVolume::LimitedVRGeometry::LimitedVRGeometry(const Data &data)
        : gradient(CFV::GradientPlan::Prepare(*data.mesh, data.metrics, CFV::GradientMethod::GreenGauss)),
          gates(CFV::LimitedVRPlan::Prepare(data.metrics))
    {
        auto read = data.mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto mesh = read.HostView();
        auto geometry = data.metrics.AcquireRead();
        const auto metric = geometry.HostView();
        index count = 0;
        for (auto entry : data.faceEntries)
            count += data.samples[entry].other != UnInitIndex;
        faces.reserve(size_t(count));
        R::Array<index> cells(R::RowLayout::Dense(2 * count, 1));
        R::Array<real> points(R::RowLayout::Dense(2 * count, 3));
        {
            auto cellWrite = cells.Write();
            auto pointWrite = points.Write();
            const auto c = cellWrite.View<R::FixedRows<1>>();
            const auto p = pointWrite.View<R::FixedRows<3>>();
            Geom::tSmallCoords coordinates;
            for (index iFace = 0; iFace < index(data.faceEntries.size()); ++iFace)
            {
                const auto &sample = data.samples[data.faceEntries[iFace]];
                if (sample.other == UnInitIndex)
                    continue;
                const auto face = data.stencilFaces[iFace];
                mesh.GetCoordsOnFace(face, coordinates);
                const auto element = mesh.GetFaceElement(face);
                // Match legacy iG=-1: mapped one-point reference center, not
                // the area-weighted centroid (different on curved elements).
                const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, 1, 0);
                const Geom::tPoint center = Geom::Elem::PPhysicsCoordD01Nj(coordinates, shape);
                Geom::tPoint normal = Geom::tPoint::Zero();
                for (index q = metric.FaceQuadratureBegin(face); q < metric.FaceQuadratureEnd(face); ++q)
                    normal += metric.GetFaceWeight(q) * metric.GetFaceNorm(q);
                DNDS_check_throw_info(normal.allFinite() && normal.norm() > 0, "LVR face has a degenerate mean normal");
                normal.normalize();
                if (sample.side)
                    normal = -sample.rotation * normal;
                const index row = 2 * index(faces.size());
                const std::array<Geom::tPoint, 2> point{
                    CFV::Detail::FacePointInCell(mesh, face, sample.side, center),
                    CFV::Detail::FacePointInCell(mesh, face, 1 - sample.side, center)};
                c(row, 0) = sample.cell;
                c(row + 1, 0) = sample.other;
                for (int side = 0; side < 2; ++side)
                    for (int d = 0; d < 3; ++d)
                        p(row + side, d) = point[side][d];
                faces.push_back({sample.cell, sample.other, normal, sample.rotation});
            }
        }
        DNDS_check_throw_info(data.reconstruction,
                              "LVR geometry requires variational reconstruction");
        sampling.emplace(*data.reconstruction, cells, points, false);
        // Gradient methods share this quadrature registry. Prove the reuse of
        // the ordinary conservative BC buffer once, not by matching row count.
        DNDS_check_throw_info(gradient.NumBoundaryQuadrature() == data.gradient.NumBoundaryQuadrature(), "LVR gradient boundary registry differs");
        for (index iPoint = 0; iPoint < gradient.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto a = gradient.GetBoundaryQuadrature(iPoint), b = data.gradient.GetBoundaryQuadrature(iPoint);
            DNDS_check_throw_info(a.cell == b.cell && a.face == b.face && a.quadrature == b.quadrature && a.point == b.point,
                                  "LVR gradient boundary quadrature order differs");
        }
    }
    FiniteVolume::LimitedVRWorkspace::LimitedVRWorkspace(const Data &data, const CFV::ReconstructionWorkspace &prototype)
        : target(data.reconstruction->BuildWorkspace(prototype.components, prototype.temporary.Owned().Layout(), prototype.periodicVectorStarts)),
          targetCoefficients(data.reconstruction->BuildCoefficients(prototype.components, prototype.temporary.Owned().Layout())),
          gradients(data.metrics.BuildField(prototype.components * data.metrics.Dimension())), scalars(data.metrics.BuildField(2)),
          branches(data.metrics.BuildField(3)), seeds(data.metrics.BuildField(3)), temporary(data.metrics.BuildField(3)),
          base(R::RowLayout::Dense(data.metrics.NumCell(), 2)),
          alpha(R::RowLayout::Dense(data.metrics.NumCell(), 1)), sampledFaces(R::RowLayout::Dense(data.limitedVRGeometry ? 2 * index(data.limitedVRGeometry->faces.size()) : 0, prototype.components)),
          sampledPP(R::RowLayout::Dense(data.positivitySamples ? data.positivitySamples->Size() : 0, prototype.components)), targetHalo(data.BindHalo(targetCoefficients)),
          branchHalo(data.BindHalo(branches)), seedHalo(data.BindHalo(seeds))
    {
        // Current physical adapter dispatches only compressible flow5. Generic
        // CFV penalties/targets and scalar stencils do not assume this width.
        DNDS_check_throw_info(prototype.components == 5 && data.limitedVRGeometry && data.positivitySamples, "LVR compressible target setup is incomplete");
        Algebra::Fill(targetCoefficients, 0.);
        Algebra::Fill(target.temporary, 0.);
        auto penalty = alpha.Write();
        for (index iCell = 0; iCell < data.metrics.NumCell(); ++iCell)
        {
            penalty.View()(iCell, 0) = 0;
        }
    }
}
