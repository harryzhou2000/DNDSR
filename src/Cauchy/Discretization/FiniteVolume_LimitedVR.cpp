#include "Cauchy/Discretization/FiniteVolume.hpp"
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
        const auto stencilRead = data.stencil.Read();
        const auto stencil = stencilRead.View();
        index count = 0;
        for (index iFace = 0; iFace < stencil.faceEntries.RowCount(); ++iFace)
            count += stencil.samples(stencil.faceEntries(iFace, 0), 0).other != UnInitIndex;
        faces = R::Array<LimitedVRFace>(R::RowLayout::Dense(count, 1));
        index nextFace = 0;
        R::Array<index> cells(R::RowLayout::Dense(2 * count, 1));
        R::Array<real> points(R::RowLayout::Dense(2 * count, 3));
        {
            auto faceWrite = faces.Write();
            const auto faceValues = faceWrite.View<R::FixedRows<1>>();
            auto cellWrite = cells.Write();
            auto pointWrite = points.Write();
            const auto c = cellWrite.View<R::FixedRows<1>>();
            const auto p = pointWrite.View<R::FixedRows<3>>();
            Geom::tSmallCoords coordinates;
            for (index iFace = 0; iFace < stencil.faceEntries.RowCount(); ++iFace)
            {
                const auto sample = FiniteVolumeDetail::ExpandSample(stencil.samples(stencil.faceEntries(iFace, 0), 0), {data.boundaryConditions.data(), index(data.boundaryConditions.size())});
                if (sample.other == UnInitIndex)
                    continue;
                const auto face = stencil.stencilFaces(iFace, 0);
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
                const index row = 2 * nextFace;
                const std::array<Geom::tPoint, 2> point{
                    CFV::Detail::FacePointInCell(mesh, face, sample.side, center),
                    CFV::Detail::FacePointInCell(mesh, face, 1 - sample.side, center)};
                c(row, 0) = sample.cell;
                c(row + 1, 0) = sample.other;
                for (int side = 0; side < 2; ++side)
                    for (int d = 0; d < 3; ++d)
                        p(row + side, d) = point[side][d];
                LimitedVRFace record{sample.cell, sample.other, {normal[0], normal[1], normal[2]}, {}};
                record.changesFrame = sample.changesFrame;
                for (int j = 0; j < 9; ++j)
                    record.rotation[j] = sample.rotation.data()[j];
                faceValues(nextFace++, 0) = record;
            }
        }
        // Allocate then fill the owned gather stencil. Storage is proportional
        // to local face incidences, never global entity count. GPU face kernels
        // write one independent sensor row; this CSR avoids atomic scatter.
        {
            const index owned = data.metrics.NumCell();
            std::vector<index> counts(owned, 0);
            const auto faceRead = faces.Read();
            const auto values = faceRead.View<R::FixedRows<1>>();
            for (index iFace = 0; iFace < count; ++iFace)
                for (const auto iCell : {values(iFace, 0).cell, values(iFace, 0).other})
                    if (iCell >= 0 && iCell < owned)
                        ++counts[iCell];
            cellFaceOffsets = R::Array<index>(R::RowLayout::Dense(owned + 1, 1));
            auto offsetWrite = cellFaceOffsets.Write();
            const auto offsets = offsetWrite.View<R::FixedRows<1>>();
            offsets(0, 0) = 0;
            for (index iCell = 0; iCell < owned; ++iCell)
            {
                offsets(iCell + 1, 0) = offsets(iCell, 0) + counts[iCell];
                counts[iCell] = offsets(iCell, 0);
            }
            cellFaces = R::Array<index>(R::RowLayout::Dense(offsets(owned, 0), 1));
            auto incidenceWrite = cellFaces.Write();
            const auto incidences = incidenceWrite.View<R::FixedRows<1>>();
            for (index iFace = 0; iFace < count; ++iFace)
                for (const auto iCell : {values(iFace, 0).cell, values(iFace, 0).other})
                    if (iCell >= 0 && iCell < owned)
                        incidences(counts[iCell]++, 0) = iFace;
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
          gradientExecution(data.limitedVRGeometry->gradient.BuildWorkspace(prototype.components, prototype.periodicVectorStarts)),
          base(R::RowLayout::Dense(data.metrics.NumCell(), 2)),
          alpha(R::RowLayout::Dense(data.metrics.NumCell(), 1)), sampledFaces(R::RowLayout::Dense(data.limitedVRGeometry ? 2 * data.limitedVRGeometry->faces.RowCount() : 0, prototype.components)),
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
