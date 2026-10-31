#pragma once

#include "FiniteVolumePlan.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "CFV/Math/PolynomialReconstruction.hpp"
#include "Geom/Access/EntityWorkPartition.hpp"

namespace DNDS::CFV::Detail
{
    /// Implementation storage. Public plans expose no mutable aliases to it.
    struct FiniteVolumeData
    {
        ssp<Geom::UnstructuredMesh> mesh;
        Runtime::VersionStamp stamp;
        Geom::EntitySpace cells;
        Algebra::DofSpace dofs;
        std::optional<Runtime::RowTransferPlan> halo;
        int dim, intOrder;
        // Cell table: volume, center[3], barycenter[3], isotropic half-box scale.
        Runtime::Array<real> cell;
        // Face quadrature table: position[3], unit normal[3], weighted measure.
        Runtime::Array<real> faceQuadrature;
        Runtime::Array<index> faceOffsets;
        FiniteVolumeData(ssp<Geom::UnstructuredMesh> m, int order, std::shared_ptr<Runtime::CommunicationContext> communication)
            : mesh(std::move(m)), stamp(mesh->versions.Stamp(Runtime::Change::Topology | Runtime::Change::Geometry |
                                                             Runtime::Change::Distribution | Runtime::Change::Layout)),
              cells(mesh->Entities(Geom::EntityKind::Cell)),
              dofs(std::move(communication), cells.NumOwned(), cells.NumGhost(), "FV cells"),
              dim(mesh->getDim()), intOrder(order) {}
    };
    struct ReconstructionData
    {
        FiniteVolumePlan metrics;
        ReconstructionSettings settings;
        ReconstructionInverseDiagnostics inverseDiagnostics;
        Runtime::Array<real> inverseConditions;
        std::optional<Geom::EntityWorkPartition> workPartition;
        int n;
        Runtime::Array<real> moments;
        // One row per owned cell-face incidence, preallocated from cell2face.
        Runtime::Array<index> offsets, neighbors, faces, boundaryOffsets;
        Runtime::Array<real> matrixAInvB, vectorAInvB, secondary;
        // Dirichlet injection already multiplied by A^{-1} and quadrature weight.
        Runtime::Array<real> boundaryInjection;
        Runtime::Array<index> boundaryPoints;
        ReconstructionData(FiniteVolumePlan m, ReconstructionSettings s)
            : metrics(std::move(m)), settings(s), n(Math::PolynomialSize(metrics.Dimension(), s.maxOrder) - 1) {}
    };
    inline Geom::tPoint FacePointInCell(const Geom::UnstructuredMeshView &mesh, index iFace, int side, const Geom::tPoint &point)
    {
        const auto zone = mesh.GetFaceZone(iFace);
        if (side == 1 && Geom::FaceIDIsPeriodicMain(zone))
            return mesh.Periodic().TransCoord(point, zone);
        if (side == 1 && Geom::FaceIDIsPeriodicDonor(zone))
            return mesh.Periodic().TransCoordBack(point, zone);
        return point;
    }
    inline Geom::tPoint OtherPointInCell(const Geom::UnstructuredMeshView &mesh, index iFace, int side, const Geom::tPoint &point)
    {
        const auto zone = mesh.GetFaceZone(iFace);
        if ((side == 1 && Geom::FaceIDIsPeriodicMain(zone)) || (side == 0 && Geom::FaceIDIsPeriodicDonor(zone)))
            return mesh.Periodic().TransCoord(point, zone);
        if ((side == 0 && Geom::FaceIDIsPeriodicMain(zone)) || (side == 1 && Geom::FaceIDIsPeriodicDonor(zone)))
            return mesh.Periodic().TransCoordBack(point, zone);
        return point;
    }
    inline void Zero(FVField &field)
    {
        for (auto *array : {&field.Owned(), &field.Ghost()})
        {
            auto write = array->Write();
            const auto view = write.View<Runtime::TableRows>();
            for (index iRow = 0; iRow < view.RowCount(); ++iRow)
                std::fill(view[iRow].begin(), view[iRow].end(), 0.);
        }
    }
}
