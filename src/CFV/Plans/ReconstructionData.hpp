#pragma once

#include "FiniteVolumePlan.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "CFV/Math/PolynomialReconstruction.hpp"
#include "CFV/Math/TaylorBasis.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
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
        bool axisymmetric;
        // Cell table: volume, center[3], barycenter[3], isotropic half-box scale.
        Runtime::Array<real> cell;
        // Integral (x-bary_r)*dA divided by radial volume, used by the weighted
        // divergence identity in axisymmetric Green-Gauss. Zero otherwise.
        Runtime::Array<real> radialMoment;
        // Face quadrature table: position[3], unit normal[3], weighted measure.
        Runtime::Array<real> faceQuadrature;
        Runtime::Array<index> faceOffsets;
        FiniteVolumeData(ssp<Geom::UnstructuredMesh> m, int order, bool axis, std::shared_ptr<Runtime::CommunicationContext> communication)
            : mesh(std::move(m)), stamp(mesh->versions.Stamp(Runtime::Change::Topology | Runtime::Change::Geometry |
                                                             Runtime::Change::Distribution | Runtime::Change::Layout)),
              cells(mesh->Entities(Geom::EntityKind::Cell)),
              dofs(std::move(communication), cells.NumOwned(), cells.NumGhost(), "FV cells"),
              dim(mesh->getDim()), intOrder(order), axisymmetric(axis) {}
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
        Runtime::Array<real> faceBasis;
        // One row per owned cell-face incidence, preallocated from cell2face.
        Runtime::Array<index> offsets, neighbors, faces, boundaryOffsets;
        Runtime::Array<real> matrixAInvB, vectorAInvB, secondary;
        Runtime::Array<real> localMatrix, localInverse;
        // Unscaled blocks retain the original assembled operator even when
        // the truncated local inverse projects out small, nonzero modes.
        Runtime::Array<real> rawMatrixB, rawVectorB, rawBoundaryInjection;
        Runtime::Array<real> neighborRotation;
        bool hasRotatedPeriodicity = false;
        // Dirichlet injection already multiplied by A^{-1} and quadrature weight.
        Runtime::Array<real> boundaryInjection;
        Runtime::Array<index> boundaryPoints;
        ReconstructionData(FiniteVolumePlan m, ReconstructionSettings s)
            : metrics(std::move(m)), settings(s), n(Math::PolynomialSize(metrics.Dimension(), s.maxOrder) - 1) {}
    };
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
