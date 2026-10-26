#pragma once
// TEST ONLY. Old wire-format decoding supplies a numerical/reference fixture;
// production code cannot include this header or export a legacy mesh backend.
#include "Geom/Mesh/MeshTransition.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
namespace DNDS::Geom::TestFixtureCodec
{
    inline std::shared_ptr<UnstructuredMesh> DecodeLegacy(UnstructuredMesh &source, bool outputCaches = false)
    {
        auto serializer = std::make_shared<Serializer::SerializerJSON>();
        serializer->CreatePath("/");
        WriteCurrentPrimaryMeshSnapshot(source, serializer, "fixture", 1, CurrentMeshCheckpointPolicy::DropAuxiliary);
        auto result = std::make_shared<UnstructuredMesh>(source.mpi, source.getDim());
        result->ReadSerialize(serializer, "fixture");
        if (result->adjPrimaryState == Adj_PointToLocal)
            result->AdjLocal2GlobalPrimary();
        result->RecoverNode2CellAndNode2Bnd();
        result->RecoverCell2CellAndBnd2Cell();
        result->BuildGhostPrimary();
        result->AdjGlobal2LocalPrimary();
        if (source.CurrentNativeRequirements().faces)
        {
            result->InterpolateFace();
            result->BuildGhostN2CB();
            result->AdjGlobal2LocalN2CB();
            result->BuildCell2CellFace();
            result->AdjGlobal2LocalC2CFace();
        }
        if (source.CurrentNativeRequirements().edges)
            result->InterpolateEdge();
        if (outputCaches)
        {
            result->RecreatePeriodicNodes();
            result->BuildVTKConnectivity();
        }
        return result;
    }
}
