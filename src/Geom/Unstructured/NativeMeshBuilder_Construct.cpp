#include "Geom/Unstructured/NativeMeshPreparation.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/IO/Metis.hpp"
#include "Geom/Transition/NativeMeshPartitionGraph.hpp"
#include "Geom/Unstructured/ReaderPartitionGraph.hpp"
#include "Geom/Connectivity/TopologySubEntityQuery.hpp"
#include "Geom/Connectivity/ConnectivityInterpolateRows.hpp"
#include "Geom/Unstructured/NativeMeshBuilder.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include <map>
#include <cmath>
#include <limits>
#include <set>

namespace DNDS::Geom
{
    std::shared_ptr<const UnstructuredMeshState> ConstructNativeMesh(NativeMeshInput input, NativeMeshBuildOptions options)
    {
        const auto mpi = input.mpi;
        auto owned = NativeMeshBuilder::Build(std::move(input), options);
        GhostSpec recipe;
        owned.LocalStage(mpi, [&]
                         { recipe = DefaultNativeMeshGhostSpec(owned, options.cellLayers, options.completeNodeStar); });
        return NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    }
}
