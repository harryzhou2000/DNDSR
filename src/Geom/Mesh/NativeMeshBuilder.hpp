#pragma once

#include "NativeMeshState.hpp"

namespace DNDS::Geom
{
    /// Owned host input in explicit rank-offset global numbering. The three
    /// spaces must be sibling families without ghosts. Cone slots follow the
    /// element topology; optional IDs are owned, one-column persistent values.
    /// Coordinates (three components) and ElemInfo (one component) accept dense,
    /// padded or uniform CSR rows; publication normalizes only their metadata.
    struct NativeMeshInput
    {
        MPIInfo mpi;
        int dimension;
        EntitySpace cells, nodes, boundaries;
        Runtime::Array<index> cellNodes, boundaryNodes;
        Runtime::Array<real> coordinates;
        Runtime::Array<ElemInfo> cellInfo, boundaryInfo;
        std::map<EntityKind, Runtime::Array<index>> ids;
        bool periodic = false;
        /// Empty namespace creates a new logical mesh; restart supplies the
        /// saved namespace and high-water marks unchanged.
        MeshIdentity identity;
    };

    struct NativeMeshBuildOptions
    {
        /// At least one layer: owned CellCells rows require local neighbors.
        int cellLayers = 1;
        /// Complete incident cells/boundaries and their node cones for every
        /// owned node, including owners with no incident owned cells.
        bool completeNodeStar = false;
    };

    /// Collective primary construction over sparse native relations. No legacy
    /// mesh/Array/transformer is allocated. All ranks, including empty ones,
    /// supply the same options and invoke Build in the same collective order.
    /// This first construction slice rejects periodic meshes and requires each
    /// external boundary's complete node set to match exactly one cell.
    class NativeMeshBuilder
    {
    public:
        static std::shared_ptr<const NativeMeshState> Build(
            NativeMeshInput input, NativeMeshBuildOptions options = {});
    };
}
