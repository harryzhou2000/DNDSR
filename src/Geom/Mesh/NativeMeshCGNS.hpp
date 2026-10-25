#pragma once

#include "NativeMeshBuilder.hpp"

namespace DNDS::Geom
{
    /// Explicit BC-zone main/donor transforms and absolute geometric tolerance.
    struct NativePeriodicImportOptions
    {
        Periodicity frames;
        real tolerance = 1e-9;
    };
    /// Explicit serial import result, held entirely in Runtime arrays. Global
    /// serial scratch is confined to the reader rank; it is not ghost storage.
    /// Numbering follows zone-major, node-major deterministic deduplication.
    struct NativeSerialMeshData
    {
        Runtime::Array<real> coordinates{Runtime::RowLayout::Dense(0, 3)};
        Runtime::Array<index> cellNodes, boundaryNodes;
        Runtime::Array<index> boundaryCells{Runtime::RowLayout::Dense(0, 2)};
        Runtime::Array<ElemInfo> cellInfo{Runtime::RowLayout::Dense(0, 1)}, boundaryInfo{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> cellIds{Runtime::RowLayout::Dense(0, 1)}, nodeIds{Runtime::RowLayout::Dense(0, 1)}, boundaryIds{Runtime::RowLayout::Dense(0, 1)};
        bool periodic = false;
        std::optional<Runtime::Array<NodePeriodicBits>> cellNodesPbi, boundaryNodesPbi;
        Periodicity periodicInfo;
        MeshIdentity identity;
    };

    /// Collective native serial ingress. Only readerRank supplies payload rows;
    /// every rank supplies identical options. Optional periodic matching runs
    /// once before sparse direct distribution, and owned IDs remain payload.
    std::shared_ptr<const NativeMeshState> DistributeNativeSerialMesh(
        const MPIInfo &mpi, NativeSerialMeshData serial, int dimension, int readerRank = 0,
        NativeMeshBuildOptions options = {}, std::optional<NativePeriodicImportOptions> periodic = {});

    /// Local file operation, no MPI collectives. Supports the existing CGNS
    /// multi-zone, mixed-element and boundary PointList/PointRange conventions.
    NativeSerialMeshData ReadNativeCGNSSerial(const std::string &path, int dimension,
                                              const t_FBCName_2_ID &boundaryId);

    /// Collective explicit serial import, native direct row distribution to
    /// even contiguous ownership per family, then native sparse construction.
    /// All ranks supply identical options/path; boundaryId runs on readerRank.
    /// Optional BC-zone main/donor identification runs on readerRank before
    /// distribution. CGNS periodic connection properties are explicitly rejected;
    /// transforms must be supplied via the BC-zone convention. No legacy
    /// mesh/Array workspace is used.
    std::shared_ptr<const NativeMeshState> ReadNativeCGNS(
        const MPIInfo &mpi, const std::string &path, int dimension,
        const t_FBCName_2_ID &boundaryId, int readerRank = 0,
        NativeMeshBuildOptions options = {},
        std::optional<NativePeriodicImportOptions> periodic = {});
}
