/** @file Mesh_Helpers.hpp
 *  @brief Unified mesh helpers: read, prepare, build boundary, serialize.
 *
 *  Thin wrappers that compose the canonical mesh assembly pipeline from
 *  lower-level UnstructuredMesh / UnstructuredMeshSerialRW methods.
 *  Mirror the Python helpers in DNDSR.Geom.utils.
 *
 *  See docs/dev/mesh_helpers_design.md for the design document.
 */
#pragma once

#include "Mesh.hpp"
#include "NativeMeshPreparation.hpp"
#include "NativeMeshCGNS.hpp"
#include "NativeMeshOpenFOAM.hpp"
#include "NativeMeshRefinement.hpp"
#include "NativeMeshArchive.hpp"
#include "NativeMeshLegacyArchive.hpp"
#include "FieldTransfer.hpp"
#include "DNDS/Serializer/SerializerFactory.hpp"

#include <functional>
#include <fstream>
#include <string>

namespace DNDS::Geom
{

    /// Apply requested order derivation before partitioning. All source owned
    /// metadata is read through the native publication; existing O2 is retained.
    inline std::shared_ptr<const NativeMeshState> DeriveAndPartitionNativeMesh(
        std::shared_ptr<const NativeMeshState> state, const MPIInfo &mpi,
        const PartitionOptions &partOpts, int elevation, int bisect)
    {
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info((elevation == 0 || elevation == 1) && bisect >= 0 && bisect <= 4,
                                                                "native mesh setup requires elevation 0/1 and bisection count 0..4"); });
        std::array<int, 2> derived{elevation, bisect}, expected = derived;
        MPI_Bcast(expected.data(), int(expected.size()), MPI_INT, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(derived == expected, "native mesh derivation options differ across ranks"); });
        auto isO2 = [&]
        {
            int local = 1;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                const auto view = state->BorrowHostView();
                for (index row = 0; row < view.NumCell(); ++row)
                    local = local && Elem::Element{view.GetCellInfo(row).getElemType()}.GetOrder() == 2; });
            MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_INT, MPI_MIN, mpi.comm);
            return bool(local);
        };
        const NativeMeshBuildOptions build{1, true};
        if (elevation && !isO2())
            state = NativeMeshRefinement::Elevate(*state, mpi, build).state;
        for (int pass = 0; pass < bisect; ++pass)
        {
            if (!isO2())
                state = NativeMeshRefinement::Elevate(*state, mpi, build).state;
            state = NativeMeshRefinement::Bisect(*state, mpi, build).state;
        }
        NativePartitionOptions partition{partOpts.metisType, partOpts.metisUfactor, partOpts.metisSeed,
                                         partOpts.edgeWeightMethod, partOpts.metisNcuts, 1};
        return NativeMeshPreparation::Partition(*state, mpi, partition).state;
    }

    inline ssp<UnstructuredMesh> ReadNativeMeshFromCGNS(
        const MPIInfo &mpi, int dimension, const std::string &meshFile,
        const PartitionOptions &partOpts, const t_FBCName_2_ID &nameMapper,
        std::optional<NativePeriodicImportOptions> periodic = {},
        int elevation = 0, int bisect = 0, int readerRank = 0)
    {
        const auto distribution = partOpts.metisType == "RB" ? NativeSerialDistribution::ReaderOwned : NativeSerialDistribution::Even;
        auto state = ReadNativeCGNS(mpi, meshFile, dimension, nameMapper, readerRank, {1, true}, periodic, distribution);
        return UnstructuredMesh::FromNative(mpi, DeriveAndPartitionNativeMesh(std::move(state), mpi, partOpts, elevation, bisect));
    }

    inline ssp<UnstructuredMesh> ReadNativeMeshFromOpenFOAM(
        const MPIInfo &mpi, const std::string &meshFile, const PartitionOptions &partOpts,
        const std::map<std::string, std::string> &nameMapping,
        const t_FBCName_2_ID &nameMapper, std::optional<NativePeriodicImportOptions> periodic = {},
        int elevation = 0, int bisect = 0, int readerRank = 0)
    {
        const auto distribution = partOpts.metisType == "RB" ? NativeSerialDistribution::ReaderOwned : NativeSerialDistribution::Even;
        auto state = ReadNativeOpenFOAM(mpi, meshFile, nameMapping, nameMapper, readerRank, {1, true}, periodic, distribution);
        return UnstructuredMesh::FromNative(mpi, DeriveAndPartitionNativeMesh(std::move(state), mpi, partOpts, elevation, bisect));
    }

    /// Open an existing full native or legacy primary archive. Schema detection
    /// is explicit and collective; legacy decode preserves original IDs but
    /// creates a new namespace because the old format stores none.
    inline ssp<UnstructuredMesh> ReadNativeMeshFromArchive(
        const MPIInfo &mpi, Serializer::SerializerFactory factory, const std::string &inputPath,
        bool repartition = false, const PartitionOptions &partOpts = {}, int ghostLayers = 1)
    {
        std::string signature;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(ghostLayers >= 1 && (factory.type == "JSON" || factory.type == "H5") && factory.hdfDeflateLevel >= 0 &&
                                  factory.hdfDeflateLevel <= 9 && factory.hdfChunkSize >= 0 &&
                                  factory.jsonBinaryDeflateLevel >= 0 && factory.jsonBinaryDeflateLevel <= 9,
                                  "native archive serializer configuration invalid");
            signature = nlohmann::json{{"path", inputPath}, {"type", factory.type}, {"deflate", factory.hdfDeflateLevel},
                {"chunk", factory.hdfChunkSize}, {"collectiveData", factory.hdfCollOnData}, {"collectiveMeta", factory.hdfCollOnMeta},
                {"jsonDeflate", factory.jsonBinaryDeflateLevel}, {"codec", factory.jsonUseCodecOnUInt8}, {"repartition", repartition}, {"ghostLayers", ghostLayers},
                {"partitionType", partOpts.metisType}, {"ufactor", partOpts.metisUfactor}, {"seed", partOpts.metisSeed},
                {"weight", partOpts.edgeWeightMethod}, {"cuts", partOpts.metisNcuts}}.dump();
            DNDS_check_throw_info(signature.size() <= size_t(INT_MAX), "native archive configuration too large"); });
        int length = mpi.rank == 0 ? int(signature.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        std::string canonical;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { canonical = mpi.rank == 0 ? signature : std::string(size_t(length), '\0'); });
        MPI_Bcast(canonical.data(), length, MPI_CHAR, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(signature == canonical, "native archive reader options differ across ranks"); });
        Serializer::SerializerBaseSSP serializer;
        std::string path;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            path = std::get<0>(factory.ModifyFilePath(inputPath, mpi, "part_%d", true));
            serializer = factory.BuildSerializer(mpi);
            std::ifstream probe(path, std::ios::binary);
            DNDS_check_throw_info(probe.good(), "native archive input file is not readable"); });
        struct CloseArchive
        {
            Serializer::SerializerBaseSSP serializer;
            bool opened = false, json = false;
            ~CloseArchive()
            {
                if (opened || json)
                    try
                    {
                        serializer->CloseFile();
                    }
                    catch (...)
                    {
                    }
            }
        } close{serializer, false, factory.type == "JSON"};
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { serializer->OpenFile(path, true); close.opened = true; });
        const auto partition = repartition ? Serializer::ArrayGlobalOffset_EvenSplit : Serializer::ArrayGlobalOffset_Unknown;
        auto state = ReadCompatibleNativeMeshArchive(mpi, serializer, "meshPart", partition, ghostLayers);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { serializer->CloseFile(); });
        close.opened = close.json = false;
        if (repartition)
        {
            NativePartitionOptions options{partOpts.metisType, partOpts.metisUfactor, partOpts.metisSeed,
                                           partOpts.edgeWeightMethod, partOpts.metisNcuts, ghostLayers};
            state = NativeMeshPreparation::Partition(*state, mpi, options).state;
        }
        return UnstructuredMesh::FromNative(mpi, std::move(state));
    }

    // -----------------------------------------------------------------------
    // BuildGhostPrimary: the 5-step connectivity+ghost sequence
    // -----------------------------------------------------------------------

    /** @brief Build connectivity and ghost layer for a freshly-distributed mesh.
     *
     *  This 5-step sequence is identical across all read paths (CGNS serial,
     *  H5 parallel, H5 distributed).
     *
     *  @param nGhostLayers  Number of cell2cell hops for ghost cells (default 1).
     */
    inline void BuildGhostPrimary(UnstructuredMesh &mesh, int nGhostLayers = 1)
    {
        mesh.RecoverNode2CellAndNode2Bnd();
        mesh.RecoverCell2CellAndBnd2Cell();
        mesh.BuildGhostPrimary(nGhostLayers);
        mesh.AdjGlobal2LocalPrimary();
        mesh.AdjGlobal2LocalN2CB();
    }

    // -----------------------------------------------------------------------
    // ReadMeshFromCGNS
    // -----------------------------------------------------------------------

    /** @brief Read a CGNS mesh, partition, and optionally elevate/bisect.
     *
     *  After this call the mesh is distributed with ghost cells and local
     *  indices, but NOT solver-ready (no faces, no ghost N2CB).
     *
     *  @param mesh         Fresh UnstructuredMesh (constructed with mpi, dim).
     *  @param reader       UnstructuredMeshSerialRW bound to @p mesh.
     *  @param meshFile     Path to the CGNS file.
     *  @param partOpts     Metis partition options.
     *  @param periodicTol  Tolerance for periodic boundary deduplication.
     *  @param elevation    0 = none, 1 = O1->O2.
     *  @param bisect       Number of bisection passes (0..4).
     *  @param nameMapper   Optional callback mapping BC names to IDs
     *                      (for the CGNS reader).  If nullptr, the reader's
     *                      default AutoAppendName2ID is used.
     */
    inline void ReadMeshFromCGNS(
        ssp<UnstructuredMesh> &mesh,
        ssp<UnstructuredMeshSerialRW> &reader,
        const std::string &meshFile,
        const PartitionOptions &partOpts = {},
        real periodicTol = 1e-9,
        int elevation = 0,
        int bisect = 0,
        std::function<t_index(const std::string &)> nameMapper = nullptr)
    {
        auto &mpi = mesh->getMPI();
        int dim = mesh->getDim();

        if (nameMapper)
            reader->ReadFromCGNSSerial(meshFile, nameMapper);
        else
            reader->ReadFromCGNSSerial(meshFile);

        reader->Deduplicate1to1Periodic(periodicTol);
        reader->BuildCell2Cell();
        reader->MeshPartitionCell2Cell(partOpts);
        reader->PartitionReorderToMeshCell2Cell();

        Geom::BuildGhostPrimary(*mesh);

        // --- Optional O2 elevation ---
        if (elevation == 1)
        {
            ssp<UnstructuredMesh> meshO2;
            DNDS_MAKE_SSP(meshO2, mpi, dim);
            meshO2->BuildO2FromO1Elevation(*mesh);
            std::swap(meshO2, mesh);

            reader->mesh = mesh;
            Geom::BuildGhostPrimary(*mesh);
        }

        // --- Optional bisection ---
        DNDS_assert(bisect >= 0 && bisect <= 4);
        for (int iter = 1; iter <= bisect; iter++)
        {
            ssp<UnstructuredMesh> meshO2;
            DNDS_MAKE_SSP(meshO2, mpi, dim);
            meshO2->BuildO2FromO1Elevation(*mesh);

            meshO2->RecoverNode2CellAndNode2Bnd();
            meshO2->RecoverCell2CellAndBnd2Cell();
            meshO2->BuildGhostPrimary();

            ssp<UnstructuredMesh> meshO1B;
            DNDS_MAKE_SSP(meshO1B, mpi, dim);
            meshO1B->BuildBisectO1FormO2(*meshO2);

            std::swap(meshO1B, mesh);
            reader->mesh = mesh;
            Geom::BuildGhostPrimary(*mesh);

            index nCell = mesh->NumCellGlobal();
            index nNode = mesh->NumNodeGlobal();
            if (mpi.rank == 0)
                log() << fmt::format("Mesh Direct Bisect {} done, nCell [{}], nNode [{}]",
                                     iter, nCell, nNode)
                      << std::endl;
        }
    }

    // -----------------------------------------------------------------------
    // ReadMeshFromH5
    // -----------------------------------------------------------------------

    /** @brief Read a mesh from DNDSR H5 with even-split + ParMetis repartition.
     *
     *  Works with any number of MPI ranks regardless of how the file was written.
     */
    inline void ReadMeshFromH5(
        UnstructuredMesh &mesh,
        Serializer::SerializerFactory factory,
        const std::string &h5Path,
        const PartitionOptions &partOpts = {})
    {
        auto &mpi = mesh.getMPI();
        auto [pathMod, pathPart] = factory.ModifyFilePath(h5Path, mpi, "part_%d", true);
        Serializer::SerializerBaseSSP serializerP = factory.BuildSerializer(mpi);

        if (mpi.rank == 0)
            log() << "ReadMeshFromH5: distributed read via [" << factory.type
                  << "]" << std::endl;

        serializerP->OpenFile(pathMod, true);
        mesh.ReadSerializeAndDistribute(serializerP, "meshPart", partOpts);
        serializerP->CloseFile();

        Geom::BuildGhostPrimary(mesh);
    }

    /** @brief Read a pre-partitioned mesh from H5 (exact np match required).
     *
     *  Uses ReadSerialize (no repartitioning).  The file must have been written
     *  with the same number of MPI ranks.
     */
    inline void ReadMeshFromH5Parallel(
        UnstructuredMesh &mesh,
        Serializer::SerializerFactory factory,
        const std::string &h5Path)
    {
        auto &mpi = mesh.getMPI();
        auto [pathMod, pathPart] = factory.ModifyFilePath(h5Path, mpi, "part_%d", true);
        Serializer::SerializerBaseSSP serializerP = factory.BuildSerializer(mpi);

        if (mpi.rank == 0)
            log() << "ReadMeshFromH5Parallel: reading via [" << factory.type
                  << "]" << std::endl;

        serializerP->OpenFile(pathMod, true);
        mesh.ReadSerialize(serializerP, "meshPart");
        serializerP->CloseFile();

        Geom::BuildGhostPrimary(mesh);
    }

    // -----------------------------------------------------------------------
    // PrepareMesh
    // -----------------------------------------------------------------------

    /** @brief Options for PrepareMesh. */
    struct PrepareMeshOptions
    {
        int reorderCells = 0;       ///< 0 = natural, 1 = reorder
        int reorderParts = 1;       ///< nParts for ReorderLocalCells
        bool buildSerialOut = true; ///< build serial output arrays
    };

    /** @brief Prepare a distributed mesh for solver use.
     *
     *  Steps: cell reorder, face interpolation, ghost N2CB, optional serial output.
     *  Does NOT include: elevation smoothing, wall distance, coord transforms,
     *  periodic nodes, VTK connectivity, boundary mesh.
     *  Those remain solver / caller responsibility.
     *
     *  @param mesh     Distributed mesh (output of ReadMeshFromCGNS / ReadMeshFromH5).
     *  @param reader   The reader bound to @p mesh.
     *  @param opts     Options controlling reordering and serial output.
     */
    inline void PrepareMesh(
        UnstructuredMesh &mesh,
        UnstructuredMeshSerialRW &reader,
        const PrepareMeshOptions &opts = {})
    {
        auto &mpi = mesh.getMPI();

        if (mesh.HasNativeState())
        {
            mesh.PrepareNativeTopology();
            if (opts.reorderCells == 1)
                mesh.ReorderLocalCells(opts.reorderParts);
            if (opts.buildSerialOut)
                reader.BuildSerialOut();
            return;
        }

        if (opts.reorderCells == 1)
            mesh.ReorderLocalCells(opts.reorderParts);

        mesh.InterpolateFace();
        mesh.AssertOnFaces();

        mesh.AdjLocal2GlobalN2CB();
        mesh.BuildGhostN2CB();
        mesh.AdjGlobal2LocalN2CB();
        log() << fmt::format("{}, NumBndGhost {}", mpi.rank, mesh.NumBndGhost())
              << std::endl;

        // Owned nodes must have all cell neighbors resolved to local indices.
        // Ghost (son) nodes may have unresolved neighbors encoded as negative.
        for (index iNode = 0; iNode < mesh.NumNode(); iNode++)
            for (index iCell : mesh.node2cell.father->operator[](iNode))
                DNDS_assert(iCell >= 0);

        if (opts.buildSerialOut)
        {
            mesh.AdjLocal2GlobalPrimary();
            reader.BuildSerialOut();
            mesh.AdjGlobal2LocalPrimary();
        }
    }

    // -----------------------------------------------------------------------
    // BuildBndMesh
    // -----------------------------------------------------------------------

    /** @brief Extract the boundary surface mesh from a solver-ready volume mesh.
     *
     *  @param mesh       The volume mesh (must have faces built).
     *  @param meshBnd    Output boundary mesh (constructed with dim-1).
     *  @param readerBnd  Reader bound to @p meshBnd.
     *  @param buildSerialOut  Whether to build serial output for the bnd mesh.
     */
    inline void BuildBndMesh(
        UnstructuredMesh &mesh,
        UnstructuredMesh &meshBnd,
        UnstructuredMeshSerialRW &readerBnd,
        bool buildSerialOut = true)
    {
        mesh.ConstructBndMesh(meshBnd);
        if (buildSerialOut)
        {
            if (meshBnd.HasNativeState())
                readerBnd.BuildSerialOut();
            else
            {
                meshBnd.AdjLocal2GlobalPrimaryForBnd();
                readerBnd.BuildSerialOut();
                meshBnd.AdjGlobal2LocalPrimaryForBnd();
            }
        }
    }

    // -----------------------------------------------------------------------
    // SerializeMesh
    // -----------------------------------------------------------------------

    /** @brief Write a partitioned mesh to H5 for later same-partition or redistributed read.
     *
     *  The Python equivalent of the C++ ``partitionMeshOnly`` path.  The
     *  caller controls where this occurs in the pipeline; in Euler solvers it
     *  occurs before coordinate transforms, boundary mesh extraction, and solver
     *  setup.  Redistribution is supported by the collective H5 serializer, not
     *  by per-rank serializers.
     */
    inline void SerializeMesh(
        UnstructuredMesh &mesh,
        const std::string &outputPath,
        Serializer::SerializerFactory &factory)
    {
        auto &mpi = mesh.getMPI();
        auto [pathMod, pathPart] = factory.ModifyFilePath(outputPath, mpi, "part_%d", false);
        Serializer::SerializerBaseSSP serializerP = factory.BuildSerializer(mpi);

        serializerP->OpenFile(pathMod, false);
        if (mesh.HasNativeState())
            mesh.WriteNativeArchive(serializerP, "meshPart");
        else
        {
            mesh.AdjLocal2GlobalPrimary();
            mesh.WriteSerialize(serializerP, "meshPart");
            mesh.AdjGlobal2LocalPrimary();
        }
        serializerP->CloseFile();
    }

    // -----------------------------------------------------------------------
    // MeshH5Path (naming convention)
    // -----------------------------------------------------------------------

    /** @brief Build the conventional H5 filename for a partitioned mesh.
     *
     *  Convention: ``{base}_part_{mpiSize}[_elevated][_bisectN]``
     *  (The caller / SerializerFactory appends the actual extension.)
     */
    inline std::string MeshH5Path(
        const std::string &base, int mpiSize,
        int elevation = 0, int bisect = 0)
    {
        using namespace std::literals;
        std::string name = base + "_part_" + std::to_string(mpiSize);
        if (elevation == 1)
            name += "_elevated";
        if (bisect > 0)
            name += "_bisect" + std::to_string(bisect);
        return name;
    }

} // namespace DNDS::Geom
