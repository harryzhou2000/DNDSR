#include "NativeMeshPreparation.hpp"
#include "NativeMeshTopology.hpp"
#include "Mesh.hpp"
#include "FieldTransfer.hpp"
#include "Geom/Metis.hpp"
#include <limits>
#include <set>

namespace DNDS::Geom
{
    void UnstructuredMesh::PrepareNativeTopology(bool faces, bool edges, int ghostLayers)
    {
        CheckCollectiveMutationAvailable(true);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(_nativeState, "native topology preparation requires a native mesh"); });
        const auto stamp = versions.Stamp();
        auto result = NativeMeshPreparation::PrepareTopology(*_nativeState, mpi, faces, edges, ghostLayers);
        std::shared_ptr<NativeMeshState> prepared;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            stamp.Validate();
            prepared = std::shared_ptr<NativeMeshState>(new NativeMeshState(*result)); });
        MeshMutationToken mutation(this, versions, Runtime::Change::Topology | Runtime::Change::Layout | Runtime::Change::Distribution | Runtime::Change::Allocation);
        mutation.Commit();
        prepared->BindPreparedEpoch(versions);
        _nativeState = std::move(prepared);
        node2bndNode.clear();
        node2parentNode.clear();
        cell2parentCell.clear();
    }

    namespace
    {
        struct ActiveCommunicator
        {
            MPI_Comm value = MPI_COMM_NULL;
            ~ActiveCommunicator()
            {
                if (value != MPI_COMM_NULL)
                    MPI_Comm_free(&value);
            }
        };
        void Agree(const MPIInfo &mpi, const NativePartitionOptions &options)
        {
            std::array<int, 7> values{options.metisUfactor, options.metisSeed, options.edgeWeightMethod,
                                      options.metisNcuts, options.ghostLayers, options.metisType == "KWAY", 0};
            auto expected = values;
            MPI_Bcast(expected.data(), int(expected.size()), MPI_INT, 0, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { DNDS_check_throw_info(values == expected && options.metisType == "KWAY" &&
                                                                        options.edgeWeightMethod == 0 && options.metisUfactor > 0 && options.metisSeed >= 0 &&
                                                                        options.metisNcuts > 0 && options.ghostLayers > 0 &&
                                                                        options.metisSeed <= std::numeric_limits<int>::max() - (options.metisNcuts - 1),
                                                                    "native distributed partition requires matching KWAY, unweighted graph, positive cuts/ufactor/layers and supported seed"); });
        }
    }

    std::shared_ptr<const NativeMeshState> NativeMeshPreparation::PrepareTopology(
        const NativeMeshState &source, const MPIInfo &mpi, bool faces, bool edges, int ghostLayers)
    {
        std::array<int, 5> flags{int(faces), int(edges), ghostLayers, 0, 0};
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            source.Validate();
            flags[3] = source.Requirements().faces;
            flags[4] = source.Requirements().edges; });
        auto expected = flags;
        MPI_Bcast(expected.data(), int(expected.size()), MPI_INT, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(flags == expected && ghostLayers > 0, "native preparation topology options differ across ranks"); });
        return NativeMeshTopology::Build(source, mpi, faces && !flags[3], edges && !flags[4], ghostLayers);
    }

    NativeMeshTransitionResult NativeMeshPreparation::Partition(
        const NativeMeshState &source, const MPIInfo &mpi, const NativePartitionOptions &options)
    {
        Agree(mpi, options);
        auto complete = PrepareTopology(source, mpi, true, false, options.ghostLayers);
        const auto cells = complete->Entities(EntityKind::Cell);
        std::vector<idx_t> xadj, adjacency, part;
        std::vector<index> counts(size_t(mpi.size));
        const index owned = cells.NumOwned();
        MPI_Allgather(&owned, 1, DNDS_MPI_INDEX, counts.data(), 1, DNDS_MPI_INDEX, mpi.comm);
        idx_t global = 0, nParts = 0;
        std::vector<idx_t> offsets;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            global = METIS::indexToIdx(cells.NumGlobal());
            nParts = METIS::indexToIdx(std::min<index>(mpi.size, global));
            offsets.push_back(0);
            for (auto count : counts)
                if (count) offsets.push_back(offsets.back() + METIS::indexToIdx(count));
            DNDS_check_throw_info(offsets.back() == global, "native partition cell ownership is inconsistent");
            xadj.push_back(0);
            const auto view = complete->BorrowHostView();
            for (index cell = 0; cell < owned; ++cell)
            {
                std::set<index> neighbors;
                for (auto face : view.CellFaces(cell))
                {
                    DNDS_check_throw_info(face >= 0 && face < view.NumFaceProc(), "partition facial cone is unavailable");
                    for (auto neighbor : view.FaceCells(face))
                    {
                        if (neighbor == UnInitIndex) continue;
                        DNDS_check_throw_info(neighbor >= 0 && neighbor < view.NumCellProc(), "partition face support is unavailable");
                        const auto target = cells.LocalToGlobal(neighbor);
                        if (target != cells.LocalToGlobal(cell)) neighbors.insert(target);
                    }
                }
                for (auto target : neighbors) adjacency.push_back(METIS::indexToIdx(target));
                xadj.push_back(METIS::indexToIdx(index(adjacency.size())));
            }
            part.resize(std::max<size_t>(size_t(owned), 1));
            if (adjacency.empty()) adjacency.push_back(0); });
        std::vector<real_t> weights;
        std::vector<idx_t> candidate;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            weights.assign(size_t(nParts), nParts ? real_t(1) / real_t(nParts) : real_t(0));
            candidate.resize(part.size()); });
        ActiveCommunicator active;
        MPI_Comm_split(mpi.comm, owned ? 1 : MPI_UNDEFINED, mpi.rank, &active.value);
        int status = METIS_OK;
        if (owned && nParts > 1)
        {
            int activeSize = 0;
            MPI_Comm_size(active.value, &activeSize);
            idx_t nCon = 1, zero = 0, edgeCut = 0;
            if (activeSize == 1)
            {
                std::array<idx_t, METIS_NOPTIONS> settings{};
                METIS_SetDefaultOptions(settings.data());
                settings[METIS_OPTION_NCUTS] = options.metisNcuts;
                settings[METIS_OPTION_UFACTOR] = options.metisUfactor;
                settings[METIS_OPTION_SEED] = options.metisSeed;
                idx_t vertices = global;
                status = METIS_PartGraphKway(&vertices, &nCon, xadj.data(), adjacency.data(), nullptr, nullptr, nullptr,
                                             &nParts, nullptr, nullptr, settings.data(), &edgeCut, part.data());
            }
            else
            {
                std::array<real_t, 1> tolerance{real_t(1) + real_t(options.metisUfactor) / real_t(1000)};
                std::array<idx_t, 3> settings{1, 0, idx_t(options.metisSeed)};
                bool best = false;
                for (int cut = 0; cut < options.metisNcuts; ++cut)
                {
                    settings[2] = idx_t(options.metisSeed + cut);
                    idx_t objective = 0;
                    status = ParMETIS_V3_PartKway(offsets.data(), xadj.data(), adjacency.data(), nullptr, nullptr,
                                                  &zero, &zero, &nCon, &nParts, weights.data(), tolerance.data(), settings.data(),
                                                  &objective, candidate.data(), &active.value);
                    if (status != METIS_OK)
                        break;
                    if (!best || objective < edgeCut)
                    {
                        best = true;
                        edgeCut = objective;
                        part = candidate;
                    }
                }
            }
        }
        ReorderInput input;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(status == METIS_OK, "native METIS/ParMETIS partition failed");
            EntityReorderMap map{EntityKind::Cell, {}};
            for (index row = 0; row < owned; ++row)
            {
                const auto target = nParts > 1 ? part[size_t(row)] : 0;
                DNDS_check_throw_info(target >= 0 && target < mpi.size, "partition returned invalid rank");
                map.targetRanks.push_back(MPI_int(target));
            }
            input.explicitMaps.push_back(std::move(map)); });
        auto result = NativeMeshTransition::Build(*complete, mpi, input, options.ghostLayers);
        // Topology completion changes appended maps, but retains every original
        // owned row verbatim. External fields must validate the caller's exact
        // original association, never the intermediate publication's maps.
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (const auto kind : {EntityKind::Cell, EntityKind::Node, EntityKind::Bnd, EntityKind::Face, EntityKind::Edge})
                if (source.HasEntities(kind))
                {
                    DNDS_check_throw_info(source.NumOwned(kind) == complete->NumOwned(kind), "topology preparation changed original owned extent");
                    for (index row = 0; row < source.NumOwned(kind); ++row)
                        DNDS_check_throw_info(source.OriginalEntityId(kind, row) == complete->OriginalEntityId(kind, row),
                                              "topology preparation changed original owned row order");
                    result.sourceSpaces.insert_or_assign(kind, source.Entities(kind));
                } });
        return result;
    }
}
