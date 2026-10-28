#pragma once

#include "Geom/Mesh/NativeMeshCGNS.hpp"
#include "Geom/Mesh/NativeMeshPreparation.hpp"
#include "Geom/Mesh/NativeMeshTransition.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/NativeMeshAssembler.hpp"
#include "Geom/Mesh/MeshTransition.hpp"
#include "Geom/Mesh/EntityField.hpp"
#include "NativePeriodicFixtures.hpp"
#include <cstring>
#include <filesystem>
#include <map>
#include <set>

namespace NativeReorderChecks
{
    namespace G = DNDS::Geom;
    namespace R = DNDS::Runtime;
    using I = DNDS::index;
    using State = std::shared_ptr<const G::NativeMeshState>;
    inline std::map<std::string, State> fixtures;
    inline std::filesystem::path Root()
    {
        return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    }

    inline State Fixture(const DNDS::MPIInfo &mpi, bool faces = false, bool edges = false,
                         bool periodic = false, int layers = 1, const std::string &file = {}, DNDS::real length = 10)
    {
        const int dim = edges ? 3 : 2;
        const std::string name = file.empty() ? edges ? "UP3D_32.cgns" : periodic ? "IV10_10.cgns"
                                                                                  : "UniformSquare_10.cgns"
                                              : file;
        const std::string key = name + ":" + std::to_string(faces) + ":" + std::to_string(edges) + ":" + std::to_string(layers) + ":" + std::to_string(length);
        auto found = fixtures.find(key);
        if (found != fixtures.end())
            return found->second;
        std::optional<G::NativePeriodicImportOptions> options;
        if (periodic || edges)
        {
            options.emplace();
            options->tolerance = 1e-8;
            if (!edges)
            {
                options->frames.translation[1].map() = G::tPoint{length, 0, 0};
                options->frames.translation[2].map() = G::tPoint{0, length, 0};
                options->frames.translation[3].map().setZero();
            }
        }
        auto state = G::ReadNativeCGNS(mpi, (Root() / "data/mesh" / name).string(), dim, G::FBC_Name_2_ID_Default, 0, {layers, true}, options);
        if (faces || edges)
        {
            G::NativePartitionOptions partition;
            partition.metisType = "KWAY";
            partition.metisUfactor = 30;
            partition.metisSeed = 42;
            partition.metisNcuts = 1;
            partition.ghostLayers = layers;
            state = G::NativeMeshPreparation::Partition(*state, mpi, partition).state;
            if (edges)
                state = G::PrepareNativeMeshTopology(*state, mpi, false, true, layers);
        }
        fixtures.emplace(key, state);
        return state;
    }

    inline I Id(const G::NativeMeshState &state, G::EntityKind kind, I local)
    {
        if (local == DNDS::UnInitIndex)
            return local;
        if (local >= 0)
            return state.OriginalEntityId(kind, local);
        const auto present = state.Entities(kind).GlobalToLocal(-1 - local);
        DNDS_check_throw_info(present, "owned topology record needs a locally available target");
        return state.OriginalEntityId(kind, *present);
    }
    inline I Word(DNDS::real value)
    {
        I result;
        static_assert(sizeof(result) == sizeof(value));
        std::memcpy(&result, &value, sizeof(result));
        return result;
    }
    template <class Row>
    inline void Targets(std::vector<I> &record, const G::NativeMeshState &state,
                        G::EntityKind target, Row row, bool unordered = false)
    {
        std::vector<I> ids;
        for (const auto value : row)
            ids.push_back(Id(state, target, value));
        if (unordered)
            std::sort(ids.begin(), ids.end());
        record.push_back(I(ids.size()));
        record.insert(record.end(), ids.begin(), ids.end());
    }
    template <class Row>
    inline void Bits(std::vector<I> &record, Row row)
    {
        record.push_back(I(row.size()));
        for (const auto value : row)
            record.push_back(uint8_t(value));
    }
    inline void CellSlots(std::vector<I> &record, const G::NativeMeshState &state,
                          const G::UnstructuredMeshView &view, I cell, bool edges)
    {
        const auto entities = edges ? view.CellEdges(cell) : view.CellFaces(cell);
        record.push_back(I(entities.size()));
        for (size_t slot = 0; slot < entities.size(); ++slot)
        {
            const I entity = entities[slot];
            const auto nodes = edges ? view.EdgeNodes(entity) : view.FaceNodes(entity);
            std::vector<std::pair<I, I>> physical;
            for (size_t node = 0; node < nodes.size(); ++node)
            {
                I bits = 0;
                if (view.IsPeriodic())
                {
                    const auto entityBits = edges ? view.EdgeNodePeriodicBits(entity) : view.FaceNodePeriodicBits(entity);
                    const auto parentBits = edges ? view.CellEdgePeriodicBits(cell) : view.CellFacePeriodicBits(cell);
                    bits = uint8_t(entityBits[node]) ^ uint8_t(parentBits[slot]);
                }
                physical.emplace_back(Id(state, G::EntityKind::Node, nodes[node]), bits);
            }
            // Preserve the element's FACE/EDGE SLOT. Orientation of a newly
            // regenerated entity may change its private first-parent frame.
            std::sort(physical.begin(), physical.end());
            record.push_back(I(physical.size()));
            for (const auto &[node, bits] : physical)
            {
                record.push_back(node);
                record.push_back(bits);
            }
        }
    }

    inline R::Array<I> Records(const G::NativeMeshState &state, G::EntityKind kind)
    {
        auto access = state.Snapshot(R::Device::Host()).AcquireRead();
        const auto view = access.HostView();
        std::vector<std::vector<I>> rows(size_t(state.NumOwned(kind)));
        for (I row = 0; row < state.NumOwned(kind); ++row)
        {
            auto &record = rows[size_t(row)];
            record.push_back(state.OriginalEntityId(kind, row));
            if (kind == G::EntityKind::Node)
            {
                for (int d = 0; d < 3; ++d)
                    record.push_back(Word(view.Coord(row)[d]));
                if (state.Requirements().wallDistance)
                    for (int d = 0; d < 3; ++d)
                        record.push_back(Word(view.NodeWallDist(row)[d]));
                if (view.HasRelation(G::MeshRelation::NodeCells))
                    Targets(record, state, G::EntityKind::Cell, view.NodeCells(row), true);
                if (view.HasRelation(G::MeshRelation::NodeBoundaries))
                    Targets(record, state, G::EntityKind::Bnd, view.NodeBoundaries(row), true);
            }
            if (kind == G::EntityKind::Cell)
            {
                const auto info = view.GetCellInfo(row);
                record.push_back(info.getElemType());
                record.push_back(info.zone);
                Targets(record, state, G::EntityKind::Node, view.CellNodes(row));
                if (view.IsPeriodic())
                    Bits(record, view.CellNodePeriodicBits(row));
                Targets(record, state, G::EntityKind::Cell, view.CellCells(row), true);
                if (view.HasRelation(G::MeshRelation::CellFaces))
                    CellSlots(record, state, view, row, false);
                if (view.HasRelation(G::MeshRelation::CellEdges))
                    CellSlots(record, state, view, row, true);
                if (view.HasRelation(G::MeshRelation::CellFaceCells))
                    Targets(record, state, G::EntityKind::Cell, view.CellFaceCells(row));
            }
            if (kind == G::EntityKind::Bnd)
            {
                const auto info = view.GetBndInfo(row);
                record.push_back(info.getElemType());
                record.push_back(info.zone);
                Targets(record, state, G::EntityKind::Node, view.BoundaryNodes(row));
                if (view.IsPeriodic())
                    Bits(record, view.BoundaryNodePeriodicBits(row));
                Targets(record, state, G::EntityKind::Cell, view.BoundaryCells(row));
            }
            if (kind == G::EntityKind::Face)
            {
                const auto info = view.GetFaceInfo(row);
                record.push_back(info.getElemType());
                record.push_back(info.zone);
                Targets(record, state, G::EntityKind::Node, view.FaceNodes(row));
                if (view.IsPeriodic())
                    Bits(record, view.FaceNodePeriodicBits(row));
                Targets(record, state, G::EntityKind::Cell, view.FaceCells(row));
                if (view.HasRelation(G::MeshRelation::FaceBoundaries))
                    Targets(record, state, G::EntityKind::Bnd, view.FaceBoundaries(row));
            }
            if (kind == G::EntityKind::Edge)
            {
                const auto info = view.GetEdgeInfo(row);
                record.push_back(info.getElemType());
                record.push_back(info.zone);
                Targets(record, state, G::EntityKind::Node, view.EdgeNodes(row));
                if (view.IsPeriodic())
                    Bits(record, view.EdgeNodePeriodicBits(row));
                Targets(record, state, G::EntityKind::Cell, view.EdgeCells(row), true);
            }
        }
        return NativePeriodicFixtures::Rows(rows);
    }
    inline std::map<G::EntityKind, R::Array<I>> RecordAll(const State &state, const std::unordered_set<G::EntityKind> &omit = {})
    {
        std::map<G::EntityKind, R::Array<I>> records;
        for (const auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd, G::EntityKind::Face, G::EntityKind::Edge})
            if (state->HasEntities(kind) && !omit.count(kind))
                records.emplace(kind, Records(*state, kind));
        return records;
    }
    inline void Equal(const R::Array<I> &a, const R::Array<I> &b)
    {
        auto x = a.Read(), y = b.Read();
        REQUIRE(x.View().Rows() == y.View().Rows());
        for (size_t row = 0; row < x.View().Rows(); ++row)
        {
            REQUIRE(x.View().RowSize(row) == y.View().RowSize(row));
            for (size_t slot = 0; slot < x.View().RowSize(row); ++slot)
                CHECK(x.View()(row, slot) == y.View()(row, slot));
        }
    }
    inline void Check(const State &before, const G::NativeMeshTransitionResult &result,
                      const G::ReorderInput &input, const DNDS::MPIInfo &mpi,
                      const std::map<G::EntityKind, R::Array<I>> &expected)
    {
        const auto &after = result.state;
        CHECK(after->MeshNamespace() == before->MeshNamespace());
        for (const auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd, G::EntityKind::Face, G::EntityKind::Edge})
        {
            if (!before->HasEntities(kind))
                continue;
            REQUIRE(after->HasEntities(kind));
            CHECK(after->NumGlobal(kind) == before->NumGlobal(kind));
            CHECK(after->NextEntityId(kind) >= before->NextEntityId(kind));
            if (input.destroyKinds.count(kind))
            {
                for (I row = 0; row < after->NumOwned(kind); ++row)
                    CHECK(after->OriginalEntityId(kind, row) >= before->NextEntityId(kind));
                continue;
            }
            CHECK(result.sourceSpaces.at(kind).SameOrder(before->Entities(kind)));
            const auto &record = expected.at(kind);
            const auto transfer = result.transfers.find(kind);
            if (transfer == result.transfers.end())
            {
                // General transitions expose plans only for placed/followed
                // families. An untouched family's owned rows retain their
                // exact order, including references to the moved families.
                Equal(record, Records(*after, kind));
                continue;
            }
            auto route = G::NativeRowTransfer(transfer->second, mpi);
            R::Array<I> moved(route.DestinationLayout(record.Layout()));
            R::PayloadPlan(route, record.Layout(), moved.Layout(), R::ElementSchema::Arithmetic<I>())
                .Bind(record, moved, R::TransferTransport::Direct, {R::TransferMode::Transient})
                .Begin()
                .Wait();
            Equal(moved, Records(*after, kind));
        }
        // Explicit placements have an independent destination-rank payload.
        // A wrong permutation plan cannot pass merely by moving all records
        // and IDs consistently to the same wrong rank.
        for (const auto &placement : input.explicitMaps)
        {
            R::Array<I> ranks(R::RowLayout::Dense(placement.targetRanks.size(), 1));
            {
                auto write = ranks.Write();
                for (size_t row = 0; row < placement.targetRanks.size(); ++row)
                    write.View()(row, 0) = placement.targetRanks[row];
            }
            auto route = G::NativeRowTransfer(result.transfers.at(placement.kind), mpi);
            R::Array<I> moved(route.DestinationLayout(ranks.Layout()));
            R::PayloadPlan(route, ranks.Layout(), moved.Layout(), R::ElementSchema::Arithmetic<I>())
                .Bind(ranks, moved, R::TransferTransport::Direct, {R::TransferMode::Transient})
                .Begin()
                .Wait();
            auto read = moved.Read();
            for (size_t row = 0; row < moved.Rows(); ++row)
                CHECK(read.View()(row, 0) == mpi.rank);
        }
        // All output compute groups stay local, including optional topology.
        auto access = after->Snapshot(R::Device::Host()).AcquireRead();
        const auto view = access.HostView();
        for (size_t i = 0; i < size_t(G::MeshRelation::Count); ++i)
            if (view.HasRelation(G::MeshRelation(i)))
                CHECK(view.RelationState(G::MeshRelation(i)) == G::Adj_PointToLocal);
    }

    inline G::ReorderInput Place(const State &source, G::EntityKind kind, const DNDS::MPIInfo &mpi, int pattern = 0)
    {
        std::vector<DNDS::MPI_int> destinations(size_t(source->NumOwned(kind)));
        for (size_t row = 0; row < destinations.size(); ++row)
            destinations[row] = pattern == 0 ? mpi.rank : pattern == 1 ? DNDS::MPI_int(row % size_t(mpi.size))
                                                      : pattern == 2   ? (mpi.rank + 1) % mpi.size
                                                                       : DNDS::MPI_int((mpi.rank + row) % size_t(mpi.size));
        G::ReorderInput input;
        input.explicitMaps.push_back({kind, std::move(destinations)});
        return input;
    }
    inline void RoundTrip(const State &state, const DNDS::MPIInfo &mpi, const std::vector<G::AdjKind> &keys)
    {
        auto facade = G::UnstructuredMesh::FromNative(mpi, state);
        auto requirements = state->Requirements();
        for (const auto &key : keys)
        {
            auto source = G::RelationReadAccess::Current(*facade, key, requirements);
            auto global = G::NativeRelation::Convert(source, G::Adj_PointToGlobal);
            auto local = G::NativeRelation::Convert(global.Read(), G::Adj_PointToLocal);
            auto read = local.Read();
            CHECK(source.Rows() == read.Rows());
            for (I row = 0; row < source.Rows(); ++row)
            {
                REQUIRE(source.RowSize(row) == read.RowSize(row));
                for (size_t slot = 0; slot < source.RowSize(row); ++slot)
                    CHECK(source.Raw(row, slot) == read.Raw(row, slot));
            }
        }
    }
    inline G::NativeMeshTransitionResult Move(const State &source, const G::ReorderInput &input,
                                              const DNDS::MPIInfo &mpi, int layers = 1)
    {
        auto expected = RecordAll(source, input.destroyKinds);
        auto retained = source->Snapshot(R::Device::Host()).AcquireRead();
        const auto stamp = retained.Stamp();
        auto result = G::NativeMeshTransition::Build(*source, mpi, input, layers);
        Check(source, result, input, mpi, expected);
        CHECK(stamp.Valid());
        CHECK(retained.HostView().NumCell() == source->NumOwned(G::EntityKind::Cell));
        return result;
    }
    inline void Follow(const State &source, const DNDS::MPIInfo &mpi, G::EntityKind leader,
                       const std::vector<std::pair<G::EntityKind, G::AdjKind>> &followers)
    {
        auto input = Place(source, leader, mpi, 1);
        auto leaderSpace = source->Entities(leader);
        R::Array<I> owned(R::RowLayout::Dense(size_t(leaderSpace.NumOwned()), 1)), ghost(R::RowLayout::Dense(size_t(leaderSpace.NumGhost()), 1));
        {
            auto write = owned.Write();
            for (I row = 0; row < leaderSpace.NumOwned(); ++row)
                write.View()(size_t(row), 0) = input.explicitMaps[0].targetRanks[size_t(row)];
        }
        auto route = leaderSpace.BuildRowTransferPlan();
        R::PayloadPlan(route, owned.Layout(), ghost.Layout(), R::ElementSchema::Arithmetic<I>())
            .Bind(owned, ghost, R::TransferTransport::Direct, {R::TransferMode::Transient})
            .Begin()
            .Wait();
        auto ownedRead = owned.Read(), ghostRead = ghost.Read();
        auto facade = G::UnstructuredMesh::FromNative(mpi, source);
        std::map<G::EntityKind, R::Array<I>> expected;
        for (const auto &[kind, key] : followers)
        {
            input.follows.push_back({kind, leader, key});
            auto support = G::RelationReadAccess::Current(*facade, key, source->Requirements());
            R::Array<I> ranks(R::RowLayout::Dense(size_t(support.NumOwned()), 1));
            {
                auto write = ranks.Write();
                for (I row = 0; row < support.NumOwned(); ++row)
                {
                    I destination = mpi.size;
                    for (size_t slot = 0; slot < support.RowSize(row); ++slot)
                    {
                        const auto entry = support.ReadEntry(row, slot);
                        if (entry.absent)
                            continue;
                        const auto local = leaderSpace.GlobalToLocal(entry.global);
                        REQUIRE(local.has_value());
                        const auto target = *local < leaderSpace.NumOwned() ? ownedRead.View()(size_t(*local), 0)
                                                                            : ghostRead.View()(size_t(*local - leaderSpace.NumOwned()), 0);
                        destination = std::min(destination, target);
                    }
                    write.View()(size_t(row), 0) = destination == mpi.size ? mpi.rank : destination;
                }
            }
            expected.emplace(kind, std::move(ranks));
        }
        auto result = Move(source, input, mpi);
        for (const auto &[kind, ranks] : expected)
        {
            auto rowRoute = G::NativeRowTransfer(result.transfers.at(kind), mpi);
            R::Array<I> moved(rowRoute.DestinationLayout(ranks.Layout()));
            R::PayloadPlan(rowRoute, ranks.Layout(), moved.Layout(), R::ElementSchema::Arithmetic<I>())
                .Bind(ranks, moved, R::TransferTransport::Direct, {R::TransferMode::Transient})
                .Begin()
                .Wait();
            auto read = moved.Read();
            for (size_t row = 0; row < moved.Rows(); ++row)
                CHECK(read.View()(row, 0) == mpi.rank);
        }
    }
}
