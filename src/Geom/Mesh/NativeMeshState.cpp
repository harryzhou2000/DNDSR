#include "NativeMeshState.hpp"
#include "Mesh.hpp"

namespace DNDS::Geom
{
    std::shared_ptr<const NativeMeshState> NativeMeshState::CaptureLegacy(UnstructuredMesh &source,
                                                                          MeshReadRequirements requirements)
    {
        auto sourceRead = source.versions.Read();
        auto result = std::shared_ptr<NativeMeshState>(new NativeMeshState);
        result->_requirements = requirements;
        result->_localPartStarts = source.localPartitionStarts;
        result->_localCellNeighbors = source.cell2cellFaceVLocalParts;
        result->_placements.emplace(-1, NativeMeshSnapshot::ImportLegacyWorkspace(source, requirements));
        result->_hostRead = std::make_shared<MeshReadAccess>(result->_placements.at(-1).AcquireRead());
        result->_hostView = result->_hostRead->HostView();
        auto capture = [&](EntityKind kind, const auto &pair)
        {
            auto identity = source.LegacyEntities(kind);
            DNDS_check_throw_info(pair.trans.pLGlobalMapping && pair.trans.pLGhostMapping,
                                  "native publication requires prepared entity indexing");
            result->_spaces.emplace(kind, Space{std::move(identity), pair.father->Size(), pair.son->Size(),
                                                std::make_shared<GlobalOffsetsMapping>(*pair.trans.pLGlobalMapping),
                                                std::make_shared<OffsetAscendIndexMapping>(*pair.trans.pLGhostMapping)});
        };
        capture(EntityKind::Cell, source.cell2node);
        capture(EntityKind::Node, source.coords);
        capture(EntityKind::Bnd, source.bnd2node);
        if (requirements.faces)
            capture(EntityKind::Face, source.face2node);
        if (requirements.edges)
            capture(EntityKind::Edge, source.edge2node);
        const auto ownedCells = result->_spaces.at(EntityKind::Cell).owned;
        DNDS_check_throw_info(result->_localPartStarts.empty() ||
                                  (result->_localPartStarts.front() == 0 && result->_localPartStarts.back() == ownedCells &&
                                   std::is_sorted(result->_localPartStarts.begin(), result->_localPartStarts.end())),
                              "local partition offsets differ from owned cell rows");
        DNDS_check_throw_info(result->_localCellNeighbors.empty() || result->_localCellNeighbors.size() == size_t(ownedCells),
                              "local neighbor cache differs from owned cell rows");
        for (auto &[kind, space] : result->_spaces)
        {
            space.identity = EntitySpace(kind, space.identity._mpi, space.owned, space.ghost, space.identity._identity,
                                         source.versions.Stamp(), space.global, space.ghosts, source.versions);
            space.identity._publishedMaps = true;
        }
        auto originals = [&](EntityKind kind, const auto &pair, const auto &canonical)
        {
            if (!pair.father && !pair.son)
                return;
            const auto &space = result->_spaces.at(kind);
            DNDS_check_throw_info(pair.father && pair.son && pair.father->Size() == space.owned && pair.son->Size() == space.ghost,
                                  "native publication original IDs differ from entity rows");
            const auto &global = pair.trans.pLGlobalMapping;
            const auto &ghosts = pair.trans.pLGhostMapping;
            DNDS_check_throw_info(pair.trans.father == pair.father && pair.trans.son == pair.son &&
                                      global && ghosts && pair.father->pLGlobalMapping == global &&
                                      global == canonical.trans.pLGlobalMapping && ghosts == canonical.trans.pLGhostMapping,
                                  "native publication original-ID source mapping differs from its entity space");
            DNDS_check_throw_info(global->RLengths().size() == size_t(source.mpi.size) &&
                                      source.mpi.rank >= 0 && source.mpi.rank < source.mpi.size &&
                                      global->RLengths()[source.mpi.rank] == space.owned &&
                                      ghosts->ghostIndex.size() == size_t(space.ghost),
                                  "native publication original-ID map extents differ from its arrays");
            auto ownedRead = pair.father->HostReadData();
            auto ghostRead = pair.son->HostReadData();
            Runtime::Array<index> ids(Runtime::RowLayout::Dense(size_t(space.owned + space.ghost), 1));
            auto write = ids.Write();
            for (index row = 0; row < space.owned + space.ghost; ++row)
                write.View()(size_t(row), 0) = row < space.owned ? (*pair.father)(row, 0) : (*pair.son)(row - space.owned, 0);
            write.Wait();
            result->_originalIds.emplace(kind, std::move(ids));
        };
        originals(EntityKind::Cell, source.cell2cellOrig, source.cell2node);
        originals(EntityKind::Node, source.node2nodeOrig, source.coords);
        originals(EntityKind::Bnd, source.bnd2bndOrig, source.bnd2node);
        for (const auto &[kind, ids] : result->_originalIds)
            result->_originalReads.emplace(kind, std::make_shared<Runtime::Array<index>::ReadLease>(ids.Read()));
        return result;
    }

    std::shared_ptr<const NativeMeshState> NativeMeshState::BindEpoch(Runtime::VersionTracker versions) const
    {
        auto result = std::shared_ptr<NativeMeshState>(new NativeMeshState(*this));
        result->BindPreparedEpoch(std::move(versions));
        return result;
    }
    void NativeMeshState::BindPreparedEpoch(Runtime::VersionTracker versions)
    {
        _versions = std::move(versions);
        _epoch = _versions.Stamp();
        for (auto &[kind, space] : _spaces)
        {
            space.identity._stamp = _versions.Stamp(Runtime::Change::Topology | Runtime::Change::Distribution | Runtime::Change::Layout);
            space.identity._versions = _versions;
        }
    }

    std::shared_ptr<const NativeMeshState> NativeMeshState::WithPlacement(Runtime::Device device, Runtime::ExecutionContext context) const
    {
        Validate();
        DNDS_check_throw_info(device == Runtime::Device::Host() ||
                                  (device.space == Runtime::MemorySpace::CUDA && device.ordinal >= 0),
                              "invalid native mesh placement device");
        DNDS_check_throw_info(context.device == device &&
                                  (device.space != Runtime::MemorySpace::Host || !context.stream),
                              "native mesh placement context mismatch");
        auto result = std::shared_ptr<NativeMeshState>(new NativeMeshState(*this));
        if (!result->_placements.count(PlacementKey(device)))
            result->_placements.emplace(PlacementKey(device), _placements.at(-1).To(device, context));
        return result;
    }

    void NativeMeshState::CheckRequirements(MeshReadRequirements r) const
    {
        Validate();
        DNDS_check_throw_info((!r.faces || _requirements.faces) && (!r.inverse || _requirements.inverse) &&
                                  (!r.edges || _requirements.edges) && (!r.cellFaceNeighbors || _requirements.cellFaceNeighbors) &&
                                  (!r.wallDistance || _requirements.wallDistance) &&
                                  (!r.allFaceCellsLocal || _requirements.allFaceCellsLocal),
                              "native publication lacks requested relation group or closure; republish explicitly");
    }

    const NativeMeshSnapshot &NativeMeshState::Snapshot(Runtime::Device device) const
    {
        Validate();
        auto found = _placements.find(PlacementKey(device));
        DNDS_check_throw_info(found != _placements.end(), "native mesh placement is not prepared; place explicitly before reading");
        return found->second;
    }

    index NativeMeshState::NumOwned(EntityKind kind) const
    {
        auto found = _spaces.find(kind);
        DNDS_check_throw_info(found != _spaces.end(), "native entity family was not published");
        return found->second.owned;
    }
    index NativeMeshState::NumGhost(EntityKind kind) const
    {
        auto found = _spaces.find(kind);
        DNDS_check_throw_info(found != _spaces.end(), "native entity family was not published");
        return found->second.ghost;
    }
    index NativeMeshState::NumGlobal(EntityKind kind) const
    {
        auto found = _spaces.find(kind);
        DNDS_check_throw_info(found != _spaces.end(), "native entity family was not published");
        return found->second.global->globalSize();
    }
    EntitySpace NativeMeshState::Entities(EntityKind kind) const
    {
        Validate();
        auto found = _spaces.find(kind);
        DNDS_check_throw_info(found != _spaces.end(), "native entity family was not published");
        return found->second.identity;
    }
    index NativeMeshState::OriginalEntityId(EntityKind kind, index row) const
    {
        auto found = _originalReads.find(kind);
        DNDS_check_throw_info(found != _originalReads.end() && row >= 0 && size_t(row) < found->second->Layout().Rows(),
                              "original entity ID outside published family");
        return found->second->View()(size_t(row), 0);
    }
}
