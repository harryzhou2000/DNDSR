#include "MeshEdit.hpp"
#include "MeshTransition.hpp"
#include "FieldTransfer.hpp"
#include "NativeConnectivity.hpp"

namespace DNDS::Geom
{
    namespace
    {
        constexpr unsigned Coordinates = 1, CellMetadata = 2, NeighborOrder = 4;

        template <class T>
        Runtime::Array<T> CloneArray(const Runtime::Array<T> &source)
        {
            Runtime::Array<T> result(source.Layout());
            auto destination = result.Storage();
            Runtime::Copy(source.Storage(), destination).Wait();
            return result;
        }

        template <class T>
        Runtime::ArrayPair<T> ClonePair(const Runtime::ArrayPair<T> &source)
        {
            return {CloneArray(source.owned), CloneArray(source.ghost)};
        }

        template <class T>
        void Refresh(Runtime::ArrayPair<T> &pair, const EntitySpace &space, const Runtime::ElementSchema &schema)
        {
            RelationGhostRoute route(space);
            auto ghost = route.TransferRows(space, pair.owned, schema);
            if (!pair.owned.Layout().IsVariable())
                ghost = Runtime::Array<T>::FromStorage(Runtime::RowLayout::Dense(ghost.Rows(), pair.owned.Layout().Width()), ghost.Storage());
            pair.ghost = std::move(ghost);
        }

        MeshReadRequirements CurrentRequirements(UnstructuredMesh &mesh)
        {
            if (mesh.HasNativeState())
                return mesh.CurrentNativeRequirements();
            auto present = [](const auto &pair)
            { return bool(pair.father) || bool(pair.son); };
            auto result = MeshReadRequirements::Primary();
            result.faces = present(mesh.face2node);
            // Recovery can leave these setup arrays in global numbering even
            // after primary/facial queries are local. The detached clone
            // rebuilds them; do not import them as a readable local group.
            result.inverse = present(mesh.node2cell) && mesh.node2cell.isLocal() && mesh.node2bnd.isLocal();
            result.edges = present(mesh.edge2node);
            result.cellFaceNeighbors = present(mesh.cell2cellFace);
            result.wallDistance = present(mesh.nodeWallDist);
            return result;
        }
    }

    void MeshEdit::Check() const
    {
        DNDS_check_throw_info(!Closed(), "mesh edit scope is closed or moved");
        _stamp.Validate();
    }
    void MeshEdit::CheckRow(index row, index count)
    {
        DNDS_check_throw_info(row >= 0 && row < count, "mesh edit row is outside owned entities");
    }

    MeshEdit MeshEdit::Begin(std::shared_ptr<UnstructuredMesh> source, int ghostLayers)
    {
        DNDS_check_throw_info(source, "mesh edit requires a source owner");
        MeshEdit result;
        result._source = std::move(source);
        auto &mesh = *result._source;
        FieldTransferDetail::LocalStage(mesh.mpi, [&]
                                        {
            result._requirements = CurrentRequirements(mesh);
            result._sourceRead = std::make_unique<MeshReadAccess>(mesh.AcquireRead(result._requirements));
            result._stamp = result._sourceRead->Stamp(); });
        if (mesh.HasNativeState())
        {
            FieldTransferDetail::LocalStage(mesh.mpi, [&]
                                            {
                DNDS_check_throw_info(ghostLayers >= 1, "mesh edit requires positive ghost_layers");
                const auto &state = mesh._nativeState->Snapshot(Runtime::Device::Host()).Live();
                auto &data = result._nativeCandidate.emplace();
                data.identity = {mesh._nativeState->MeshNamespace(), mesh._nativeState->IdHighWater()};
                data.dimension = state.semantics.dim;
                data.periodic = state.semantics.isPeriodic;
                data.periodicInfo = state.semantics.periodicInfo;
                data.relations = state.relations;
                data.coords = ClonePair(*state.coords);
                data.cellInfo = ClonePair(*state.cellInfo);
                data.relations[size_t(MeshRelation::CellCells)] = ClonePair(*state.relations[size_t(MeshRelation::CellCells)]);
                data.nodeWallDist = state.nodeWallDist;
                data.bndInfo = state.bndInfo;
                data.faceInfo = state.faceInfo;
                data.edgeInfo = state.edgeInfo;
                data.cellNodesPbi = state.cellNodesPbi;
                data.bndNodesPbi = state.bndNodesPbi;
                data.cellFacesPbi = state.cellFacesPbi;
                data.faceNodesPbi = state.faceNodesPbi;
                data.cellEdgesPbi = state.cellEdgesPbi;
                data.edgeNodesPbi = state.edgeNodesPbi;
                const auto view = result._sourceRead->HostView();
                for (index row = 0; row < view.NumCell(); ++row)
                {
                    const auto neighbors = view.CellCells(row);
                    result._neighbors.emplace_back(neighbors.begin(), neighbors.end());
                    result._cellTypes.push_back(view.GetCellInfo(row).getElemType());
                } });
            return result;
        }
        result._candidate = CloneCurrentMeshForSetup(mesh, ghostLayers, TransitionCachePolicy::Invalidate);
        auto &candidate = *result._candidate;
        FieldTransferDetail::LocalStage(mesh.mpi, [&]
                                        {
            // A staged edit preserves numbering. Reconstruction may need an
            // explicit ghost depth; never silently reinterpret its local IDs.
            for (auto kind : {EntityKind::Cell, EntityKind::Node, EntityKind::Bnd, EntityKind::Face, EntityKind::Edge})
            {
                if ((kind == EntityKind::Face && !result._requirements.faces) ||
                    (kind == EntityKind::Edge && !result._requirements.edges)) continue;
                auto before = mesh.Entities(kind), after = candidate.Entities(kind);
                DNDS_check_throw_info(before.NumOwned() == after.NumOwned() && before.NumLocal() == after.NumLocal(),
                                      "mesh edit ghost extent changed; specify matching ghost_layers");
                for (index row = 0; row < before.NumLocal(); ++row)
                    DNDS_check_throw_info(before.LocalToGlobal(row) == after.LocalToGlobal(row),
                                          "mesh edit candidate changed entity order");
            }
            candidate._entityIdentity = mesh._entityIdentity;
            candidate.localPartitionStarts.clear();
            for (int part = 0; part < mesh.NLocalParts(); ++part)
                candidate.localPartitionStarts.push_back(mesh.LocalPartStart(part));
            candidate.localPartitionStarts.push_back(mesh.LocalPartEnd(mesh.NLocalParts() - 1));
            candidate.cell2cellFaceVLocalParts = mesh.LocalCellNeighbors();
            const auto view = result._sourceRead->HostView();
            // Reconstruction may sort neighbor rows. Restore their exact source
            // ordering before exposing this narrowly scoped order editor.
            for (index row = 0; row < view.NumCellProc(); ++row)
            {
                const auto neighbors = view.CellCells(row);
                DNDS_check_throw_info(candidate.cell2cell.RowSize(row) == rowsize(neighbors.size()),
                                      "mesh edit candidate changed neighbor row shape");
                for (rowsize slot = 0; slot < rowsize(neighbors.size()); ++slot)
                    candidate.cell2cell(row, slot) = neighbors[size_t(slot)];
                if (row < view.NumCell())
                {
                    result._neighbors.emplace_back(neighbors.begin(), neighbors.end());
                    result._cellTypes.push_back(view.GetCellInfo(row).getElemType());
                }
            } });
        return result;
    }

    index MeshEdit::NumNode() const
    {
        Check();
        return _nativeCandidate ? index(_nativeCandidate->coords->owned.Rows()) : _candidate->NumNode();
    }
    index MeshEdit::NumCell() const
    {
        Check();
        return _nativeCandidate ? index(_nativeCandidate->cellInfo->owned.Rows()) : _candidate->NumCell();
    }
    tPoint MeshEdit::Coord(index row) const
    {
        Check();
        CheckRow(row, NumNode());
        if (_nativeCandidate)
        {
            auto read = _nativeCandidate->coords->owned.Read();
            return tPoint{read.View()(size_t(row), 0), read.View()(size_t(row), 1), read.View()(size_t(row), 2)};
        }
        return _candidate->Coord(row);
    }
    void MeshEdit::SetCoord(index row, const tPoint &value)
    {
        Check();
        CheckRow(row, NumNode());
        if (_nativeCandidate)
        {
            auto write = _nativeCandidate->coords->owned.Write();
            for (size_t component = 0; component < 3; ++component)
                write.View()(size_t(row), component) = value[component];
            _changes |= Coordinates;
            return;
        }
        auto write = _candidate->coords.father->HostWriteData();
        _candidate->coords[row] = value;
        _changes |= Coordinates;
    }
    ElemInfo MeshEdit::CellInfo(index row) const
    {
        Check();
        CheckRow(row, NumCell());
        if (_nativeCandidate)
            return _nativeCandidate->cellInfo->owned.Read().View()(size_t(row), 0);
        return _candidate->GetCellInfo(row);
    }
    void MeshEdit::SetCellInfo(index row, const ElemInfo &value)
    {
        Check();
        CheckRow(row, NumCell());
        if (_nativeCandidate)
        {
            _nativeCandidate->cellInfo->owned.Write().View()(size_t(row), 0) = value;
            _changes |= CellMetadata;
            return;
        }
        auto write = _candidate->cellElemInfo.father->HostWriteData();
        _candidate->cellElemInfo(row, 0) = value;
        _changes |= CellMetadata;
    }
    std::vector<index> MeshEdit::CellNeighbors(index row) const
    {
        Check();
        CheckRow(row, NumCell());
        if (_nativeCandidate)
        {
            auto read = _nativeCandidate->relations[size_t(MeshRelation::CellCells)]->owned.Read();
            const auto values = read.View()[size_t(row)];
            return {values.begin(), values.end()};
        }
        auto values = _candidate->cell2cell[row];
        return {values.begin(), values.end()};
    }
    void MeshEdit::SetCellNeighbor(index row, rowsize slot, index value)
    {
        Check();
        CheckRow(row, NumCell());
        if (_nativeCandidate)
        {
            auto &array = _nativeCandidate->relations[size_t(MeshRelation::CellCells)]->owned;
            DNDS_check_throw_info(slot >= 0 && size_t(slot) < array.Layout().RowSize(size_t(row)), "mesh neighbor slot outside row");
            array.Write().View()(size_t(row), size_t(slot)) = value;
            _changes |= NeighborOrder;
            return;
        }
        DNDS_check_throw_info(slot >= 0 && slot < _candidate->cell2cell.RowSize(row), "mesh neighbor slot outside row");
        auto write = _candidate->cell2cell.father->HostWriteData();
        _candidate->cell2cell(row, slot) = value;
        _changes |= NeighborOrder;
    }

    void MeshEdit::Commit()
    {
        // A moved-from scope has no communicator and is a caller error. A
        // closed/aborted scope still retains its source: agree that failure
        // before any peer enters validation or publication collectives.
        DNDS_check_throw_info(_source, "cannot commit a moved-from mesh edit");
        auto &mesh = *_source;
        FieldTransferDetail::LocalStage(mesh.mpi, [&]
                                        {
            Check();
            for (index node = 0; node < NumNode(); ++node)
                DNDS_check_throw_info(Coord(node).allFinite(), "mesh edit coordinates must be finite");
            for (index cell = 0; cell < NumCell(); ++cell)
            {
                DNDS_check_throw_info(CellInfo(cell).getElemType() == _cellTypes[size_t(cell)],
                                      "cell metadata edit cannot change element type");
                auto before = _neighbors[size_t(cell)], after = CellNeighbors(cell);
                std::sort(before.begin(), before.end());
                std::sort(after.begin(), after.end());
                DNDS_check_throw_info(before == after, "neighbor-order edit must preserve the exact multiset including sentinels");
            } });
        unsigned changes = _changes;
        MPI_Allreduce(MPI_IN_PLACE, &changes, 1, MPI_UNSIGNED, MPI_BOR, mesh.mpi.comm);
        auto requirements = _requirements;
        if (changes & Coordinates)
            requirements.wallDistance = false;
        std::shared_ptr<NativeMeshState> prepared;
        if (_nativeCandidate)
        {
            auto &data = *_nativeCandidate;
            if (changes & Coordinates)
            {
                Refresh(*data.coords, mesh.Entities(EntityKind::Node), Runtime::ElementSchema::Arithmetic<real>());
                data.nodeWallDist.reset();
            }
            if (changes & CellMetadata)
            {
                MPI_Datatype type;
                MPI_Type_contiguous(ElemInfo::CommMult(), ElemInfo::CommType(), &type);
                MPI_Type_commit(&type);
                Runtime::ElementSchema schema(type, sizeof(ElemInfo), "Geom.ElemInfo.v1", Runtime::ElementSchema::TypeIdentity<ElemInfo>());
                MPI_Type_free(&type);
                Refresh(*data.cellInfo, mesh.Entities(EntityKind::Cell), schema);
            }
            if (changes & NeighborOrder)
            {
                const auto cells = mesh.Entities(EntityKind::Cell);
                auto &pair = *data.relations[size_t(MeshRelation::CellCells)];
                std::optional<Runtime::Array<index>> globals;
                FieldTransferDetail::LocalStage(mesh.mpi, [&]
                                                {
                    RelationReadAccess read(Adj::Cell2Cell, cells, cells, Adj_PointToLocal, pair.owned, &pair.ghost);
                    Runtime::RowBuilder rows(size_t(cells.NumOwned()), sizeof(index));
                    for (index row = 0; row < cells.NumOwned(); ++row)
                    {
                        rows.ResizeRow(size_t(row), read.RowSize(row));
                        auto output = rows.Row<index>(size_t(row));
                        for (size_t slot = 0; slot < output.size; ++slot)
                        {
                            const auto entry = read.ReadEntry(row, slot);
                            output[slot] = entry.absent ? UnInitIndex : entry.global;
                        }
                    }
                    globals.emplace(Runtime::Array<index>::FromSharedStorage(rows.Commit())); });
                RelationGhostRoute route(cells);
                auto ghost = route.TransferRows(cells, *globals, Runtime::ElementSchema::Arithmetic<index>());
                FieldTransferDetail::LocalStage(mesh.mpi, [&]
                                                {
                    auto read = ghost.Read();
                    Runtime::RowBuilder rows(ghost.Rows(), sizeof(index));
                    for (size_t row = 0; row < ghost.Rows(); ++row)
                    {
                        rows.ResizeRow(row, read.View().RowSize(row));
                        auto output = rows.Row<index>(row);
                        for (size_t slot = 0; slot < output.size; ++slot)
                        {
                            const auto global = read.View()(row, slot);
                            if (global == UnInitIndex) output[slot] = UnInitIndex;
                            else if (auto local = cells.GlobalToLocal(global)) output[slot] = *local;
                            else
                            {
                                output[slot] = -1 - global;
                                DNDS_check_throw_info(output[slot] != UnInitIndex, "edited neighbor target collides with absent sentinel");
                            }
                        }
                    }
                    pair.ghost = Runtime::Array<index>::FromSharedStorage(rows.Commit()); });
            }
            FieldTransferDetail::LocalStage(mesh.mpi, [&]
                                            {
                prepared = std::shared_ptr<NativeMeshState>(new NativeMeshState(*mesh._nativeState));
                // Only the explicit historical facade key is an authority alias.
                // Native sorted/set variants keep their own unchanged ordering.
                if ((changes & NeighborOrder) && prepared->_namedRelations.count(Adj::Cell2Cell))
                {
                    auto &relation = prepared->_namedRelations.at(Adj::Cell2Cell);
                    relation.data = {*data.relations[size_t(MeshRelation::CellCells)], Adj_PointToLocal};
                    relation.owned = std::make_shared<Runtime::Array<index>::ReadLease>(relation.data.rows.owned.Read());
                    relation.ghost = std::make_shared<Runtime::Array<index>::ReadLease>(relation.data.rows.ghost.Read());
                }
                prepared->_requirements = requirements;
                prepared->_placements.clear();
                prepared->_placements.emplace(-1, NativeMeshSnapshot::FromNative(data, requirements));
                prepared->_hostRead = std::make_shared<MeshReadAccess>(prepared->_placements.at(-1).AcquireRead());
                prepared->_hostView = prepared->_hostRead->HostView();
                prepared->_versions.RebindPrepared(mesh.versions); });
        }
        else
        {
            if (changes & Coordinates)
                _candidate->coords.trans.pullOnce();
            if (changes & CellMetadata)
                _candidate->cellElemInfo.trans.pullOnce();
            if (changes & NeighborOrder)
            {
                _candidate->AdjLocal2GlobalPrimary();
                _candidate->cell2cell.trans.pullOnce();
                _candidate->AdjGlobal2LocalPrimary();
            }
            std::shared_ptr<const NativeMeshState> next = NativeMeshState::CaptureLegacy(*_candidate, requirements);
            FieldTransferDetail::LocalStage(mesh.mpi, [&]
                                            {
                prepared = std::shared_ptr<NativeMeshState>(new NativeMeshState(*next));
                prepared->_versions.RebindPrepared(mesh.versions); });
        }
        _closed = true;
        _sourceRead.reset();
        mesh.CheckCollectiveMutationAvailable(true);
        FieldTransferDetail::LocalStage(mesh.mpi, [&]
                                        { _stamp.Validate(); });
        auto mask = Runtime::Change::Allocation;
        if (!mesh._nativeState)
            mask = mask | Runtime::Change::Layout;
        if (changes & Coordinates)
            mask = mask | Runtime::Change::Geometry | Runtime::Change::Values;
        if (changes & CellMetadata)
            mask = mask | Runtime::Change::Topology | Runtime::Change::Values | Runtime::Change::Interface;
        if (changes & NeighborOrder)
            mask = mask | Runtime::Change::Layout;
        // This publishes an independent current-state candidate. It deliberately
        // never enters or reads the source's detached legacy edit workspace.
        MeshMutationToken mutation(&mesh, mesh.versions, mask);
        mutation.Commit();
        try
        {
            // Owner containers were allocated before mutation. Only stamp and
            // shared-owner assignment remains after the generation commit.
            FieldTransferDetail::LocalStage(mesh.mpi, [&]
                                            { prepared->BindPreparedEpoch(mesh.versions); });
        }
        catch (...)
        {
            mesh._nativeState.reset();
            auto failed = mesh.versions.BeginMutation(Runtime::Change::All);
            throw;
        }
        mesh._nativeState = std::move(prepared);
        _candidate.reset();
        _nativeCandidate.reset();
    }
    void MeshEdit::Abort()
    {
        _closed = true;
        _sourceRead.reset();
        _candidate.reset();
        _nativeCandidate.reset();
    }
}
