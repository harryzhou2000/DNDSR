#include "MeshEdit.hpp"
#include "MeshTransition.hpp"
#include "FieldTransfer.hpp"

namespace DNDS::Geom
{
    namespace
    {
        constexpr unsigned Coordinates = 1, CellMetadata = 2, NeighborOrder = 4;

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
        return _candidate->NumNode();
    }
    index MeshEdit::NumCell() const
    {
        Check();
        return _candidate->NumCell();
    }
    tPoint MeshEdit::Coord(index row) const
    {
        Check();
        CheckRow(row, _candidate->NumNode());
        return _candidate->Coord(row);
    }
    void MeshEdit::SetCoord(index row, const tPoint &value)
    {
        Check();
        CheckRow(row, _candidate->NumNode());
        auto write = _candidate->coords.father->HostWriteData();
        _candidate->coords[row] = value;
        _changes |= Coordinates;
    }
    ElemInfo MeshEdit::CellInfo(index row) const
    {
        Check();
        CheckRow(row, _candidate->NumCell());
        return _candidate->GetCellInfo(row);
    }
    void MeshEdit::SetCellInfo(index row, const ElemInfo &value)
    {
        Check();
        CheckRow(row, _candidate->NumCell());
        auto write = _candidate->cellElemInfo.father->HostWriteData();
        _candidate->cellElemInfo(row, 0) = value;
        _changes |= CellMetadata;
    }
    std::vector<index> MeshEdit::CellNeighbors(index row) const
    {
        Check();
        CheckRow(row, _candidate->NumCell());
        auto values = _candidate->cell2cell[row];
        return {values.begin(), values.end()};
    }
    void MeshEdit::SetCellNeighbor(index row, rowsize slot, index value)
    {
        Check();
        CheckRow(row, _candidate->NumCell());
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
            for (index node = 0; node < _candidate->NumNode(); ++node)
                DNDS_check_throw_info(_candidate->Coord(node).allFinite(), "mesh edit coordinates must be finite");
            for (index cell = 0; cell < _candidate->NumCell(); ++cell)
            {
                DNDS_check_throw_info(_candidate->GetCellInfo(cell).getElemType() == _cellTypes[size_t(cell)],
                                      "cell metadata edit cannot change element type");
                auto before = _neighbors[size_t(cell)], after = CellNeighbors(cell);
                std::sort(before.begin(), before.end());
                std::sort(after.begin(), after.end());
                DNDS_check_throw_info(before == after, "neighbor-order edit must preserve the exact multiset including sentinels");
            } });
        unsigned changes = _changes;
        MPI_Allreduce(MPI_IN_PLACE, &changes, 1, MPI_UNSIGNED, MPI_BOR, mesh.mpi.comm);
        if (changes & Coordinates)
            _candidate->coords.trans.pullOnce();
        if (changes & CellMetadata)
            _candidate->cellElemInfo.trans.pullOnce();
        if (changes & NeighborOrder)
        {
            // Local IDs differ across ranks. Refresh adjacency rows in global
            // numbering, then resolve them in each receiver's retained maps.
            _candidate->AdjLocal2GlobalPrimary();
            _candidate->cell2cell.trans.pullOnce();
            _candidate->AdjGlobal2LocalPrimary();
        }
        auto requirements = _requirements;
        // Arbitrary coordinate motion invalidates a captured wall-distance field.
        if (changes & Coordinates)
            requirements.wallDistance = false;
        std::shared_ptr<const NativeMeshState> next;
        FieldTransferDetail::LocalStage(mesh.mpi, [&]
                                        { next = NativeMeshState::CaptureLegacy(*_candidate, requirements); });
        std::shared_ptr<NativeMeshState> prepared;
        FieldTransferDetail::LocalStage(mesh.mpi, [&]
                                        {
            prepared = std::shared_ptr<NativeMeshState>(new NativeMeshState(*next));
            if (mesh._nativeState)
            {
                // Exact ordering was checked before editing. Retain private
                // canonical map identity and unchanged original-ID ownership.
                prepared->_spaces = mesh._nativeState->_spaces;
                prepared->_originalIds = mesh._nativeState->_originalIds;
                prepared->_originalReads = mesh._nativeState->_originalReads;
            }
            prepared->_versions = mesh.versions; });
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
    }
    void MeshEdit::Abort()
    {
        _closed = true;
        _sourceRead.reset();
        _candidate.reset();
    }
}
