#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Access/MeshAccess.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include <nlohmann/json.hpp>

namespace DNDS::Geom
{
    MeshReadAccess::MeshReadAccess(MeshReadAccess &&other) noexcept
        : _meshRead(std::move(other._meshRead)), _sourceRead(std::move(other._sourceRead)),
          _nativeOwner(std::move(other._nativeOwner)), _stamp(std::move(other._stamp)),
          _context(other._context), _storage(other._storage), _holds(std::move(other._holds)),
          _completion(std::move(other._completion)), _completed(other._completed), _active(other._active)
    {
        other._active = false;
    }
    void MeshReadAccess::Validate(const MeshReadRequirements &requirements, MeshAuditOptions audit) const
    {
        auto total = [](const auto &pair)
        { return pair.owned.Rows() + pair.ghost.Rows(); };
        auto sameRows = [](const auto &a, const auto &b)
        {
            return a.owned.Rows() == b.owned.Rows() && a.ghost.Rows() == b.ghost.Rows();
        };
        auto width = [](const auto &pair, size_t expected)
        {
            // Layouts live on host even for device arrays. Shape validation
            // below uses captured ArrayLayout through the construction helpers.
            DNDS_check_throw_info(pair.owned.width == expected && (!pair.ghost.rows || pair.ghost.width == expected),
                                  "mesh uniform component count mismatch");
        };
        width(_storage.coords, 3);
        width(_storage.cellElemInfo, 1);
        width(_storage.bndElemInfo, 1);
        auto cellPairWidth = [&](MeshRelation relation)
        {
            const auto &pair = _storage.relations[size_t(relation)];
            width(pair, 2);
            for (const auto *segment : {&pair.owned, &pair.ghost})
                DNDS_check_throw_info(!segment->offsets && !segment->sizes && segment->width == 2 && segment->stride >= 2,
                                      "physical cell incidence must use a two-slot table");
        };
        cellPairWidth(MeshRelation::BoundaryCells);
        auto rows = [&](MeshRelation kind)
        { return total(_storage.relations[static_cast<size_t>(kind)]); };
        DNDS_check_throw_info(sameRows(_storage.cellElemInfo, _storage.relations[static_cast<size_t>(MeshRelation::CellNodes)]) &&
                                  sameRows(_storage.bndElemInfo, _storage.relations[static_cast<size_t>(MeshRelation::BoundaryNodes)]),
                              "mesh element information does not match entity rows");
        if (requirements.faces)
        {
            width(_storage.faceElemInfo, 1);
            cellPairWidth(MeshRelation::FaceCells);
            DNDS_check_throw_info(sameRows(_storage.faceElemInfo, _storage.relations[static_cast<size_t>(MeshRelation::FaceNodes)]), "face information row mismatch");
        }
        if (requirements.edges)
        {
            width(_storage.edgeElemInfo, 1);
            DNDS_check_throw_info(sameRows(_storage.edgeElemInfo, _storage.relations[static_cast<size_t>(MeshRelation::EdgeNodes)]), "edge information row mismatch");
        }
        if (requirements.wallDistance)
        {
            width(_storage.nodeWallDist, 3);
            DNDS_check_throw_info(sameRows(_storage.nodeWallDist, _storage.coords), "wall distance row mismatch");
        }
        const auto nodeCount = total(_storage.coords);
        const auto cellCount = rows(MeshRelation::CellNodes);
        const auto bndCount = rows(MeshRelation::BoundaryNodes);
        const auto faceCount = rows(MeshRelation::FaceNodes);
        const auto edgeCount = rows(MeshRelation::EdgeNodes);
        const auto &cellRows = _storage.relations[static_cast<size_t>(MeshRelation::CellNodes)];
        const auto &bndRows = _storage.relations[static_cast<size_t>(MeshRelation::BoundaryNodes)];
        const auto &faceRows = _storage.relations[static_cast<size_t>(MeshRelation::FaceNodes)];
        const auto &edgeRows = _storage.relations[static_cast<size_t>(MeshRelation::EdgeNodes)];
        if (_context.device == Runtime::Device::Host())
        {
            auto elementRows = [&](const auto &info, const auto &nodes, int dimension)
            {
                for (size_t row = 0; row < total(nodes); ++row)
                {
                    const auto type = info(row, 0).type;
                    // Current supported element IDs are the contiguous range
                    // [Line2, ElemType_NUM); reject before dispatch can assert.
                    DNDS_check_throw_info(type > Elem::UnknownElem && type < Elem::ElemType_NUM,
                                          "mesh element information has an unsupported type");
                    const Elem::Element element{static_cast<Elem::ElemType>(type)};
                    DNDS_check_throw_info(element.GetDim() == dimension,
                                          "mesh element dimension differs from its entity role");
                    DNDS_check_throw_info(nodes.RowSize(row) == size_t(element.GetNumNodes()) &&
                                              nodes.RowSize(row) <= size_t(Elem::CellNumNodeMax),
                                          "mesh node incidence width differs from element type");
                }
            };
            elementRows(_storage.cellElemInfo, cellRows, _storage.dim);
            elementRows(_storage.bndElemInfo, bndRows, _storage.dim - 1);
            if (requirements.faces)
            {
                elementRows(_storage.faceElemInfo, faceRows, _storage.dim - 1);
                const auto &cellFaces = _storage.relations[static_cast<size_t>(MeshRelation::CellFaces)];
                DNDS_check_throw_info(sameRows(cellFaces, cellRows), "cell face rows differ from cell distribution");
                for (size_t row = 0; row < total(cellRows); ++row)
                    DNDS_check_throw_info(cellFaces.RowSize(row) == size_t(Elem::Element{_storage.cellElemInfo(row, 0).getElemType()}.GetNumFaces()),
                                          "mesh face incidence width differs from cell element type");
            }
            if (requirements.edges)
                elementRows(_storage.edgeElemInfo, edgeRows, 1);
        }
        const size_t sourceOwned[]{cellRows.owned.Rows(), bndRows.owned.Rows(), bndRows.owned.Rows(), cellRows.owned.Rows(),
                                   _storage.coords.owned.Rows(), _storage.coords.owned.Rows(), cellRows.owned.Rows(),
                                   faceRows.owned.Rows(), faceRows.owned.Rows(), faceRows.owned.Rows(), bndRows.owned.Rows(),
                                   cellRows.owned.Rows(), edgeRows.owned.Rows(), edgeRows.owned.Rows(), cellRows.owned.Rows()};
        const size_t sourceGhost[]{cellRows.ghost.Rows(), bndRows.ghost.Rows(), bndRows.ghost.Rows(), cellRows.ghost.Rows(),
                                   _storage.coords.ghost.Rows(), _storage.coords.ghost.Rows(), cellRows.ghost.Rows(),
                                   faceRows.ghost.Rows(), faceRows.ghost.Rows(), faceRows.ghost.Rows(), bndRows.ghost.Rows(),
                                   cellRows.ghost.Rows(), edgeRows.ghost.Rows(), edgeRows.ghost.Rows(), cellRows.ghost.Rows()};
        const size_t targetCounts[]{nodeCount, nodeCount, cellCount, cellCount, cellCount, bndCount,
                                    faceCount, nodeCount, cellCount, bndCount, faceCount,
                                    edgeCount, nodeCount, cellCount, cellCount};
        std::vector<bool> requiredFaces(faceCount, false);
        if (requirements.faces && _context.device == Runtime::Device::Host())
        {
            const auto &faceCells = _storage.relations[static_cast<size_t>(MeshRelation::FaceCells)];
            for (size_t face = 0; face < faceCount; ++face)
                DNDS_check_throw_info((_storage.isPeriodic && FaceIDIsPeriodic(_storage.faceElemInfo(face, 0).zone)) || faceCells(face, 0) < 0 ||
                                          faceCells(face, 0) != faceCells(face, 1),
                                      "only a periodic face can reference the same cell on both sides");
            for (size_t i = 0; i < (requirements.allFaceCellsLocal ? faceCount : faceRows.owned.Rows()); ++i)
                requiredFaces[i] = true;
            const auto &cellFaces = _storage.relations[static_cast<size_t>(MeshRelation::CellFaces)];
            for (size_t i = 0; i < cellFaces.owned.Rows(); ++i)
                for (size_t j = 0; j < cellFaces.RowSize(i); ++j)
                {
                    const auto face = cellFaces(i, j);
                    DNDS_check_throw_info(face >= 0 && size_t(face) < faceCount,
                                          "owned cell face is outside requested local closure");
                    if (audit.closure)
                        DNDS_check_throw_info(faceCells(size_t(face), 0) == index(i) || faceCells(size_t(face), 1) == index(i),
                                              "owned cell face does not reference its source cell");
                    requiredFaces[size_t(face)] = true;
                }
        }
        for (size_t r = 0; r < static_cast<size_t>(MeshRelation::Count); ++r)
        {
            if (!_storage.relationPresent[r])
                continue;
            const auto &relation = _storage.relations[r];
            DNDS_check_throw_info(relation.owned.Rows() == sourceOwned[r] && relation.ghost.Rows() == sourceGhost[r],
                                  "mesh relation owned/ghost rows differ from source entity space");
            if (_context.device != Runtime::Device::Host())
                continue;
            const auto kind = static_cast<MeshRelation>(r);
            const bool requireLocal = kind == MeshRelation::CellNodes || kind == MeshRelation::BoundaryNodes ||
                                      kind == MeshRelation::FaceNodes || kind == MeshRelation::EdgeNodes;
            for (size_t i = 0; i < total(relation); ++i)
                for (size_t j = 0; j < relation.RowSize(i); ++j)
                {
                    const auto target = relation(i, j);
                    const bool validLocal = target >= 0 && size_t(target) < targetCounts[r];
                    // Optional outer rows keep the legacy encoded-unavailable
                    // convention. Inverse support lists can be incomplete even
                    // for owned nodes unless a wider cell closure was requested.
                    bool unavailableAllowed = !requireLocal &&
                                              (i >= relation.owned.Rows() || kind == MeshRelation::NodeCells ||
                                               kind == MeshRelation::NodeBoundaries);
                    bool absentAllowed = kind == MeshRelation::FaceBoundaries ||
                                         kind == MeshRelation::BoundaryFaces ||
                                         kind == MeshRelation::CellFaceCells ||
                                         (kind == MeshRelation::BoundaryCells && j == 1);
                    if (kind == MeshRelation::FaceCells)
                    {
                        const bool external = FaceIDIsExternalBC(_storage.faceElemInfo(i, 0).zone);
                        absentAllowed = j == 1 && external;
                        unavailableAllowed = !requiredFaces[i] && target != UnInitIndex;
                        DNDS_check_throw_info(j != 1 || external == (target == UnInitIndex),
                                              "mesh face second-cell sentinel disagrees with its boundary zone");
                    }
                    DNDS_check_throw_info(validLocal || (target == UnInitIndex && absentAllowed) ||
                                              (target < 0 && unavailableAllowed),
                                          "mesh relation target is not a valid local entity or permitted sentinel");
                }
        }
        auto periodicShape = [&](const auto &bits, MeshRelation kind)
        {
            const auto &relation = _storage.relations[static_cast<size_t>(kind)];
            DNDS_check_throw_info(bits.owned.Rows() == relation.owned.Rows() && bits.ghost.Rows() == relation.ghost.Rows(),
                                  "mesh periodic row distribution mismatch");
            if (_context.device == Runtime::Device::Host())
                for (size_t i = 0; i < total(relation); ++i)
                    DNDS_check_throw_info(bits.RowSize(i) == relation.RowSize(i), "mesh periodic incidence width mismatch");
        };
        if (_storage.isPeriodic)
        {
            periodicShape(_storage.cell2nodePbi, MeshRelation::CellNodes);
            periodicShape(_storage.bnd2nodePbi, MeshRelation::BoundaryNodes);
            if (requirements.faces)
            {
                periodicShape(_storage.cell2facePbi, MeshRelation::CellFaces);
                periodicShape(_storage.face2nodePbi, MeshRelation::FaceNodes);
                if (_context.device == Runtime::Device::Host())
                {
                    const auto &cellFaces = _storage.relations[static_cast<size_t>(MeshRelation::CellFaces)];
                    const auto &faceCells = _storage.relations[static_cast<size_t>(MeshRelation::FaceCells)];
                    for (size_t cell = 0; cell < cellFaces.owned.Rows(); ++cell)
                        for (size_t slot = 0; slot < cellFaces.RowSize(cell); ++slot)
                        {
                            const auto face = cellFaces(cell, slot);
                            if (faceCells(size_t(face), 0) != index(cell) || faceCells(size_t(face), 1) != index(cell))
                                continue;
                            size_t incidences = 0, backSides = 0;
                            for (size_t other = 0; other < cellFaces.RowSize(cell); ++other)
                                if (cellFaces(cell, other) == face)
                                {
                                    ++incidences;
                                    backSides += !bool(_storage.cell2facePbi(cell, other));
                                }
                            DNDS_check_throw_info(incidences == 2 && backSides == 1,
                                                  "self-periodic face requires exactly two cell-face incidences with opposite side bits");
                        }
                }
            }
            if (requirements.edges)
            {
                periodicShape(_storage.cell2edgePbi, MeshRelation::CellEdges);
                periodicShape(_storage.edge2nodePbi, MeshRelation::EdgeNodes);
            }
        }
    }

    MeshReadAccess::~MeshReadAccess()
    {
        // Keep the mesh/version lease until all device uses have finished, not
        // merely until launch returns. Worker CPU users retain this owner.
        Wait();
    }
    UnstructuredMeshView MeshReadAccess::HostView() const
    {
        DNDS_check_throw_info(_active && !_completed, "mesh read access was moved or sealed");
        DNDS_check_throw_info(_context.device == Runtime::Device::Host(), "HostView requires host acquisition");
        _stamp.Validate();
        return UnstructuredMeshView(_storage);
    }
    UnstructuredMeshView MeshReadAccess::DeviceView() const
    {
        DNDS_check_throw_info(_active && !_completed, "mesh read access was moved or sealed");
        DNDS_check_throw_info(_context.device.space == Runtime::MemorySpace::CUDA, "DeviceView requires CUDA acquisition");
        _stamp.Validate();
        return UnstructuredMeshView(_storage);
    }
    void MeshReadAccess::Complete(Runtime::Completion completion)
    {
        DNDS_check_throw_info(_active && !_completed && completion.Matches(_context), "mesh completion context mismatch or already attached");
        _completion = std::move(completion);
        _completed = true;
        for (auto &hold : _holds)
            hold->Complete(_completion);
    }
    void MeshReadAccess::Wait()
    {
        if (!_active)
            return;
        if (!_completed)
            Complete(Runtime::Completion::Record(_context));
        _completion.Wait();
    }

    void MeshReadAccess::Restrict(const MeshReadRequirements &requirements)
    {
        auto hide = [&](std::initializer_list<MeshRelation> relations)
        {
            for (auto relation : relations)
            {
                const auto slot = size_t(relation);
                _storage.relationPresent[slot] = false;
                _storage.relationStates[slot] = Adj_Unknown;
                _storage.relations[slot] = {};
            }
        };
        if (!requirements.faces)
        {
            hide({MeshRelation::CellFaces, MeshRelation::FaceNodes, MeshRelation::FaceCells,
                  MeshRelation::FaceBoundaries, MeshRelation::BoundaryFaces});
            _storage.faceElemInfo = {};
            _storage.cell2facePbi = {};
            _storage.face2nodePbi = {};
        }
        if (!requirements.inverse)
            hide({MeshRelation::NodeCells, MeshRelation::NodeBoundaries});
        if (!requirements.edges)
        {
            hide({MeshRelation::CellEdges, MeshRelation::EdgeNodes, MeshRelation::EdgeCells});
            _storage.edgeElemInfo = {};
            _storage.cell2edgePbi = {};
            _storage.edge2nodePbi = {};
        }
        if (!requirements.cellFaceNeighbors)
            hide({MeshRelation::CellFaceCells});
        if (!requirements.wallDistance)
        {
            _storage.nodeWallDist = {};
            _storage.hasNodeWallDist = false;
        }
    }

    MeshReadAccess UnstructuredMesh::AcquireRead(MeshReadRequirements requirements, Runtime::ExecutionContext context)
    {
        DNDS_check_throw_info(_nativeState, "mesh read requires native authority");
        auto sourceRead = versions.Read();
        _nativeState->CheckRequirements(requirements);
        MeshReadAccess result(_nativeState->Snapshot(context.device), context, std::move(sourceRead), versions.Stamp());
        result._nativeOwner = _nativeState;
        result.Restrict(requirements);
        return result;
    }

    std::shared_ptr<UnstructuredMesh> UnstructuredMesh::FromNative(const MPIInfo &mpi,
                                                                   std::shared_ptr<const UnstructuredMeshState> state)
    {
        DNDS_check_throw_info(state, "native mesh state is missing");
        state->Validate();
        const auto nodes = state->Entities(EntityKind::Node);
        EntitySpace::CheckCommunicator(nodes._mpi.comm, mpi.comm);
        auto result = std::make_shared<UnstructuredMesh>(mpi, state->BorrowHostView().Dimension());
        result->_entityIdentity = nodes._identity;
        result->isPeriodic = state->BorrowHostView().IsPeriodic();
        result->periodicInfo = state->BorrowHostView().Periodic();
        result->_nativeState = state->BindEpoch(result->versions);
        result->_nativeSmoothing.reset();
        return result;
    }

    void UnstructuredMesh::ReplaceNativeForSetup(std::shared_ptr<const UnstructuredMeshState> source)
    {
        CheckCollectiveMutationAvailable(true);
        std::shared_ptr<UnstructuredMeshState> prepared;
        std::shared_ptr<const uint8_t> identity;
        Periodicity periodic;
        bool nextPeriodic = false;
        std::string signature;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            int actualRank = -1, actualSize = 0;
            MPI_Comm_rank(mpi.comm, &actualRank);
            MPI_Comm_size(mpi.comm, &actualSize);
            DNDS_check_throw_info(actualRank == mpi.rank && actualSize == mpi.size && source,
                                  "native setup replacement requires valid MPIInfo and source");
            source->Validate();
            const auto nodes = source->Entities(EntityKind::Node);
            EntitySpace::CheckCommunicator(nodes._mpi.comm, mpi.comm);
            const auto &view = source->BorrowHostView();
            DNDS_check_throw_info(view.Dimension() == getDim(), "native setup replacement dimension differs");
            signature = source->MeshNamespace() + ":" + std::to_string(view.Dimension()) + ":" + std::to_string(view.IsPeriodic());
            for (const auto &[kind, family] : source->_spaces)
            {
                EntitySpace::CheckCommunicator(family.identity._mpi.comm, mpi.comm);
                signature += "k" + std::to_string(int(kind)) + ":";
                for (auto count : family.global->RLengths()) signature += std::to_string(count) + ",";
            }
            for (const auto &[kind, next] : source->IdHighWater()) signature += "i" + std::to_string(int(kind)) + ":" + std::to_string(next) + ";";
            const auto r = source->Requirements();
            signature += "q" + std::to_string(r.faces) + std::to_string(r.inverse) + std::to_string(r.edges) +
                         std::to_string(r.cellFaceNeighbors) + std::to_string(r.wallDistance) + std::to_string(r.allFaceCellsLocal);
            for (size_t slot = 0; slot < size_t(MeshRelation::Count); ++slot)
                signature += view.HasRelation(static_cast<MeshRelation>(slot)) ? "r1" : "r0";
            for (const auto &key : source->RelationKeys())
            { const auto value = key.Signature(); signature += "a" + std::to_string(value.size()) + ":" + value; }
            std::vector<real> frame;
            for (size_t f = 0; f < view.Periodic().rotation.size(); ++f)
            {
                for (int entry = 0; entry < 9; ++entry) frame.push_back(view.Periodic().rotation[f].map().data()[entry]);
                for (int entry = 0; entry < 3; ++entry) frame.push_back(view.Periodic().translation[f].map()[entry]);
                for (int entry = 0; entry < 3; ++entry) frame.push_back(view.Periodic().rotationCenter[f].map()[entry]);
            }
            signature += nlohmann::json(frame).dump();
            signature += source->Provenance() ? source->Provenance()->Signature() : "none";
            DNDS_check_throw_info(signature.size() <= size_t(INT_MAX), "native setup replacement metadata too large");
            prepared = std::shared_ptr<UnstructuredMeshState>(new UnstructuredMeshState(*source));
            // Prepare all allocations and descriptor bindings before publication.
            prepared->BindPreparedEpoch(versions);
            identity = nodes._identity;
            periodic = view.Periodic();
            nextPeriodic = view.IsPeriodic(); });
        int size = mpi.rank == 0 ? int(signature.size()) : 0;
        MPI_Bcast(&size, 1, MPI_INT, 0, mpi.comm);
        std::string expected;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { expected = mpi.rank == 0 ? signature : std::string(size_t(size), '\0'); });
        MPI_Bcast(expected.data(), size, MPI_CHAR, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(signature == expected, "native setup source metadata differs across ranks"); });
        MeshMutationToken mutation(this, versions, Runtime::Change::All);
        mutation.Commit();
        // Refresh stamps after commit; these operations only rebind shared
        // handles or clear existing storage and cannot allocate.
        prepared->BindPreparedEpoch(versions);
        _entityIdentity = std::move(identity);
        isPeriodic = nextPeriodic;
        periodicInfo = periodic;
        _nativeState = std::move(prepared);
        _nativeSmoothing.reset();
        node2bndNode.clear();
        node2parentNode.clear();
        cell2parentCell.clear();
    }

    void UnstructuredMesh::PlaceNative(Runtime::Device device, Runtime::ExecutionContext context)
    {
        auto read = versions.Read();
        DNDS_check_throw_info(_nativeState, "publish native mesh state before placement");
        _nativeState = _nativeState->WithPlacement(device, context);
        _nativeSmoothing.reset();
    }

}
