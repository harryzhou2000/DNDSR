#include "Mesh.hpp"
#include "MeshAccess.hpp"
#include "DNDS/Runtime/LegacyArrayBridge.hpp"

namespace DNDS::Geom
{
    MeshReadAccess::MeshReadAccess(MeshReadAccess &&other) noexcept
        : _meshRead(std::move(other._meshRead)), _sourceRead(std::move(other._sourceRead)), _stamp(std::move(other._stamp)),
          _context(other._context), _storage(other._storage), _holds(std::move(other._holds)),
          _completion(std::move(other._completion)), _completed(other._completed), _active(other._active)
    {
        other._active = false;
    }
    template <class Pair>
    auto MeshReadAccess::Capture(Pair &pair) -> Runtime::PairReadView<typename Pair::t_arr::value_type>
    {
        using T = typename Pair::t_arr::value_type;
        auto capture = [&](const auto &legacy) -> Runtime::ArrayView<const T>
        {
            if (!legacy)
                return {};
            auto array = Runtime::ShareLegacyHost(*legacy);
            // Explicit placement conversion keeps the transitional source read
            // alive during the copy. Native topology backing will bypass this.
            if (_context.device != Runtime::Device::Host())
                array = array.To(_context.device, _context);
            auto hold = std::make_unique<ArrayHold<T>>(std::move(array), _context);
            auto view = hold->read.View();
            _holds.push_back(std::move(hold));
            return view;
        };
        return {capture(pair.father), capture(pair.son)};
    }

    MeshReadAccess::MeshReadAccess(UnstructuredMesh &mesh, MeshReadRequirements requirements,
                                   Runtime::ExecutionContext context)
        : _meshRead(mesh.versions.Read()), _stamp(mesh.versions.Stamp()), _context(context)
    {
        DNDS_check_throw_info(context.device.space != Runtime::MemorySpace::Host || !context.stream,
                              "host mesh acquisition cannot use a device stream");
        // Validate actual indices on host before device placement. This temporary
        // owner uses the same controllers; it is discarded after synchronous
        // uploads. Native backing will cache validated shape/order descriptors.
        std::unique_ptr<MeshReadAccess> hostPreflight;
        if (context.device != Runtime::Device::Host())
            hostPreflight = std::make_unique<MeshReadAccess>(mesh, requirements, Runtime::ExecutionContext::Host());
        _storage.dim = mesh.dim;
        _storage.isPeriodic = mesh.isPeriodic;
        _storage.periodicInfo = mesh.periodicInfo;
        auto source = [&](const auto &pair, const auto &canonical, bool equivalentEdgeMaps)
        {
            DNDS_check_throw_info(pair.father && pair.son && canonical.father && canonical.son,
                                  "mesh source entity storage is not initialized");
            DNDS_check_throw_info(pair.trans.father == pair.father && pair.trans.son == pair.son,
                                  "mesh source mapping is attached to different arrays");
            const auto &global = pair.trans.pLGlobalMapping;
            const auto &ghost = pair.trans.pLGhostMapping;
            const auto &expectedGlobal = canonical.trans.pLGlobalMapping;
            const auto &expectedGhost = canonical.trans.pLGhostMapping;
            DNDS_check_throw_info(global && ghost && expectedGlobal && expectedGhost &&
                                      pair.father->pLGlobalMapping == global,
                                  "mesh source mapping is not prepared");
            DNDS_check_throw_info(global->RLengths().size() == size_t(mesh.mpi.size) &&
                                      mesh.mpi.rank >= 0 && mesh.mpi.rank < mesh.mpi.size &&
                                      global->RLengths()[mesh.mpi.rank] == pair.father->Size() &&
                                      ghost->ghostIndex.size() == size_t(pair.son->Size()),
                                  "mesh source mapping extents differ from its arrays");
            bool same = global == expectedGlobal && ghost == expectedGhost;
            // BuildGhostEdge historically constructs independent route objects
            // for edge companions from the same exact ordered global/ghost IDs.
            // Confine semantic equivalence to that explicit legacy boundary.
            if (!same && equivalentEdgeMaps)
                same = global->RLengths() == expectedGlobal->RLengths() &&
                       global->ROffsets() == expectedGlobal->ROffsets() &&
                       ghost->ghostIndex == expectedGhost->ghostIndex &&
                       ghost->ghostStart == expectedGhost->ghostStart &&
                       ghost->ghostSizes == expectedGhost->ghostSizes &&
                       ghost->pushIndexStarts == expectedGhost->pushIndexStarts &&
                       ghost->pushIndexSizes == expectedGhost->pushIndexSizes &&
                       ghost->pushingIndexGlobal == expectedGhost->pushingIndexGlobal;
            DNDS_check_throw_info(same, "mesh source row mapping differs from the published entity space");
        };
        auto field = [&](auto &pair, const auto &canonical, bool equivalentEdgeMaps = false)
        {
            source(pair, canonical, equivalentEdgeMaps);
            return Capture(pair);
        };
        auto relation = [&](MeshRelation kind, auto &pair, const auto &targetMap, const auto &canonical,
                            bool equivalentEdgeMaps = false)
        {
            DNDS_check_throw_info(pair.isLocal(), "mesh query requires a published local adjacency");
            // A local index is meaningful only in the published target order.
            // Equal row counts cannot establish that correspondence.
            DNDS_check_throw_info(pair.idx.isWired() && targetMap && pair.idx.mapping() == targetMap,
                                  "mesh relation target mapping differs from the published entity space");
            auto slot = static_cast<size_t>(kind);
            _storage.relations[slot] = field(pair, canonical, equivalentEdgeMaps);
            _storage.relationStates[slot] = pair.idx.state();
            _storage.relationPresent[slot] = true;
        };
        _storage.coords = field(mesh.coords, mesh.coords);
        relation(MeshRelation::CellNodes, mesh.cell2node, mesh.coords.trans.pLGhostMapping, mesh.cell2node);
        relation(MeshRelation::BoundaryNodes, mesh.bnd2node, mesh.coords.trans.pLGhostMapping, mesh.bnd2node);
        relation(MeshRelation::BoundaryCells, mesh.bnd2cell, mesh.cell2node.trans.pLGhostMapping, mesh.bnd2node);
        relation(MeshRelation::CellCells, mesh.cell2cell, mesh.cell2node.trans.pLGhostMapping, mesh.cell2node);
        _storage.cellElemInfo = field(mesh.cellElemInfo, mesh.cell2node);
        _storage.bndElemInfo = field(mesh.bndElemInfo, mesh.bnd2node);
        if (mesh.isPeriodic)
        {
            _storage.cell2nodePbi = field(mesh.cell2nodePbi, mesh.cell2node);
            _storage.bnd2nodePbi = field(mesh.bnd2nodePbi, mesh.bnd2node);
        }
        if (requirements.faces)
        {
            relation(MeshRelation::CellFaces, mesh.cell2face, mesh.face2node.trans.pLGhostMapping, mesh.cell2node);
            relation(MeshRelation::FaceNodes, mesh.face2node, mesh.coords.trans.pLGhostMapping, mesh.face2node);
            relation(MeshRelation::FaceCells, mesh.face2cell, mesh.cell2node.trans.pLGhostMapping, mesh.face2node);
            relation(MeshRelation::FaceBoundaries, mesh.face2bnd, mesh.bnd2node.trans.pLGhostMapping, mesh.face2node);
            relation(MeshRelation::BoundaryFaces, mesh.bnd2face, mesh.face2node.trans.pLGhostMapping, mesh.bnd2node);
            _storage.faceElemInfo = field(mesh.faceElemInfo, mesh.face2node);
            if (mesh.isPeriodic)
            {
                _storage.cell2facePbi = field(mesh.cell2facePbi, mesh.cell2node);
                _storage.face2nodePbi = field(mesh.face2nodePbi, mesh.face2node);
            }
        }
        if (requirements.inverse)
        {
            relation(MeshRelation::NodeCells, mesh.node2cell, mesh.cell2node.trans.pLGhostMapping, mesh.coords);
            relation(MeshRelation::NodeBoundaries, mesh.node2bnd, mesh.bnd2node.trans.pLGhostMapping, mesh.coords);
        }
        if (requirements.edges)
        {
            relation(MeshRelation::CellEdges, mesh.cell2edge, mesh.edge2node.trans.pLGhostMapping, mesh.cell2node);
            relation(MeshRelation::EdgeNodes, mesh.edge2node, mesh.coords.trans.pLGhostMapping, mesh.edge2node);
            relation(MeshRelation::EdgeCells, mesh.edge2cell, mesh.cell2node.trans.pLGhostMapping, mesh.edge2node, true);
            _storage.edgeElemInfo = field(mesh.edgeElemInfo, mesh.edge2node, true);
            if (mesh.isPeriodic)
            {
                _storage.cell2edgePbi = field(mesh.cell2edgePbi, mesh.cell2node);
                _storage.edge2nodePbi = field(mesh.edge2nodePbi, mesh.edge2node, true);
            }
        }
        if (requirements.cellFaceNeighbors)
            relation(MeshRelation::CellFaceCells, mesh.cell2cellFace, mesh.cell2node.trans.pLGhostMapping, mesh.cell2node);
        if (requirements.wallDistance)
        {
            _storage.nodeWallDist = field(mesh.nodeWallDist, mesh.coords);
            _storage.hasNodeWallDist = true;
        }
        DNDS_check_throw_info(!requirements.allFaceCellsLocal || requirements.faces,
                              "all-face cell closure requires facial relations");
        Validate(requirements);
    }

    void MeshReadAccess::Validate(const MeshReadRequirements &requirements) const
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
        auto rows = [&](MeshRelation kind)
        { return total(_storage.relations[static_cast<size_t>(kind)]); };
        DNDS_check_throw_info(sameRows(_storage.cellElemInfo, _storage.relations[static_cast<size_t>(MeshRelation::CellNodes)]) &&
                                  sameRows(_storage.bndElemInfo, _storage.relations[static_cast<size_t>(MeshRelation::BoundaryNodes)]),
                              "mesh element information does not match entity rows");
        if (requirements.faces)
        {
            width(_storage.faceElemInfo, 1);
            width(_storage.relations[static_cast<size_t>(MeshRelation::FaceCells)], 2);
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

    MeshReadAccess UnstructuredMesh::AcquireRead(MeshReadRequirements requirements, Runtime::ExecutionContext context)
    {
        if (_nativeState)
        {
            auto sourceRead = versions.Read();
            _nativeState->CheckRequirements(requirements);
            return MeshReadAccess(_nativeState->Snapshot(context.device), context, std::move(sourceRead), versions.Stamp());
        }
        return MeshReadAccess(*this, requirements, context);
    }

    std::shared_ptr<UnstructuredMesh> UnstructuredMesh::FromNative(const MPIInfo &mpi,
                                                                   std::shared_ptr<const NativeMeshState> state)
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
        return result;
    }

    void UnstructuredMesh::PublishNative(MeshReadRequirements requirements)
    {
        CheckCollectiveMutationAvailable(true);
        std::shared_ptr<const NativeMeshState> next;
        std::exception_ptr error;
        int valid = 1;
        try
        {
            next = NativeMeshState::CaptureLegacy(*this, requirements);
        }
        catch (...)
        {
            valid = 0;
            error = std::current_exception();
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        if (!valid)
        {
            if (error)
                std::rethrow_exception(error);
            DNDS_check_throw_info(false, "native workspace import failed on another rank");
        }
        MeshMutationToken mutation(this, versions, Runtime::Change::All, [this]
                                   { _nativeState.reset(); });
        mutation.Commit();
        std::shared_ptr<const NativeMeshState> bound;
        error = {};
        valid = 1;
        try
        {
            bound = next->BindEpoch(versions);
        }
        catch (...)
        {
            valid = 0;
            error = std::current_exception();
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        if (!valid)
        {
            // Publication must never leave a silently usable mixed state if
            // post-commit owner allocation fails. Make failure sticky.
            auto failed = versions.BeginMutation(Runtime::Change::All);
            if (error)
                std::rethrow_exception(error);
            DNDS_check_throw_info(false, "native publication binding failed on another rank");
        }
        _nativeState = std::move(bound);
    }
    void UnstructuredMesh::PlaceNative(Runtime::Device device, Runtime::ExecutionContext context)
    {
        auto read = versions.Read();
        DNDS_check_throw_info(_nativeState, "publish native mesh state before placement");
        _nativeState = _nativeState->WithPlacement(device, context);
    }
    void UnstructuredMesh::UseLegacyWorkspace()
    {
        CheckCollectiveMutationAvailable(true);
        MeshMutationToken mutation(this, versions, Runtime::Change::All, [this]
                                   { _nativeState.reset(); });
        mutation.Commit();
    }
}
