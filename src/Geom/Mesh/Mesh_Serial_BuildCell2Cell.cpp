#include "Mesh.hpp"
#include "NativeMeshPeriodicImport.hpp"
#include "FieldTransfer.hpp"
#include "DNDS/Runtime/LegacyArrayBridge.hpp"

#include <set>

#include <nanoflann.hpp>

#include "Geom/PointCloud.hpp"
#include <unordered_map>
#include <omp.h>

namespace DNDS::Geom
{

    void UnstructuredMeshSerialRW::
        Deduplicate1to1Periodic(real search_eps)
    {
        mesh->CheckCollectiveMutationAvailable();
        NativeSerialMeshData prepared;
        FieldTransferDetail::LocalStage(mesh->mpi, [&]
                                        {
            DNDS_check_throw_info(dataIsSerialIn, "periodic identification requires serial input");
            NativeSerialMeshData source;
            source.coordinates = Runtime::ShareLegacyHost(*coordSerial);
            source.cellNodes = Runtime::ShareLegacyHost(*cell2nodeSerial);
            source.boundaryNodes = Runtime::ShareLegacyHost(*bnd2nodeSerial);
            source.cellInfo = Runtime::ShareLegacyHost(*cellElemInfoSerial);
            source.boundaryInfo = Runtime::ShareLegacyHost(*bndElemInfoSerial);
            source.nodeIds = Runtime::ShareLegacyHost(*node2nodeOrigSerial);
            source.cellIds = Runtime::ShareLegacyHost(*cell2cellOrigSerial);
            source.boundaryIds = Runtime::ShareLegacyHost(*bnd2bndOrigSerial);
            prepared = CanonicalizeNativePeriodic(std::move(source), {mesh->periodicInfo, search_eps}); });
        int periodic = prepared.periodic;
        MPI_Allreduce(MPI_IN_PLACE, &periodic, 1, MPI_INT, MPI_MAX, mesh->mpi.comm);
        auto nextCoords = coordSerial;
        auto nextCells = cell2nodeSerial, nextBoundaries = bnd2nodeSerial;
        auto nextIds = node2nodeOrigSerial;
        decltype(cell2nodePbiSerial) nextCellBits;
        decltype(bnd2nodePbiSerial) nextBoundaryBits;
        if (periodic)
            FieldTransferDetail::LocalStage(mesh->mpi, [&]
                                            {
                auto copy = [&](auto &destination, const auto &source, MPI_Datatype datatype, int count)
                {
                    using Array = typename std::decay_t<decltype(destination)>::element_type;
                    auto next = std::make_shared<Array>(ObjName{"native periodic compatibility"}, datatype, count, mesh->mpi);
                    next->Resize(index(source.Rows()));
                    auto read = source.Read();
                    for (size_t row = 0; row < source.Rows(); ++row)
                    {
                        if constexpr (Array::IsCSR()) next->ResizeRow(index(row), rowsize(read.View().RowSize(row)));
                        for (size_t slot = 0; slot < read.View().RowSize(row); ++slot) (*next)(index(row), rowsize(slot)) = read.View()(row, slot);
                    }
                    next->Compress();
                    destination = std::move(next);
                };
                copy(nextCoords, prepared.coordinates, MPI_DOUBLE, 1);
                copy(nextCells, prepared.cellNodes, DNDS_MPI_INDEX, 1);
                copy(nextBoundaries, prepared.boundaryNodes, DNDS_MPI_INDEX, 1);
                copy(nextIds, prepared.nodeIds, DNDS_MPI_INDEX, 1);
                Runtime::Array<NodePeriodicBits> empty(Runtime::RowLayout::Ragged({}));
                copy(nextCellBits, prepared.cellNodesPbi ? *prepared.cellNodesPbi : empty, NodePeriodicBits::CommType(), 1);
                copy(nextBoundaryBits, prepared.boundaryNodesPbi ? *prepared.boundaryNodesPbi : empty, NodePeriodicBits::CommType(), 1); });
        auto mutation = mesh->BeginManagedMutation(Runtime::Change::Geometry | Runtime::Change::Interface);
        mesh->isPeriodic = periodic;
        if (periodic)
        {
            coordSerial = std::move(nextCoords);
            cell2nodeSerial = std::move(nextCells);
            bnd2nodeSerial = std::move(nextBoundaries);
            node2nodeOrigSerial = std::move(nextIds);
            cell2nodePbiSerial = std::move(nextCellBits);
            bnd2nodePbiSerial = std::move(nextBoundaryBits);
        }
        mutation.Commit();
    }

    void UnstructuredMeshSerialRW::
        BuildCell2Cell() // currently does not handle parallel input mode
    {
        DNDS_assert(this->dataIsSerialIn);
        real t0 = MPI_Wtime();
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  BuildCell2Cell "
#ifdef DNDS_USE_OMP
                        << fmt::format("Using OMP [{}]", omp_get_max_threads())
#endif
                        << std::endl;
        cell2cellSerial = make_ssp<decltype(cell2cellSerial)::element_type>(ObjName{"BuildCell2Cell::cell2cellSerial"}, mesh->getMPI());
        // if (mRank != mesh->getMPI().rank)
        //     return;
        /// TODO: abstract these: invert cone (like node 2 cell -> cell 2 node) (also support operating on pair)
        /**********************************************************************************************************************/
        // getting node2cell // only primary vertices
        std::vector<std::vector<DNDS::index>> node2cell(coordSerial->Size());
        // std::vector<DNDS::rowsize> node2cellSz(coordSerial->Size(), 0);
        // for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
        //     for (DNDS::rowsize iN = 0; iN < Elem::Element{(*cellElemInfoSerial)(iCell, 0).getElemType()}.GetNumVertices(); iN++)
        //         node2cellSz[(*cell2nodeSerial)(iCell, iN)]++;
        // for (DNDS::index iNode = 0; iNode < node2cell.size(); iNode++)
        //     node2cell[iNode].reserve(node2cellSz[iNode]);
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            for (DNDS::rowsize iN = 0; iN < Elem::Element{(*cellElemInfoSerial)(iCell, 0).getElemType()}.GetNumVertices(); iN++)
                node2cell[(*cell2nodeSerial)(iCell, iN)].push_back(iCell);
        // if (mesh->getMPI().rank == mRank)
        //     for (auto &row : node2cell)
        //         if (row.size() != 3)
        //             std::cout << "fucked not 4" << std::endl;
        // node2cellSz.clear();
        /**********************************************************************************************************************/
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  BuildCell2Cell Part 1" << std::endl;
        cell2cellSerial->Resize(cell2nodeSerial->Size(), 32); // 32 is a guess for the number of nodal neighbors

        index nCells = cell2nodeSerial->Size();
        index nCellsDone = 0;
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
        {
            auto CellElem = Elem::Element{(*cellElemInfoSerial)[iCell]->getElemType()};
            std::vector<DNDS::index> cell2nodeRow{
                (*cell2nodeSerial)[iCell].begin(),
                (*cell2nodeSerial)[iCell].begin() + CellElem.GetNumVertices()};
            std::sort(cell2nodeRow.begin(), cell2nodeRow.end());
            // only primary vertices
            std::unordered_set<DNDS::index> c_neighbors_hash;
            c_neighbors_hash.reserve(64);
            std::vector<DNDS::index>
                c_neighbors;
            c_neighbors.reserve(64);
            // std::set<DNDS::index> c_neighbors; // could optimize?
            //                                    // std::vector<DNDS::index> c_neighbors;
            //                                    // c_neighbors.reserve(30);
            //                                    /****/
            for (auto iNode : cell2nodeRow)
                for (auto iCellOther : node2cell[iNode])
                {
                    if (iCellOther == iCell || c_neighbors_hash.count(iCellOther) == 1)
                        continue;
                    //! ** override: point-complete cell2cell info!
                    c_neighbors.push_back(iCellOther);
                    c_neighbors_hash.insert(iCellOther);
                    continue;
                    //! ** override: point-complete cell2cell info!

                    auto CellElemO = Elem::Element{(*cellElemInfoSerial)[iCellOther]->getElemType()};
                    std::vector<DNDS::index> cell2nodeRowO{
                        (*cell2nodeSerial)[iCellOther].begin(),
                        (*cell2nodeSerial)[iCellOther].begin() + CellElemO.GetNumVertices()};
                    std::sort(cell2nodeRowO.begin(), cell2nodeRowO.end());
                    std::vector<DNDS::index> interSect;
                    interSect.reserve(32);
                    std::set_intersection(cell2nodeRowO.begin(), cell2nodeRowO.end(), cell2nodeRow.begin(), cell2nodeRow.end(),
                                          std::back_inserter(interSect));
                    if ((iCellOther != iCell) && interSect.size() > (mesh->dim - 1)) //! mesh->dim - 1 is a simple method

                    // (false ||
                    //  Elem::cellsAreFaceConnected(
                    //      cell2nodeSerial->operator[](iCell),
                    //      cell2nodeSerial->operator[](iCellOther),
                    //      CellElem,
                    //      Elem::Element{(*cellElemInfoSerial)[iCellOther]->getElemType()}))
                    {
                        c_neighbors_hash.insert(iCellOther);
                        c_neighbors.push_back(iCellOther);
                    }
                }
            // if (!c_neighbors.erase(iCell))
            //     DNDS_assert_info(false, "neighbors should include myself now");
            // std::sort(c_neighbors.begin(), c_neighbors.end());
            // auto last = std::unique(c_neighbors.begin(), c_neighbors.end());
            // c_neighbors.erase(last, c_neighbors.end());

            std::sort(c_neighbors.begin(), c_neighbors.end());
            cell2cellSerial->ResizeRow(iCell, c_neighbors.size());
            DNDS::rowsize ic2c = 0;
            for (auto iCellOther : c_neighbors)
                (*cell2cellSerial)(iCell, ic2c++) = iCellOther;
            index myDone;
#ifdef DNDS_USE_OMP
#    pragma omp atomic capture
#endif
            myDone = nCellsDone++;

            if (myDone % (nCells / 50 + 1) == 0 || myDone == nCells - 1)
            {
#ifdef DNDS_USE_OMP
#    pragma omp critical
#endif
                {
                    auto fmt = DNDS::log().flags();
                    print_progress(log(), double(myDone + 1) / nCells);
                    DNDS::log().setf(fmt);
                }
            }
        }
        (*cell2cellSerial).Compress();
        if (mesh->getMPI().rank == mRank)
            log() << std::endl;

        /*************************************************************************************************/
        cell2cellSerialFacial = make_ssp<decltype(cell2cellSerialFacial)::element_type>(ObjName{"BuildCell2Cell::cell2cellSerialFacial"}, mesh->getMPI());
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  BuildCell2Cell Part 2" << std::endl;
        cell2cellSerialFacial->Resize(cell2cellSerial->Size(), 6);
        nCellsDone = 0;
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
        for (index iCell = 0; iCell < cell2cellSerial->Size(); iCell++)
        {
            std::vector<index> facialNeighbors;
            facialNeighbors.reserve(6);
            std::vector<index> c2ni((*cell2nodeSerial)[iCell].begin(),
                                    (*cell2nodeSerial)[iCell].begin() +
                                        Elem::Element{(*cellElemInfoSerial)[iCell]->getElemType()}.GetNumVertices());
            std::sort(c2ni.begin(), c2ni.end());
            for (auto iCellOther : (*cell2cellSerial)[iCell])
            {
                std::vector<index> c2nj((*cell2nodeSerial)[iCellOther].begin(),
                                        (*cell2nodeSerial)[iCellOther].begin() +
                                            Elem::Element{(*cellElemInfoSerial)[iCellOther]->getElemType()}.GetNumVertices());
                std::sort(c2nj.begin(), c2nj.end());
                std::vector<index> intersect;
                intersect.reserve(9);
                std::set_intersection(c2ni.begin(), c2ni.end(), c2nj.begin(), c2nj.end(), std::back_inserter(intersect));
                if (intersect.size() >= mesh->dim) // for 2d, exactly 2; for 3d, 3 or 4
                    facialNeighbors.push_back(iCellOther);
            }
            (*cell2cellSerialFacial).ResizeRow(iCell, facialNeighbors.size());
            (*cell2cellSerialFacial)[iCell] = facialNeighbors;
            index myDone;
#ifdef DNDS_USE_OMP
#    pragma omp atomic capture
#endif
            myDone = nCellsDone++;

            if (myDone % (nCells / 50 + 1) == 0 || myDone == nCells - 1)
            {
#ifdef DNDS_USE_OMP
#    pragma omp critical
#endif
                {
                    auto fmt = DNDS::log().flags();
                    print_progress(log(), double(myDone + 1) / nCells);
                    DNDS::log().setf(fmt);
                }
            }
        }
        if (mesh->getMPI().rank == mRank)
            log() << std::endl;
        /*************************************************************************************************/

        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Done  BuildCell2Cell"
                        << fmt::format(" time [{:.4g}]", MPI_Wtime() - t0) << std::endl;
    }

    void UnstructuredMesh::RecreatePeriodicNodes()
    {
        CheckCollectiveMutationAvailable();
        auto mutationToken = BeginManagedMutation(Runtime::Change::Interface | Runtime::Change::Values);

        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(cell2node.isLocal());
        DNDS_assert(cell2node.trans.pLGhostMapping);
        DNDS_assert(coords.trans.pLGhostMapping && coords.trans.pLGlobalMapping);
        if (!isPeriodic)
        {
            mutationToken.Commit();
            return;
        }
        tAdj1Pair nodeNeedCreate;
        nodeNeedCreate.InitPair("RecreatePeriodicNodes::nodeNeedCreate", mpi);
        nodeNeedCreate.TransAttach();
        nodeNeedCreate.father->Resize(coords.father->Size());
        nodeNeedCreate.trans.BorrowGGIndexing(coords.trans);
        nodeNeedCreate.trans.createMPITypes();
        // std::cout << "here X0.5" << std::endl;
        for (index i = 0; i < nodeNeedCreate.Size(); i++)
            nodeNeedCreate(i, 0) = 0x01U;
        for (index iC = 0; iC < cell2node.father->Size(); iC++)
            for (rowsize ic2n = 0; ic2n < cell2node.RowSize(iC); ic2n++)
                nodeNeedCreate(cell2node(iC, ic2n), 0) |= (0x01U << uint8_t(cell2nodePbi(iC, ic2n)));
        DNDS::ArrayTransformerType<tAdj1::element_type>::Type nodeNeedCreatePastTrans;
        tAdj1 nodeNeedCreatePast;
        nodeNeedCreatePast = make_ssp<decltype(nodeNeedCreatePast)::element_type>(ObjName{"RecreatePeriodicNodes::nodeNeedCreatePast"}, mpi);
        nodeNeedCreatePastTrans.setFatherSon(nodeNeedCreate.son, nodeNeedCreatePast);
        nodeNeedCreatePastTrans.createFatherGlobalMapping();
        std::vector<index> pushSonSeries(nodeNeedCreate.son->Size());
        for (index i = 0; i < pushSonSeries.size(); i++)
            pushSonSeries[i] = i;
        nodeNeedCreatePastTrans.createGhostMapping(pushSonSeries, nodeNeedCreate.trans.pLGhostMapping->ghostStart);
        nodeNeedCreatePastTrans.createMPITypes();
        nodeNeedCreatePastTrans.pullOnce();
        for (index i = 0; i < nodeNeedCreatePast->Size(); i++)
        {
            index iNodeG = nodeNeedCreate.trans.pLGhostMapping->pushingIndexGlobal[i]; //?should be right
            auto [ret, rank, iNode] = nodeNeedCreate.trans.pLGhostMapping->search_indexAppend(iNodeG);
            DNDS_assert(rank == -1); // must be in main
            nodeNeedCreate.father->operator()(iNode, 0) |= nodeNeedCreatePast->operator()(i, 0);
        }
        // std::cout << "here X1" << std::endl;

        index nCreatedNodes{0};
        tAdj8Pair node2recreatedNodes;
        node2recreatedNodes.InitPair("RecreatePeriodicNodes::node2recreatedNodes", mpi);
        node2recreatedNodes.TransAttach();
        node2recreatedNodes.father->Resize(coords.father->Size());
        node2recreatedNodes.trans.BorrowGGIndexing(coords.trans);
        node2recreatedNodes.trans.createMPITypes();
        for (index i = 0; i < coords.father->Size(); i++)
        {
            DNDS_assert(nodeNeedCreate(i, 0) & 0x01U);
            node2recreatedNodes(i, 0) = i;
            for (uint8_t bitCur = 1; bitCur < 8; bitCur++)
            {
                if (nodeNeedCreate(i, 0) & (0x01U << bitCur))
                    node2recreatedNodes(i, rowsize(bitCur)) = coords.father->Size() + nCreatedNodes++;
                else
                    node2recreatedNodes(i, rowsize(bitCur)) = UnInitIndex;
            }
        }
        // node2recreatedNodes now points to local new coords

        coordsPeriodicRecreated.InitPair("RecreatePeriodicNodes::coordsPeriodicRecreated", mpi);
        coordsPeriodicRecreated.TransAttach();
        coordsPeriodicRecreated.father->Resize(nCreatedNodes + coords.father->Size());
        coordsPeriodicRecreated.trans.createFatherGlobalMapping();
        nodeRecreated2nodeLocal.resize(coordsPeriodicRecreated.father->Size());
        for (index iN = 0; iN < coords.father->Size(); iN++) // fill the coords
        {
            coordsPeriodicRecreated[iN] = coords[iN];
            nodeRecreated2nodeLocal[iN] = iN;
            for (uint8_t bitCur = 1; bitCur < 8; bitCur++)
            {
                if (node2recreatedNodes(iN, rowsize(bitCur)) != UnInitIndex)
                {
                    coordsPeriodicRecreated[node2recreatedNodes(iN, rowsize(bitCur))] =
                        periodicInfo.GetCoordByBits(coords[iN], NodePeriodicBits{bitCur});
                    nodeRecreated2nodeLocal[node2recreatedNodes(iN, rowsize(bitCur))] = iN;
                }
            }
        }
        for (index iN = 0; iN < coords.father->Size(); iN++) // convert node2recreatedNodes into global
            for (int i = 0; i < node2recreatedNodes.RowSize(iN); i++)
                if (node2recreatedNodes(iN, i) != UnInitIndex)
                    node2recreatedNodes(iN, i) = coordsPeriodicRecreated.trans.pLGlobalMapping->operator()(
                        mpi.rank, node2recreatedNodes(iN, i));
        node2recreatedNodes.trans.pullOnce(); // for cell2node query
        // std::cout << "here X2" << std::endl;
        cell2nodePeriodicRecreated.InitPair("RecreatePeriodicNodes::cell2nodePeriodicRecreated", mpi);
        cell2nodePeriodicRecreated.TransAttach();
        cell2nodePeriodicRecreated.father->Resize(cell2node.father->Size());
        for (index iC = 0; iC < cell2node.father->Size(); iC++)
        {
            cell2nodePeriodicRecreated.ResizeRow(iC, cell2node.RowSize(iC));
            for (rowsize ic2n = 0; ic2n < cell2node.RowSize(iC); ic2n++)
            {
                index iNode = cell2node(iC, ic2n);
                index iNodeNewGlobal = node2recreatedNodes(iNode, 0);
                if (cell2nodePbi(iC, ic2n))
                {
                    iNodeNewGlobal = node2recreatedNodes(iNode, rowsize(uint8_t(cell2nodePbi(iC, ic2n))));
                    DNDS_assert(iNodeNewGlobal != UnInitIndex);
                }
                cell2nodePeriodicRecreated(iC, ic2n) = iNodeNewGlobal;
            }
        }
        // std::cout << "here X3" << std::endl;
        // here, cell2nodePeriodicRecreated point to global coordsPeriodicRecreated
        // coordsPeriodicRecreated has no son

        index nNodeRecreatedGlobal = coordsPeriodicRecreated.father->globalSize();
        if (mpi.rank == mRank)
            log() << fmt::format("=== RecreatePeriodicNodes: new nNodes [{}]", nNodeRecreatedGlobal) << std::endl;
        mutationToken.Commit();
    }

    void UnstructuredMesh::BuildVTKConnectivity()
    {
        CheckCollectiveMutationAvailable();
        auto mutationToken = BeginManagedMutation(Runtime::Change::Interface | Runtime::Change::Values);

        DNDS_assert(cell2node.father);
        DNDS_assert(coords.trans.pLGhostMapping); // for this->NodeIndexLocal2Global()
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(cell2node.isLocal());
        if (isPeriodic)
        {
            DNDS_assert(cell2nodePeriodicRecreated.father);
            DNDS_assert(coordsPeriodicRecreated.father);
        }
        vtkCell2nodeOffsets.resize(cell2node.father->Size() + 1);
        vtkCell2nodeOffsets[0] = 0;
        vtkCellType.resize(cell2node.father->Size());
        for (index iC = 0; iC < cell2node.father->Size(); iC++)
        {
            auto eCell = GetCellElement(iC);
            auto [vtkType, vtkC2n] = Elem::ToVTKVertsAndData(eCell, cell2node[iC]);
            vtkCellType[iC] = vtkType;
            vtkCell2nodeOffsets[iC + 1] = vtkCell2nodeOffsets[iC] + vtkC2n.size();
        }
        vtkCell2node.clear();
        vtkCell2node.reserve(vtkCell2nodeOffsets.back()); // now offsets are local
        for (index iC = 0; iC < cell2node.father->Size(); iC++)
        {
            auto eCell = GetCellElement(iC);
            if (!isPeriodic)
            {
                auto [vtkType, vtkC2n] = Elem::ToVTKVertsAndData(eCell, cell2node[iC]);
                for (auto iN : vtkC2n)
                    vtkCell2node.push_back(this->NodeIndexLocal2Global(iN));
            }
            else
            { // cell2nodePeriodicRecreated points to global
                auto [vtkType, vtkC2n] = Elem::ToVTKVertsAndData(eCell, cell2nodePeriodicRecreated[iC]);
                for (auto iN : vtkC2n)
                    vtkCell2node.push_back(iN);
            }
        }
        index localVtkCellOffset = vtkCell2nodeOffsets.back();
        index sumPrevVtkCellOffset{0};
        MPI::Scan(&localVtkCellOffset, &sumPrevVtkCellOffset, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        sumPrevVtkCellOffset -= localVtkCellOffset;
        for (auto &v : vtkCell2nodeOffsets)
            v += sumPrevVtkCellOffset;

        // get some offsets
        index curNNode = coords.father->Size();
        if (isPeriodic)
            curNNode = coordsPeriodicRecreated.father->Size();
        MPI::Scan(&curNNode, &vtkNodeOffset, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        vtkNodeOffset -= curNNode;
        index curNCell = cell2node.father->Size();
        MPI::Scan(&curNCell, &vtkCellOffset, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        vtkCellOffset -= curNCell;

        index vtkCell2NodeSize = vtkCell2node.size();
        MPI::Allreduce(&vtkCell2NodeSize, &vtkCell2NodeGlobalSiz, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);

        index nCellsLocal = cell2node.father->Size();
        MPI_Allreduce(&nCellsLocal, &vtkNCellGlobal, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        tCoord coordOut = isPeriodic ? coordsPeriodicRecreated.father : coords.father;
        index nNodesLocal = coordOut->Size();
        MPI_Allreduce(&nNodesLocal, &vtkNNodeGlobal, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);

        // std::cout << mpi.rank << " " << sumPrevVtkCellOffset << std::endl;
        mutationToken.Commit();
    }
}
