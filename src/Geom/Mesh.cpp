#include "Mesh.hpp"

#include <cstdlib>
#include <string>
#include <map>
#include <omp.h>
#include <filesystem>
#include <unordered_map>

namespace DNDS::Geom
{

    // void UnstructuredMeshSerialRW::InterpolateTopology() //!could be useful for parallel?
    // {
    //     // count node 2 face
    //     DNDS_MAKE_SSP(cell2faceSerial, mesh->mpi);
    //     DNDS_MAKE_SSP(face2cellSerial, mesh->mpi);
    //     DNDS_MAKE_SSP(face2nodeSerial, mesh->mpi);
    //     DNDS_MAKE_SSP(faceElemInfoSerial, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);

    //     if (mRank != mesh->mpi.rank)
    //         return;

    //     for (DNDS::index iCell = 0; iCell <cell2nodeSerial->Size(); iCell++)
    //     {

    //         // TODO
    //     }
    // }

    /*******************************************************************************************************************/
    /*******************************************************************************************************************/
    /*******************************************************************************************************************/

    /**
     * Used for generating pushing data structure
     *  @todo: //TODO test on parallel re-distributing
     */
    // for one-shot usage, partition data corresponds to mpi

    template <class TPartitionIdx>
    void Partition2LocalIdx(
        const std::vector<TPartitionIdx> &partition,
        std::vector<DNDS::index> &localPush,
        std::vector<DNDS::index> &localPushStart, const DNDS::MPIInfo &mpi)
    {
        // localPushStart.resize(mpi.size);
        std::vector<DNDS::index> localPushSizes(mpi.size, 0);
        for (auto r : partition)
        {
            DNDS_assert(r < mpi.size);
            localPushSizes[r]++;
        }
        DNDS::AccumulateRowSize(localPushSizes, localPushStart);
        localPush.resize(localPushStart[mpi.size]);
        localPushSizes.assign(mpi.size, 0);
        DNDS_assert(partition.size() == localPush.size());
        for (DNDS::index i = 0; i < partition.size(); i++)
            localPush[localPushStart[partition[i]] + (localPushSizes[partition[i]]++)] = i;
    }

    /**
     * Serial2Global is used for converting adj data to point to reordered global
     * @todo: //TODO: get fully serial version
     *  @todo: //TODO test on parallel re-distributing
     */
    template <class TPartitionIdx>
    void Partition2Serial2Global(
        const std::vector<TPartitionIdx> &partition,
        std::vector<DNDS::index> &serial2Global, const DNDS::MPIInfo &mpi, DNDS::MPI_int nPart)
    {
        serial2Global.resize(partition.size());
        /****************************************/
        std::vector<DNDS::index> numberAtLocal(nPart, 0);
        for (auto r : partition)
            numberAtLocal[r]++;
        std::vector<DNDS::index> numberTotal(nPart), numberPrev(nPart);
        MPI::Allreduce(numberAtLocal.data(), numberTotal.data(), nPart, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        MPI_Scan(numberAtLocal.data(), numberPrev.data(), nPart, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        std::vector<DNDS::index> numberTotalPlc(nPart + 1);
        numberTotalPlc[0] = 0;
        for (DNDS::MPI_int r = 0; r < nPart; r++)
            numberTotalPlc[r + 1] = numberTotalPlc[r] + numberTotal[r], numberPrev[r] -= numberAtLocal[r];
        // 2 things here: accumulate total and subtract local from prev
        /****************************************/
        numberAtLocal.assign(numberAtLocal.size(), 0);
        DNDS::index iFill = 0;
        for (auto r : partition)
            serial2Global[iFill++] = (numberAtLocal[r]++) + numberTotalPlc[r] + numberPrev[r];
    }

    /**
     * In this section, Serial means un-reordered index, i-e
     * pointing to the *serial data, cell2node then pointing to the
     * cell2node global indices:
     * 0 1; 2 3
     *
     * Global means the re-ordered data's global indices
     * if partition ==
     * 0 1; 0 1
     * then Serial 2 Global is:
     * 0 2; 1 3
     *
     * if in proc 0, a cell refers to node 2, then must be seen in JSG ghost, gets global output == 1
     *
     * comm complexity: same as data comm
     * @todo: //TODO test on parallel re-distributing
     * @todo: //TODO: fully serial emulator
     */

    template <class TAdj = tAdj1>
    void ConvertAdjSerial2Global(TAdj &arraySerialAdj,
                                 const std::vector<DNDS::index> &partitionJSerial2Global,
                                 const DNDS::MPIInfo &mpi)
    {
        // IndexArray JSG(IndexArray::tContext(partitionJSerial2Global.size()), mpi);
        // forEachInArray(
        //     JSG, [&](IndexArray::tComponent &e, index i)
        //     { e[0] = partitionJSerial2Global[i]; });
        // JSGGhost.createGlobalMapping();
        tAdj1 JSG, JSGGhost;
        DNDS_MAKE_SSP(JSG, mpi);
        DNDS_MAKE_SSP(JSGGhost, mpi);
        JSG->Resize(partitionJSerial2Global.size());
        for (DNDS::index i = 0; i < JSG->Size(); i++)
            (*JSG)(i, 0) = partitionJSerial2Global[i];
        JSG->createGlobalMapping();
        std::vector<DNDS::index> ghostJSerialQuery;

        // get ghost
        DNDS::index nGhost = 0;
        for (DNDS::index i = 0; i < arraySerialAdj->Size(); i++)
        {
            for (DNDS::rowsize j = 0; j < arraySerialAdj->RowSize(i); j++)
            {
                DNDS::index v = (*arraySerialAdj)(i, j);
                if (v == DNDS::UnInitIndex)
                    break;
                DNDS::MPI_int rank = -1;
                DNDS::index val = -1;
                if (!JSG->pLGlobalMapping->search(v, rank, val))
                    DNDS_assert_info(false, "search failed");
                if (rank != mpi.rank) //! excluding self
                    nGhost++;
            }
        }
        ghostJSerialQuery.reserve(nGhost);
        for (DNDS::index i = 0; i < arraySerialAdj->Size(); i++)
        {
            for (DNDS::rowsize j = 0; j < arraySerialAdj->RowSize(i); j++)
            {
                DNDS::index v = (*arraySerialAdj)(i, j);
                if (v == DNDS::UnInitIndex)
                    break;
                DNDS::MPI_int rank = -1;
                DNDS::index val = -1;
                JSG->pLGlobalMapping->search(v, rank, val);
                if (rank != mpi.rank) //! excluding self
                    ghostJSerialQuery.push_back(v);
            }
        }
        // PrintVec(ghostJSerialQuery, std::cout);
        typename DNDS::ArrayTransformerType<tAdj1::element_type>::Type JSGTrans;
        JSGTrans.setFatherSon(JSG, JSGGhost);
        JSGTrans.createGhostMapping(ghostJSerialQuery);
        JSGTrans.createMPITypes();
        JSGTrans.pullOnce();

        for (DNDS::index i = 0; i < arraySerialAdj->Size(); i++)
        {
            for (DNDS::rowsize j = 0; j < arraySerialAdj->RowSize(i); j++)
            {
                DNDS::index &v = (*arraySerialAdj)(i, j);
                if (v == DNDS::UnInitIndex)
                    break;
                DNDS::MPI_int rank = -1;
                DNDS::index val = -1;
                if (!JSGTrans.pLGhostMapping->search(v, rank, val))
                    DNDS_assert_info(false, "search failed");
                if (rank == -1)
                    v = (*JSG)(val, 0);
                else
                    v = (*JSGGhost)(val, 0);
            }
        }
    }

    /**
     * @todo: //TODO: fully serial emulator
     */
    template <class TArr = tAdj1>
    void TransferDataSerial2Global(TArr &arraySerial,
                                   TArr &arrayDist,
                                   const std::vector<DNDS::index> &pushIndex,
                                   const std::vector<DNDS::index> &pushIndexStart,
                                   const DNDS::MPIInfo &mpi)
    {
        typename DNDS::ArrayTransformerType<typename TArr::element_type>::Type trans;
        trans.setFatherSon(arraySerial, arrayDist);
        trans.createFatherGlobalMapping();
        trans.createGhostMapping(pushIndex, pushIndexStart);
        trans.createMPITypes();
        trans.pullOnce();
    }

    //! inefficient, use Partition2Serial2Global ! only used for convenient comparison
    void PushInfo2Serial2Global(std::vector<DNDS::index> &serial2Global,
                                DNDS::index localSize,
                                const std::vector<DNDS::index> &pushIndex,
                                const std::vector<DNDS::index> &pushIndexStart,
                                const DNDS::MPIInfo &mpi)
    {
        tIndPair Serial2Global;
        DNDS_MAKE_SSP(Serial2Global.father, mpi);
        DNDS_MAKE_SSP(Serial2Global.son, mpi);
        Serial2Global.father->Resize(localSize);
        Serial2Global.TransAttach();
        Serial2Global.trans.createFatherGlobalMapping();
        Serial2Global.trans.createGhostMapping(pushIndex, pushIndexStart);
        Serial2Global.trans.createMPITypes();
        Serial2Global.son->createGlobalMapping();
        // Set son to son's global
        for (DNDS::index iSon = 0; iSon < Serial2Global.son->Size(); iSon++)
            (*Serial2Global.son)[iSon] = Serial2Global.son->pLGlobalMapping->operator()(mpi.rank, iSon);
        Serial2Global.trans.pushOnce();
        serial2Global.resize(localSize);
        for (DNDS::index iFat = 0; iFat < Serial2Global.father->Size(); iFat++)
            serial2Global[iFat] = Serial2Global.father->operator[](iFat);
    }

    // template <class TAdj = tAdj1>
    // void ConvertAdjSerial2Global(TAdj &arraySerialAdj,
    //                              const std::vector<DNDS::index> &partitionJSerial2Global,
    //                              const DNDS::MPIInfo &mpi)
    // {
    // }
    /*******************************************************************************************************************/
    /*******************************************************************************************************************/
    /*******************************************************************************************************************/

    void UnstructuredMeshSerialRW::
        PartitionReorderToMeshCell2Cell()
    {
        if (mesh->mpi.rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  PartitionReorderToMeshCell2Cell" << std::endl;
        DNDS_assert(cnPart == mesh->mpi.size);
        // * 1: get the nodal partition
        nodePartition.resize(coordSerial->Size(), static_cast<DNDS::MPI_int>(INT32_MAX));
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            for (DNDS::rowsize ic2n = 0; ic2n < (*cell2nodeSerial).RowSize(iCell); ic2n++)
                nodePartition[(*cell2nodeSerial)(iCell, ic2n)] = std::min(nodePartition[(*cell2nodeSerial)(iCell, ic2n)], cellPartition.at(iCell));
        // * 1: get the bnd partition
        bndPartition.resize(bnd2cellSerial->Size());
        for (DNDS::index iBnd = 0; iBnd < bnd2cellSerial->Size(); iBnd++)
            bndPartition[iBnd] = cellPartition[(*bnd2cellSerial)(iBnd, 0)];

        std::vector<DNDS::index> cell_push, cell_pushStart, node_push, node_pushStart, bnd_push, bnd_pushStart;
        Partition2LocalIdx(cellPartition, cell_push, cell_pushStart, mesh->mpi);
        Partition2LocalIdx(nodePartition, node_push, node_pushStart, mesh->mpi);
        Partition2LocalIdx(bndPartition, bnd_push, bnd_pushStart, mesh->mpi);
        std::vector<DNDS::index> cell_Serial2Global, node_Serial2Global, bnd_Serial2Global;
        Partition2Serial2Global(cellPartition, cell_Serial2Global, mesh->mpi, mesh->mpi.size);
        Partition2Serial2Global(nodePartition, node_Serial2Global, mesh->mpi, mesh->mpi.size);
        // Partition2Serial2Global(bndPartition, bnd_Serial2Global, mesh->mpi, mesh->mpi.size);//seems not needed for now
        // PushInfo2Serial2Global(cell_Serial2Global, cellPartition.size(), cell_push, cell_pushStart, mesh->mpi);//*safe validation version
        // PushInfo2Serial2Global(node_Serial2Global, nodePartition.size(), node_push, node_pushStart, mesh->mpi);//*safe validation version
        // PushInfo2Serial2Global(bnd_Serial2Global, bndPartition.size(), bnd_push, bnd_pushStart, mesh->mpi);    //*safe validation version

        ConvertAdjSerial2Global(cell2nodeSerial, node_Serial2Global, mesh->mpi);
        ConvertAdjSerial2Global(cell2cellSerial, cell_Serial2Global, mesh->mpi);
        ConvertAdjSerial2Global(bnd2nodeSerial, node_Serial2Global, mesh->mpi);
        ConvertAdjSerial2Global(bnd2cellSerial, cell_Serial2Global, mesh->mpi);

        DNDS_MAKE_SSP(mesh->coords.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->coords.son, mesh->mpi);
        DNDS_MAKE_SSP(mesh->cellElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(mesh->cellElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(mesh->bndElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(mesh->bndElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(mesh->cell2node.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->cell2node.son, mesh->mpi);
        if (mesh->isPeriodic)
        {
            DNDS_MAKE_SSP(mesh->cell2nodePbi.father, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->mpi);
            DNDS_MAKE_SSP(mesh->cell2nodePbi.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->mpi);
        }
        DNDS_MAKE_SSP(mesh->cell2cell.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->cell2cell.son, mesh->mpi);
        DNDS_MAKE_SSP(mesh->bnd2node.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->bnd2node.son, mesh->mpi);
        DNDS_MAKE_SSP(mesh->bnd2cell.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->bnd2cell.son, mesh->mpi);

        // coord transferring
        TransferDataSerial2Global(coordSerial, mesh->coords.father, node_push, node_pushStart, mesh->mpi);

        // cells transferring
        TransferDataSerial2Global(cell2cellSerial, mesh->cell2cell.father, cell_push, cell_pushStart, mesh->mpi);
        TransferDataSerial2Global(cell2nodeSerial, mesh->cell2node.father, cell_push, cell_pushStart, mesh->mpi);
        if (mesh->isPeriodic)
            TransferDataSerial2Global(cell2nodePbiSerial, mesh->cell2nodePbi.father, cell_push, cell_pushStart, mesh->mpi);
        TransferDataSerial2Global(cellElemInfoSerial, mesh->cellElemInfo.father, cell_push, cell_pushStart, mesh->mpi);

        // bnds transferring
        TransferDataSerial2Global(bnd2cellSerial, mesh->bnd2cell.father, bnd_push, bnd_pushStart, mesh->mpi);
        TransferDataSerial2Global(bnd2nodeSerial, mesh->bnd2node.father, bnd_push, bnd_pushStart, mesh->mpi);
        TransferDataSerial2Global(bndElemInfoSerial, mesh->bndElemInfo.father, bnd_push, bnd_pushStart, mesh->mpi);

        DNDS::MPISerialDo(mesh->mpi, [&]()
                          { std::cout << "Rank " << mesh->mpi.rank << " : nCell " << mesh->cell2cell.father->Size() << std::endl; });
        DNDS::MPISerialDo(mesh->mpi, [&]()
                          { std::cout << " Rank " << mesh->mpi.rank << " : nNode " << mesh->coords.father->Size() << std::endl; });
        DNDS::MPISerialDo(mesh->mpi, [&]()
                          { std::cout << " Rank " << mesh->mpi.rank << " : nBnd " << mesh->bnd2node.father->Size() << std::endl; });
        mesh->adjPrimaryState = Adj_PointToGlobal;
        if (mesh->mpi.rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Done  PartitionReorderToMeshCell2Cell" << std::endl;
    }

    void UnstructuredMeshSerialRW::
        BuildSerialOut()
    {
        DNDS_assert(mesh->adjPrimaryState == Adj_PointToGlobal);
        mode = SerialOutput;
        dataIsSerialIn = false;
        dataIsSerialOut = true;

        std::vector<DNDS::index> serialPullCell;
        std::vector<DNDS::index> serialPullNode;
        // std::vector<DNDS::index> serialPullBnd;

        DNDS::index numCellGlobal = mesh->cellElemInfo.father->globalSize();
        // DNDS::index numBndGlobal = mesh->bndElemInfo.father->globalSize();
        DNDS::index numNodeGlobal = mesh->coords.father->globalSize();

        if (mesh->mpi.rank == mRank)
        {
            serialPullCell.resize(numCellGlobal);
            serialPullNode.resize(numNodeGlobal);
            // serialPullBnd.reserve(numBndGlobal);
            for (DNDS::index i = 0; i < numCellGlobal; i++)
                serialPullCell[i] = i;
            for (DNDS::index i = 0; i < numNodeGlobal; i++)
                serialPullNode[i] = i;
            // for (DNDS::index i = 0; i < numBndGlobal; i++)
            //     serialPullBnd[i] = i;
        }
        DNDS_MAKE_SSP(cell2nodeSerial, mesh->mpi);
        if (mesh->isPeriodic)
            DNDS_MAKE_SSP(cell2nodePbiSerial, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->mpi);
        // DNDS_MAKE_SSP(bnd2nodeSerial, mesh->mpi);
        DNDS_MAKE_SSP(coordSerial, mesh->mpi);
        DNDS_MAKE_SSP(cellElemInfoSerial, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        // DNDS_MAKE_SSP(bndElemInfoSerial, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        // DNDS_MAKE_SSP(bnd2cellSerial, mesh->mpi);// not needed yet

        coordSerialOutTrans.setFatherSon(mesh->coords.father, coordSerial);
        cell2nodeSerialOutTrans.setFatherSon(mesh->cell2node.father, cell2nodeSerial);
        if (mesh->isPeriodic)
            cell2nodePbiSerialOutTrans.setFatherSon(mesh->cell2nodePbi.father, cell2nodePbiSerial);
        // bnd2nodeSerialOutTrans.setFatherSon(mesh->bnd2node.father, bnd2nodeSerial);
        cellElemInfoSerialOutTrans.setFatherSon(mesh->cellElemInfo.father, cellElemInfoSerial);
        // bndElemInfoSerialOutTrans.setFatherSon(mesh->bndElemInfo.father, bndElemInfoSerial);

        // Father could already have global mapping, result should be the same
        coordSerialOutTrans.createFatherGlobalMapping();
        cell2nodeSerialOutTrans.createFatherGlobalMapping();
        if (mesh->isPeriodic)
            cell2nodePbiSerialOutTrans.createFatherGlobalMapping();
        // bnd2nodeSerialOutTrans.createFatherGlobalMapping();
        cellElemInfoSerialOutTrans.createFatherGlobalMapping();
        // bndElemInfoSerialOutTrans.createFatherGlobalMapping();

        coordSerialOutTrans.createGhostMapping(serialPullNode);
        cell2nodeSerialOutTrans.createGhostMapping(serialPullCell);
        if (mesh->isPeriodic)
            cell2nodePbiSerialOutTrans.createGhostMapping(serialPullCell);
        // bnd2nodeSerialOutTrans.createGhostMapping(serialPullBnd);
        cellElemInfoSerialOutTrans.BorrowGGIndexing(cell2nodeSerialOutTrans); // accidentally rewrites mesh->cellElemInfo.father's global mapping but ok
        // bndElemInfoSerialOutTrans.BorrowGGIndexing(bnd2nodeSerialOutTrans);

        coordSerialOutTrans.createMPITypes();
        cell2nodeSerialOutTrans.createMPITypes();
        if (mesh->isPeriodic)
            cell2nodePbiSerialOutTrans.createMPITypes();
        // bnd2nodeSerialOutTrans.createMPITypes();
        cellElemInfoSerialOutTrans.createMPITypes();
        // bndElemInfoSerialOutTrans.createMPITypes();

        coordSerialOutTrans.pullOnce();
        cell2nodeSerialOutTrans.pullOnce();
        if (mesh->isPeriodic)
            cell2nodePbiSerialOutTrans.pullOnce();
        // bnd2nodeSerialOutTrans.pullOnce();
        cellElemInfoSerialOutTrans.pullOnce();
        // bndElemInfoSerialOutTrans.pullOnce();
        if (mesh->mpi.rank == mRank)
        {
            std::cout << "UnstructuredMeshSerialRW === BuildSerialOut Done " << std::endl;
        }
    }

}

namespace DNDS::Geom
{
    void UnstructuredMesh::
        BuildGhostPrimary()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
        /********************************/
        // cells
        {
            cell2cell.TransAttach();
            cell2node.TransAttach();
            if (isPeriodic)
                cell2nodePbi.TransAttach();
            cellElemInfo.TransAttach();

            cell2cell.trans.createFatherGlobalMapping();

            std::vector<DNDS::index> ghostCells;
            for (DNDS::index iCell = 0; iCell < cell2cell.father->Size(); iCell++)
            {
                for (DNDS::rowsize ic2c = 0; ic2c < cell2cell.father->RowSize(iCell); ic2c++)
                {
                    auto iCellOther = (*cell2cell.father)(iCell, ic2c);
                    DNDS::MPI_int rank;
                    DNDS::index val;
                    if (!cell2cell.trans.pLGlobalMapping->search(iCellOther, rank, val))
                        DNDS_assert_info(false, "search failed");
                    if (rank != mpi.rank)
                        ghostCells.push_back(iCellOther);
                }
            }
            cell2cell.trans.createGhostMapping(ghostCells);

            cell2node.trans.BorrowGGIndexing(cell2cell.trans);
            if (isPeriodic)
                cell2nodePbi.trans.BorrowGGIndexing(cell2cell.trans);
            cellElemInfo.trans.BorrowGGIndexing(cell2cell.trans);

            cell2cell.trans.createMPITypes();
            cell2node.trans.createMPITypes();
            if (isPeriodic)
                cell2nodePbi.trans.createMPITypes();
            cellElemInfo.trans.createMPITypes();

            cell2cell.trans.pullOnce();
            cell2node.trans.pullOnce();
            if (isPeriodic)
                cell2nodePbi.trans.pullOnce();
            cellElemInfo.trans.pullOnce();
        }

        /********************************/
        // cells done, go on to nodes
        {
            coords.TransAttach();
            coords.trans.createFatherGlobalMapping();

            std::vector<DNDS::index> ghostNodes;
            for (DNDS::index iCell = 0; iCell < cell2cell.Size(); iCell++) // note doing full (son + father) traverse
            {
                for (DNDS::rowsize ic2c = 0; ic2c < cell2node.RowSize(iCell); ic2c++)
                {
                    auto iNode = cell2node(iCell, ic2c);
                    DNDS::MPI_int rank;
                    DNDS::index val;
                    if (!coords.trans.pLGlobalMapping->search(iNode, rank, val))
                        DNDS_assert_info(false, "search failed");
                    if (rank != mpi.rank)
                        ghostNodes.push_back(iNode);
                }
            }
            coords.trans.createGhostMapping(ghostNodes);
            coords.trans.createMPITypes();
            coords.trans.pullOnce();
        }

        /********************************/
        // bnds: dummy now, no actual comm
        {

            bnd2cell.TransAttach();
            bnd2node.TransAttach();
            bndElemInfo.TransAttach();

            bnd2cell.trans.createFatherGlobalMapping();

            std::vector<DNDS::index> ghostBnds; // no ghosted bnds now
            bnd2cell.trans.createGhostMapping(ghostBnds);

            bnd2node.trans.BorrowGGIndexing(bnd2cell.trans);
            bndElemInfo.trans.BorrowGGIndexing(bnd2cell.trans);

            bnd2cell.trans.createMPITypes();
            bnd2node.trans.createMPITypes();
            bndElemInfo.trans.createMPITypes();

            bnd2cell.trans.pullOnce();
            bnd2node.trans.pullOnce();
            bndElemInfo.trans.pullOnce();
        }
    }

    void UnstructuredMesh::
        AdjGlobal2LocalPrimary()
    {
        // needs results of BuildGhostPrimary()
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);

        /**********************************/
        // convert bnd2cell, bnd2node, cell2cell, cell2node ptrs global to local
        auto CellIndexGlobal2Local = [&](DNDS::index &iCellOther)
        {
            if (iCellOther == UnInitIndex)
                return;
            DNDS::MPI_int rank;
            DNDS::index val;
            // if (!cell2cell.trans.pLGlobalMapping->search(iCellOther, rank, val))
            //     DNDS_assert_info(false, "search failed");
            // if (rank != mpi.rank)
            //     iCellOther = -1 - iCellOther;
            auto result = cell2cell.trans.pLGhostMapping->search_indexAppend(iCellOther, rank, val);
            if (result)
                iCellOther = val;
            else
                iCellOther = -1 - iCellOther; // mapping to un-found in father-son
        };
        auto NodeIndexGlobal2Local = [&](DNDS::index &iNodeOther)
        {
            if (iNodeOther == UnInitIndex)
                return;
            DNDS::MPI_int rank;
            DNDS::index val;
            // if (!cell2cell.trans.pLGlobalMapping->search(iCellOther, rank, val))
            //     DNDS_assert_info(false, "search failed");
            // if (rank != mpi.rank)
            //     iCellOther = -1 - iCellOther;
            auto result = coords.trans.pLGhostMapping->search_indexAppend(iNodeOther, rank, val);
            if (result)
                iNodeOther = val;
            else
                iNodeOther = -1 - iNodeOther; // mapping to un-found in father-son
        };

        for (DNDS::index iCell = 0; iCell < cell2cell.Size(); iCell++)
            for (DNDS::rowsize j = 0; j < cell2cell.RowSize(iCell); j++)
                CellIndexGlobal2Local(cell2cell(iCell, j));

        for (DNDS::index iBnd = 0; iBnd < bnd2cell.Size(); iBnd++)
            for (DNDS::rowsize j = 0; j < bnd2cell.RowSize(iBnd); j++)
                CellIndexGlobal2Local(bnd2cell(iBnd, j)), DNDS_assert(j == 0 ? bnd2cell(iBnd, j) >= 0 : true); // must be inside

        for (DNDS::index iCell = 0; iCell < cell2node.Size(); iCell++)
            for (DNDS::rowsize j = 0; j < cell2node.RowSize(iCell); j++)
                NodeIndexGlobal2Local(cell2node(iCell, j)), DNDS_assert(cell2node(iCell, j) >= 0);

        for (DNDS::index iBnd = 0; iBnd < bnd2node.Size(); iBnd++)
            for (DNDS::rowsize j = 0; j < bnd2node.RowSize(iBnd); j++)
                NodeIndexGlobal2Local(bnd2node(iBnd, j)), DNDS_assert(bnd2node(iBnd, j) >= 0);
        /**********************************/

        adjPrimaryState = Adj_PointToLocal;
    }

    void UnstructuredMesh::
        AdjLocal2GlobalPrimary()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        /**********************************/
        // convert bnd2cell, bnd2node, cell2cell, cell2node ptrs local to global
        /**********************************/
        // convert bnd2cell, bnd2node, cell2cell, cell2node ptrs global to local
        auto CellIndexLocal2Global = [&](DNDS::index &iCellOther)
        {
            if (iCellOther == UnInitIndex)
                return;
            if (iCellOther < 0) // mapping to un-found in father-son
                iCellOther = -1 - iCellOther;
            else
                iCellOther = cell2cell.trans.pLGhostMapping->operator()(-1, iCellOther);
        };
        auto NodeIndexLocal2Global = [&](DNDS::index &iNodeOther)
        {
            if (iNodeOther == UnInitIndex)
                return;
            if (iNodeOther < 0) // mapping to un-found in father-son
                iNodeOther = -1 - iNodeOther;
            else
                iNodeOther = coords.trans.pLGhostMapping->operator()(-1, iNodeOther);
        };

        for (DNDS::index iCell = 0; iCell < cell2cell.Size(); iCell++)
            for (DNDS::rowsize j = 0; j < cell2cell.RowSize(iCell); j++)
                CellIndexLocal2Global(cell2cell(iCell, j));

        for (DNDS::index iBnd = 0; iBnd < bnd2cell.Size(); iBnd++)
            for (DNDS::rowsize j = 0; j < bnd2cell.RowSize(iBnd); j++)
                CellIndexLocal2Global(bnd2cell(iBnd, j)), DNDS_assert(j == 0 ? bnd2cell(iBnd, j) >= 0 : true); // must be inside

        for (DNDS::index iCell = 0; iCell < cell2node.Size(); iCell++)
            for (DNDS::rowsize j = 0; j < cell2node.RowSize(iCell); j++)
                NodeIndexLocal2Global(cell2node(iCell, j)), DNDS_assert(cell2node(iCell, j) >= 0);

        for (DNDS::index iBnd = 0; iBnd < bnd2node.Size(); iBnd++)
            for (DNDS::rowsize j = 0; j < bnd2node.RowSize(iBnd); j++)
                NodeIndexLocal2Global(bnd2node(iBnd, j)), DNDS_assert(bnd2node(iBnd, j) >= 0);
        /**********************************/
        /**********************************/
        adjPrimaryState = Adj_PointToGlobal;
    }

    void UnstructuredMesh::
        AdjGlobal2LocalPrimaryForBnd() // a reduction of primary version
    {
        // needs results of BuildGhostPrimary()
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);

        /**********************************/
        // convert cell2node ptrs global to local
        auto NodeIndexGlobal2Local = [&](DNDS::index &iNodeOther)
        {
            if (iNodeOther == UnInitIndex)
                return;
            DNDS::MPI_int rank;
            DNDS::index val;
            // if (!cell2cell.trans.pLGlobalMapping->search(iCellOther, rank, val))
            //     DNDS_assert_info(false, "search failed");
            // if (rank != mpi.rank)
            //     iCellOther = -1 - iCellOther;
            auto result = coords.trans.pLGhostMapping->search_indexAppend(iNodeOther, rank, val);
            if (result)
                iNodeOther = val;
            else
                iNodeOther = -1 - iNodeOther; // mapping to un-found in father-son
        };

        for (DNDS::index iCell = 0; iCell < cell2node.Size(); iCell++)
            for (DNDS::rowsize j = 0; j < cell2node.RowSize(iCell); j++)
                NodeIndexGlobal2Local(cell2node(iCell, j)), DNDS_assert(cell2node(iCell, j) >= 0);
        /**********************************/

        adjPrimaryState = Adj_PointToLocal;
    }

    void UnstructuredMesh::
        AdjLocal2GlobalPrimaryForBnd() // a reduction of primary version
    {
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        /**********************************/
        // convert cell2node ptrs local to global
        auto NodeIndexLocal2Global = [&](DNDS::index &iNodeOther)
        {
            if (iNodeOther == UnInitIndex)
                return;
            if (iNodeOther < 0) // mapping to un-found in father-son
                iNodeOther = -1 - iNodeOther;
            else
                iNodeOther = coords.trans.pLGhostMapping->operator()(-1, iNodeOther);
        };
        for (DNDS::index iCell = 0; iCell < cell2node.Size(); iCell++)
            for (DNDS::rowsize j = 0; j < cell2node.RowSize(iCell); j++)
                NodeIndexLocal2Global(cell2node(iCell, j)), DNDS_assert(cell2node(iCell, j) >= 0);
        /**********************************/
        /**********************************/
        adjPrimaryState = Adj_PointToGlobal;
    }

    void UnstructuredMesh::
        AdjGlobal2LocalFacial()
    {
        DNDS_assert(adjFacialState == Adj_PointToGlobal);
        /**********************************/
        // convert face2cell ptrs and face2node ptrs global to local
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iFace = 0; iFace < face2cell.Size(); iFace++)
        {
            for (rowsize if2n = 0; if2n < face2node.RowSize(iFace); if2n++)
            {
                index &iNode = face2node(iFace, if2n);
                index val;
                MPI_int rank;
                auto ret = coords.trans.pLGhostMapping->search_indexAppend(iNode, rank, val);
                DNDS_assert(ret);
                iNode = val;
            }
            for (rowsize if2c = 0; if2c < 2; if2c++)
            {
                index &iCell = face2cell(iFace, if2c);
                if (iCell != UnInitIndex) // is not a bnd
                {
                    index val;
                    MPI_int rank;
                    auto ret = cell2node.trans.pLGhostMapping->search_indexAppend(iCell, rank, val);
                    DNDS_assert(ret);
                    iCell = val;
                }
            }
        }
        /**********************************/
        adjFacialState = Adj_PointToLocal;
    }

    void UnstructuredMesh::
        AdjLocal2GlobalFacial()
    {
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        /**********************************/
        // convert face2cell ptrs and face2node ptrs to global
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iFace = 0; iFace < face2cell.Size(); iFace++)
        {
            for (rowsize if2n = 0; if2n < face2node.RowSize(iFace); if2n++)
            {
                index &iNode = face2node(iFace, if2n);
                iNode = coords.trans.pLGhostMapping->operator()(-1, iNode);
            }
            for (rowsize if2c = 0; if2c < 2; if2c++)
            {
                index &iCell = face2cell(iFace, if2c);
                if (iCell != UnInitIndex) // is not a bnd
                    iCell = cell2node.trans.pLGhostMapping->operator()(-1, iCell);
            }
        }
        // MPI_Barrier(mpi.comm);
        /**********************************/
        adjFacialState = Adj_PointToGlobal;
    }

    /// @todo //TODO: handle periodic cases
    void UnstructuredMesh::
        InterpolateFace()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToLocal); // And also should have primary ghost comm

        DNDS_MAKE_SSP(cell2face.father, mpi);
        DNDS_MAKE_SSP(cell2face.son, mpi);
        DNDS_MAKE_SSP(face2cell.father, mpi);
        DNDS_MAKE_SSP(face2cell.son, mpi);
        DNDS_MAKE_SSP(face2node.father, mpi);
        DNDS_MAKE_SSP(face2node.son, mpi);
        if (isPeriodic)
        {
            DNDS_MAKE_SSP(face2nodePbi.father, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
            DNDS_MAKE_SSP(face2nodePbi.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
        }
        DNDS_MAKE_SSP(faceElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        DNDS_MAKE_SSP(faceElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);

        cell2face.father->Resize(cell2cell.father->Size()); //!
        cell2face.son->Resize(cell2cell.son->Size());
        std::vector<std::vector<DNDS::index>> node2face(coords.Size());
        std::vector<std::vector<DNDS::index>> face2nodeV;
        std::vector<std::vector<NodePeriodicBits>> face2nodePbiV;
        std::vector<std::pair<DNDS::index, DNDS::index>> face2cellV;
        std::vector<ElemInfo> faceElemInfoV;

        DNDS::index nFaces = 0;
        for (DNDS::index iCell = 0; iCell < cell2cell.Size(); iCell++)
        {
            auto eCell = Elem::Element{cellElemInfo[iCell]->getElemType()};
            cell2face.ResizeRow(iCell, eCell.GetNumFaces());
            for (int ic2f = 0; ic2f < eCell.GetNumFaces(); ic2f++)
            {
                auto eFace = eCell.ObtainFace(ic2f);
                std::vector<DNDS::index> faceNodes(eFace.GetNumNodes());
                eCell.ExtractFaceNodes(ic2f, cell2node[iCell], faceNodes);
                DNDS::index iFound = -1;
                std::vector<DNDS::index> faceVerts(faceNodes.begin(), faceNodes.begin() + eFace.GetNumVertices());
                std::sort(faceVerts.begin(), faceVerts.end());

                std::vector<NodePeriodicBits> faceNodePeriodicBits;
                std::vector<std::pair<index, NodePeriodicBits>> faceNodeNodePeriodicBits;
                if (isPeriodic)
                {
                    faceNodePeriodicBits.resize(eFace.GetNumNodes());
                    faceNodeNodePeriodicBits.resize(eFace.GetNumNodes());
                    eCell.ExtractFaceNodes(ic2f, cell2nodePbi[iCell], faceNodePeriodicBits);
                    for (rowsize i = 0; i < faceNodeNodePeriodicBits.size(); i++)
                        faceNodeNodePeriodicBits[i].first = faceNodes[i], faceNodeNodePeriodicBits[i].second = faceNodePeriodicBits[i];
                    std::sort(faceNodeNodePeriodicBits.begin(), faceNodeNodePeriodicBits.end(),
                              [](auto &L, auto &R)
                              { return L.first < R.first; });
                }

                for (auto iV : faceVerts)
                    if (iFound < 0)
                        for (auto iFOther : node2face[iV])
                        {
                            auto eFaceOther = Elem::Element{faceElemInfoV[iFOther].getElemType()};
                            if (eFaceOther.type != eFace.type)
                                continue;
                            std::vector<DNDS::index> faceVertsOther(
                                face2nodeV[iFOther].begin(),
                                face2nodeV[iFOther].begin() + eFace.GetNumVertices());
                            std::sort(faceVertsOther.begin(), faceVertsOther.end());
                            if (std::equal(faceVerts.begin(), faceVerts.end(), faceVertsOther.begin(), faceVertsOther.end()))
                            {
                                if (isPeriodic)
                                {
                                    std::vector<std::pair<index, NodePeriodicBits>> faceNodeNodePeriodicBitsOther(eFaceOther.GetNumNodes());
                                    for (rowsize i = 0; i < faceNodeNodePeriodicBitsOther.size(); i++)
                                        faceNodeNodePeriodicBitsOther[i].first = face2nodeV[iFOther][i],
                                        faceNodeNodePeriodicBitsOther[i].second = face2nodePbiV[iFOther][i];
                                    std::sort(faceNodeNodePeriodicBitsOther.begin(), faceNodeNodePeriodicBitsOther.end(),
                                              [](auto &L, auto &R)
                                              { return L.first < R.first; });
                                    auto v0 = faceNodeNodePeriodicBits.at(0).second ^ faceNodeNodePeriodicBitsOther.at(0).second;
                                    DNDS_assert(faceNodeNodePeriodicBitsOther.size() == faceNodeNodePeriodicBits.size());
                                    bool collaborating = true;
                                    for (rowsize i = 1; i < faceNodeNodePeriodicBitsOther.size(); i++)
                                        if ((faceNodeNodePeriodicBits[i].second ^ faceNodeNodePeriodicBitsOther[i].second) != v0)
                                            collaborating = false;
                                    if (collaborating)
                                        iFound = iFOther;
                                }
                                else
                                    iFound = iFOther;
                            }
                        }
                if (iFound < 0)
                {
                    // * face not existent yet
                    face2nodeV.emplace_back(faceNodes); // note: faceVerts invalid here!
                    if (isPeriodic)
                        face2nodePbiV.emplace_back(faceNodePeriodicBits);
                    face2cellV.emplace_back(std::make_pair(iCell, DNDS::UnInitIndex));
                    faceElemInfoV.emplace_back(ElemInfo{eFace.type, 0});
                    for (auto iV : faceVerts)
                        node2face[iV].push_back(nFaces);
                    cell2face(iCell, ic2f) = nFaces;
                    nFaces++;
                }
                else
                {
                    DNDS_assert(face2cellV[iFound].second == DNDS::UnInitIndex);
                    face2cellV[iFound].second = iCell;
                    cell2face(iCell, ic2f) = iFound;
                }
            }
        }
        node2face.clear(); // no need

        /*************************************/
        // ! collect!
        std::vector<index> iFaceAllToCollected(nFaces);
        std::vector<std::vector<index>> faceSendLocals(mpi.size);
        index nFacesNew = 0;
        for (index iFace = 0; iFace < nFaces; iFace++)
        {
            if (faceElemInfoV[iFace].zone <= 0) // if internal
            {
                if (face2cellV[iFace].second == UnInitIndex && face2cellV[iFace].first >= cell2face.father->Size()) // has not other cell with ghost parent
                    iFaceAllToCollected[iFace] = UnInitIndex;                                                       // * discard
                else if (face2cellV[iFace].first >= cell2face.father->Size() &&
                         face2cellV[iFace].second >= cell2face.father->Size()) // both sides ghost
                    iFaceAllToCollected[iFace] = UnInitIndex;                  // * discard
                else if (face2cellV[iFace].first >= cell2face.father->Size() ||
                         face2cellV[iFace].second >= cell2face.father->Size())
                {
                    DNDS_assert(face2cellV[iFace].second >= cell2face.father->Size()); // should only be the internal as first
                    // * check both sided's info //TODO: optimize so that pLGhostMapping returns rank directly ?
                    index cellGlobL = cell2node.trans.pLGhostMapping->operator()(-1, face2cellV[iFace].first);
                    index cellGlobR = cell2node.trans.pLGhostMapping->operator()(-1, face2cellV[iFace].second);
                    MPI_int rankL, rankR;
                    index valL, valR;
                    auto retL = cell2node.father->pLGlobalMapping->search(cellGlobL, rankL, valL);
                    auto retR = cell2node.father->pLGlobalMapping->search(cellGlobR, rankR, valR);
                    DNDS_assert(retL && retR && (rankL != rankR));
                    if (rankL > rankR)
                    {
                        iFaceAllToCollected[iFace] = -1; // * discard but with ghost
                    }
                    else
                    {
                        DNDS_assert(rankL == mpi.rank);
                        faceSendLocals[rankR].push_back(nFacesNew);
                        iFaceAllToCollected[iFace] = nFacesNew++; //*use
                    }
                }
                else
                {
                    iFaceAllToCollected[iFace] = nFacesNew++; //*use
                }
            }
            else // all bnds would be non duplicate
            {
                iFaceAllToCollected[iFace] = nFacesNew++; //*use
            }
        }

        /**********************************/
        face2cell.father->Resize(nFacesNew);
        face2node.father->Resize(nFacesNew);
        if (isPeriodic)
            face2nodePbi.father->Resize(nFacesNew);
        faceElemInfo.father->Resize(nFacesNew); //! considering globally duplicate faces
        nFacesNew = 0;
        for (DNDS::index iFace = 0; iFace < nFaces; iFace++)
        {
            if (iFaceAllToCollected[iFace] >= 0) // ! -1 is also ignored!
            {
                face2node.ResizeRow(nFacesNew, face2nodeV[iFace].size());
                for (DNDS::rowsize if2n = 0; if2n < face2node.RowSize(nFacesNew); if2n++)
                    face2node(nFacesNew, if2n) = face2nodeV[iFace][if2n];
                if (isPeriodic)
                {
                    DNDS_assert(face2nodeV[iFace].size() == face2nodePbiV[iFace].size());
                    face2nodePbi.ResizeRow(nFacesNew, face2nodePbiV[iFace].size());
                    for (DNDS::rowsize if2n = 0; if2n < face2nodePbi.RowSize(nFacesNew); if2n++)
                        face2nodePbi(nFacesNew, if2n) = face2nodePbiV[iFace][if2n];
                }
                face2cell(nFacesNew, 0) = face2cellV[iFace].first;
                face2cell(nFacesNew, 1) = face2cellV[iFace].second;
                faceElemInfo(nFacesNew, 0) = faceElemInfoV[iFace];
                nFacesNew++;
            }
        }

        MPI_Barrier(mpi.comm);
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iCell = 0; iCell < cell2face.Size(); iCell++) // convert face indices pointers
        {
            for (rowsize ic2f = 0; ic2f < cell2face.RowSize(iCell); ic2f++)
            {
                cell2face(iCell, ic2f) = iFaceAllToCollected[cell2face(iCell, ic2f)]; // Uninit if to discard
            }
        }
        adjFacialState = Adj_PointToLocal;

        // now face built except for the ghost part
        /*****************************************/

        /**********************************/
        // put bnd elem info into faces
        bnd2face.resize(bndElemInfo.Size(), -1);
        std::unordered_map<index, index> iFace2iBnd;
        for (DNDS::index iBnd = 0; iBnd < bndElemInfo.Size(); iBnd++)
        {
            DNDS::index pCell = bnd2cell(iBnd, 0);
            std::vector<DNDS::index> b2nRow = bnd2node[iBnd];
            std::sort(b2nRow.begin(), b2nRow.end());
            int nFound = 0;
            auto faceID = bndElemInfo[iBnd]->zone;
            for (DNDS::index ic2f = 0; ic2f < cell2face.RowSize(pCell); ic2f++)
            {
                auto iFace = cell2face(pCell, ic2f);
                if (iFace < 0) //==-1, pointing to ghost face
                    continue;
                std::vector<DNDS::index> f2nRow = face2node[iFace];
                std::sort(f2nRow.begin(), f2nRow.end());
                if (std::equal(b2nRow.begin(), b2nRow.end(), f2nRow.begin(), f2nRow.end()))
                {
                    if (iFace2iBnd.count(iFace))
                    {
                        DNDS_assert(FaceIDIsPeriodic(faceID)); // only periodic gets to be duplicated
                        index iBndOther = iFace2iBnd[iFace];
                        index iCellA = bnd2cell[iBnd][0];
                        index iCellB = bnd2cell[iBndOther][0];
                        DNDS_assert(iCellA < cell2node.father->Size()); // both points to local non-ghost cells
                        DNDS_assert(iCellB < cell2node.father->Size()); // both points to local non-ghost cells
                        // std::cout << iCellA << " vs " << iCellB << std::endl;
                        if (iCellA > iCellB)
                            continue;
                    }
                    iFace2iBnd[iFace] = iBnd;
                    nFound++; // two things:
                    // if is periodic, then only gets the bnd info of the main cell's bnd;
                    // if is external bc, then must be non-ghost face
                    faceElemInfo(iFace, 0) = bndElemInfo(iBnd, 0);
                    bnd2face[iBnd] = iFace;
                    DNDS_assert_info(FaceIDIsExternalBC(faceID) ||
                                         FaceIDIsPeriodic(faceID),
                                     "bnd elem should have a BC id not interior");
                }
            }
            DNDS_assert(nFound == 1 || (FaceIDIsPeriodic(faceID) && nFound == 0)); // periodic could miss the face
        }

        /**********************************/
        // alter face2node and face2cell to point to global
        this->AdjLocal2GlobalFacial();

        // comm on the faces
        std::vector<index> faceSendLocalsIdx;
        std::vector<index> faceSendLocalsStarts(mpi.size + 1);
        faceSendLocalsStarts[0] = 0;
        for (MPI_int r = 0; r < mpi.size; r++)
            faceSendLocalsStarts[r + 1] = faceSendLocalsStarts[r] + faceSendLocals[r].size();
        faceSendLocalsIdx.resize(faceSendLocalsStarts.back());
        for (MPI_int r = 0; r < mpi.size; r++)
            std::copy(faceSendLocals[r].begin(), faceSendLocals[r].end(), faceSendLocalsIdx.begin() + faceSendLocalsStarts[r]);

        face2node.father->Compress(); // before comm
        if (isPeriodic)
            face2nodePbi.father->Compress();
        face2cell.TransAttach();
        face2node.TransAttach();
        if (isPeriodic)
            face2nodePbi.TransAttach();
        faceElemInfo.TransAttach();

        face2cell.trans.createFatherGlobalMapping();
        face2cell.trans.createGhostMapping(faceSendLocalsIdx, faceSendLocalsStarts);
        face2node.trans.BorrowGGIndexing(face2cell.trans);
        if (isPeriodic)
            face2nodePbi.trans.BorrowGGIndexing(face2cell.trans);
        faceElemInfo.trans.BorrowGGIndexing(face2cell.trans);

        face2cell.trans.createMPITypes();
        face2node.trans.createMPITypes();
        if (isPeriodic)
            face2nodePbi.trans.createMPITypes();
        faceElemInfo.trans.createMPITypes();

        face2cell.trans.pullOnce();
        face2node.trans.pullOnce();
        if (isPeriodic)
            face2nodePbi.trans.pullOnce();
        faceElemInfo.trans.pullOnce();

        this->AdjGlobal2LocalFacial();
        // alter face2node and face2cell to point to local
        /**********************************/

        /**********************************/
        // tend to unattended cell2face with pointing to ghost
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iFace = 0; iFace < face2cell.son->Size(); iFace++) // face2cell points to local now
        {
            // before: first points to inner, //!relies on the order of setting face2cell
            DNDS_assert((*face2cell.son)(iFace, 0) >= cell2node.father->Size());
            auto eFace = Elem::Element{(*faceElemInfo.son)(iFace, 0).getElemType()};
            auto faceVerts = std::vector<index>((*face2node.son)[iFace].begin(), (*face2node.son)[iFace].begin() + eFace.GetNumVertices());
            std::sort(faceVerts.begin(), faceVerts.end()); //* do not forget to do set operation sort first
            for (rowsize if2c = 0; if2c < 2; if2c++)
            {
                index iCell = (*face2cell.son)(iFace, if2c);
                auto cell2faceRow = cell2face[iCell];
                auto cellNodes = cell2node[iCell];
                auto eCell = Elem::Element{cellElemInfo(iCell, 0).getElemType()};
                bool found = false;
                for (rowsize ic2f = 0; ic2f < cell2face.RowSize(iCell); ic2f++)
                {
                    auto eFace = eCell.ObtainFace(ic2f);
                    std::vector<index> faceNodesC(eFace.GetNumNodes());
                    eCell.ExtractFaceNodes(ic2f, cellNodes, faceNodesC);
                    std::sort(faceNodesC.begin(), faceNodesC.end());
                    if (std::includes(faceNodesC.begin(), faceNodesC.end(), faceVerts.begin(), faceVerts.end()))
                    {
                        DNDS_assert(cell2face(iCell, ic2f) == -1);
                        cell2face(iCell, ic2f) = iFace + face2cell.father->Size(); // remember is ghost
                        found = true;
                    }
                }
                DNDS_assert(found);
            }
        }

        cell2face.father->Compress();
        cell2face.son->Compress();

        for (DNDS::index iFace = 0; iFace < faceElemInfo.Size(); iFace++)
        {
            if (FaceIDIsPeriodicMain(faceElemInfo(iFace, 0).zone))
            {
                // DNDS_assert(false);
            }
        }
        for (DNDS::index iFace = 0; iFace < faceElemInfo.Size(); iFace++)
        {
            if (FaceIDIsPeriodicDonor(faceElemInfo(iFace, 0).zone))
            {
                // DNDS_assert(false);
            }
        }

        // DNDS::MPISerialDo(mpi, [&]()
        //                   { std::cout << "Rank " << mpi.rank << " : nFace " << face2node.Size() << std::endl; });
        // DNDS::MPISerialDo(mpi, [&]()
        //                   { std::cout << "Rank " << mpi.rank << " : nC2C " << cell2cell.father->DataSize() << std::endl; });
        auto gSize = face2node.father->globalSize(); //! sync call!!!
        if (mpi.rank == 0)
            log() << "UnstructuredMesh === InterpolateFace: total faces " << gSize << std::endl;
    }

    void UnstructuredMesh::
        AssertOnFaces()
    {

        //* some assertions on faces
        std::vector<uint16_t> cCont(cell2cell.Size(), 0); // simulate flux
        for (DNDS::index iFace = 0; iFace < faceElemInfo.Size(); iFace++)
        {
            auto faceID = faceElemInfo(iFace, 0).zone;
            if (FaceIDIsInternal(faceID))
            {
                // if (FaceIDIsPeriodic(faceID))
                // {
                //     // TODO: tend to the case of face is PeriodicDonor with Main in same proc
                //     continue;
                // }
                // if (face2cell[iFace][0] < cell2cell.father->Size()) // other side prime cell, periodic also
                DNDS_assert(face2cell[iFace][1] != DNDS::UnInitIndex); // Assert has enough cell donors
                DNDS_assert(face2cell[iFace][0] >= 0 && face2cell[iFace][0] < cell2cell.Size());
                DNDS_assert(face2cell[iFace][1] >= 0 && face2cell[iFace][1] < cell2cell.Size());
                cCont[face2cell[iFace][0]]++;
                cCont[face2cell[iFace][1]]++;
            }
            else // a external BC
            {
                DNDS_assert(face2cell[iFace][1] == DNDS::UnInitIndex);
                DNDS_assert(face2cell[iFace][0] >= 0 && face2cell[iFace][0] < cell2cell.father->Size());
                cCont[face2cell[iFace][0]]++;
            }
        }
        for (DNDS::index iCell = 0; iCell < cellElemInfo.father->Size(); iCell++) // for every non-ghost
        {
            for (auto iFace : cell2face[iCell])
            {
                DNDS_assert(iFace >= 0 && iFace < face2cell.Size());
                DNDS_assert(face2cell[iFace][0] == iCell || face2cell[iFace][1] == iCell);
            }
            DNDS_assert(cCont[iCell] == cell2face.RowSize(iCell));
        }
    }

    void UnstructuredMesh::
        WriteSerialize(SerializerBase *serializer, const std::string &name)
    {
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);

        auto cwd = serializer->GetCurrentPath();
        serializer->CreatePath(name);
        serializer->GoToPath(name);

        serializer->WriteString("mesh", "UnstructuredMesh");
        serializer->WriteIndex("dim", dim);
        serializer->WriteIndex("MPIRank", mpi.rank);
        serializer->WriteIndex("MPISize", mpi.size);
        serializer->WriteInt("isPeriodic", isPeriodic);

        coords.WriteSerialize(serializer, "coords");
        cell2node.WriteSerialize(serializer, "cell2node");
        cell2cell.WriteSerialize(serializer, "cell2cell");
        cellElemInfo.WriteSerialize(serializer, "cellElemInfo");
        bnd2node.WriteSerialize(serializer, "bnd2node");
        bnd2cell.WriteSerialize(serializer, "bnd2cell");
        bndElemInfo.WriteSerialize(serializer, "bndElemInfo");
        if (isPeriodic)
        {
            cell2nodePbi.WriteSerialize(serializer, "cell2nodePbi");
            periodicInfo.WriteSerializer(serializer, "periodicInfo");
        }

        serializer->GoToPath(cwd);
    }

    void UnstructuredMesh::
        ReadSerialize(SerializerBase *serializer, const std::string &name)
    {
        auto cwd = serializer->GetCurrentPath();
        // serializer->CreatePath(name);//! remember no create!
        serializer->GoToPath(name);

        std::string meshRead;
        index dimRead, rankRead, sizeRead;
        int isPeriodicRead;
        serializer->ReadString("mesh", meshRead);
        serializer->ReadIndex("dim", dimRead);
        serializer->ReadIndex("MPIRank", rankRead);
        serializer->ReadIndex("MPISize", sizeRead);
        serializer->ReadInt("isPeriodic", isPeriodicRead);
        isPeriodic = bool(isPeriodicRead);
        DNDS_assert(meshRead == "UnstructuredMesh");
        DNDS_assert(dimRead == dim);
        DNDS_assert(rankRead == mpi.rank && sizeRead == mpi.size);

        // make the empty arrays
        auto mesh = this;
        DNDS_MAKE_SSP(mesh->coords.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->coords.son, mesh->mpi);
        DNDS_MAKE_SSP(mesh->cellElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(mesh->cellElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(mesh->bndElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(mesh->bndElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(mesh->cell2node.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->cell2node.son, mesh->mpi);
        if (isPeriodic)
        {
            DNDS_MAKE_SSP(mesh->cell2nodePbi.father, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->mpi);
            DNDS_MAKE_SSP(mesh->cell2nodePbi.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->mpi);
        }
        DNDS_MAKE_SSP(mesh->cell2cell.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->cell2cell.son, mesh->mpi);
        DNDS_MAKE_SSP(mesh->bnd2node.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->bnd2node.son, mesh->mpi);
        DNDS_MAKE_SSP(mesh->bnd2cell.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->bnd2cell.son, mesh->mpi);

        coords.ReadSerialize(serializer, "coords");
        cell2node.ReadSerialize(serializer, "cell2node");
        cell2cell.ReadSerialize(serializer, "cell2cell");
        cellElemInfo.ReadSerialize(serializer, "cellElemInfo");
        bnd2node.ReadSerialize(serializer, "bnd2node");
        bnd2cell.ReadSerialize(serializer, "bnd2cell");
        bndElemInfo.ReadSerialize(serializer, "bndElemInfo");
        if (isPeriodic)
        {
            cell2nodePbi.ReadSerialize(serializer, "cell2nodePbi");
            periodicInfo.ReadSerializer(serializer, "periodicInfo");
        }

        // after matters:
        coords.trans.createMPITypes();
        cell2node.trans.createMPITypes();
        cell2cell.trans.createMPITypes();
        cellElemInfo.trans.createMPITypes();
        bnd2node.trans.createMPITypes();
        bnd2cell.trans.createMPITypes();
        bndElemInfo.trans.createMPITypes();
        if (isPeriodic)
            cell2nodePbi.trans.createMPITypes();
        adjPrimaryState = Adj_PointToLocal; // the file is pointing to local

        index nCellG = this->NumCellGlobal(); // collective call!
        index nNodeG = this->NumNodeGlobal(); // collective call!
        if (mpi.rank == mRank)
        {
            log() << "UnstructuredMesh === ReadSerialize "
                  << "Global NumCell [ " << nCellG << " ]" << std::endl;
            log() << "UnstructuredMesh === ReadSerialize "
                  << "Global NumNode [ " << nNodeG << " ]" << std::endl;
        }
        MPISerialDo(mpi, [&]()
                    { log() << "    Rank: " << mpi.rank << " nCell " << this->NumCell() << " nCellGhost " << this->NumCellGhost() << std::endl; });
        MPISerialDo(mpi, [&]()
                    { log() << "    Rank: " << mpi.rank << " nNode " << this->NumNode() << " nNodeGhost " << this->NumNodeGhost() << std::endl; });

        serializer->GoToPath(cwd);
    }

    void UnstructuredMesh::ConstructBndMesh(UnstructuredMesh &bMesh)
    {
        DNDS_assert(bMesh.dim == dim - 1 && bMesh.mpi == mpi);
        DNDS_MAKE_SSP(bMesh.cell2node.father, mpi);
        DNDS_MAKE_SSP(bMesh.cell2node.son, mpi);
        DNDS_MAKE_SSP(bMesh.coords.father, mpi);
        DNDS_MAKE_SSP(bMesh.coords.son, mpi);

        bMesh.cellElemInfo.father = bndElemInfo.father;
        bMesh.cellElemInfo.son = bndElemInfo.son;

        node2bndNode.resize(this->NumNodeProc(), -1);
        index bndNodeCount{0};
        for (index iBnd = 0; iBnd < this->NumBnd(); iBnd++) //! bnd has no ghost!
            for (auto iNode : bnd2node.father->operator[](iBnd))
                if (node2bndNode.at(iNode) == -1)
                    node2bndNode.at(iNode) = bndNodeCount++;
        bMesh.node2parentNode.resize(bndNodeCount);
        for (index iNode = 0; iNode < node2bndNode.size(); iNode++)
            if (node2bndNode[iNode] >= 0)
                bMesh.node2parentNode.at(node2bndNode[iNode]) = iNode;
        bMesh.coords.father->Resize(bndNodeCount);
        // std::cout << bndNodeCount << std::endl;
        for (index iBNode = 0; iBNode < bndNodeCount; iBNode++)
            bMesh.coords[iBNode] = coords[bMesh.node2parentNode[iBNode]];
        bMesh.cell2node.father->Resize(this->NumBnd());
        for (index iB = 0; iB < this->NumBnd(); iB++)
        {
            bMesh.cell2node.ResizeRow(iB, bnd2node.father->RowSize(iB));
            for (rowsize ib2n = 0; ib2n < bnd2node.father->RowSize(iB); ib2n++)
                bMesh.cell2node[iB][ib2n] = node2bndNode.at(bnd2node[iB][ib2n]),
                DNDS_assert(node2bndNode.at(bnd2node[iB][ib2n]) >= 0);
        }

        bMesh.cell2node.father->Compress();

        bMesh.coords.father->createGlobalMapping();
        bMesh.cell2node.father->createGlobalMapping();

        bMesh.coords.TransAttach();
        bMesh.cell2node.TransAttach();
        bMesh.coords.trans.createGhostMapping(std::vector<int>{});
        bMesh.cell2node.trans.createGhostMapping(std::vector<int>{});

        bMesh.adjPrimaryState = Adj_PointToLocal;
    }
}