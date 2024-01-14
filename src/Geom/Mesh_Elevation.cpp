#include "Mesh.hpp"

#include <omp.h>
#include <fmt/core.h>

namespace DNDS::Geom
{
    void UnstructuredMesh::BuildO2FromO1Elevation(UnstructuredMesh &meshO1)
    {
        real epsSqrDist = 1e-15;

        bool O1MeshIsO1 = meshO1.IsO1();
        DNDS_assert(O1MeshIsO1);
        // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX Phase0" << std::endl;
        DNDS_assert(meshO1.adjPrimaryState == Adj_PointToLocal);
        mRank = meshO1.mRank;
        mpi = meshO1.mpi;
        dim = meshO1.dim;
        isPeriodic = meshO1.isPeriodic;
        periodicInfo = meshO1.periodicInfo;

        tAdjPair cellNewNodes; // records iNode - iNodeO1
        index localNewNodeNum{0};
        DNDS_MAKE_SSP(cellNewNodes.father, mpi);
        DNDS_MAKE_SSP(cellNewNodes.son, mpi);
        cellNewNodes.father->Resize(meshO1.cell2node.father->Size());
        std::vector<Eigen::Vector<real, 3>> newCoords;

        // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX Phase1" << std::endl;

        /**********************************/ // TODO
        // add new nodes to cellNewNodes, edge and face O2 nodes belong to smallest neighboring cell, cellNewNodes record local node indices
        for (index iCell = 0; iCell < meshO1.cell2node.father->Size(); iCell++)
        {
            Elem::Element eCell = meshO1.GetCellElement(iCell);
            auto c2n = meshO1.cell2node[iCell];
            auto c2c = meshO1.cell2cell[iCell];
            index iCellGlob = meshO1.cell2node.trans.pLGhostMapping->operator()(-1, iCell);

            std::vector<index> cellNewNodeRowV;
            cellNewNodeRowV.resize(eCell.GetNumElev_O1O2());
            SmallCoordsAsVector coordsC;
            meshO1.GetCoordsOnCell(iCell, coordsC);

            for (index iNNode = 0; iNNode < eCell.GetNumElev_O1O2(); iNNode++)
            {
                auto eSpan = eCell.ObtainElevNodeSpan(iNNode);
                std::array<index, Elem::CellNumNodeMax> spanNodes;
                eCell.ExtractElevNodeSpanNodes(iNNode, c2n, spanNodes);
                auto spanNodesSrt = spanNodes;
                std::sort(spanNodesSrt.begin(), spanNodesSrt.begin() + eSpan.GetNumNodes());
                tPoint newNodeCoord;

                { // the new node method, should cope with shape func
                    SmallCoordsAsVector coordsSpan;
                    coordsSpan.resize(3, eSpan.GetNumNodes());
                    eCell.ExtractElevNodeSpanNodes(iNNode, coordsC, coordsSpan);
                    newNodeCoord = coordsSpan.rowwise().mean();

                    if (isPeriodic)
                    {
                        std::vector<NodePeriodicBits> pbi;
                        pbi.resize(eSpan.GetNumNodes());
                        eCell.ExtractElevNodeSpanNodes(iNNode, meshO1.cell2nodePbi[iCell], pbi);
                        NodePeriodicBits pbiR;
                        pbiR.setP1True(), pbiR.setP2True(), pbiR.setP3True();
                        for (auto pbie : pbi)
                            pbiR = pbiR & pbie;
                        newNodeCoord = periodicInfo.GetCoordBackByBits(newNodeCoord, pbiR);
                    }
                }

                index curMinGlobiCell = iCellGlob;
                index curMinic2c = -1;
                for (int ic2c = 0; ic2c < c2c.size(); ic2c++)
                {
                    index iCellOther = c2c[ic2c];
                    if (iCellOther == iCell)
                        continue;
                    std::vector<index> c2nOther = meshO1.cell2node[iCellOther];
                    index iCellGlobOther = meshO1.cell2node.trans.pLGhostMapping->operator()(-1, iCellOther);
                    std::sort(c2nOther.begin(), c2nOther.end());
                    if (iCellGlobOther < curMinGlobiCell &&
                        std::includes(c2nOther.begin(), c2nOther.end(), spanNodesSrt.begin(), spanNodesSrt.begin() + eSpan.GetNumNodes()))
                    {
                        curMinGlobiCell = iCellGlobOther;
                        curMinic2c = ic2c;
                    }
                }
                if (curMinGlobiCell == iCellGlob)
                {
                    newCoords.push_back(newNodeCoord);
                    cellNewNodeRowV.at(iNNode) = (localNewNodeNum++); //*
                }
                else
                {
                    cellNewNodeRowV.at(iNNode) = (-1 - curMinic2c);
                }
            }
            cellNewNodes.ResizeRow(iCell, cellNewNodeRowV.size());
            cellNewNodes[iCell] = cellNewNodeRowV;
            // std::cout << "iCell " << iCell << ":  ";
            // for (auto i : cellNewNodeRowV)
            //     std::cout << i << ", ";
            // std::cout << std::endl;
        }
        // std::cout << fmt::format("Num NewNode {} ", localNewNodeNum) << std::endl;
        // for (auto v : newCoords)
        //     std::cout << v.transpose() << std::endl;
        index numNewNode;
        MPI::Allreduce(&localNewNodeNum, &numNewNode, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        if (mpi.rank == mRank)
            log() << fmt::format("=== Mesh Elevation: Num NewNode {} ", numNewNode) << std::endl;

        /**********************************/
        // build each proc coordO2 from cellNewNodes
        DNDS_MAKE_SSP(coords.father, mpi);
        DNDS_MAKE_SSP(coords.son, mpi);
        coords.father->Resize(meshO1.coords.father->Size() + localNewNodeNum);
        for (index iC = 0; iC < meshO1.coords.father->Size(); iC++)
            coords[iC] = meshO1.coords[iC];
        for (index iC = meshO1.coords.father->Size(); iC < meshO1.coords.father->Size() + localNewNodeNum; iC++)
            coords[iC] = newCoords.at(iC - meshO1.coords.father->Size());
        coords.father->createGlobalMapping();

        // tAdj1Pair nodeLocalIdxOld;
        // DNDS_MAKE_SSP(nodeLocalIdxOld.father, mpi);
        // DNDS_MAKE_SSP(nodeLocalIdxOld.son, mpi);
        // nodeLocalIdxOld.father->Resize(meshO1.coords.father->Size());
        // for (index iC = 0; iC < meshO1.coords.father->Size(); iC++)
        //     nodeLocalIdxOld[iC][0] = iC;
        // nodeLocalIdxOld.TransAttach();
        // nodeLocalIdxOld.trans.BorrowGGIndexing(mesh->cell2node.trans);
        // nodeLocalIdxOld.trans.createMPITypes();
        // nodeLocalIdxOld.trans.pullOnce();

        /**********************************/
        // from coordO2 get cellNewNodesGlobal, and get cellNewNodesGlobalPair
        for (index iCell = 0; iCell < meshO1.cell2node.father->Size(); iCell++)
        {
            auto cellNewNodesRow = cellNewNodes[iCell];
            for (auto &iNodeNew : cellNewNodesRow)
            {
                // nodes here must be at local proc; now use global idx
                iNodeNew =
                    iNodeNew < 0
                        ? iNodeNew // negative field was filled with indicating which cell to seek the node
                        : coords.father->pLGlobalMapping->operator()(mpi.rank, iNodeNew + meshO1.coords.father->Size());
            }
        }
        cellNewNodes.TransAttach();
        cellNewNodes.trans.BorrowGGIndexing(meshO1.cell2node.trans);
        cellNewNodes.trans.createMPITypes();
        cellNewNodes.trans.pullOnce();

        coords.TransAttach();
        std::vector<DNDS::index> ghostNodesTmp;
        for (index iCell = 0; iCell < meshO1.cell2node.Size(); iCell++)
        {
            auto cellNewNodesRow = cellNewNodes[iCell];
            for (auto iNodeNew : cellNewNodesRow)
            {
                if (iNodeNew < 0)
                    continue;
                MPI_int rank;
                index val;
                if (!coords.trans.pLGlobalMapping->search(iNodeNew, rank, val))
                    DNDS_assert_info(false, "search failed");
                if (rank != mpi.rank)
                    ghostNodesTmp.push_back(iNodeNew);
            }
        }
        coords.trans.createGhostMapping(ghostNodesTmp);
        coords.trans.createMPITypes();
        coords.trans.pullOnce();

        /**********************************/
        // each cell obtain new global cell2node global state with cellNewNodesGlobalPair
        DNDS_MAKE_SSP(cell2node.father, mpi);
        DNDS_MAKE_SSP(cell2node.son, mpi);
        DNDS_MAKE_SSP(cellElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        DNDS_MAKE_SSP(cellElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        if (isPeriodic)
        {
            DNDS_MAKE_SSP(cell2nodePbi.father, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), getMPI());
            DNDS_MAKE_SSP(cell2nodePbi.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), getMPI());
        }

        cell2node.father->Resize(meshO1.cell2node.father->Size());
        cellElemInfo.father->Resize(meshO1.cell2node.father->Size());
        if (isPeriodic)
            cell2nodePbi.father->Resize(meshO1.cell2node.father->Size());
        for (index iCell = 0; iCell < meshO1.cell2node.father->Size(); iCell++)
        {
            Elem::Element eCell = meshO1.GetCellElement(iCell);
            Elem::Element eCellO2 = eCell.ObtainElevatedElem();
            cell2node.father->ResizeRow(iCell, eCellO2.GetNumNodes());
            if (isPeriodic)
                cell2nodePbi.father->ResizeRow(iCell, eCellO2.GetNumNodes());
            auto c2n = meshO1.cell2node[iCell];
            for (int ic2n = 0; ic2n < c2n.size(); ic2n++)
            {
                // fill in the O1 nodes, note that global indices for O1 nodes have changed
                index iNodeOldGlobal = meshO1.coords.trans.pLGhostMapping->operator()(-1, c2n[ic2n]);
                index nodeOldOrigLocalIdx{-1};
                int nodeOldOrigRank{-1};
                if (!meshO1.coords.trans.pLGlobalMapping->search(iNodeOldGlobal, nodeOldOrigRank, nodeOldOrigLocalIdx))
                    DNDS_assert_info(false, "search failed");
                // nodeOldOrigRank and nodeOldOrigLocalIdx is same in new
                cell2node(iCell, ic2n) =
                    coords.father->pLGlobalMapping->operator()(nodeOldOrigRank, nodeOldOrigLocalIdx); // now point to global

                // fill in node pbi
                if (isPeriodic)
                    cell2nodePbi(iCell, ic2n) = meshO1.cell2nodePbi(iCell, ic2n);
            }
            for (int ic2n = c2n.size(); ic2n < cell2node[iCell].size(); ic2n++)
                cell2node(iCell, ic2n) = -1;

            cellElemInfo(iCell, 0) = meshO1.cellElemInfo(iCell, 0);
            cellElemInfo(iCell, 0).setElemType(eCellO2.type); // update cell elem info
        }
        // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
        for (index iCell = 0; iCell < meshO1.cell2node.father->Size(); iCell++)
        {
            Elem::Element eCell = meshO1.GetCellElement(iCell);
            auto c2n = meshO1.cell2node[iCell];
            auto c2c = meshO1.cell2cell[iCell];
            index iCellGlob = meshO1.cell2node.trans.pLGhostMapping->operator()(-1, iCell);

            // std::vector<index> cellNewNodeRowV;
            // cellNewNodeRowV.resize(eCell.GetNumElev_O1O2());
            SmallCoordsAsVector coordsC;
            meshO1.GetCoordsOnCell(iCell, coordsC);

            for (index iNNode = 0; iNNode < eCell.GetNumElev_O1O2(); iNNode++)
            {
                auto eSpan = eCell.ObtainElevNodeSpan(iNNode);
                std::array<index, Elem::CellNumNodeMax> spanNodes;
                eCell.ExtractElevNodeSpanNodes(iNNode, c2n, spanNodes);
                auto spanNodesSrt = spanNodes;
                std::sort(spanNodesSrt.begin(), spanNodesSrt.begin() + eSpan.GetNumNodes());
                tPoint newNodeCoord;
                { // the new node method, should cope with shape func
                    SmallCoordsAsVector coordsSpan;
                    coordsSpan.resize(3, eSpan.GetNumNodes());
                    eCell.ExtractElevNodeSpanNodes(iNNode, coordsC, coordsSpan);
                    newNodeCoord = coordsSpan.rowwise().mean();

                    // std::cout << fmt::format("############\niCell {}, iNNode {}", iCell, iNNode) << std::endl;
                    // std::cout << newNodeCoord.transpose() << std::endl;

                    if (isPeriodic)
                    {
                        std::vector<NodePeriodicBits> pbi;
                        pbi.resize(eSpan.GetNumNodes());
                        eCell.ExtractElevNodeSpanNodes(iNNode, meshO1.cell2nodePbi[iCell], pbi);
                        NodePeriodicBits pbiR;
                        pbiR.setP1True(), pbiR.setP2True(), pbiR.setP3True();
                        for (auto pbie : pbi)
                            pbiR = pbiR & pbie;
                        // std::cout << uint(uint8_t(pbiR)) << std::endl;
                        newNodeCoord = periodicInfo.GetCoordBackByBits(newNodeCoord, pbiR);
                        cell2nodePbi[iCell][c2n.size() + iNNode] = pbiR; //* fill In the O2 cell2nodePbi
                        // ** cell2nodePbiO2: edge using edge common, face using face common
                        // ** note that this technique requires at least 2 layer each periodic direction
                    }
                }
                if (cellNewNodes[iCell][iNNode] >= 0)
                {
                    cell2node[iCell][c2n.size() + iNNode] = cellNewNodes[iCell][iNNode];
                    continue;
                }

                int nFound = 0;
                real minSqrDist = veryLargeReal;
                for (auto iNNodeC : cellNewNodes[c2c[-1 - cellNewNodes[iCell][iNNode]]])
                {
                    if (iNNodeC < 0)
                        continue;
                    // iNNodeC is global iNode
                    MPI_int rank{-1};
                    index val{-1};
                    if (!coords.trans.pLGhostMapping->search_indexAppend(iNNodeC, rank, val))
                        DNDS_assert_info(false, "search failed");
                    // std::cout << fmt::format("val {}, iNNodeC {}", val, iNNodeC) << std::endl;
                    // std::cout << (coords[val] - newNodeCoord).squaredNorm() << std::endl;
                    // std::cout << coords[val].transpose() << ", " << newNodeCoord.transpose() << std::endl;
                    real sqrDist = (coords[val] - newNodeCoord).squaredNorm();
                    minSqrDist = std::min(sqrDist, minSqrDist);
                    if (sqrDist < epsSqrDist)
                    {
                        nFound++;
                        cell2node[iCell][c2n.size() + iNNode] = iNNodeC;
                    }
                }
                DNDS_assert_info(nFound == 1, fmt::format("geometric search for elevated point failed, nFound = {}, minSqrDist = {}, pos ({},{},{})", nFound, minSqrDist,
                                                          newNodeCoord(0), newNodeCoord(1), newNodeCoord(2)));
                //* comment: way of ridding of the geometric search:
                //* use interpolated topo: face/edge
                //* or record the full vertex string for each O2 nodes' span (O2 nodes do not have siblings in the same span)
            }
        }
        /**********************************/
        // cell2cell, bnd2cell can be copied; bnd2nodeO2 created with bnd2cell and bnd2node;
        // cell2cell = meshO1.cell2cell;
        // bnd2cell = meshO1.bnd2cell;
        DNDS_MAKE_SSP(cell2cell.father, mpi);
        DNDS_MAKE_SSP(cell2cell.son, mpi);
        cell2cell.father->Resize(meshO1.cell2cell.father->Size());
        for (index iCell = 0; iCell < meshO1.cell2node.father->Size(); iCell++)
        {
            cell2cell.father->ResizeRow(iCell, meshO1.cell2cell.RowSize(iCell));
            cell2cell[iCell] = std::vector<index>{meshO1.cell2cell[iCell]};
            for (auto &iCellOther : cell2cell[iCell])
            {
                iCellOther = meshO1.cell2cell.trans.pLGhostMapping->operator()(-1, iCellOther);
            }
        }
        DNDS_MAKE_SSP(bnd2cell.father, mpi);
        DNDS_MAKE_SSP(bnd2cell.son, mpi);
        bnd2cell.father->Resize(meshO1.bnd2cell.father->Size());
        for (index iBnd = 0; iBnd < meshO1.bnd2node.father->Size(); iBnd++)
        {
            bnd2cell(iBnd, 0) = meshO1.bnd2cell(iBnd, 0);
            bnd2cell(iBnd, 0) = meshO1.cell2cell.trans.pLGhostMapping->operator()(-1, bnd2cell(iBnd, 0));
        }

        DNDS_MAKE_SSP(bnd2node.father, mpi);
        DNDS_MAKE_SSP(bnd2node.son, mpi);
        DNDS_MAKE_SSP(bndElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        DNDS_MAKE_SSP(bndElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        bnd2node.father->Resize(meshO1.bnd2node.father->Size());
        bndElemInfo.father->Resize(meshO1.bnd2node.father->Size());
        for (index iBnd = 0; iBnd < meshO1.bnd2node.father->Size(); iBnd++)
        {
            index iCell = meshO1.bnd2cell(iBnd, 0); // my own bnd2cell is to global!
            auto eCellO2 = this->GetCellElement(iCell);
            auto b2n = meshO1.bnd2node[iBnd];
            auto c2nO2 = this->cell2node[iCell];
            std::vector<index> b2nv = b2n;
            for (auto &i : b2nv)
            {
                //* note that b2nv holds O1 nodes' old and local indices
                index iNodeOldGlobal = meshO1.coords.trans.pLGhostMapping->operator()(-1, i);
                index nodeOldOrigLocalIdx{-1};
                int nodeOldOrigRank{-1};
                if (!meshO1.coords.trans.pLGlobalMapping->search(iNodeOldGlobal, nodeOldOrigRank, nodeOldOrigLocalIdx))
                    DNDS_assert_info(false, "search failed");
                // nodeOldOrigRank and nodeOldOrigLocalIdx is same in new
                i = coords.father->pLGlobalMapping->operator()(nodeOldOrigRank, nodeOldOrigLocalIdx); // now point to global
            }
            std::sort(b2nv.begin(), b2nv.end());

            int nFound{0};
            int c2fFound{-1};
            std::vector<index> b2nO2Found;
            Elem::Element eBndFound;

            for (int ic2f = 0; ic2f < eCellO2.GetNumFaces(); ic2f++)
            {
                auto eFaceO2 = eCellO2.ObtainFace(ic2f);
                std::vector<index> f2nO2;
                f2nO2.resize(eFaceO2.GetNumNodes());
                eCellO2.ExtractFaceNodes(ic2f, c2nO2, f2nO2);
                std::sort(f2nO2.begin(), f2nO2.end());
                if (std::includes(f2nO2.begin(), f2nO2.end(), b2nv.begin(), b2nv.end()))
                {
                    nFound++;
                    c2fFound = ic2f;
                    b2nO2Found = f2nO2;
                    eBndFound = eFaceO2;
                }
            }
            DNDS_assert(nFound == 1);
            bnd2node.father->ResizeRow(iBnd, b2nO2Found.size());
            bnd2node[iBnd] = b2nO2Found;
            bndElemInfo(iBnd, 0) = meshO1.bndElemInfo(iBnd, 0);
            bndElemInfo(iBnd, 0).setElemType(eBndFound.type);
        }
        adjPrimaryState = Adj_PointToGlobal;

        DNDS_MAKE_SSP(coords.son, mpi);

        // this->BuildGhostPrimary();

        // this->AdjGlobal2LocalPrimary();
        // if (mpi.rank == 0)
        // {
        //     for (index iCell = 0; iCell < meshO1.cell2node.Size(); iCell++)
        //     {
        //         std::vector<index> v1 = meshO1.cell2node[iCell];
        //         std::vector<index> v2 = cell2node[iCell];
        //         std::cout << "iCell " << iCell << std::endl;
        //         for (auto i : v1)
        //             std::cout << i << "(" << meshO1.coords[i].transpose() << ")"
        //                       << ", ";
        //         std::cout << std::endl;
        //         for (auto i : v2)
        //             std::cout << i << "(" << coords[i].transpose() << ")"
        //                       << ", ";
        //         std::cout << std::endl;

        //         std::vector<index> v3 = meshO1.cell2cell[iCell];
        //         std::vector<index> v4 = cell2cell[iCell];
        //         for (auto i : v3)
        //             std::cout << i
        //                       << ", ";
        //         std::cout << std::endl;
        //         for (auto i : v4)
        //             std::cout << i
        //                       << ", ";
        //         std::cout << std::endl;
        //     }
        // }
        // // this->AdjLocal2GlobalPrimary();
    }
}