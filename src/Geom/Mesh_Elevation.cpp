#include "Mesh.hpp"
#include "Quadrature.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "RadialBasisFunction.hpp"

#include <omp.h>
#include <fmt/core.h>
#include <functional>
#include <unordered_set>
#include <Solver/Linear.hpp>

namespace DNDS::Geom
{
    void UnstructuredMesh::BuildO2FromO1Elevation(UnstructuredMesh &meshO1)
    {
        real epsSqrDist = 1e-20;

        bool O1MeshIsO1 = meshO1.IsO1();
        DNDS_assert(O1MeshIsO1);
        // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX Phase0" << std::endl;
        DNDS_assert(meshO1.adjPrimaryState == Adj_PointToLocal);
        mRank = meshO1.mRank;
        mpi = meshO1.mpi;
        dim = meshO1.dim;
        isPeriodic = meshO1.isPeriodic;
        periodicInfo = meshO1.periodicInfo;
        nNodeO1 = meshO1.coords.father->Size();
        elevState = Elevation_O1O2;

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

    static tPoint HermiteInterpolateMidPointOnLine2WithNorm(tPoint c0, tPoint c1, tPoint n0, tPoint n1)
    {
        tPoint c01 = c1 - c0;
        tPoint c01U = c01.stableNormalized();
        real c01Len = c01.stableNorm();
        tPoint t0 = -c01U.cross(n0).cross(n0);
        tPoint t1 = -c01U.cross(n1).cross(n1);
        t0 = t0.stableNormalized() * c01Len;
        t1 = t1.stableNormalized() * c01Len;
        if (n0.norm() == 0 && n1.norm() != 0)
            return 0.25 * c0 + 0.75 * c1 - 0.25 * t1;
        if (n0.norm() != 0 && n1.norm() == 0)
            return 0.75 * c0 + 0.25 * c1 + 0.25 * t0;
        if (n0.norm() == 0 && n1.norm() == 0)
            return 0.5 * (c0 + c1);
        return 0.5 * (c0 + c1) + 0.125 * (t0 - t1);
    }

    static tPoint HermiteInterpolateMidPointOnQuad4WithNorm(
        tPoint c0, tPoint c1, tPoint c2, tPoint c3,
        tPoint n0, tPoint n1, tPoint n2, tPoint n3)
    {
        tPoint c01 = HermiteInterpolateMidPointOnLine2WithNorm(c0, c1, n0, n1);
        tPoint c12 = HermiteInterpolateMidPointOnLine2WithNorm(c1, c2, n1, n2);
        tPoint c23 = HermiteInterpolateMidPointOnLine2WithNorm(c2, c3, n2, n3);
        tPoint c30 = HermiteInterpolateMidPointOnLine2WithNorm(c3, c0, n3, n0);
        return 0.5 * (c01 + c12 + c23 + c30) - 0.25 * (c0 + c1 + c2 + c3);
    }

    void UnstructuredMesh::ElevatedNodesGetBoundarySmooth(const std::function<bool(t_index)> &FiFBndIdNeedSmooth)
    {
        // if (mpi.rank == 1)
        //     for (index iCell = 0; iCell < cell2face.Size(); iCell++)
        //     {
        //         std::cout << iCell << ": ";
        //         for (auto i : cell2face[iCell])
        //             std::cout << i << ", ";
        //         std::cout << std::endl;
        //     }
        DNDS_assert(elevState == Elevation_O1O2);
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        DNDS_assert(face2node.father);

        DNDS_MAKE_SSP(coordsElevDisp.father, mpi);
        DNDS_MAKE_SSP(coordsElevDisp.son, mpi);
        coordsElevDisp.father->Resize(coords.father->Size());
        coordsElevDisp.TransAttach();
        coordsElevDisp.trans.BorrowGGIndexing(coords.trans);
        coordsElevDisp.trans.createMPITypes();

        for (index iN = 0; iN < coords.Size(); iN++)
            coordsElevDisp[iN].setConstant(largeReal);

        // tAdj1Pair nCoordBndNum;
        // DNDS_MAKE_SSP(nCoordBndNum.father, mpi);
        // DNDS_MAKE_SSP(nCoordBndNum.son, mpi);
        // nCoordBndNum.father->Resize(coords.father->Size());
        // nCoordBndNum.TransAttach();
        // nCoordBndNum.trans.BorrowGGIndexing(coords.trans);
        // nCoordBndNum.trans.createMPITypes();

        // for (index iN = 0; iN < coords.size(); iN++)
        //     nCoordBndNum[iN][0] = 0;

        /***********************************/
        // build faceExteded info
        tAdjPair face2nodeExtended;
        tElemInfoArrayPair faceElemInfoExtended;
        tPbiPair face2nodePbiExtended;
        face2nodeExtended.father = face2node.father;
        faceElemInfoExtended.father = faceElemInfo.father;
        if (isPeriodic)
            face2nodePbiExtended.father = face2nodePbi.father;
        DNDS_MAKE_SSP(face2nodeExtended.son, mpi);
        DNDS_MAKE_SSP(faceElemInfoExtended.son, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        if (isPeriodic)
            DNDS_MAKE_SSP(face2nodePbiExtended.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);

        this->AdjLocal2GlobalFacial();
        this->AdjLocal2GlobalC2F();

        std::vector<index> faceGhostExt;
        for (index iCell = 0; iCell < cell2face.Size(); iCell++)
            for (auto iFace : cell2face[iCell])
            {
                DNDS_assert_info(iFace >= 0 || iFace == UnInitIndex, fmt::format("iFace {}", iFace));
                if (iFace == UnInitIndex) // old cell2face could contain void pointing
                    continue;
                DNDS::MPI_int rank;
                DNDS::index val;
                if (!face2node.trans.pLGlobalMapping->search(iFace, rank, val))
                    DNDS_assert_info(false, "search failed");
                if (rank != mpi.rank)
                    faceGhostExt.push_back(iFace);
                // if (mpi.rank == 1)
                // std::cout << "added Face " << iFace << std::endl;
            }
        face2nodeExtended.TransAttach();
        faceElemInfoExtended.TransAttach();
        if (isPeriodic)
            face2nodePbiExtended.TransAttach();
        face2nodeExtended.trans.createGhostMapping(faceGhostExt);
        faceElemInfoExtended.trans.BorrowGGIndexing(face2nodeExtended.trans);
        if (isPeriodic)
            face2nodePbiExtended.trans.BorrowGGIndexing(face2nodeExtended.trans);
        face2nodeExtended.trans.createMPITypes();
        faceElemInfoExtended.trans.createMPITypes();
        if (isPeriodic)
            face2nodePbiExtended.trans.createMPITypes();
        face2nodeExtended.trans.pullOnce();
        faceElemInfoExtended.trans.pullOnce();
        if (isPeriodic)
            face2nodePbiExtended.trans.pullOnce();

        // std::cout << fmt::format("rank {} faceElemInfo {} {}, face2node {} {}",
        //                          mpi.rank,
        //                          faceElemInfo.father->Size(), faceElemInfo.son->Size(),
        //                          face2node.father->Size(), face2node.son->Size())
        //           << std::endl;
        // std::cout << fmt::format("rank {} faceElemInfoExt {} {}, face2nodeExt {} {}",
        //                          mpi.rank,
        //                          faceElemInfoExtended.father->Size(), faceElemInfoExtended.son->Size(),
        //                          face2nodeExtended.father->Size(), face2nodeExtended.son->Size())
        //           << std::endl;

        this->AdjGlobal2LocalFacial();
        this->AdjGlobal2LocalC2F();
        //** a direct copy from AdjGlobal2LocalPrimary()
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

        // std::cout << fmt::format("rank {} Sizes ext : extFaceProc: {}->{}", mpi.rank, faceElemInfo.Size(), faceElemInfoExtended.Size()) << std::endl;
        // if (mpi.rank == 1)
        //     for (index iFace = 0; iFace < face2nodeExtended.Size(); iFace++)
        //     {
        //         std::cout << fmt::format("rank {}, face {}", mpi.rank, face2nodeExtended.trans.pLGhostMapping->operator()(-1, iFace)) << std::endl;
        //     }

        // this->AdjGlobal2LocalPrimary();

        for (index iFace = face2nodeExtended.father->Size(); iFace < face2nodeExtended.Size(); iFace++)
        {
            for (auto &iN : face2nodeExtended[iFace])
            {
                DNDS_assert_info(iN >= 0, fmt::format("rank {}, iN {}", mpi.rank, iN)); // can't be unfound node
                NodeIndexGlobal2Local(iN);
                DNDS_assert_info(iN >= 0, fmt::format("rank {}, iN {}", mpi.rank, iN)); // can't be unfound node
            }
        }
        // std::cout << mpi.rank << " rank Here XXXX 1" << std::endl;
        /***********************************/
        // build nodeNormClusters
        using t3VecsPair = ArrayPair<ArrayEigenUniMatrixBatch<3, 1>>;
        t3VecsPair nodeNormClusters;
        DNDS_MAKE_SSP(nodeNormClusters.father, mpi);
        nodeNormClusters.father->Resize(coords.father->Size(), 3, 1);
        DNDS_MAKE_SSP(nodeNormClusters.son, mpi);

        std::vector<int> nodeBndNum(coords.father->Size(), 0); //? need row-appending methods for NonUniform arrays?
        for (index iFace = 0; iFace < face2nodeExtended.Size(); iFace++)
        {
            auto faceBndID = faceElemInfoExtended(iFace, 0).zone;
            if (!FiFBndIdNeedSmooth(faceBndID))
                continue;
            for (auto iNode : face2nodeExtended[iFace])
                if (iNode < coords.father->Size() && iNode < nNodeO1) //* being local node and O1 node
                    nodeBndNum.at(iNode)++;
        }
        // if (mpi.rank == 0)
        // {
        //     // std::cout << "faceBndID: \n";
        //     // for (index iFace = 0; iFace < face2nodeExtended.Size(); iFace++)
        //     // {
        //     //     auto faceBndID = faceElemInfoExtended(iFace, 0).zone;
        //     //     std::cout << faceBndID << ", " << FiFBndIdNeedSmooth(faceBndID) << "; "
        //     //               << coords[face2node(iFace, 0)].transpose()
        //     //               << " ||| " << coords[face2node(iFace, 1)].transpose() << std::endl;
        //     // }
        //     // std::cout << "XXXXXXXXXXXX" << std::endl;
        //     std::cout << "nodeBndNum: ";
        //     for (auto i : nodeBndNum)
        //         std::cout
        //             << i << ";";
        //     std::cout << "XXXXXXXXXXXX" << std::endl;
        // }

        // std::cout << mpi.rank << " rank Here XXXX 2" << std::endl;
        for (index iNode = 0; iNode < coords.father->Size(); iNode++)
            nodeNormClusters.father->ResizeBatch(iNode, std::max(nodeBndNum.at(iNode), 0));
        for (auto &v : nodeBndNum)
            v = 0; // set to zero
        // std::cout << mpi.rank << " rank Here XXXX 3" << std::endl;
        auto GetCoordsOnFaceExtended = [&](index iFace, tSmallCoords &cs)
        {
            if (!isPeriodic)
                __GetCoords(face2nodeExtended[iFace], cs);
            else
                __GetCoordsOnElem(face2nodeExtended[iFace], face2nodePbiExtended[iFace], cs);
        };

        for (index iFace = 0; iFace < face2nodeExtended.Size(); iFace++)
        {
            auto faceBndID = faceElemInfoExtended(iFace, 0).zone;
            if (!FiFBndIdNeedSmooth(faceBndID))
                continue;
            auto eFace = Elem::Element{faceElemInfoExtended(iFace, 0).getElemType()}; // O1 faces could use only one norm (not strict for Quad4)
            auto qFace = Elem::Quadrature{eFace, 1};
            real faceArea{0};
            tPoint uNorm;
            tSmallCoords coordsF;
            GetCoordsOnFaceExtended(iFace, coordsF);
            qFace.Integration(
                faceArea,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coordsF, DiNj);
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsF, DiNj);
                    tPoint np;
                    if (dim == 2)
                        np = FacialJacobianToNormVec<2>(J);
                    else
                        np = FacialJacobianToNormVec<3>(J);
                    // np.stableNormalize(); // do not normalize to preserve face area
                    uNorm = np;
                    vInc = np.norm();
                });

            for (int if2n = 0; if2n < face2nodeExtended[iFace].size(); if2n++)
            {
                index iNode = face2nodeExtended[iFace][if2n];
                if (iNode < coords.father->Size() && iNode < nNodeO1) //* being local node and O1 node
                {
                    tPoint uNormC = uNorm;
                    if (isPeriodic)
                        uNormC = periodicInfo.GetVectorBackByBits(uNorm, face2nodePbiExtended[iFace][if2n]);
                    nodeNormClusters(iNode, nodeBndNum.at(iNode)++) = uNormC;
                }
            }
        }
        nodeNormClusters.TransAttach();
        nodeNormClusters.trans.BorrowGGIndexing(coords.trans);
        nodeNormClusters.trans.createMPITypes();
        MPI_Barrier(mpi.comm);
        nodeNormClusters.trans.pullOnce();

        /***********************************/
        // do interpolation
        tPoint bboxMove;
        bboxMove.setZero();
        index nMoved{0};
        std::unordered_set<index> moveds;
        for (index iFace = 0; iFace < face2nodeExtended.Size(); iFace++)
        {
            auto faceBndID = faceElemInfoExtended(iFace, 0).zone;
            if (!FiFBndIdNeedSmooth(faceBndID))
                continue;
            auto eFace = Elem::Element{faceElemInfoExtended(iFace, 0).getElemType()}; // O1 faces could use only one norm (not strict for Quad4)
            auto qFace = Elem::Quadrature{eFace, 1};
            Elem::SummationNoOp noOp;
            tPoint uNorm;
            SmallCoordsAsVector coordsF;
            GetCoordsOnFaceExtended(iFace, coordsF);
            qFace.Integration(
                noOp,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coordsF, DiNj);
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsF, DiNj);
                    tPoint np;
                    if (dim == 2)
                        np = FacialJacobianToNormVec<2>(J);
                    else
                        np = FacialJacobianToNormVec<3>(J);
                    np.stableNormalize();
                    uNorm = np;
                });

            auto eFaceO1 = eFace.ObtainO1Elem();

            auto f2n = face2nodeExtended[iFace];

            std::vector<int> f2nVecSeq(f2n.size());
            for (int i = 0; i < f2n.size(); i++)
                f2nVecSeq[i] = i;

            for (int if2n = eFaceO1.GetNumNodes(); if2n < f2n.size(); if2n++)
            {
                index iNode = f2n[if2n];
                if (iNode >= coords.father->Size())
                    continue;
                DNDS_assert(iNode >= nNodeO1);

                std::vector<index> spanO2Node;
                std::vector<int> spanO2if2n;
                SmallCoordsAsVector coordsSpan;
                int iElev = if2n - eFaceO1.GetNumNodes();
                int spanNnode = eFaceO1.ObtainElevNodeSpan(iElev).GetNumNodes();
                spanO2Node.resize(spanNnode);
                coordsSpan.resize(3, spanNnode);
                spanO2if2n.resize(spanNnode);

                eFaceO1.ExtractElevNodeSpanNodes(iElev, f2n, spanO2Node); // should use O1f2n, but O2f2n is equivalent
                eFaceO1.ExtractElevNodeSpanNodes(iElev, coordsF, coordsSpan);
                eFaceO1.ExtractElevNodeSpanNodes(iElev, f2nVecSeq, spanO2if2n);
                tPoint cooUnsmooth = coordsF(Eigen::all, if2n);

                std::vector<tPoint> edges;
                if (spanO2Node.size() == 2)
                {
                    edges.push_back((coordsSpan[1] - coordsSpan[0]).stableNormalized());
                }
                else if (spanO2Node.size() == 4)
                {
                    edges.push_back((coordsSpan[1] - coordsSpan[0]).stableNormalized());
                    edges.push_back((coordsSpan[2] - coordsSpan[1]).stableNormalized());
                    edges.push_back((coordsSpan[3] - coordsSpan[2]).stableNormalized());
                    edges.push_back((coordsSpan[0] - coordsSpan[3]).stableNormalized());
                }
                else
                    DNDS_assert(false);

                Eigen::Matrix<real, 3, 4> norms;
                Eigen::Vector<real, 4> nAdd;
                nAdd.setZero();
                norms.setZero();
                // if (mpi.rank == 1)
                //     std::cout << nodeNormClusters.RowSize(spanO2Node[0]) << " nodeNormClustersRowSize" << std::endl;

                for (int iN = 0; iN < spanO2Node.size(); iN++)
                {
                    real sinValFFMax{0};
                    for (int iNorm = 0; iNorm < nodeNormClusters.RowSize(spanO2Node[iN]); iNorm++)
                        for (int jNorm = 0; jNorm < nodeNormClusters.RowSize(spanO2Node[iN]); jNorm++)
                            sinValFFMax = std::max(
                                sinValFFMax,
                                std::sqrt(1 - sqr(std::abs(nodeNormClusters(spanO2Node[iN], iNorm)
                                                               .stableNormalized()
                                                               .dot(nodeNormClusters(spanO2Node[iN], jNorm).stableNormalized())))));
                    for (int iNorm = 0; iNorm < nodeNormClusters.RowSize(spanO2Node[iN]); iNorm++)
                    {
                        tPoint normN = nodeNormClusters(spanO2Node[iN], iNorm);
                        if (isPeriodic)
                            normN = periodicInfo.GetVectorByBits(normN, face2nodePbiExtended[iFace][if2n]);
                        real sinValMax = 0;
                        for (auto e : edges)
                            sinValMax = std::max(sinValMax, std::abs(e.dot(normN.stableNormalized())));
                        //! angle is here
                        //* now using FFMaxAngle also
                        if (std::max(sinValMax, sinValFFMax) > std::sin(pi / 180. * elevationInfo.MaxIncludedAngle))
                            continue;
                        nAdd[iN] += 1.;
                        // norms(Eigen::all, iN) += normN * 1.;
                        norms(Eigen::all, iN) += normN.stableNormalized();

                        // std::cout << fmt::format("iFace {}, if2n {}, iN {}, iNorm{}, sinValMax {}; ====",
                        // iFace, if2n, iN, iNorm, sinValMax) << normN.transpose() << std::endl;
                    }
                }
                // norms.array().rowwise() /= nAdd.transpose().array();
                norms.colwise().normalize();
                for (int iN = 0; iN < spanO2Node.size(); iN++)
                {
                    if (nAdd[iN] < 0.1) // no found, then no mov
                        norms(Eigen::all, iN).setZero();
                    // DNDS_assert(nAdd[iN] > 0.1);
                }

                tPoint cooSmooth;
                tPoint cooInc;
                if (spanO2Node.size() == 2)
                {
                    cooSmooth = HermiteInterpolateMidPointOnLine2WithNorm(
                        coordsSpan[0], coordsSpan[1], norms(Eigen::all, 0), norms(Eigen::all, 1));
                    cooInc = cooSmooth - cooUnsmooth;
                }
                else // definitely is spanO2Node.size() == 4
                {
                    f2n[if2n - 1];
                    DNDS_assert(f2n.size() == 9); // has to be a Quad9
                    cooInc = 0.5 * (coordsElevDisp[f2n[if2n - 4]] +
                                    coordsElevDisp[f2n[if2n - 3]] +
                                    coordsElevDisp[f2n[if2n - 2]] +
                                    coordsElevDisp[f2n[if2n - 1]]);
                    if (isPeriodic)
                    {
                        cooInc =
                            0.5 *
                            (periodicInfo.GetVectorByBits<3, 1>(coordsElevDisp[f2n[if2n - 4]], face2nodePbiExtended[iFace][if2n - 4]) +
                             periodicInfo.GetVectorByBits<3, 1>(coordsElevDisp[f2n[if2n - 3]], face2nodePbiExtended[iFace][if2n - 3]) +
                             periodicInfo.GetVectorByBits<3, 1>(coordsElevDisp[f2n[if2n - 2]], face2nodePbiExtended[iFace][if2n - 2]) +
                             periodicInfo.GetVectorByBits<3, 1>(coordsElevDisp[f2n[if2n - 1]], face2nodePbiExtended[iFace][if2n - 1]));
                    }

                    cooSmooth = HermiteInterpolateMidPointOnQuad4WithNorm(
                        coordsSpan[0], coordsSpan[1], coordsSpan[2], coordsSpan[3],
                        norms(Eigen::all, 0), norms(Eigen::all, 1), norms(Eigen::all, 2), norms(Eigen::all, 3));

                    cooInc = cooSmooth - cooUnsmooth;
                }

                if (isPeriodic)
                    cooInc = periodicInfo.GetVectorBackByBits(cooInc, face2nodePbiExtended[iFace][if2n]);
                // if (cooInc.stableNorm() > 0) //! could use a threshold
                {

                    coordsElevDisp[iNode] = cooInc * 1; // maybe output the increment directly?
                    nMoved++;
                    moveds.insert(iNode);
                    bboxMove = bboxMove.array().max(cooInc.array().abs());
                }

                // std::cout << mpi.rank << " rank; iNode " << iNode << " alterwith " << coordsElevDisp[iNode].transpose() << std::endl;
            }
        }
        tPoint bboxMoveT;
        MPI::Allreduce(bboxMove.data(), bboxMoveT.data(), 3, DNDS_MPI_REAL, MPI_MAX, mpi.comm);

        coordsElevDisp.trans.pullOnce();
        nMoved = moveds.size();
        MPI::Allreduce(&nMoved, &nTotalMoved, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        if (mpi.rank == mRank)
            log() << fmt::format("UnstructuredMesh === ElevatedNodesGetBoundarySmooth: Smoothing Complete, total Moved [{}], moving Vec Bnd {},{},{}",
                                 nTotalMoved, bboxMoveT(0), bboxMoveT(1), bboxMoveT(2))
                  << std::endl;

        // std::cout << mpi.rank << " rank Here XXXX -1" << std::endl;
    }

    struct CoordPairDOF : public tCoordPair
    {
        real dot(CoordPairDOF &R)
        {
            real ret = 0;
            for (index i = 0; i < this->father->Size(); i++)
                ret += (*this)[i].dot(R[i]);
            real retSum;
            MPI::Allreduce(&ret, &retSum, 1, DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            return retSum;
        }

        real norm2()
        {
            return std::sqrt(this->dot(*this));
        }

        void addTo(CoordPairDOF &R, real alpha)
        {
            for (index i = 0; i < this->Size(); i++)
                (*this)[i] += R[i] * alpha;
        }

        void operator=(CoordPairDOF &R)
        {
            for (index i = 0; i < this->Size(); i++)
                (*this)[i] = R[i];
        }

        void operator*=(real r)
        {
            for (index i = 0; i < this->Size(); i++)
                (*this)[i] *= r;
        }
    };

    void UnstructuredMesh::ElevatedNodesSolveInternalSmooth()
    {
        DNDS_assert(elevState == Elevation_O1O2);
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        DNDS_assert(face2node.father);
        DNDS_assert(nTotalMoved >= 0);
        if (!nTotalMoved)
        {
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth() early exit for no nodes were moved";
            return;
        }

        std::vector<std::unordered_set<index>> node2nodeV;
        node2nodeV.resize(coords.father->Size());
        for (index iCell = 0; iCell < cell2node.Size(); iCell++)
        {
            for (auto iN : cell2node[iCell])
                for (auto iNOther : cell2node[iCell])
                {
                    if (iN == iNOther)
                        continue;
                    if (iN < coords.father->Size())
                    {
                        node2nodeV[iN].insert(iNOther);
                    }
                }
        }

        DNDS_MPI_InsertCheck(mpi, "Got node2node");

        coordsElevDisp.trans.initPersistentPull(); // only holds local nodes

        int nSplit = 5;
        for (index iN = nNodeO1; iN < coords.father->Size(); iN++)
            if (coordsElevDisp[iN](0) != largeReal)
                coordsElevDisp[iN] /= real(nSplit);
        coordsElevDisp.trans.startPersistentPull();
        coordsElevDisp.trans.waitPersistentPull();

        std::unordered_set<index> nodesBoundInterpolated;
        for (index iN = nNodeO1; iN < coords.father->Size(); iN++)
        {
            if (coordsElevDisp[iN](0) != largeReal)
            {
                nodesBoundInterpolated.insert(iN);
            }
        }

        for (int iSplit = 1; iSplit < nSplit; iSplit++)
        {
            std::unordered_set<index> nodesLeft, ndoesLeftNew;
            for (index iN = nNodeO1; iN < coords.father->Size(); iN++)
                nodesLeft.insert(iN);
            std::unordered_set<index> currentFront;
            for (auto iN : nodesBoundInterpolated)
                nodesLeft.erase(iN);
            for (auto iN : nodesLeft)
                coordsElevDisp[iN].setConstant(largeReal);

            // for (index iN = nNodeO1; iN < coords.father->Size(); iN++)
            // {
            //     if (coordsElevDisp[iN](0) != largeReal)
            //     {
            //         nodesLeft.erase(iN);
            //     }
            // }
            index nodesLeftSz = nodesLeft.size();
            index nodesLeftSzAll;
            MPI::Allreduce(&nodesLeftSz, &nodesLeftSzAll, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            if (mpi.rank == 0)
                std::cout << "nodesLeftSzAll at init " << nodesLeftSzAll << std::endl;

            for (index iN = 0; iN < nNodeO1; iN++)
                coordsElevDisp[iN](1) = 3 * largeReal; // marks O1 nodes
            coordsElevDisp.trans.startPersistentPull();
            coordsElevDisp.trans.waitPersistentPull();
            index sumMov{0};
            index sumLoc{0};
            for (int iIter = 1; iIter <= elevationInfo.nIter; iIter++)
            {
                currentFront.clear();
                for (index iN : nodesLeft)
                {
                    int nOtherSmoothCount{0};
                    for (auto iNOther : node2nodeV[iN])
                        if (coordsElevDisp[iNOther](0) != largeReal)
                            nOtherSmoothCount++;
                    if (nOtherSmoothCount)
                    {
                        currentFront.insert(iN);
                        coordsElevDisp[iN](1) = 2 * largeReal; // marks front nodes
                    }
                }
                coordsElevDisp.trans.startPersistentPull();
                coordsElevDisp.trans.waitPersistentPull();
                // interpolation
                for (index iN : currentFront)
                {
                    auto &n2nSet = node2nodeV[iN];
                    std::vector<index> n2nVec, n2nVecKnown;
                    n2nVec.reserve(n2nSet.size());
                    for (auto i : n2nSet)
                        n2nVec.push_back(i);
                    for (auto i : n2nSet)
                        if (coordsElevDisp[i](1) == 3 * largeReal ||
                            (coordsElevDisp[i](0) != largeReal && coordsElevDisp[i](1) != 2 * largeReal))
                            // if ((coordsElevDisp[i](0) != largeReal && currentFront.count(i) == 0))
                            n2nVecKnown.push_back(i);
                    tSmallCoords dispsNKnown;
                    dispsNKnown.resize(3, n2nVecKnown.size());
                    for (int in2n = 0; in2n < n2nVecKnown.size(); in2n++)
                        if (coordsElevDisp[n2nVecKnown[in2n]](1) == 3 * largeReal)
                            dispsNKnown(Eigen::all, in2n).setZero();
                        else
                            dispsNKnown(Eigen::all, in2n) = coordsElevDisp[n2nVecKnown[in2n]];

                    tSmallCoords coordsN, coordsNKnown;
                    __GetCoords(n2nVec, coordsN); // !using original coords, disregarding the periodic situation!
                    __GetCoords(n2nVecKnown, coordsNKnown);
                    tPoint coordsNC = coordsN.rowwise().mean();
                    tSmallCoords coordsNKnownRel = coordsNKnown.colwise() - coordsNC;

                    real rijMax = RBF::GetMaxRij(coordsNKnownRel, coordsNKnownRel);
                    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>
                        coefs = RBF::RBFInterpolateSolveCoefsNoPoly(coordsNKnownRel, dispsNKnown.transpose(), elevationInfo.RBFRadius * rijMax, elevationInfo.kernel); //! R here

                    tSmallCoords qs;
                    qs.resize(3, 1);
                    qs = coords[iN] - coordsNC;

                    tPoint dCur = (RBF::RBFCPC2(coordsNKnownRel, qs, elevationInfo.RBFRadius * rijMax, elevationInfo.kernel).transpose() * coefs.topRows(coordsNKnownRel.cols()) +
                                   coefs(Eigen::last - 3, Eigen::all) +
                                   qs.transpose() * coefs.bottomRows(3))
                                      .transpose();
                    if (mpi.rank == 0)
                    {
                        // std::cout << coordsElevDisp[n2nVecKnown[4]].transpose() << std::endl;
                        // std::cout << dCur.transpose() << std::endl;
                        // std::cout << "=== XNode\n";
                        // std::cout << coordsNKnownRel << "\n";
                        // std::cout << "=== vNode\n"
                        //           << std::endl;
                        // std::cout << dispsNKnown << std::endl;
                        // std::cout << "qs " << qs.transpose() << std::endl;
                        // std::abort();
                    }
                    real maxDispOrig = dispsNKnown.colwise().norm().array().maxCoeff() * 0.9999;
                    if (dCur.norm() > maxDispOrig)
                        dCur = dCur.normalized() * maxDispOrig;
                    coordsElevDisp[iN] = dCur;
                    nodesLeft.erase(iN);
                }
                index frtSz = currentFront.size();
                index frtSzTot{0};
                MPI::Allreduce(&frtSz, &frtSzTot, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
                sumMov += frtSzTot, sumLoc += frtSzTot;
                if (frtSzTot == 0)
                {
                    if (mpi.rank == mRank)
                    {
                        log() << fmt::format("=== Iter {}, moved zero Nodes, stopping", iIter)
                              << std::endl;
                    }
                    break;
                }
                if (iIter % 10 == 0)
                    if (mpi.rank == mRank)
                    {
                        log() << fmt::format("=== Iter {}, moved numNodes {}", iIter, sumLoc)
                              << std::endl;
                        sumLoc = 0;
                    }
                coordsElevDisp.trans.startPersistentPull();
                coordsElevDisp.trans.waitPersistentPull();
            }

            if (mpi.rank == mRank)
                log() << fmt::format("UnstructuredMesh === ElevatedNodesSolveInternalSmooth() Step {} Iter Done Total {}", iSplit, sumMov)
                      << std::endl;

            auto determineCellBadJNum = [&](index iCell) -> int
            {
                auto eCell = GetCellElement(iCell);
                auto qCellMax = Elem::Quadrature{GetCellElement(iCell), 3};
                auto qCellO1 = Elem::Quadrature{GetCellElement(iCell), 1};

                tSmallCoords coordsCell;
                GetCoordsOnCell(iCell, coordsCell);
                tSmallCoords coordsCellNew = coordsCell;
                for (int ic2n = 0; ic2n < cell2node[iCell].size(); ic2n++)
                    if (coordsElevDisp[cell2node[iCell][ic2n]](0) != largeReal) // has disp
                        coordsCellNew(Eigen::all, ic2n) += coordsElevDisp[cell2node[iCell][ic2n]];
                real vol{0};
                qCellO1.Integration(
                    vol,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        tJacobi J = Elem::ShapeJacobianCoordD01Nj(coordsCell, DiNj);
                        real JDet;
                        if (dim == 2)
                            JDet = J(Eigen::all, 0).cross(J(Eigen::all, 1)).stableNorm();
                        else
                            JDet = J.determinant();
                        vInc = JDet;
                    });

                Elem::SummationNoOp noOp;
                int nBadJ{0};
                qCellMax.Integration(
                    noOp,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        tJacobi J = Elem::ShapeJacobianCoordD01Nj(coordsCellNew, DiNj);
                        real JDet;
                        if (dim == 2)
                            JDet = J(Eigen::all, 0).cross(J(Eigen::all, 1)).stableNorm();
                        else
                            JDet = J.determinant();

                        tJacobi J_orig = Elem::ShapeJacobianCoordD01Nj(coordsCell, DiNj);
                        real JDet_orig;
                        if (dim == 2)
                            JDet_orig = J_orig(Eigen::all, 0).cross(J(Eigen::all, 1)).stableNorm();
                        else
                            JDet_orig = J_orig.determinant();
                        DNDS_assert(JDet_orig > 0);
                        // DNDS_assert(JDet > 0);

                        // if (JDet * Geom::Elem::ParamSpaceVol(eCell.GetParamSpace()) < 1e-2 * vol)
                        if (JDet <= 0)
                            nBadJ++;
                    });
                return nBadJ;
            };

            // for (int iIter = 1; iIter <= elevationInfo.nIter; iIter++)
            // {
            //     std::unordered_set<index> resetNodes;
            //     for (index iCell = 0; iCell < cell2node.father->Size(); iCell++)
            //     {
            //         int nBadJ = determineCellBadJNum(iCell);
            //         if (nBadJ)
            //             for (auto iN : cell2node[iCell])
            //                 if (coordsElevDisp[iN](0) != largeReal)
            //                     resetNodes.insert(iN);
            //     }
            //     index nResetNode = resetNodes.size();
            //     MPI::AllreduceOneIndex(nResetNode, MPI_SUM, mpi);
            //     if (nResetNode == 0)
            //     {
            //         if (mpi.rank == mRank)
            //             log() << fmt::format("=== Fixing Iter {}, nodes reset is zero, stopping", iIter);
            //         break;
            //     }
            //     for (index iN : resetNodes)
            //     {
            //         coordsElevDisp[iN](0) = largeReal;     // won't be used
            //         coordsElevDisp[iN](1) = 4 * largeReal; // mark as reset
            //     }
            //     if (mpi.rank == mRank)
            //     {
            //         log() << fmt::format("=== Fixing Iter {}, nodes reset {}", iIter, nResetNode)
            //               << std::endl;
            //     }
            //     coordsElevDisp.trans.startPersistentPull();
            //     coordsElevDisp.trans.waitPersistentPull();
            // }

            for (index iN = nNodeO1; iN < coords.father->Size(); iN++)
            {
                // if(dim == 2)
                //     dispO2[iN](2) = 0;
                // if(dispO2[iN].norm() != 0)
                //     std::cout << dispO2[iN].transpose() << std::endl;
                if (coordsElevDisp[iN](0) != largeReal)
                    coords[iN] += coordsElevDisp[iN];
            }
            coords.trans.pullOnce();
        }
    }
}