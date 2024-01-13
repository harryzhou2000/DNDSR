#include "Mesh.hpp"

#include <omp.h>
#include <fmt/core.h>

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
            SmallCoordsAsVector coords;
            meshO1.GetCoordsOnCell(iCell, coords);

            for (index iNNode = 0; iNNode < eCell.GetNumElev_O1O2(); iNNode++)
            {
                auto eSpan = eCell.ObtainElevNodeSpan(iNNode);
                std::array<index, Elem::CellNumNodeMax> spanNodes;
                eCell.ExtractElevNodeSpanNodes(iNNode, c2n, spanNodes);
                auto spanNodesSrt = spanNodes;
                std::sort(spanNodesSrt.begin(), spanNodesSrt.begin() + eSpan.GetNumNodes());
                tPoint newNodeCoord

                { // the new node method, should cope with shape func
                    SmallCoordsAsVector coordsSpan;
                    coordsSpan.resize(3, eSpan.GetNumNodes());
                    eCell.ExtractElevNodeSpanNodes(iNNode, coords, coordsSpan);
                    newNodeCoord = coordsSpan.rowwise().mean();
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
                    if (iCellGlobOther < iCellGlob &&
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
        }
        // std::cout << fmt::format("Num NewNode {} ", localNewNodeNum) << std::endl;
        // for (auto v : newCoords)
        //     std::cout << v.transpose() << std::endl;

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
                    iNodesNew < 0
                        ? iNodesNew // negative field was filled with indicating which cell to seek the node
                        : coords.father->pLGlobalMapping->operator()(mpi.rank, iNodeNew + meshO1.coords.father->Size());
            }
        }
        cellNewNodes.TransAttach();
        cellNewNodes.trans.BorrowGGIndexing(mesh->cell2node.trans);
        cellNewNodes.trans.createMPITypes();
        cellNewNodes.trans.pullOnce();

        /**********************************/
        // each cell obtain new global cell2node global state with cellNewNodesGlobalPair
        DNDS_MAKE_SSP(cell2node.father, mpi);
        DNDS_MAKE_SSP(cell2node.son, mpi);
        DNDS_MAKE_SSP(cellElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        DNDS_MAKE_SSP(cellElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);

        cell2node.father->Resize(meshO1.cell2node.father->Size());
        for (index iCell = 0; iCell < meshO1.cell2node.father->Size(); iCell++)
        {
            Elem::Element eCell = meshO1.GetCellElement(iCell);
            Elem::Element eCellO2 = eCell.ObtainElevatedElem();
            cell2node.father->ResizeRow(iCell, eCellO2.GetNumNodes());
            auto c2n = meshO1.cell2node[iCell];
            for (int ic2n = 0; ic2n < c2n.size(); ic2n++)
            {
                // fill in the O1 nodes
                index iNodeOldGlobal = meshO1.coords.trans.pLGhostMapping->operator()(-1, c2n[ic2n]);
                index nodeOldOrigLocalIdx{-1};
                int nodeOldOrigRank{-1};
                if (!meshO1.coords.trans.pLGlobalMapping->search(iNodeOldGlobal, nodeOldOrigRank, nodeOldOrigLocalIdx))
                    DNDS_assert_info(false, "search failed");
                // nodeOldOrigRank and nodeOldOrigLocalIdx is same in new
                cell2node(iCell, ic2n) =
                    coords.father->pLGlobalMapping->operator()(nodeOldOrigRank, nodeOldOrigLocalIdx); // now point to global
            }
        }
        for (index iCell = 0; iCell < meshO1.cell2node.father->Size(); iCell++)
        {
            Elem::Element eCell = meshO1.GetCellElement(iCell);
            auto c2n = meshO1.cell2node[iCell];
            auto c2c = meshO1.cell2cell[iCell];
            index iCellGlob = meshO1.cell2node.trans.pLGhostMapping->operator()(-1, iCell);

            // std::vector<index> cellNewNodeRowV;
            // cellNewNodeRowV.resize(eCell.GetNumElev_O1O2());
            SmallCoordsAsVector coords;
            meshO1.GetCoordsOnCell(iCell, coords);

            for (index iNNode = 0; iNNode < eCell.GetNumElev_O1O2(); iNNode++)
            {
                if (cellNewNodes[iCell][iNNode] >= 0)
                {
                    cell2node[iCell][c2c.size() + iNNode] = cellNewNodes[iCell][iNNode];
                    continue;
                }
                auto eSpan = eCell.ObtainElevNodeSpan(iNNode);
                std::array<index, Elem::CellNumNodeMax> spanNodes;
                eCell.ExtractElevNodeSpanNodes(iNNode, c2n, spanNodes);
                auto spanNodesSrt = spanNodes;
                std::sort(spanNodesSrt.begin(), spanNodesSrt.begin() + eSpan.GetNumNodes());
                tPoint newNodeCoord
                { // the new node method, should cope with shape func
                    SmallCoordsAsVector coordsSpan;
                    coordsSpan.resize(3, eSpan.GetNumNodes());
                    eCell.ExtractElevNodeSpanNodes(iNNode, coords, coordsSpan);
                    newNodeCoord = coordsSpan.rowwise().mean();
                }
                int nFound = 0;
                for (auto iNNodeC : cellNewNodes[-1 - cellNewNodes[iCell][iNNode]])
                {
                    if (iNNodeC < 0)
                        continue;
                    // ! iNNodeC is a global index, could be not local!!
                }
            }
            /**********************************/
            // cell2cell, bnd2cell can be copied; bnd2nodeO2 created with bnd2cell and bnd2node; cell2nodePbiO2: edge using edge common, face using face common
        }
    }