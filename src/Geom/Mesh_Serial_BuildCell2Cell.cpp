#include "Mesh.hpp"

#include <set>

#include <nanoflann.hpp>

#include "PointCloud.hpp"
#include <unordered_map>

namespace DNDS::Geom
{
    void UnstructuredMeshSerialRW::
        BuildCell2Cell()
    {
        if (mesh->mpi.rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  BuildCell2Cell" << std::endl;
        DNDS_MAKE_SSP(cell2cellSerial, mesh->mpi);
        // if (mRank != mesh->mpi.rank)
        //     return;
        /// TODO: abstract these: invert cone (like node 2 cell -> cell 2 node) (also support operating on pair)
        /**********************************************************************************************************************/
        // getting node2cell // only primary vertices
        std::vector<std::vector<DNDS::index>>
            node2cell(coordSerial->Size());
        std::vector<DNDS::rowsize> node2cellSz(coordSerial->Size(), 0);
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            for (DNDS::rowsize iN = 0; iN < Elem::Element{(*cellElemInfoSerial)(iCell, 0).getElemType()}.GetNumVertices(); iN++)
                node2cellSz[(*cell2nodeSerial)(iCell, iN)]++;
        for (DNDS::index iNode = 0; iNode < node2cell.size(); iNode++)
            node2cell[iNode].reserve(node2cellSz[iNode]);
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            for (DNDS::rowsize iN = 0; iN < Elem::Element{(*cellElemInfoSerial)(iCell, 0).getElemType()}.GetNumVertices(); iN++)
                node2cell[(*cell2nodeSerial)(iCell, iN)].push_back(iCell);
        node2cellSz.clear();
        /**********************************************************************************************************************/
        if (mesh->mpi.rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  BuildCell2Cell Part 1" << std::endl;
        cell2cellSerial->Resize(cell2nodeSerial->Size());

        index nCells = cell2nodeSerial->Size();
        index nCellsDone = 0;
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
        {
            auto CellElem = Elem::Element{(*cellElemInfoSerial)[iCell]->getElemType()};
            std::vector<DNDS::index> cell2nodeRow{
                (*cell2nodeSerial)[iCell].begin(),
                (*cell2nodeSerial)[iCell].begin() + CellElem.GetNumVertices()};
            std::sort(cell2nodeRow.begin(), cell2nodeRow.end());
            // only primary vertices
            std::set<DNDS::index> c_neighbors; // could optimize?
                                               // std::vector<DNDS::index> c_neighbors;
                                               // c_neighbors.reserve(30);
                                               /****/
#ifdef DNDS_USE_OMP
            // #pragma omp single
#pragma omp critical
#endif
            {
                if (nCellsDone % (nCells / 1000 + 1) == 0)
                {
                    auto fmt = DNDS::log().flags();
                    DNDS::log() << "\r\033[K" << std::setw(5) << double(nCellsDone) / nCells * 100 << "%";
                    DNDS::log().flush();
                    DNDS::log().setf(fmt);
                }

                if (nCellsDone == nCells - 1)
                    DNDS::log()
                        // << "\r\033[K"
                        << std::endl;
                        }
            /****/

            for (auto iNode : cell2nodeRow)
                for (auto iCellOther : node2cell[iNode])
                {
                    if (iCellOther == iCell || c_neighbors.count(iCellOther) == 1)
                        continue;
                    //! ** override: point-complete cell2cell info!
                    c_neighbors.insert(iCellOther);
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
                        c_neighbors.insert(iCellOther);
                    // c_neighbors.push_back(iCellOther);
                }
            // if (!c_neighbors.erase(iCell))
            //     DNDS_assert_info(false, "neighbors should include myself now");
            // std::sort(c_neighbors.begin(), c_neighbors.end());
            // auto last = std::unique(c_neighbors.begin(), c_neighbors.end());
            // c_neighbors.erase(last, c_neighbors.end());

            cell2cellSerial->ResizeRow(iCell, c_neighbors.size());
            DNDS::rowsize ic2c = 0;
            for (auto iCellOther : c_neighbors)
                (*cell2cellSerial)(iCell, ic2c++) = iCellOther;
#ifdef DNDS_USE_OMP
#pragma omp atomic
#endif
            nCellsDone++;
        }

        /*************************************************************************************************/
        DNDS_MAKE_SSP(cell2cellSerialFacial, mesh->mpi);
        if (mesh->mpi.rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  BuildCell2Cell Part 2" << std::endl;
        cell2cellSerialFacial->Resize(cell2cellSerial->Size());
        nCellsDone = 0;
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (index iCell = 0; iCell < cell2cellSerial->Size(); iCell++)
        {
#ifdef DNDS_USE_OMP
#pragma omp critical
#endif
            {
                if (nCellsDone % (nCells / 1000 + 1) == 0)
                {
                    auto fmt = DNDS::log().flags();
                    DNDS::log() << "\r\033[K" << std::setw(5) << double(nCellsDone) / nCells * 100 << "%";
                    DNDS::log().flush();
                    DNDS::log().setf(fmt);
                }

                if (nCellsDone == nCells - 1)
                    DNDS::log()
                        // << "\r\033[K"
                        << std::endl;
            }
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
#ifdef DNDS_USE_OMP
#pragma omp atomic
#endif
            nCellsDone++;
        }
        /*************************************************************************************************/

        // TODO: build periodic donor oct-trees
        // TODO: search in donors if periodic main and connect cell2cell
        /*
        PointCloudKDTree periodicDonorCenter1;
        PointCloudKDTree periodicDonorCenter2;
        PointCloudKDTree periodicDonorCenter3;
        std::vector<index> periodicDonorIBnd1;
        std::vector<index> periodicDonorIBnd2;
        std::vector<index> periodicDonorIBnd3;

        for (DNDS::index iBnd = 0; iBnd < bnd2cellSerial->Size(); iBnd++)
        {
            auto faceID = bndElemInfoSerial->operator()(iBnd, 0).zone;
            Eigen::Matrix<real, 3, Eigen::Dynamic> coordBnd;
            coordBnd.resize(3, bnd2nodeSerial->RowSize(iBnd));
            for (rowsize ib2n = 0; ib2n < bnd2nodeSerial->RowSize(iBnd); ib2n++)
                coordBnd(Eigen::all, ib2n) = coordSerial->operator[]((*bnd2nodeSerial)(iBnd, ib2n));
            tPoint faceCent = coordBnd.rowwise().maxCoeff();

            if (FaceIDIsPeriodicMain(faceID))
            {
            }
            if (FaceIDIsPeriodicDonor(faceID))
            {
                if (faceID == BC_ID_PERIODIC_1_DONOR)
                    periodicDonorCenter1.pts.push_back(faceCent), periodicDonorIBnd1.push_back(iBnd);
                if (faceID == BC_ID_PERIODIC_2_DONOR)
                    periodicDonorCenter2.pts.push_back(faceCent), periodicDonorIBnd2.push_back(iBnd);
                if (faceID == BC_ID_PERIODIC_3_DONOR)
                    periodicDonorCenter3.pts.push_back(faceCent), periodicDonorIBnd3.push_back(iBnd);
            }
        }
        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTree>,
            PointCloudKDTree,
            3,
            index>;
        kdtree_t periodicDonorTree1(3, periodicDonorCenter1);
        kdtree_t periodicDonorTree2(3, periodicDonorCenter2);
        kdtree_t periodicDonorTree3(3, periodicDonorCenter3);

        std::unordered_map<index, index> periodicMainToDonor1;
        std::unordered_map<index, index> periodicMainToDonor2;
        std::unordered_map<index, index> periodicMainToDonor3;
        for (DNDS::index iBnd = 0; iBnd < bnd2cellSerial->Size(); iBnd++)
        {
            auto faceID = bndElemInfoSerial->operator()(iBnd, 0).zone;
            Eigen::Matrix<real, 3, Eigen::Dynamic> coordBnd;
            coordBnd.resize(3, bnd2nodeSerial->RowSize(iBnd));
            for (rowsize ib2n = 0; ib2n < bnd2nodeSerial->RowSize(iBnd); ib2n++)
                coordBnd(Eigen::all, ib2n) = coordSerial->operator[]((*bnd2nodeSerial)(iBnd, ib2n));
            tPoint faceCent = coordBnd.rowwise().maxCoeff();

            if (FaceIDIsPeriodicMain(faceID))
            {
                tPoint faceCentTrans = mesh->periodicInfo.TransCoord(faceCent, faceID);
                std::vector<index> outIndices(1);
                Eigen::Vector<real, Eigen::Dynamic> outDistancesSqr;
                outDistancesSqr.resize(1);
                size_t nResults{0};
                if (faceID == BC_ID_PERIODIC_1)
                    nResults = periodicDonorTree1.knnSearch(faceCentTrans.data(), 1, outIndices.data(), outDistancesSqr.data());
                if (faceID == BC_ID_PERIODIC_2)
                    nResults = periodicDonorTree2.knnSearch(faceCentTrans.data(), 1, outIndices.data(), outDistancesSqr.data());
                if (faceID == BC_ID_PERIODIC_3)
                    nResults = periodicDonorTree3.knnSearch(faceCentTrans.data(), 1, outIndices.data(), outDistancesSqr.data());
                DNDS_assert_info(nResults == 1, "Tree Search of periodic result number not 1");
                real search_eps = 1e-8;
                DNDS_assert_info(outDistancesSqr[0] < sqr(search_eps), "nearest neighbour not matching under the threshold");
                index donorIBnd{-1};
                if (faceID == BC_ID_PERIODIC_1)
                    donorIBnd = periodicDonorIBnd1.at(outIndices[0]), periodicMainToDonor1[iBnd] = donorIBnd;
                if (faceID == BC_ID_PERIODIC_2)
                    donorIBnd = periodicDonorIBnd2.at(outIndices[0]), periodicMainToDonor2[iBnd] = donorIBnd;
                if (faceID == BC_ID_PERIODIC_3)
                    donorIBnd = periodicDonorIBnd3.at(outIndices[0]), periodicMainToDonor3[iBnd] = donorIBnd;
                index iCellA = (*bnd2cellSerial)[iBnd][0];
                index iCellB = (*bnd2cellSerial)[donorIBnd][0];
                std::vector<index> cellA2cells = (*cell2cellSerial)[iCellA];
                std::vector<index> cellB2cells = (*cell2cellSerial)[iCellB];
                cellA2cells.push_back(iCellB);
                cellB2cells.push_back(iCellA);
                (*cell2cellSerial).ResizeRow(iCellA, cellA2cells.size());
                (*cell2cellSerial)[iCellA] = cellA2cells;
                (*cell2cellSerial).ResizeRow(iCellB, cellB2cells.size());
                (*cell2cellSerial)[iCellB] = cellB2cells; //! todo: bad here, should do some node altering
            }
            if (FaceIDIsPeriodicDonor(faceID))
            {
            }
        }
        */

        if (mesh->mpi.rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Done  BuildCell2Cell" << std::endl;
    }
}