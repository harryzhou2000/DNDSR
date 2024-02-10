#include "Mesh.hpp"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/graph_utility.hpp"
#include "boost/graph/minimum_degree_ordering.hpp"

namespace _METIS
{
#include "metis.h"
#include "parmetis.h"

    static idx_t indexToIdx(DNDS::index v)
    {
        if constexpr (sizeof(DNDS::index) <= sizeof(idx_t))
            return v;
        else
            return DNDS::checkedIndexTo32(v);
    }
}

namespace DNDS::Geom
{
    void UnstructuredMeshSerialRW::
        MeshPartitionCell2Cell()
    {
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  MeshPartitionCell2Cell" << std::endl;
        //! preset hyper config, should be optional in the future
        bool isSerial = true;
        _METIS::idx_t nPart = mesh->getMPI().size;
        cnPart = nPart;

        //! assuming all adj point to local numbers now
        // * Tend to local-global issues putting into
        // cell2cellSerial->Compress();
        // cell2cellSerial->AssertConsistent();
        // cell2cellSerial->createGlobalMapping();
        cell2cellSerialFacial->Compress();
        cell2cellSerialFacial->AssertConsistent();
        cell2cellSerialFacial->createGlobalMapping();

        std::vector<_METIS::idx_t> vtxdist(mesh->getMPI().size + 1);
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::MPI_int r = 0; r <= mesh->getMPI().size; r++)
            vtxdist[r] = _METIS::indexToIdx(cell2cellSerialFacial->pLGlobalMapping->ROffsets().at(r));
        std::vector<_METIS::idx_t> xadj(cell2cellSerialFacial->Size() + 1);
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iCell = 0; iCell < xadj.size(); iCell++)
            xadj[iCell] = _METIS::indexToIdx(cell2cellSerialFacial->rowPtr(iCell) - cell2cellSerialFacial->rowPtr(0));
        std::vector<_METIS::idx_t> adjncy(xadj.back());
        DNDS_assert(cell2cellSerialFacial->DataSize() == xadj.back());
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iAdj = 0; iAdj < xadj.back(); iAdj++)
            adjncy[iAdj] = _METIS::indexToIdx(cell2cellSerialFacial->data()[iAdj]);
        if (adjncy.size() == 0)
            adjncy.resize(1, -1); //*coping with zero sized data

        _METIS::idx_t nCell = _METIS::indexToIdx(cell2cellSerialFacial->Size());
        _METIS::idx_t nCon{1}, options[METIS_NOPTIONS];
        _METIS::METIS_SetDefaultOptions(options);
        {
            options[_METIS::METIS_OPTION_OBJTYPE] = _METIS::METIS_OBJTYPE_CUT;
            options[_METIS::METIS_OPTION_CTYPE] = _METIS::METIS_CTYPE_RM; //? could try shem?
            options[_METIS::METIS_OPTION_IPTYPE] = _METIS::METIS_IPTYPE_GROW;
            options[_METIS::METIS_OPTION_RTYPE] = _METIS::METIS_RTYPE_FM;
            // options[METIS_OPTION_NO2HOP] = 0; // only available in metis 5.1.0
            options[_METIS::METIS_OPTION_NCUTS] = 1;
            options[_METIS::METIS_OPTION_NITER] = 10;
            options[_METIS::METIS_OPTION_UFACTOR] = 30;
            options[_METIS::METIS_OPTION_MINCONN] = 0;
            options[_METIS::METIS_OPTION_CONTIG] = 1; // ! forcing contigious partition now ? necessary?
            options[_METIS::METIS_OPTION_SEED] = 0;   // ! seeding 0 for determined result
            options[_METIS::METIS_OPTION_NUMBERING] = 0;
            options[_METIS::METIS_OPTION_DBGLVL] = _METIS::METIS_DBG_TIME | _METIS::METIS_DBG_IPART;
        }
        std::vector<_METIS::idx_t> partOut(nCell);
        if (nCell == 0)
            partOut.resize(1, -1); //*coping with zero sized data
        if (nPart > 1)
        {
            if (mesh->getMPI().size == 1 || (isSerial && mesh->getMPI().rank == mRank))
            {
                _METIS::idx_t objval;
                int ret = _METIS::METIS_PartGraphKway(
                    &nCell, &nCon, xadj.data(), adjncy.data(), NULL, NULL, NULL,
                    &nPart, NULL, NULL, options, &objval, partOut.data());
                if (ret != _METIS::METIS_OK)
                {
                    DNDS::log() << "METIS returned not OK: [" << ret << "]" << std::endl;
                    DNDS_assert(false);
                }
            }
            else if (mesh->getMPI().size != 1 && (!isSerial))
            {
                ///@todo //TODO: parmetis needs testing!
                for (int i = 0; i < vtxdist.size() - 1; i++)
                    DNDS_assert_info(vtxdist[i + 1] - vtxdist[i] > 0, "need more than zero cells on each proc!");
                std::vector<_METIS::real_t> tpWeights(nPart * nCon, 1.0 / nPart); //! assuming homogenous
                _METIS::real_t ubVec[1]{1.05};
                DNDS_assert(nCon == 1);
                _METIS::idx_t optsC[3];
                _METIS::idx_t wgtflag{0}, numflag{0};
                optsC[0] = 1;
                optsC[1] = 1;
                optsC[2] = 0;
                _METIS::idx_t objval;
                int ret = _METIS::ParMETIS_V3_PartKway(
                    vtxdist.data(), xadj.data(), adjncy.data(), NULL, NULL, &wgtflag, &numflag,
                    &nCon, &nPart, tpWeights.data(), ubVec, optsC, &objval, partOut.data(),
                    &mesh->getMPI().comm);
                if (ret != _METIS::METIS_OK)
                {
                    DNDS::log() << "METIS returned not OK: [" << ret << "]" << std::endl;
                    DNDS_assert(false);
                }
            }
        }
        else
        {
            partOut.assign(partOut.size(), 0);
        }
        cellPartition.resize(cell2cellSerialFacial->Size());
        for (DNDS::index i = 0; i < cellPartition.size(); i++)
            cellPartition[i] = partOut[i];
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Done  MeshPartitionCell2Cell" << std::endl;
    }

    // void UnstructuredMesh::ReorderCellLocal()
    // {
    //     //! currently not used

    //     /****************************/
    //     // got reordering, iCellNew = iPerm[iCell], iCell = perm[iCellNew], A(perm,perm) = Anew

    //     //! remember to add all cell related indices here for altering
    //     // DNDS_assert(this->adjPrimaryState == Adj_PointToLocal);
    //     // DNDS_assert(this->adjFacialState == Adj_PointToLocal);
    //     // DNDS_assert(this->adjC2FState == Adj_PointToLocal);
    // }

    void UnstructuredMesh::ObtainLocalFactFillOrdering(int method)
    {
        cell2cellFaceVLocal = this->GetCell2CellFaceVLocal();
        if (!this->NumCell())
            return;
        if (method == -1)
        {
            localFillOrderingNew2Old.reserve(this->NumCell());
            for (index i = 0; i < this->NumCell(); i++)
                if (i % 2 == 0)
                    localFillOrderingNew2Old.push_back(i);
            for (index i = 0; i < this->NumCell(); i++)
                if (i % 2 != 0)
                    localFillOrderingNew2Old.push_back(i);
            localFillOrderingOld2New.resize(this->NumCell());
            for (index i = 0; i < this->NumCell(); i++)
                localFillOrderingOld2New.at(localFillOrderingNew2Old.at(i)) = i;
        }
        if (method == 1)
        {
            _METIS::idx_t nCell = _METIS::indexToIdx(this->NumCell());
            _METIS::idx_t nCon{1}, options[METIS_NOPTIONS];
            _METIS::METIS_SetDefaultOptions(options);
            {
                options[_METIS::METIS_OPTION_CTYPE] = _METIS::METIS_CTYPE_RM;
                options[_METIS::METIS_OPTION_RTYPE] = _METIS::METIS_RTYPE_FM;
                options[_METIS::METIS_OPTION_IPTYPE] = _METIS::METIS_IPTYPE_EDGE;
                options[_METIS::METIS_OPTION_RTYPE] = _METIS::METIS_RTYPE_SEP1SIDED;
                options[_METIS::METIS_OPTION_NSEPS] = 1;
                options[_METIS::METIS_OPTION_NITER] = 10;
                options[_METIS::METIS_OPTION_UFACTOR] = 30;
                options[_METIS::METIS_OPTION_COMPRESS] = 0; // do not compress
                // options[_METIS::METIS_OPTION_CCORDER] = 0; //use default?
                options[_METIS::METIS_OPTION_SEED] = 0;    // ! seeding 0 for determined result
                options[_METIS::METIS_OPTION_PFACTOR] = 0; // not removing large vertices
                options[_METIS::METIS_OPTION_NUMBERING] = 0;
                // options[_METIS::METIS_OPTION_DBGLVL] = _METIS::METIS_DBG_TIME | _METIS::METIS_DBG_IPART;
            }
            std::vector<std::vector<index>> &cell2cellFaceV = cell2cellFaceVLocal;
            std::vector<_METIS::idx_t> adjncy, xadj, perm, iPerm;
            xadj.resize(nCell + 1);
            xadj[0] = 0;
            for (_METIS::idx_t iC = 0; iC < nCell; iC++)
                xadj[iC + 1] = xadj[iC] + cell2cellFaceV[iC].size(); //! check overflow!
            adjncy.resize(xadj.back());
            for (_METIS::idx_t iC = 0; iC < nCell; iC++)
                std::copy(cell2cellFaceV[iC].begin(), cell2cellFaceV[iC].end(), adjncy.begin() + xadj[iC]);
            perm.resize(nCell);
            iPerm.resize(nCell);

            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling metis" << std::endl;

            int ret = _METIS::METIS_NodeND(&nCell, xadj.data(), adjncy.data(), NULL, options, perm.data(), iPerm.data());
            DNDS_assert_info(ret == _METIS::METIS_OK, fmt::format("Metis return not ok, [{}]", ret));

            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): metis done" << std::endl;

            localFillOrderingNew2Old.resize(nCell);
            localFillOrderingOld2New.resize(nCell);
            for (index i = 0; i < this->NumCell(); i++)
            {
                localFillOrderingNew2Old[i] = perm[i];
                localFillOrderingOld2New[i] = iPerm[i];
            }
        }
        else if (method == 2)
        {
            using namespace boost;
            typedef adjacency_list<vecS, vecS, directedS> Graph;
            Graph cell2cellG(this->NumCell());
            std::vector<std::vector<index>> &cell2cellFaceV = cell2cellFaceVLocal;
            for (index iCell = 0; iCell < this->NumCell(); iCell++)
                for (auto iCOther : cell2cellFaceV[iCell])
                    add_edge(iCell, iCOther, cell2cellG);
            std::vector<index> supernodeSizes(this->NumCell(), 1), degree(this->NumCell(), 0);
            localFillOrderingNew2Old.resize(this->NumCell(), 0);
            localFillOrderingOld2New.resize(this->NumCell(), 0);
            boost::property_map<Graph, vertex_index_t>::type id = get(vertex_index, cell2cellG);
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling boost" << std::endl;
            minimum_degree_ordering(
                cell2cellG,
                make_iterator_property_map(degree.data(), id, degree[0]),
                localFillOrderingOld2New.data(),
                localFillOrderingNew2Old.data(),
                make_iterator_property_map(supernodeSizes.data(), id, supernodeSizes[0]),
                0,
                id);
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): boost done" << std::endl;
        }
    }
}